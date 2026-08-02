"""RQ1 pilot: how much does causal masking at lookahead L damage the content
representation itself — before any accent-conversion model exists?

The argument
------------
Every streaming FAC system is a causal content encoder followed by a converter
followed by a vocoder. If the *encoder* has already destroyed the information
that distinguishes /iy/ from /ih/ by the time it emits a frame, no downstream
converter can recover it. So the encoder's own degradation curve is a **lower
bound on the lookahead requirement** of the whole system, and it is measurable
today: no trained converter, no golden targets, no listening test.

Concretely, for real L2-ARCTIC speech we compute WavLM layer-9 features under

  (a) full bidirectional attention  -> the reference the offline literature uses
  (b) chunked causal attention with lookahead L, for L in 0..640 ms

and measure how far (b) drifts from (a). Then we look for a knee.

Two methodological findings this script exists to expose
--------------------------------------------------------
**Masking attention does not make WavLM causal. There are two other leaks.**

1. **The positional convolution.** `pos_conv_embed` is a depthwise Conv1d with
   kernel 128 and symmetric padding 64. At a 20 ms frame rate that is
   **1.28 s of future** entering every frame, before the transformer stack is
   reached. Fixed by left-only padding.

2. **The feature-encoder GroupNorm.** wav2vec2/WavLM *base* checkpoints use
   `feat_extract_norm="group"`: `nn.GroupNorm(num_groups=C, num_channels=C)`
   over a (B, C, T) tensor normalises each channel by statistics computed over
   the **entire utterance**. Every output frame depends on every input frame —
   an unbounded, global leak underneath everything else. Fixed by cumulative
   (running) normalisation, applied identically in the bidirectional reference
   so the sweep stays unconfounded. (The *-large* checkpoints use
   `feat_extract_norm="layer"`, which is per-frame and causal-safe.)

Neither is mentioned in the streaming-VC/AC papers we have read. A system that
masks self-attention and calls itself causal, on a base-sized SSL encoder, has
an unreported second of lookahead plus an utterance-global dependency.

We verify rather than assert: `--selftest` runs a **truncation proof** — delete
the audio after frame t, check whether the output at frame t-L-1 changed — over
four ablations (mask only / +pos_conv / +groupnorm / both), so any residual
leak is attributed to a named component instead of just failing.

Outputs
-------
results/raw/content_degradation_<tag>.csv   one row per (L, utterance)
results/raw/content_degradation_<tag>_summary.json

Usage
-----
    python3 bench/bench_content_degradation.py --selftest
    python3 bench/bench_content_degradation.py --n-utts 48 --tag m4
    python3 bench/bench_content_degradation.py --n-utts 200 --device cuda --tag gpu
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from sfac.causal import StreamGeometry, chunked_lookahead_mask  # noqa: E402
from sfac.latency import StageTimer, write_csv                  # noqa: E402
from phoneme_analysis import find_knee                          # noqa: E402


SWEEP_L = (0, 20, 40, 80, 160, 320, 640)
LAYER = 9          # WavLM layer 9: the usual phonetic-content tap
FRAME_MS = 20.0


# ==========================================================================
# Making WavLM actually causal
# ==========================================================================

def make_group_norm_causal(model) -> int:
    """Replace the feature encoder's GroupNorm with a cumulative (causal) one.

    THE SECOND LEAK, and the one nobody talks about.

    wav2vec2/WavLM *base* checkpoints use `feat_extract_norm="group"`: the
    first convolutional layer is followed by
    `nn.GroupNorm(num_groups=C, num_channels=C)` applied to a (B, C, T) tensor.
    With one group per channel, each channel is normalised by its mean and
    variance **over the entire time axis**. Every output frame therefore
    depends on every input frame, including all future ones — a global,
    utterance-level dependency sitting underneath the transformer.

    Consequence: masking attention and fixing the positional convolution is
    still not enough. A wav2vec2/WavLM-base encoder cannot be made causal at
    all while this layer is present. (The *large* checkpoints use
    `feat_extract_norm="layer"`, which is per-frame and causal-safe — that
    difference is itself worth a sentence in the paper.)

    We replace it with a cumulative normalisation: frame t is normalised by the
    running mean and variance over frames 0..t. That is what an actual
    streaming deployment must do, it is exactly causal, and — critically — we
    apply it in the **bidirectional reference condition too**, so the only
    thing differing across the sweep remains the attention mask.

    Returns the number of channels re-normalised (0 if the model has no
    GroupNorm, i.e. a layer-norm checkpoint).
    """
    import torch
    import torch.nn as nn

    # Defined here, not at module scope: `class _Fn(torch.nn.Module)` at import
    # time forces a torch import just to load this file, which broke --help and
    # defeats the promise that the bench tools run on a torch-less machine
    # (a Raspberry Pi, or any box where the CUDA wheel cannot load).
    class _Fn(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x):
            return self.fn(x)

    fe = model.feature_extractor
    n = 0
    for layer in fe.conv_layers:
        gn = getattr(layer, "layer_norm", None)
        if not isinstance(gn, nn.GroupNorm):
            continue
        weight, bias, eps = gn.weight, gn.bias, gn.eps

        def cumulative_norm(x, _w=weight, _b=bias, _eps=eps):
            # x: (B, C, T)
            t = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
            csum = x.cumsum(-1)
            csq = (x * x).cumsum(-1)
            mean = csum / t
            var = (csq / t - mean * mean).clamp_min(0.0)
            y = (x - mean) / torch.sqrt(var + _eps)
            return y * _w[None, :, None] + _b[None, :, None]

        layer.layer_norm = _Fn(cumulative_norm)
        n += gn.num_channels
    return n


def make_pos_conv_causal(model) -> int:
    """Left-pad the positional conv so it cannot see the future.

    HF implements `pos_conv_embed` as Conv1d(k=128, groups=16, padding=64)
    followed by `WavLMSamePadLayer`, which trims one element when the kernel is
    even. Net effect: symmetric context, ~64 frames each side.

    We rewrite it as: pad (k-1) on the LEFT only, zero on the right, and drop
    the trim. Output length is preserved and every output position depends only
    on positions <= itself.

    Returns the number of future frames that were leaking, for the record.
    """
    import torch
    import torch.nn as nn

    pce = model.encoder.pos_conv_embed
    conv = pce.conv
    k = conv.kernel_size[0]
    leak = k // 2

    # neutralise the existing padding + same-pad trim
    conv.padding = (0,)
    if hasattr(pce, "padding"):
        pce.padding = nn.Identity()

    orig_forward = pce.forward

    def causal_forward(hidden_states):
        # (B, T, D) -> (B, D, T)
        x = hidden_states.transpose(1, 2)
        x = nn.functional.pad(x, (k - 1, 0))     # left-only
        x = conv(x)
        x = pce.activation(x)
        return x.transpose(1, 2)

    pce.forward = causal_forward
    return leak


NEG = -1e9   # not finfo.min: that underflows once a position bias is added


class MaskInjector:
    """Force every encoder layer to attend only within the lookahead window.

    Finding the right injection point is version-dependent and getting it
    wrong is silent, so this probes rather than assumes.

    Current `transformers` implements `WavLMAttention` on top of
    `F.multi_head_attention_forward`, and passes the layer's `attention_mask`
    as **key_padding_mask** -- which must be 2-D (B, T). Handing it a 4-D
    additive mask raises, which is how we found this. But that same call
    already passes `gated_position_bias`, shape (B*H, T, T), as the *additive*
    `attn_mask` argument. So the correct injection is to **add** our lookahead
    mask to the gated position bias: same tensor, same pre-softmax slot, and
    the model's own padding mask keeps working untouched.

    Older builds took an additive 4-D `attention_mask` on the encoder layer.
    We fall back to that if `torch_multi_head_self_attention` is absent.
    """

    def __init__(self, model, geom: Optional[StreamGeometry]):
        import torch
        self.model = model
        self.geom = geom
        self._cache: Dict[Tuple[int, str], "torch.Tensor"] = {}
        self._orig = []
        if geom is None:
            self.strategy = "none"
            return
        layers = list(model.encoder.layers)
        if hasattr(layers[0].attention, "torch_multi_head_self_attention"):
            self.strategy = "position_bias"
            for layer in layers:
                att = layer.attention
                self._orig.append((att, "torch_multi_head_self_attention",
                                   att.torch_multi_head_self_attention))
                att.torch_multi_head_self_attention = self._wrap_mha(
                    att.torch_multi_head_self_attention)
        else:
            self.strategy = "layer_attention_mask"
            for layer in layers:
                self._orig.append((layer, "forward", layer.forward))
                layer.forward = self._wrap_layer(layer.forward)

    def _mask2d(self, T: int, dtype, device):
        """(T, T) additive mask. Broadcasts over the (B*H, T, T) position bias."""
        import torch
        key = (T, f"{device}{dtype}")
        m = self._cache.get(key)
        if m is None:
            bm = chunked_lookahead_mask(
                T,
                chunk_frames=self.geom.chunk_frames,
                lookahead_frames=self.geom.lookahead_frames,
                lookback_frames=self.geom.lookback_frames,
            )
            m = torch.from_numpy(np.where(bm, 0.0, NEG)).to(dtype=dtype, device=device)
            self._cache[key] = m
        return m

    def _wrap_mha(self, orig):
        def mha(hidden_states, attention_mask, gated_position_bias,
                output_attentions, *a, **kw):
            m = self._mask2d(hidden_states.shape[1], gated_position_bias.dtype,
                             gated_position_bias.device)
            return orig(hidden_states, attention_mask,
                        gated_position_bias + m, output_attentions, *a, **kw)
        return mha

    def _wrap_layer(self, orig):
        def fwd(hidden_states, attention_mask=None, *a, **kw):
            m = self._mask2d(hidden_states.shape[1], hidden_states.dtype,
                             hidden_states.device)[None, None]
            return orig(hidden_states, m, *a, **kw)
        return fwd

    def restore(self):
        for obj, name, f in self._orig:
            setattr(obj, name, f)
        self._orig = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()


def load_encoder(name: str = "microsoft/wavlm-base-plus", device: str = "cpu",
                 causal_pos_conv: bool = True, causal_group_norm: bool = True):
    """Load WavLM and (optionally) make it genuinely causal.

    Returns (model, info) where info records both leaks so every result file
    carries the provenance of its own causality.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoModel
    model = AutoModel.from_pretrained(name, attn_implementation="eager")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    k = model.encoder.pos_conv_embed.conv.kernel_size[0]
    info = {
        "pos_conv_kernel": int(k),
        "pos_conv_leak_frames": int(k // 2),
        "pos_conv_leak_ms": (k // 2) * FRAME_MS,
        "pos_conv_fixed": bool(causal_pos_conv),
        "has_group_norm": any(
            isinstance(getattr(l, "layer_norm", None), nn.GroupNorm)
            for l in model.feature_extractor.conv_layers),
        "group_norm_fixed": False,
        "group_norm_channels": 0,
    }
    if causal_pos_conv:
        make_pos_conv_causal(model)
    if causal_group_norm:
        n = make_group_norm_causal(model)
        info["group_norm_fixed"] = n > 0
        info["group_norm_channels"] = n
    return model, info


def features(model, wav: np.ndarray, geom: Optional[StreamGeometry],
             device: str, layer: int = LAYER):
    """WavLM hidden states at `layer` under the given lookahead geometry.

    geom=None -> unmasked (full bidirectional attention), the reference.
    """
    import torch
    x = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    with torch.no_grad(), MaskInjector(model, geom):
        out = model(x, output_hidden_states=True)
    return out.hidden_states[layer].squeeze(0)     # (T, D)


# ==========================================================================
# Causality self-test — the part that proves the numbers mean anything
# ==========================================================================

def causality_selftest(device: str = "cpu", tol: float = 1e-4,
                       encoder: str = "microsoft/wavlm-base-plus") -> Dict[str, object]:
    """Deleting future audio must not change past outputs.

    This is a proof, not a smoke test: every lookahead label in the study is
    wrong unless it passes.

    Four ablations, so the report attributes any leak to a named component
    instead of just reporting "not causal":

        mask_only        attention masked, nothing else patched
        mask_posconv     + causal positional convolution
        mask_groupnorm   + cumulative feature-encoder normalisation
        mask_both        both -- the configuration the sweep actually uses

    Expectation: only `mask_both` passes. If `mask_only` passes, the test is
    broken, not WavLM.
    """
    import torch
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(16_000 * 4).astype(np.float32) * 0.1
    cut = 16_000 * 2                       # delete the second half
    geom = StreamGeometry(chunk_ms=20, lookahead_ms=0, lookback_ms=None)

    variants = {"mask_only": (False, False), "mask_posconv": (True, False),
                "mask_groupnorm": (False, True), "mask_both": (True, True)}
    report: Dict[str, object] = {"encoder": encoder, "tolerance": tol}
    for name, (pc, gn) in variants.items():
        model, info = load_encoder(encoder, device, causal_pos_conv=pc,
                                   causal_group_norm=gn)
        with MaskInjector(model, geom) as mi:
            strategy = mi.strategy
        f_full = features(model, wav, geom, device)
        f_cut = features(model, wav[:cut], geom, device)
        n = f_cut.shape[0]
        margin = 8      # covers the conv frontend's own 25 ms receptive field
        a, b = f_full[: n - margin], f_cut[: n - margin]
        rel = float((a - b).norm() / a.norm())
        report[name] = {
            "mask_injection_strategy": strategy,
            "pos_conv_causal": pc, "group_norm_causal": gn,
            "pos_conv_kernel": info["pos_conv_kernel"],
            "pos_conv_leak_ms_if_unpatched": info["pos_conv_leak_ms"],
            "has_group_norm": info["has_group_norm"],
            "max_abs_delta": float((a - b).abs().max()),
            "relative_l2_delta": rel,
            "causal": bool(rel < tol),
        }
        del model

    both = report["mask_both"]["causal"]
    only = report["mask_only"]["causal"]
    k = report["mask_both"]["pos_conv_kernel"]
    if both and not only:
        verdict = (
            f"PASS. Masking attention alone does NOT make {encoder} causal. The "
            f"positional convolution (kernel {k}) and the feature-encoder "
            "GroupNorm each leak future context; with both patched the "
            "truncation proof passes and the lookahead labels are valid.")
    elif both and only:
        verdict = ("SUSPECT: the unpatched model passes too, so the truncation "
                   "test is probably not exercising the leak. Fix the test "
                   "before trusting any sweep.")
    else:
        verdict = (
            "FAIL: still not causal with both patches (relative L2 "
            f"{report['mask_both']['relative_l2_delta']:.2e} > {tol:g}). Do NOT "
            "run the sweep. Look for a third non-causal op, or switch to a "
            "layer-norm checkpoint (the -large variants use "
            "feat_extract_norm='layer', which is per-frame and causal-safe).")
    report["verdict"] = verdict
    # legacy keys so downstream summaries keep working
    report["with_fix"] = report["mask_both"]
    report["without_fix"] = report["mask_only"]
    return report


# ==========================================================================
# Frame bucketing (proxy for phoneme class, no aligner required)
# ==========================================================================

def frame_acoustics(wav: np.ndarray, n_frames: int, sr: int = 16_000) -> Dict[str, np.ndarray]:
    """Per-frame voicing and spectral-flux proxies, aligned to WavLM's rate.

    We do not have time-aligned phone labels (the HF L2-ARCTIC release ships
    IPA strings, not TextGrids), so a true per-phone attribution needs forced
    alignment and is deferred to the full study. What we can compute now is an
    acoustic proxy that separates the two populations H2 cares about:

      * voiced + spectrally steady   ~ vowels, nasals, approximants
      * unvoiced or spectrally rapid ~ stops, fricatives, affricates, transitions

    H2 predicts the first group degrades faster as lookahead shrinks, because
    formant trajectories need right context while local obstruent gestures do
    not. That is a weaker test than the annotated one, and it is labelled as a
    proxy everywhere it is reported.
    """
    hop = int(sr * FRAME_MS / 1000)
    win = 2 * hop
    T = n_frames
    voiced = np.zeros(T, np.float32)
    energy = np.zeros(T, np.float32)
    spec = np.zeros((T, win // 2 + 1), np.float32)
    hann = np.hanning(win).astype(np.float32)
    for t in range(T):
        s = t * hop
        seg = wav[s:s + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        seg = seg * hann
        energy[t] = float(np.sqrt((seg ** 2).mean()) + 1e-9)
        S = np.abs(np.fft.rfft(seg))
        spec[t] = S
        # voicing via normalised autocorrelation peak in the 60-400 Hz range
        ac = np.correlate(seg, seg, "full")[win - 1:]
        ac = ac / (ac[0] + 1e-9)
        lo, hi = sr // 400, sr // 60
        voiced[t] = float(ac[lo:hi].max()) if hi < len(ac) else 0.0
    sn = spec / (np.linalg.norm(spec, axis=1, keepdims=True) + 1e-9)
    flux = np.zeros(T, np.float32)
    flux[1:] = np.linalg.norm(sn[1:] - sn[:-1], axis=1)
    return {"voiced": voiced, "flux": flux, "energy": energy}


def bucket_frames(ac: Dict[str, np.ndarray]) -> np.ndarray:
    """-> array of {'silence','sonorant_steady','obstruent_or_transient'}."""
    e, v, f = ac["energy"], ac["voiced"], ac["flux"]
    thr_e = max(1e-4, float(np.percentile(e, 20)))
    med_f = float(np.median(f))
    out = np.empty(len(e), dtype=object)
    for i in range(len(e)):
        if e[i] < thr_e:
            out[i] = "silence"
        elif v[i] > 0.35 and f[i] < med_f:
            out[i] = "sonorant_steady"
        else:
            out[i] = "obstruent_or_transient"
    return out


# ==========================================================================
# Divergence metrics
# ==========================================================================

def frame_divergence(ref, hyp) -> np.ndarray:
    """1 - cosine similarity, per frame. 0 = identical, 1 = orthogonal."""
    import torch
    n = min(ref.shape[0], hyp.shape[0])
    a, b = ref[:n], hyp[:n]
    return (1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)).cpu().numpy()


def linear_cka(X, Y) -> float:
    """Centred-kernel-alignment between two frame x dim matrices.

    Cosine-per-frame answers "did the representation move?". CKA answers
    "did the *geometry* of the representation change?" -- i.e. are the same
    frames still near each other. A converter can absorb a global rotation
    (high per-frame drift, high CKA) but not a collapse of phonetic structure
    (high drift, low CKA). Reporting both distinguishes those.
    """
    import torch
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n].double(), Y[:n].double()
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = (X.T @ Y).norm(p="fro") ** 2
    xx = (X.T @ X).norm(p="fro")
    yy = (Y.T @ Y).norm(p="fro")
    return float(xty / (xx * yy + 1e-12))


# ==========================================================================
# Data
# ==========================================================================

def decode_audio(a) -> Tuple[np.ndarray, int]:
    """Return (mono float32 waveform, sample_rate) from an HF audio column.

    Four shapes to handle, because `datasets` changed its Audio representation
    and the new one has a hard third-party dependency:

      * <4.0            -> dict with 'array' / 'sampling_rate'
      * >=4.0           -> torchcodec AudioDecoder (needs torchcodec + FFmpeg)
      * decode=False    -> dict with raw 'bytes' (our fallback; no torchcodec)
      * dict with 'path'-> read from disk

    The `bytes` path is what `undecoded_audio()` below uses: it sidesteps
    torchcodec entirely, which matters because torchcodec needs a matching
    FFmpeg and is a common install failure on macOS.
    """
    if isinstance(a, dict) and a.get("array") is not None:
        return np.asarray(a["array"], dtype=np.float32), int(a["sampling_rate"])
    if hasattr(a, "get_all_samples"):                      # torchcodec
        s = a.get_all_samples()
        w = s.data
        w = w.mean(0) if w.ndim > 1 else w
        return w.numpy().astype(np.float32), int(s.sample_rate)
    if isinstance(a, dict) and a.get("bytes"):
        import io
        import soundfile as sf
        w, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
        return (w.mean(1) if w.ndim > 1 else w).astype(np.float32), int(sr)
    if isinstance(a, dict) and a.get("path"):
        import soundfile as sf
        w, sr = sf.read(a["path"], dtype="float32", always_2d=False)
        return (w.mean(1) if w.ndim > 1 else w).astype(np.float32), int(sr)
    raise TypeError(f"unrecognised audio column type: {type(a)}")


def open_l2arctic(split: str = "scripted"):
    """load_dataset with a torchcodec-free fallback.

    `datasets>=4` decodes audio through torchcodec, which needs a compatible
    FFmpeg and frequently is not installed. Rather than make the whole pilot
    depend on that, try the normal path once and, if decoding raises, re-open
    with `decode=False` and hand raw bytes to soundfile.
    """
    from datasets import load_dataset
    ds = load_dataset("KoelLabs/L2Arctic")[split]
    try:
        decode_audio(ds[0]["audio"])
        return ds, "decoded"
    except Exception as e:
        print(f"  audio auto-decode unavailable ({type(e).__name__}); "
              f"falling back to raw bytes + soundfile")
        from datasets import Audio
        return ds.cast_column("audio", Audio(decode=False)), "bytes"


def load_l2arctic(n_utts: int, seed: int = 0, split: str = "scripted",
                  l1_filter: Optional[List[str]] = None):
    """KoelLabs/L2Arctic: audio + ipa (produced) + g2p (canonical) + speaker meta.

    Note this HF release is the **phoneme-annotated subset** (3599 scripted
    utterances), not the full 26,867-utterance corpus. That is exactly the
    subset the H2 analysis depends on, so its size is worth recording: ~600
    utterances per L1 across six L1s. Licence is CC-BY-NC-4.0 -- non-commercial,
    which matters for the model-weights release decision.
    """
    # The repo is gated (auto-approve), so an HF token must be present.
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))):
        raise RuntimeError(
            "KoelLabs/L2Arctic is gated. Run `hf auth login` once, or "
            "export HF_TOKEN=hf_...  (accept the terms on the dataset page first).")
    ds, _mode = open_l2arctic(split)
    idx = list(range(len(ds)))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    out = []
    per_l1: Dict[str, int] = defaultdict(int)
    cap = max(1, n_utts // max(1, len(l1_filter or [1] * 6)))
    for i in idx:
        r = ds[int(i)]
        l1 = r.get("speaker_native_language", "?")
        if l1_filter and l1 not in l1_filter:
            continue
        if per_l1[l1] >= cap:
            continue
        wav, sr = decode_audio(r["audio"])
        if sr != 16_000:
            g = math.gcd(sr, 16_000)
            from scipy.signal import resample_poly
            wav = resample_poly(wav, 16_000 // g, sr // g).astype(np.float32)
        if len(wav) < 16_000 or len(wav) > 16_000 * 12:
            continue
        out.append({
            "wav": wav,
            "l1": l1,
            "speaker": r.get("speaker_code", "?"),
            "text": r.get("text", ""),
            "ipa": r.get("ipa", ""),
            "g2p": r.get("g2p", ""),
        })
        per_l1[l1] += 1
        if len(out) >= n_utts:
            break
    return out, dict(per_l1), len(ds)


def synthetic_fallback(n: int):
    """Used only when the dataset is unavailable, so --selftest always runs."""
    rng = np.random.default_rng(0)
    out = []
    for k in range(n):
        t = np.arange(int(2.5 * 16_000)) / 16_000
        f0 = 110 + 40 * np.sin(2 * np.pi * (0.6 + 0.1 * k) * t)
        ph = 2 * np.pi * np.cumsum(f0) / 16_000
        x = sum((1.0 / h) * np.sin(h * ph) for h in range(1, 14))
        env = (0.5 * (1 + np.sin(2 * np.pi * 3.0 * t))) ** 2
        x = x * env + 0.02 * rng.standard_normal(len(t))
        out.append({"wav": (0.3 * x / np.abs(x).max()).astype(np.float32),
                    "l1": "SYNTHETIC", "speaker": f"syn{k}", "text": "", "ipa": "", "g2p": ""})
    return out, {"SYNTHETIC": n}, n


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-utts", type=int, default=48)
    ap.add_argument("--chunk-ms", type=float, default=40.0)
    ap.add_argument("--lookback-ms", type=float, default=2000.0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--tag", default="local")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--encoder", default="microsoft/wavlm-base-plus")
    ap.add_argument("--lookaheads", type=float, nargs="+", default=None,
                    help="explicit L grid in ms; overrides the default sweep")
    ap.add_argument("--dense", action="store_true",
                    help="dense grid 0..200 ms in 10 ms steps, plus 320/640. "
                         "Tests whether a knee narrower than one octave is "
                         "hiding between the geometric sample points -- the "
                         "one caveat that could overturn the no-knee finding.")
    a = ap.parse_args()

    global SWEEP_L
    if a.dense:
        SWEEP_L = tuple(list(range(0, 201, 10)) + [240, 280, 320, 480, 640])
    elif a.lookaheads:
        SWEEP_L = tuple(a.lookaheads)

    outdir = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
    os.makedirs(outdir, exist_ok=True)

    # ---------------- causality self-test ----------------
    print("=" * 68)
    print("CAUSALITY SELF-TEST  (does attention masking alone make WavLM causal?)")
    print("=" * 68)
    st = causality_selftest(device="cpu")
    print(json.dumps(st, indent=2))
    with open(os.path.join(outdir, f"causality_selftest_{a.tag}.json"), "w") as f:
        json.dump(st, f, indent=2)
    if a.selftest:
        return

    # ---------------- data ----------------
    if a.synthetic:
        utts, per_l1, total = synthetic_fallback(a.n_utts)
    else:
        try:
            utts, per_l1, total = load_l2arctic(a.n_utts)
        except Exception as e:
            print(f"\nL2-ARCTIC load failed ({type(e).__name__}: {e})")
            print("Falling back to synthetic probe audio. Results labelled SYNTHETIC.")
            utts, per_l1, total = synthetic_fallback(a.n_utts)
    print(f"\n{len(utts)} utterances (corpus split size {total}); per L1: {per_l1}")

    # ---------------- sweep ----------------
    model, encinfo = load_encoder(a.encoder, a.device, causal_pos_conv=True,
                                  causal_group_norm=True)
    print(f"encoder {a.encoder} on {a.device}")
    print(f"  pos_conv kernel {encinfo['pos_conv_kernel']} -> would leak "
          f"{encinfo['pos_conv_leak_ms']:.0f} ms; patched to left-only padding")
    print(f"  feature-encoder GroupNorm present={encinfo['has_group_norm']} "
          f"-> replaced with cumulative norm on {encinfo['group_norm_channels']} channels")
    if not st["mask_both"]["causal"]:
        print("\n!! The causality proof FAILED. Lookahead labels are not valid. "
              "Results below are recorded but must not be published.\n")

    timer = StageTimer()
    rows: List[Dict[str, object]] = []
    by_L: Dict[float, List[float]] = defaultdict(list)
    by_L_cka: Dict[float, List[float]] = defaultdict(list)
    by_L_bucket: Dict[float, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_L_l1: Dict[float, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    t_start = time.time()
    for ui, u in enumerate(utts):
        ref = features(model, u["wav"], None, a.device)           # bidirectional
        ac = frame_acoustics(u["wav"], ref.shape[0])
        buckets = bucket_frames(ac)
        for L in SWEEP_L:
            geom = StreamGeometry(chunk_ms=a.chunk_ms, lookahead_ms=L,
                                  lookback_ms=a.lookback_ms)
            with timer.stage(f"fwd_L{L}"):
                hyp = features(model, u["wav"], geom, a.device)
            d = frame_divergence(ref, hyp)
            cka = linear_cka(ref, hyp)
            n = min(len(d), len(buckets))
            row = {
                "utt": ui, "speaker": u["speaker"], "l1": u["l1"],
                "lookahead_ms": L, "chunk_ms": a.chunk_ms,
                "t_algorithmic_ms": geom.algorithmic_ms,
                "n_frames": int(n),
                "divergence_mean": float(d[:n].mean()),
                "divergence_p95": float(np.percentile(d[:n], 95)),
                "cka_to_bidirectional": cka,
            }
            for b in ("silence", "sonorant_steady", "obstruent_or_transient"):
                m = buckets[:n] == b
                row[f"div_{b}"] = float(d[:n][m].mean()) if m.any() else float("nan")
                if m.any():
                    by_L_bucket[L][b].append(row[f"div_{b}"])
            rows.append(row)
            by_L[L].append(row["divergence_mean"])
            by_L_cka[L].append(cka)
            by_L_l1[L][u["l1"]].append(row["divergence_mean"])
        if (ui + 1) % 8 == 0 or ui == len(utts) - 1:
            el = time.time() - t_start
            print(f"  {ui+1}/{len(utts)} utts  ({el:.0f}s, "
                  f"{el/(ui+1):.1f}s/utt)")

    # ---------------- summarise ----------------
    Ls = list(SWEEP_L)
    mean_div = [float(np.mean(by_L[L])) for L in Ls]
    mean_cka = [float(np.mean(by_L_cka[L])) for L in Ls]
    knee_div = find_knee(Ls, mean_div)
    knee_cka = find_knee(Ls, [1 - c for c in mean_cka])

    # bootstrap CI on the knee -- 7 points gives a wide interval and the paper
    # must say so rather than quoting a point estimate
    rng = np.random.default_rng(0)
    boots = []
    n_utt = len(utts)
    for _ in range(2000):
        pick = rng.integers(0, n_utt, n_utt)
        curve = [float(np.mean([by_L[L][i] for i in pick])) for L in Ls]
        boots.append(find_knee(Ls, curve)["knee_ms"])
    boots = np.array([b for b in boots if b == b])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    modal = int(np.bincount([Ls.index(b) for b in boots]).argmax()) if len(boots) else 0

    bucket_curves = {
        b: [float(np.mean(by_L_bucket[L][b])) if by_L_bucket[L][b] else float("nan")
            for L in Ls]
        for b in ("sonorant_steady", "obstruent_or_transient")
    }
    son, obs = bucket_curves["sonorant_steady"], bucket_curves["obstruent_or_transient"]
    gain = lambda c: (c[0] - c[-1]) / c[0] if c[0] else float("nan")  # noqa: E731

    summary = {
        "tag": a.tag,
        "device": a.device,
        "encoder": a.encoder,
        "n_utts": len(utts),
        "per_l1": per_l1,
        "corpus_split_size": total,
        "chunk_ms": a.chunk_ms,
        "lookback_ms": a.lookback_ms,
        "layer": LAYER,
        "encoder_causality": encinfo,
        "causality_selftest": st,
        "lookaheads_ms": Ls,
        "divergence_from_bidirectional": dict(zip(map(str, Ls), mean_div)),
        "cka_to_bidirectional": dict(zip(map(str, Ls), mean_cka)),
        "knee_divergence": knee_div,
        "knee_cka": knee_cka,
        "knee_bootstrap_ci95_ms": ci,
        "knee_bootstrap_modal_ms": Ls[modal],
        "bucket_curves": bucket_curves,
        "relative_gain_sonorant_steady": gain(son),
        "relative_gain_obstruent_transient": gain(obs),
        "h2_proxy_direction_supported": bool(gain(son) > gain(obs)),
        "per_l1_curves": {
            l1: [float(np.mean(by_L_l1[L][l1])) for L in Ls]
            for l1 in sorted({u["l1"] for u in utts})
        },
        "forward_latency_ms": timer.summary(drop_warmup=3),
        "wall_seconds": round(time.time() - t_start, 1),
    }

    write_csv(os.path.join(outdir, f"content_degradation_{a.tag}.csv"), rows)
    with open(os.path.join(outdir, f"content_degradation_{a.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---------------- report ----------------
    print("\n" + "=" * 68)
    print("CONTENT DEGRADATION vs LOOKAHEAD  (WavLM-base-plus layer 9)")
    print("=" * 68)
    print(f"{'L (ms)':>8}{'t_algo':>9}{'divergence':>13}{'CKA':>9}"
          f"{'sonorant':>11}{'obstruent':>11}")
    for i, L in enumerate(Ls):
        print(f"{L:>8}{StreamGeometry(a.chunk_ms, L).algorithmic_ms:>9.0f}"
              f"{mean_div[i]:>13.4f}{mean_cka[i]:>9.4f}"
              f"{son[i]:>11.4f}{obs[i]:>11.4f}")
    print(f"\nknee (divergence): {knee_div['knee_ms']:.0f} ms   "
          f"bootstrap 95% CI [{ci[0]:.0f}, {ci[1]:.0f}] ms, modal {Ls[modal]} ms")
    print(f"knee (1-CKA):      {knee_cka['knee_ms']:.0f} ms")
    print(f"relative gain 0->640ms:  sonorant {gain(son):.3f}  "
          f"obstruent {gain(obs):.3f}  ->  H2-proxy "
          f"{'SUPPORTED' if gain(son) > gain(obs) else 'NOT supported'}")
    print(f"\nwrote results/raw/content_degradation_{a.tag}.csv")


if __name__ == "__main__":
    main()
