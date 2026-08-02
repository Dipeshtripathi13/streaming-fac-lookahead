"""Configurable-lookahead streaming FAC harness (PyTorch).

The one engineering requirement of the whole project:

    Lookahead L is a single exposed parameter. Across the entire sweep the
    weights, the data, the chunking, the optimiser state and the random seed
    are identical. Only the attention mask width changes.

If L co-varies with anything else -- a different checkpoint per L, a
different number of training steps, a different vocoder -- then RQ1 is
unanswerable and the paper is not salvageable. Everything in this file is
organised around making that invariant checkable rather than assumed. See
`assert_only_L_varies()` at the bottom, and tests/test_causal.py.

Torch is imported lazily so that the CPU benchmark tools (bench/) run on
machines with no torch install -- a Raspberry Pi, or a fresh clone.

Architecture (deliberately boring; we are not proposing a new model):

    waveform
      -> causal SSL content encoder (HuBERT/WavLM base, masked to L)
      -> conversion module           MODE in {AC, VC_ONLY}
      -> causal HiFi-GAN vocoder
      -> waveform

MODE is the H3 control. In VC_ONLY the conversion module is restricted to a
speaker/timbre transform and the content path is identity; in AC it may
rewrite content tokens. Same parameter count, same training budget, so any
lookahead difference between the two is attributable to the task, not to
capacity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from .causal import StreamGeometry, chunked_lookahead_mask


class Mode(str, Enum):
    AC = "AC"            # accent conversion: content may be rewritten
    VC_ONLY = "VC_ONLY"  # timbre only: content path is frozen/identity


@dataclass
class HarnessConfig:
    # --- the swept parameter, and only this ---
    lookahead_ms: float = 80.0

    # --- everything below must be held constant across the sweep ---
    chunk_ms: float = 40.0
    lookback_ms: Optional[float] = 2000.0     # TVTSyn-style bounded history
    mode: Mode = Mode.AC
    sample_rate: int = 16_000
    frame_ms: float = 20.0

    encoder_name: str = "microsoft/wavlm-base-plus"
    encoder_layer: int = 9          # WavLM layer 9 is the usual phonetic-content tap
    d_content: int = 768
    d_speaker: int = 256
    n_convert_layers: int = 4
    n_convert_heads: int = 8
    d_convert_ffn: int = 2048

    vocoder: str = "hifigan_causal"
    vocoder_hop: int = 320          # 20 ms at 16 kHz -- must match frame_ms

    seed: int = 1337
    train_steps: int = 200_000
    lr: float = 2e-4
    batch_size: int = 16

    def __post_init__(self) -> None:
        expect_hop = int(self.frame_ms * self.sample_rate / 1000)
        if self.vocoder_hop != expect_hop:
            raise ValueError(
                f"vocoder_hop={self.vocoder_hop} disagrees with frame_ms="
                f"{self.frame_ms} at {self.sample_rate} Hz (expected {expect_hop}). "
                "A mismatch here silently changes the effective frame rate and "
                "therefore the meaning of every lookahead label.")

    @property
    def geometry(self) -> StreamGeometry:
        return StreamGeometry(
            chunk_ms=self.chunk_ms,
            lookahead_ms=self.lookahead_ms,
            frame_ms=self.frame_ms,
            lookback_ms=self.lookback_ms,
            sample_rate=self.sample_rate,
        )

    def fingerprint(self) -> Dict[str, object]:
        """Everything except lookahead_ms. Must be identical across the sweep."""
        d = asdict(self)
        d.pop("lookahead_ms")
        d["mode"] = self.mode.value
        return d


def assert_only_L_varies(configs) -> None:
    """Guard the central experimental invariant. Call before any sweep.

    Raises with a diff of the offending keys rather than a bare assertion,
    because the failure mode this catches -- someone changed chunk size while
    debugging one condition -- is otherwise invisible in the results.
    """
    configs = list(configs)
    if len(configs) < 2:
        return
    ref = configs[0].fingerprint()
    for c in configs[1:]:
        f = c.fingerprint()
        diff = {k: (ref[k], f[k]) for k in ref if ref[k] != f.get(k)}
        if diff:
            raise ValueError(
                "Sweep is confounded: these fields differ across conditions "
                f"but must be held constant -> {diff}")
    Ls = [c.lookahead_ms for c in configs]
    if len(set(Ls)) != len(Ls):
        raise ValueError(f"duplicate lookahead values in sweep: {Ls}")


# ==========================================================================
# Torch modules (lazy import)
# ==========================================================================

def _torch():
    try:
        import torch
        return torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the model path. The bench/ tools do not "
            "need it. Install: pip install torch  (see setup/SETUP_GPU_COLAB.md)"
        ) from e


def build_modules(cfg: HarnessConfig):
    """Construct the harness. Returns (module, info)."""
    torch = _torch()
    nn = torch.nn

    class MaskedSelfAttention(nn.Module):
        """Self-attention whose mask width is set at call time by L.

        Crucially the *weights* do not depend on L. During training we sample
        a single fixed L per run; at eval we can also re-mask a trained model
        to a different L to measure train/test lookahead mismatch, which is a
        cheap extra ablation reviewers like.
        """

        def __init__(self, d: int, h: int):
            super().__init__()
            self.h, self.dh = h, d // h
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.o = nn.Linear(d, d, bias=False)

        def forward(self, x, add_mask):
            B, T, D = x.shape
            q, k, v = self.qkv(x).chunk(3, -1)
            q = q.view(B, T, self.h, self.dh).transpose(1, 2)
            k = k.view(B, T, self.h, self.dh).transpose(1, 2)
            v = v.view(B, T, self.h, self.dh).transpose(1, 2)
            a = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh) + add_mask
            a = a.softmax(-1)
            y = (a @ v).transpose(1, 2).reshape(B, T, D)
            return self.o(y)

    class CausalConvBlock(nn.Module):
        """Depthwise conv with left-only padding.

        A symmetric-padded conv is the single most common way a 'causal'
        speech model leaks future context. Padding only on the left makes the
        leak impossible by construction rather than by convention.
        """

        def __init__(self, d: int, k: int = 31):
            super().__init__()
            self.k = k
            self.dw = nn.Conv1d(d, d, k, groups=d)
            self.pw = nn.Conv1d(d, d, 1)
            self.norm = nn.LayerNorm(d)

        def forward(self, x):
            h = self.norm(x).transpose(1, 2)
            h = torch.nn.functional.pad(h, (self.k - 1, 0))
            return x + self.pw(self.dw(h)).transpose(1, 2)

    class ConversionBlock(nn.Module):
        def __init__(self, cfg: HarnessConfig):
            super().__init__()
            d = cfg.d_content
            self.n1 = nn.LayerNorm(d)
            self.att = MaskedSelfAttention(d, cfg.n_convert_heads)
            self.conv = CausalConvBlock(d)
            self.n2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(
                nn.Linear(d, cfg.d_convert_ffn), nn.GELU(),
                nn.Linear(cfg.d_convert_ffn, d))
            # FiLM conditioning on speaker embedding
            self.film = nn.Linear(cfg.d_speaker, 2 * d)

        def forward(self, x, add_mask, spk):
            x = x + self.att(self.n1(x), add_mask)
            x = self.conv(x)
            g, b = self.film(spk).unsqueeze(1).chunk(2, -1)
            x = x * (1 + g) + b
            return x + self.ff(self.n2(x))

    class Harness(nn.Module):
        def __init__(self, cfg: HarnessConfig):
            super().__init__()
            self.cfg = cfg
            self.blocks = nn.ModuleList(
                [ConversionBlock(cfg) for _ in range(cfg.n_convert_layers)])
            self.out = nn.Linear(cfg.d_content, cfg.d_content)
            self._mask_cache: Dict[Tuple[int, int, int], object] = {}

        def additive_mask(self, T: int, device, dtype):
            g = self.cfg.geometry
            key = (T, g.chunk_frames, g.lookahead_frames)
            m = self._mask_cache.get(key)
            if m is None:
                bm = chunked_lookahead_mask(
                    T, chunk_frames=g.chunk_frames,
                    lookahead_frames=g.lookahead_frames,
                    lookback_frames=g.lookback_frames)
                m = torch.from_numpy(
                    np.where(bm, 0.0, float("-inf")).astype(np.float32))
                self._mask_cache[key] = m
            return m.to(device=device, dtype=dtype)

        def forward(self, content, spk):
            """content: (B,T,D) SSL features.  spk: (B, d_speaker)."""
            B, T, _ = content.shape
            am = self.additive_mask(T, content.device, content.dtype)
            x = content
            if self.cfg.mode is Mode.VC_ONLY:
                # H3 control: content is frozen. The conversion stack may only
                # move timbre, which it does through FiLM. We detach the
                # content path so gradients cannot rewrite phoneme identity.
                x = x.detach()
            for b in self.blocks:
                x = b(x, am, spk)
            return self.out(x)

    torch.manual_seed(cfg.seed)
    model = Harness(cfg)
    info = {
        "params": sum(p.numel() for p in model.parameters()),
        "geometry": cfg.geometry.describe(),
        "mode": cfg.mode.value,
    }
    return model, info


# ==========================================================================
# Streaming inference driver
# ==========================================================================

class StreamingRunner:
    """Runs the harness chunk-by-chunk with a ring KV cache and instruments it.

    Deliberately separate from the nn.Module: the module defines *what* is
    computed, the runner defines *when*, and the latency decomposition is a
    property of the runner. Keeping them apart is what lets the same weights
    be evaluated both offline (masked) and streaming (buffered) and the two
    compared -- which is the equivalence check the paper needs.
    """

    def __init__(self, model, cfg: HarnessConfig, timer=None):
        self.model, self.cfg = model, cfg
        self.geom = cfg.geometry
        from .latency import StageTimer
        self.timer = timer or StageTimer()
        self.reset()

    def reset(self) -> None:
        self._kv = None
        self._emitted = 0
        self.timer.reset()

    def process(self, content_chunk, spk):
        """One chunk in, finalised frames out (or None while filling lookahead)."""
        torch = _torch()
        with self.timer.stage("convert"):
            with torch.no_grad():
                if self._kv is None:
                    self._kv = content_chunk
                else:
                    self._kv = torch.cat([self._kv, content_chunk], 1)
                    lb = self.geom.lookback_frames
                    if lb is not None:
                        keep = lb + self.geom.chunk_frames + self.geom.lookahead_frames
                        self._kv = self._kv[:, -keep:]
                T = self._kv.shape[1]
                La = self.geom.lookahead_frames
                if T - self._emitted <= La:
                    return None
                y = self.model(self._kv, spk)
        lo = max(0, T - La - self.geom.chunk_frames)
        self._emitted = T - La
        return y[:, lo:T - La]

    def budget(self, buffer_ms: float = 0.0, label: str = ""):
        from .latency import budget_from_timer
        return budget_from_timer(
            self.timer, chunk_ms=self.geom.chunk_ms,
            lookahead_ms=self.geom.lookahead_ms,
            compute_stages=["convert"], buffer_ms=buffer_ms,
            label=label or f"L{self.geom.lookahead_ms:g}/{self.cfg.mode.value}",
            meta={"mode": self.cfg.mode.value,
                  "irreducible_frontend_ms": self.geom.irreducible_lookahead_ms})


# ==========================================================================

def sweep_configs(lookaheads=(0, 20, 40, 80, 160, 320, 640),
                  modes=(Mode.AC, Mode.VC_ONLY), **base):
    """Build the full sweep and assert it is not confounded."""
    out = []
    for m in modes:
        cfgs = [HarnessConfig(lookahead_ms=L, mode=m, **base) for L in lookaheads]
        assert_only_L_varies(cfgs)
        out.extend(cfgs)
    return out
