"""Causal accent translator — the trainable core of the lookahead sweep.

The task, and why this one
--------------------------
The HF release of L2-ARCTIC (`KoelLabs/L2Arctic`) gives us, per utterance:

    text  : the prompt
    g2p   : the CANONICAL phone sequence  (what a native speaker would produce)
    ipa   : the PRODUCED phone sequence   (what this L2 speaker actually said)

It does *not* give us native reference audio. So the PHONOS golden-target
recipe (DTW a native rendition onto the source timbre) is not available from
this release alone.

What *is* available is better than a workaround — it is the supervised core of
PHONOS stated directly:

    causal accent translator: non-native audio  ->  NATIVE phone sequence

Train with CTC against `g2p`. That is literally "map non-native content tokens
to native equivalents", which is PHONOS's own description of its translator,
and it is exactly the component whose lookahead requirement RQ1 asks about.

The control that makes RQ3 answerable
-------------------------------------
Run the identical architecture, identical capacity, identical data, identical
step count, with only the *target* swapped:

    TARGET_NATIVE    (g2p)  -> accent CONVERSION. Must decide what the speaker
                               *should* have said, which depends on the lexical
                               and coarticulatory context.
    TARGET_PRODUCED  (ipa)  -> accent-faithful TRANSCRIPTION. Must only report
                               the local articulatory gesture.

If H3 is right — that conversion needs more right context than surface-level
speech processing — then the PER-vs-lookahead curve for TARGET_NATIVE should
be steeper than for TARGET_PRODUCED. Same model, same data, one flag. This is
a cleaner control than AC-vs-VC-only because the two arms are byte-identical
apart from the label tensor.

And the gap between the two heads is itself the accent-conversion signal: a
model that scores well against `ipa` and badly against `g2p` has learned to
transcribe, not to convert.

Cost
----
This is a CTC head over a frozen/partially-frozen SSL encoder, not a
generative vocoder pipeline. On a T4 each (L, target) run converges in roughly
30–60 minutes on the 3599-utterance annotated subset, so the full
7 lookaheads x 2 targets = 14 runs is a single-day job, not 200 GPU-hours.
That buys the RQ1 and RQ3 curves months earlier than the full synthesis
pipeline, and the full pipeline can then be aimed at the question the curves
raise rather than at discovering it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .causal import StreamGeometry, chunked_lookahead_mask


class Target(str, Enum):
    NATIVE = "native"       # g2p  -> accent conversion
    PRODUCED = "produced"   # ipa  -> accent-faithful transcription (control)


@dataclass
class TranslatorConfig:
    # ---- the swept parameter ----
    lookahead_ms: float = 80.0

    # ---- held constant across the sweep ----
    target: Target = Target.NATIVE
    chunk_ms: float = 40.0
    lookback_ms: Optional[float] = 2000.0
    frame_ms: float = 20.0
    sample_rate: int = 16_000

    encoder_name: str = "microsoft/wavlm-base-plus"
    freeze_encoder: bool = True     # frozen -> the sweep is about the head's
                                    # access to context, not about the encoder
                                    # re-learning. Set False for the full study.
    encoder_layer: int = 9
    d_model: int = 768
    n_layers: int = 4
    n_heads: int = 8
    d_ffn: int = 2048
    dropout: float = 0.1

    seed: int = 1337
    lr: float = 3e-4
    warmup_steps: int = 500
    train_steps: int = 8000
    batch_size: int = 8
    grad_clip: float = 1.0

    def __post_init__(self):
        if self.chunk_ms % self.frame_ms:
            raise ValueError(
                f"chunk_ms={self.chunk_ms} must be a multiple of frame_ms="
                f"{self.frame_ms}; otherwise the chunk boundary falls mid-frame "
                "and the effective lookahead differs from the label.")

    @property
    def geometry(self) -> StreamGeometry:
        return StreamGeometry(chunk_ms=self.chunk_ms, lookahead_ms=self.lookahead_ms,
                              frame_ms=self.frame_ms, lookback_ms=self.lookback_ms,
                              sample_rate=self.sample_rate)

    def fingerprint(self) -> Dict[str, object]:
        d = asdict(self)
        d.pop("lookahead_ms")
        d["target"] = self.target.value
        return d

    def tag(self) -> str:
        """Run label. Includes the seed so multi-seed runs cannot overwrite
        each other's checkpoints -- a silent way to lose 2/3 of a sweep."""
        return f"L{self.lookahead_ms:g}_{self.target.value}_s{self.seed}"


def assert_only_L_varies(cfgs: Sequence[TranslatorConfig]) -> None:
    cfgs = list(cfgs)
    if len(cfgs) < 2:
        return
    ref = cfgs[0].fingerprint()
    for c in cfgs[1:]:
        f = c.fingerprint()
        diff = {k: (ref[k], f[k]) for k in ref if ref[k] != f.get(k)}
        if diff:
            raise ValueError(f"Sweep is confounded on {diff}")
    Ls = [c.lookahead_ms for c in cfgs]
    if len(set(Ls)) != len(Ls):
        raise ValueError(f"duplicate lookaheads: {Ls}")


# ==========================================================================
# Phone vocabulary
# ==========================================================================

class PhoneVocab:
    """Shared symbol table for both targets.

    Both arms MUST use the same vocabulary, or their PERs are not comparable
    and RQ3 collapses. Build it once from the union of g2p and ipa over the
    whole corpus, freeze it, and save it beside every checkpoint.
    """

    BLANK = "<blank>"

    def __init__(self, symbols: Sequence[str]):
        syms = [self.BLANK] + sorted(set(symbols) - {self.BLANK})
        self.itos = syms
        self.stoi = {s: i for i, s in enumerate(syms)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, phones: Sequence[str]) -> List[int]:
        return [self.stoi[p] for p in phones if p in self.stoi]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.itos[i] for i in ids if i != 0]

    def to_dict(self) -> Dict[str, object]:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d) -> "PhoneVocab":
        v = cls([])
        v.itos = list(d["itos"])
        v.stoi = {s: i for i, s in enumerate(v.itos)}
        return v

    @classmethod
    def build(cls, sequences: Sequence[Sequence[str]]) -> "PhoneVocab":
        syms = set()
        for s in sequences:
            syms.update(s)
        return cls(sorted(syms))


_STRESS = re.compile(r"[0-9ˈˌ]")
_SEP = re.compile(r"[\s/|,]+")


def tokenize_phones(s: str, strip_stress: bool = True) -> List[str]:
    """Split a phone string into symbols.

    The HF release stores these as strings; the exact separator varies between
    the `g2p` and `ipa` columns. Handle whitespace, slashes and pipes, and
    fall back to per-character IPA segmentation (keeping combining marks and
    common digraph ties attached) when there is no separator.

    Strip stress digits/marks by default: stress is prosodic, this task is
    segmental, and leaving it in triples the effective vocabulary while adding
    noise the model cannot predict from a 40 ms window.
    """
    if not s:
        return []
    s = s.strip()
    if strip_stress:
        s = _STRESS.sub("", s)
    parts = [p for p in _SEP.split(s) if p]
    if len(parts) > 1:
        return parts
    # single blob -> IPA character segmentation, keeping modifiers attached
    out: List[str] = []
    for ch in s:
        if out and (
            "̀" <= ch <= "ͯ"        # combining diacritics
            or ch in "ʰʲʷˠˤ̝̞̃͜͡"
        ):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def per(ref: Sequence[str], hyp: Sequence[str]) -> float:
    """Phone error rate (Levenshtein / len(ref))."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


# ==========================================================================
# Model
# ==========================================================================

def build_translator(cfg: TranslatorConfig, vocab: PhoneVocab):
    """Causal conversion stack + CTC head on top of a masked SSL encoder."""
    import torch
    import torch.nn as nn

    class MaskedBlock(nn.Module):
        def __init__(self, c: TranslatorConfig):
            super().__init__()
            d = c.d_model
            self.n1 = nn.LayerNorm(d)
            self.att = nn.MultiheadAttention(d, c.n_heads, dropout=c.dropout,
                                             batch_first=True)
            self.k = 31
            self.dw = nn.Conv1d(d, d, self.k, groups=d)
            self.pw = nn.Conv1d(d, d, 1)
            self.nc = nn.LayerNorm(d)
            self.n2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, c.d_ffn), nn.GELU(),
                                    nn.Dropout(c.dropout), nn.Linear(c.d_ffn, d))
            self.drop = nn.Dropout(c.dropout)

        def forward(self, x, attn_mask, key_padding_mask=None):
            # `key_padding_mask` is honoured by attention but NOT by the conv or
            # the feed-forward. With a batch padded to its longest utterance,
            # the causal depthwise conv would pull padded positions into the
            # last (k-1) real frames of every shorter utterance -- a
            # batch-composition-dependent artefact that shows up as noise
            # between otherwise identical runs. Zero the padding explicitly at
            # every stage instead of trusting the attention mask to cover it.
            def _z(t):
                return t if key_padding_mask is None else t.masked_fill(
                    key_padding_mask.unsqueeze(-1), 0.0)

            x = _z(x)
            h = self.n1(x)
            a, _ = self.att(h, h, h, attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)
            x = _z(x + self.drop(a))
            # causal depthwise conv: left-only padding, never symmetric
            h = _z(self.nc(x)).transpose(1, 2)
            h = torch.nn.functional.pad(h, (self.k - 1, 0))
            x = _z(x + self.pw(self.dw(h)).transpose(1, 2))
            return _z(x + self.drop(self.ff(self.n2(x))))

    class Translator(nn.Module):
        def __init__(self, c: TranslatorConfig, V: int):
            super().__init__()
            self.cfg = c
            self.blocks = nn.ModuleList([MaskedBlock(c) for _ in range(c.n_layers)])
            self.norm = nn.LayerNorm(c.d_model)
            self.head = nn.Linear(c.d_model, V)
            self._mask_cache: Dict[int, "torch.Tensor"] = {}

        def attn_mask(self, T: int, device, dtype):
            """Float mask for nn.MultiheadAttention: 0 = attend, -inf = block."""
            m = self._mask_cache.get(T)
            if m is None:
                g = self.cfg.geometry
                bm = chunked_lookahead_mask(
                    T, chunk_frames=g.chunk_frames,
                    lookahead_frames=g.lookahead_frames,
                    lookback_frames=g.lookback_frames)
                m = torch.from_numpy(np.where(bm, 0.0, float("-inf")).astype(np.float32))
                self._mask_cache[T] = m
            return m.to(device=device, dtype=dtype)

        def forward(self, feats, key_padding_mask=None):
            T = feats.shape[1]
            am = self.attn_mask(T, feats.device, feats.dtype)
            x = feats
            for b in self.blocks:
                x = b(x, am, key_padding_mask)
            return self.head(self.norm(x))          # (B, T, V) logits

    torch.manual_seed(cfg.seed)
    model = Translator(cfg, len(vocab))
    info = {"params": sum(p.numel() for p in model.parameters()),
            "vocab": len(vocab), "geometry": cfg.geometry.describe(),
            "target": cfg.target.value}
    return model, info


def greedy_ctc_decode(logits, vocab: PhoneVocab) -> List[List[str]]:
    """Collapse repeats, drop blanks."""
    import torch
    ids = logits.argmax(-1).cpu().numpy()
    out = []
    for row in ids:
        seq, prev = [], -1
        for i in row:
            if i != prev and i != 0:
                seq.append(int(i))
            prev = int(i)
        out.append(vocab.decode(seq))
    return out


def sweep_configs(lookaheads=(0, 20, 40, 80, 160, 320, 640),
                  targets=(Target.NATIVE, Target.PRODUCED), **base):
    out: List[TranslatorConfig] = []
    for t in targets:
        arm = [TranslatorConfig(lookahead_ms=L, target=t, **base) for L in lookaheads]
        assert_only_L_varies(arm)
        out.extend(arm)
    return out


if __name__ == "__main__":
    # self-test: no torch required for the parts below
    assert tokenize_phones("HH AH0 L OW1") == ["HH", "AH", "L", "OW"]
    assert tokenize_phones("") == []
    assert tokenize_phones("ˈhɛloʊ")[:2] == ["h", "ɛ"], tokenize_phones("ˈhɛloʊ")
    assert abs(per(["a", "b", "c"], ["a", "x", "c"]) - 1 / 3) < 1e-9
    assert per([], []) == 0.0

    v = PhoneVocab.build([["AH", "B"], ["B", "K"]])
    assert v.itos[0] == PhoneVocab.BLANK and len(v) == 4
    assert v.decode(v.encode(["AH", "B", "K"])) == ["AH", "B", "K"]

    assert TranslatorConfig(lookahead_ms=80, target=Target.NATIVE, seed=7).tag() \
        != TranslatorConfig(lookahead_ms=80, target=Target.NATIVE, seed=8).tag(), \
        "tags must differ by seed or checkpoints collide"
    cfgs = sweep_configs()
    assert len(cfgs) == 14
    nat = [c for c in cfgs if c.target is Target.NATIVE]
    pro = [c for c in cfgs if c.target is Target.PRODUCED]
    for a, b in zip(nat, pro):
        fa, fb = a.fingerprint(), b.fingerprint()
        fa.pop("target"), fb.pop("target")
        assert fa == fb, "the two arms differ in more than the target"
    try:
        assert_only_L_varies([TranslatorConfig(lookahead_ms=0, chunk_ms=20),
                              TranslatorConfig(lookahead_ms=80, chunk_ms=40)])
        raise AssertionError("confound not caught")
    except ValueError as e:
        assert "chunk_ms" in str(e)
    try:
        TranslatorConfig(chunk_ms=30, frame_ms=20)
        raise AssertionError("non-multiple chunk accepted")
    except ValueError:
        pass
    print("ok  translator self-test (tokenizer, PER, vocab, sweep invariants)")
