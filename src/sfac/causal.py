"""Lookahead-controlled causal attention masks and streaming frame arithmetic.

This is the load-bearing module of the whole project. RQ1 requires that
lookahead `L` be the *only* thing that varies across the sweep: same weights,
same data, same chunking, same everything else. If anything else co-varies,
the comparison is confounded and the paper is dead.

Everything here is framework-agnostic (numpy). `masks.py` consumers convert to
torch/ONNX as needed.

Frame arithmetic
----------------
A self-supervised speech encoder (HuBERT / WavLM / wav2vec2 base) has a
convolutional feature extractor with total stride 320 at 16 kHz, i.e. one
latent frame per 20 ms, and a receptive field of 400 samples (25 ms).
So:

    frame_ms = 20
    L_frames = ceil(L_ms / frame_ms)

A lookahead of 0 ms is *not* the same as zero context: the conv frontend
still looks 25 ms ahead of the frame centre. We call that the "irreducible
frontend lookahead" and report it separately, because a paper that claims
"0 ms lookahead" while using a 25 ms conv window is misreporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


FRAME_MS_DEFAULT = 20.0          # HuBERT/WavLM/wav2vec2 base latent rate
CONV_RECEPTIVE_MS_DEFAULT = 25.0  # 400 samples @16k, the frontend window


# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamGeometry:
    """All the frame bookkeeping for one (chunk, lookahead) configuration."""

    chunk_ms: float
    lookahead_ms: float
    frame_ms: float = FRAME_MS_DEFAULT
    conv_receptive_ms: float = CONV_RECEPTIVE_MS_DEFAULT
    lookback_ms: Optional[float] = None   # None => unbounded (full history)
    sample_rate: int = 16_000

    # ---- derived ----
    @property
    def chunk_frames(self) -> int:
        return max(1, int(round(self.chunk_ms / self.frame_ms)))

    @property
    def lookahead_frames(self) -> int:
        """Ceil, not round.

        Rounding down would silently give the model less context than the
        condition label claims. Ceil means the *reported* lookahead is an
        upper bound on the true one, which is the safe direction for a
        latency claim.
        """
        return int(math.ceil(self.lookahead_ms / self.frame_ms - 1e-9))

    @property
    def lookback_frames(self) -> Optional[int]:
        if self.lookback_ms is None:
            return None
        return int(math.floor(self.lookback_ms / self.frame_ms))

    @property
    def chunk_samples(self) -> int:
        return int(round(self.chunk_ms * self.sample_rate / 1000.0))

    @property
    def lookahead_samples(self) -> int:
        return int(round(self.lookahead_ms * self.sample_rate / 1000.0))

    @property
    def irreducible_lookahead_ms(self) -> float:
        """Frontend lookahead you cannot remove without changing the encoder.

        Reported alongside `lookahead_ms` in every table. The honest
        "true algorithmic delay" is chunk + lookahead + this.
        """
        return max(0.0, self.conv_receptive_ms - self.frame_ms)

    @property
    def algorithmic_ms(self) -> float:
        return self.chunk_ms + self.lookahead_frames * self.frame_ms

    @property
    def algorithmic_ms_honest(self) -> float:
        return self.algorithmic_ms + self.irreducible_lookahead_ms

    def describe(self) -> str:
        lb = "inf" if self.lookback_frames is None else str(self.lookback_frames)
        return (
            f"chunk={self.chunk_ms:g}ms({self.chunk_frames}f) "
            f"L={self.lookahead_ms:g}ms({self.lookahead_frames}f) "
            f"lookback={lb}f "
            f"t_algo={self.algorithmic_ms:g}ms "
            f"(+{self.irreducible_lookahead_ms:g}ms frontend)"
        )


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------


def lookahead_mask(
    n: int,
    lookahead_frames: int,
    lookback_frames: Optional[int] = None,
    dtype=bool,
) -> np.ndarray:
    """Boolean attention mask, shape (n, n). True == may attend.

    Query i may attend to key j iff:

        i - lookback  <=  j  <=  i + lookahead

    lookahead_frames = 0   -> strictly causal (band-limited from the right)
    lookahead_frames = n   -> effectively bidirectional
    lookback_frames = None -> unbounded history

    Note this is the *offline-equivalent* mask: it reproduces exactly what a
    streaming implementation with the same L would see, which is what makes
    non-streaming training and streaming inference consistent.  Verifying
    that equivalence is the job of tests/test_causal.py.
    """
    if lookahead_frames < 0:
        raise ValueError("lookahead_frames must be >= 0")
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    m = j <= (i + lookahead_frames)
    if lookback_frames is not None:
        m &= j >= (i - lookback_frames)
    return m.astype(dtype)


def additive_mask(m: np.ndarray, neg: float = -1e9) -> np.ndarray:
    """Boolean mask -> additive float mask for pre-softmax logits."""
    return np.where(m.astype(bool), 0.0, neg).astype(np.float32)


def chunked_lookahead_mask(
    n: int,
    chunk_frames: int,
    lookahead_frames: int,
    lookback_frames: Optional[int] = None,
) -> np.ndarray:
    """Mask for *chunkwise* streaming, which is what real systems do.

    In chunkwise streaming, frames are emitted a chunk at a time. Every frame
    in chunk c waits for the whole chunk boundary anyway, so frames early in
    the chunk get free extra right-context. Ignoring this makes training and
    inference disagree -- the model trained with a per-frame mask sees LESS
    context at train time than at test time, which shows up as a small
    unexplained quality gain at large chunk sizes and muddies the L sweep.

    Query i (in chunk c = i // chunk_frames) may attend to key j iff:

        j <= (c + 1) * chunk_frames - 1 + lookahead_frames

    plus the lookback constraint.

    Use this one for the main experiments. Use `lookahead_mask` only when
    explicitly studying the per-frame (chunk_frames == 1) limit.
    """
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be >= 1")
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    chunk_end = ((i // chunk_frames) + 1) * chunk_frames - 1
    m = j <= (chunk_end + lookahead_frames)
    if lookback_frames is not None:
        m &= j >= (i - lookback_frames)
    return m


def effective_lookahead_frames(mask: np.ndarray) -> np.ndarray:
    """Per-query realised right-context, for auditing a mask.

    Returns an array of length n: for query i, (max attendable j) - i.
    Lets you assert that a mask you built actually implements the L you
    think it does -- including the chunk-boundary bonus.
    """
    n = mask.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        js = np.nonzero(mask[i])[0]
        out[i] = (js.max() - i) if js.size else -1
    return out


def mask_audit(mask: np.ndarray) -> dict:
    """Summary statistics a reviewer might ask for."""
    eff = effective_lookahead_frames(mask)
    return {
        "n": int(mask.shape[0]),
        "eff_lookahead_min": int(eff.min()),
        "eff_lookahead_max": int(eff.max()),
        "eff_lookahead_mean": float(eff.mean()),
        "density": float(mask.mean()),
    }


# --------------------------------------------------------------------------
# Streaming buffer that reproduces the mask exactly
# --------------------------------------------------------------------------


class LookaheadBuffer:
    """Ring buffer implementing chunkwise streaming with L frames lookahead.

    Push audio chunks; it yields (context_window, emit_slice) pairs where
    `emit_slice` indexes the frames whose output is now final. The delay
    between pushing chunk c and it becoming emittable is exactly
    `lookahead_frames` frames -- this is `t_algorithmic` made concrete.

    Why a class and not a generator: the same object is reused across the
    whole sweep so allocation cost does not leak into t_compute.
    """

    def __init__(self, geom: StreamGeometry, feat_dim: int = 1, dtype=np.float32):
        self.geom = geom
        self.feat_dim = feat_dim
        self.dtype = dtype
        lb = geom.lookback_frames
        # capacity: history + current chunk + lookahead, with slack
        self.capacity = (lb if lb is not None else 200) + geom.chunk_frames + geom.lookahead_frames + 8
        self._buf = np.zeros((self.capacity, feat_dim), dtype=dtype)
        self._n = 0          # total frames written
        self._emitted = 0    # total frames finalised

    def reset(self) -> None:
        self._buf[:] = 0
        self._n = 0
        self._emitted = 0

    @property
    def frames_written(self) -> int:
        return self._n

    @property
    def frames_emitted(self) -> int:
        return self._emitted

    def push(self, frames: np.ndarray):
        """Append `frames` (T, D). Return (window, emit_lo, emit_hi) or None.

        `window` is the slice of context the model should run over.
        [emit_lo, emit_hi) are absolute frame indices now finalised.
        Returns None when not enough lookahead has accumulated yet.
        """
        if frames.ndim == 1:
            frames = frames[:, None]
        t = frames.shape[0]
        if self._n + t > self.capacity:
            keep = self.capacity - t
            self._buf[:keep] = self._buf[self._n - keep:self._n]
            self._shift = self._n - keep
            self._n = keep
        self._buf[self._n:self._n + t] = frames
        self._n += t

        La = self.geom.lookahead_frames
        # frames [0, n - La) are finalisable
        finalisable = max(0, self._n - La)
        if finalisable <= self._emitted:
            return None
        lo, hi = self._emitted, finalisable
        lb = self.geom.lookback_frames
        ctx_lo = 0 if lb is None else max(0, lo - lb)
        window = self._buf[ctx_lo:min(self._n, hi + La)]
        self._emitted = hi
        return window, lo, hi

    def flush(self):
        """End of stream: emit remaining frames with truncated right context."""
        if self._emitted >= self._n:
            return None
        lo, hi = self._emitted, self._n
        lb = self.geom.lookback_frames
        ctx_lo = 0 if lb is None else max(0, lo - lb)
        self._emitted = hi
        return self._buf[ctx_lo:self._n], lo, hi


# --------------------------------------------------------------------------

SWEEP_LOOKAHEAD_MS = (0, 20, 40, 80, 160, 320, 640)
SWEEP_CHUNK_MS = (20, 40, 80)
