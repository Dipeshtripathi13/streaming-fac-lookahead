"""How does t_compute scale with lookahead and chunk size?

This is the RQ4 instrument. It answers a question that is *independent of
model quality* and therefore can be measured today, on any machine, with no
trained accent-conversion model:

    Given a transformer content encoder of realistic size, how much does the
    per-chunk compute cost change as you vary lookahead L, chunk size C, and
    left-context (lookback) W -- and which of the three dominates?

Why this matters for the paper
------------------------------
There is a widespread assumption that shrinking lookahead makes a streaming
system cheaper. It does not, or at least not much: lookahead changes the
*attention mask*, not the number of frames you push through the feed-forward
and convolution stacks. What actually drives per-chunk CPU cost is chunk size
(how many frames per forward pass, and therefore how well the matmuls
amortise) and left-context length (which sets the KV-cache size).

If that holds, it has a sharp consequence: **algorithmic latency and
computational latency are traded against each other, not together.** Cutting
L buys you algorithmic latency for free in compute terms, but costs quality.
Cutting C buys you algorithmic latency and *costs* compute, because small
matmuls waste the vector units. That is a genuinely useful thing to tell a
practitioner, and it is exactly the kind of claim the "report one number"
convention hides.

Implementation
--------------
A pure-NumPy pre-norm transformer encoder block, dimensions matched to
wav2vec2/HuBERT-base (d_model=768, 12 heads, ffn=3072, 12 layers) and to a
smaller streaming-realistic config. NumPy dispatches matmuls to BLAS, so the
FLOP-bound parts run at hardware-realistic speed; this is a scaling study,
not an attempt to beat an optimised runtime. Absolute numbers should be read
as an upper bound on a well-optimised implementation, and we report a
measured BLAS efficiency figure so the reader can discount appropriately.

Usage
-----
    python3 bench/bench_encoder_scaling.py --preset base --reps 20
    python3 bench/bench_encoder_scaling.py --preset small --threads 1
    python3 bench/bench_encoder_scaling.py --quick        # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfac.causal import StreamGeometry, chunked_lookahead_mask  # noqa: E402
from sfac.latency import StageTimer, LatencyBudget, write_csv, write_jsonl  # noqa: E402


# --------------------------------------------------------------------------
# Model presets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EncoderCfg:
    name: str
    d_model: int
    n_heads: int
    d_ffn: int
    n_layers: int
    conv_kernel: int = 31          # conformer-style depthwise conv

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


PRESETS = {
    # HuBERT / WavLM / wav2vec2 base -- what most FAC content encoders use
    "base": EncoderCfg("base", 768, 12, 3072, 12),
    # A streaming-sized encoder: what you would actually deploy on a Pi
    "small": EncoderCfg("small", 384, 6, 1536, 6),
    # DarkStream / TVTSyn-scale contextual stack
    "tiny": EncoderCfg("tiny", 256, 4, 1024, 4),
}


# --------------------------------------------------------------------------
# Pure-NumPy encoder forward pass
# --------------------------------------------------------------------------

class NumpyEncoderBlock:
    """Pre-norm transformer block with masked self-attention + depthwise conv.

    Weights are random; we are measuring time, not accuracy. Everything is
    float32 and allocated once so allocation does not leak into the timing.
    """

    def __init__(self, cfg: EncoderCfg, rng: np.random.Generator):
        d, f = cfg.d_model, cfg.d_ffn
        s = 1.0 / np.sqrt(d)
        self.cfg = cfg
        self.wq = (rng.standard_normal((d, d)) * s).astype(np.float32)
        self.wk = (rng.standard_normal((d, d)) * s).astype(np.float32)
        self.wv = (rng.standard_normal((d, d)) * s).astype(np.float32)
        self.wo = (rng.standard_normal((d, d)) * s).astype(np.float32)
        self.w1 = (rng.standard_normal((d, f)) * s).astype(np.float32)
        self.w2 = (rng.standard_normal((f, d)) * s).astype(np.float32)
        self.dw = (rng.standard_normal((cfg.conv_kernel, d)) * s).astype(np.float32)
        self.g1 = np.ones(d, np.float32)
        self.b1 = np.zeros(d, np.float32)

    @staticmethod
    def _ln(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        mu = x.mean(-1, keepdims=True)
        sd = x.std(-1, keepdims=True) + 1e-5
        return (x - mu) / sd * g + b

    def __call__(self, x: np.ndarray, add_mask: np.ndarray,
                 kv: Optional[np.ndarray] = None) -> np.ndarray:
        """x: (T_q, D) queries.  kv: (T_kv, D) context, defaults to x.

        add_mask: (T_q, T_kv) additive float mask (0 / -1e9).
        """
        c = self.cfg
        h = self._ln(x, self.g1, self.b1)
        ctx = h if kv is None else kv

        q = h @ self.wq
        k = ctx @ self.wk
        v = ctx @ self.wv

        Tq, Tk = q.shape[0], k.shape[0]
        q = q.reshape(Tq, c.n_heads, c.d_head).transpose(1, 0, 2)
        k = k.reshape(Tk, c.n_heads, c.d_head).transpose(1, 0, 2)
        v = v.reshape(Tk, c.n_heads, c.d_head).transpose(1, 0, 2)

        att = (q @ k.transpose(0, 2, 1)) / np.sqrt(c.d_head)
        att += add_mask[None, :, :]
        att -= att.max(-1, keepdims=True)
        np.exp(att, out=att)
        att /= att.sum(-1, keepdims=True)

        o = (att @ v).transpose(1, 0, 2).reshape(Tq, c.d_model) @ self.wo
        x = x + o

        # depthwise conv, causal-padded (streaming-legal)
        h = self._ln(x, self.g1, self.b1)
        pad = np.zeros((c.conv_kernel - 1, c.d_model), np.float32)
        hp = np.concatenate([pad, h], 0)
        acc = np.zeros_like(h)
        for t in range(c.conv_kernel):
            acc += hp[t:t + h.shape[0]] * self.dw[t]
        x = x + acc

        h = self._ln(x, self.g1, self.b1)
        ff = h @ self.w1
        np.maximum(ff, 0, out=ff)          # ReLU stands in for GELU; same FLOPs
        return x + ff @ self.w2


class NumpyEncoder:
    def __init__(self, cfg: EncoderCfg, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.blocks = [NumpyEncoderBlock(cfg, rng) for _ in range(cfg.n_layers)]

    def __call__(self, x, add_mask, kv=None):
        for b in self.blocks:
            x = b(x, add_mask, kv)
        return x


def attention_flops(cfg: EncoderCfg, Tq: int, Tk: int) -> float:
    d, f = cfg.d_model, cfg.d_ffn
    proj = 2 * Tq * d * d * 2 + 2 * Tk * d * d * 2      # q,o over Tq; k,v over Tk
    scores = 2 * Tq * Tk * d * 2                        # QK^T and AV
    ffn = 2 * Tq * d * f * 2
    conv = 2 * Tq * d * cfg.conv_kernel
    return (proj + scores + ffn + conv) * cfg.n_layers


# --------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------

def run_condition(
    enc: NumpyEncoder,
    geom: StreamGeometry,
    n_chunks: int,
    reps: int,
    timer: StageTimer,
) -> Dict[str, object]:
    """Simulate steady-state streaming for one (chunk, L, lookback) config.

    Each 'step' processes one chunk of queries against a KV context of
    (lookback + chunk + lookahead) frames -- exactly what a ring-KV-cache
    implementation does at inference.
    """
    cfg = enc.cfg
    cf = geom.chunk_frames
    La = geom.lookahead_frames
    lb = geom.lookback_frames if geom.lookback_frames is not None else 100

    Tq = cf
    Tk = lb + cf + La
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((Tq, cfg.d_model)).astype(np.float32)
    kv = rng.standard_normal((Tk, cfg.d_model)).astype(np.float32)

    # queries are the last cf frames of the window (before the lookahead tail)
    full = chunked_lookahead_mask(Tk, chunk_frames=cf, lookahead_frames=La,
                                  lookback_frames=lb)
    sub = full[lb:lb + cf, :]                          # (Tq, Tk)
    add = np.where(sub, 0.0, -1e9).astype(np.float32)

    # Warm properly. One call is not enough: OpenBLAS picks its kernel and
    # allocates its thread-local buffers on first use for each new matrix
    # shape, and each (chunk, L) pair is a new shape. A single warm call left
    # a visible 50% outlier at the first condition of each shape family.
    for _ in range(3):
        enc(x, add, kv)
    stage = f"encoder_L{geom.lookahead_ms:g}_C{geom.chunk_ms:g}"
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        enc(x, add, kv)
        timer.record(stage, (time.perf_counter_ns() - t0) / 1e6)

    summ = timer.summary(drop_warmup=2)[stage]
    fl = attention_flops(cfg, Tq, Tk)
    gflops = fl / (summ["p50"] / 1000.0) / 1e9

    budget = LatencyBudget(
        chunk_ms=geom.chunk_ms,
        lookahead_ms=geom.lookahead_ms,
        compute_ms_p50=summ["p50"],
        compute_ms_p95=summ["p95"],
        label=f"{cfg.name}/L{geom.lookahead_ms:g}/C{geom.chunk_ms:g}",
    )
    row = budget.to_row()
    row.pop("meta.per_stage", None)
    row.update({
        "preset": cfg.name,
        "lookback_frames": lb,
        "T_query": Tq,
        "T_kv": Tk,
        "gflop_per_chunk": round(fl / 1e9, 4),
        "achieved_gflops": round(gflops, 1),
        "t_algorithmic_honest_ms": round(geom.algorithmic_ms_honest, 2),
        "compute_stdev_ms": round(summ["stdev"], 3),
        "n_reps": int(summ["n"]),
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=list(PRESETS) + ["all"])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--lookback-ms", type=float, default=2000.0,
                    help="TVTSyn uses 2 s of look-back; that is the default here.")
    ap.add_argument("--threads", type=int, default=None,
                    help="Pin BLAS threads. Set BEFORE numpy import via env for full effect.")
    ap.add_argument("--out-prefix", default="results/raw/encoder_scaling")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    if a.threads:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
            os.environ[v] = str(a.threads)

    lookaheads = (0, 40, 160, 640) if a.quick else (0, 20, 40, 80, 160, 320, 640)
    chunks = (20, 80) if a.quick else (20, 40, 80)
    presets = list(PRESETS) if a.preset == "all" else [a.preset]
    reps = 5 if a.quick else a.reps

    rows: List[Dict[str, object]] = []
    for pname in presets:
        cfg = PRESETS[pname]
        enc = NumpyEncoder(cfg)
        print(f"\n=== preset {pname}: d={cfg.d_model} h={cfg.n_heads} "
              f"ffn={cfg.d_ffn} layers={cfg.n_layers} ===", file=sys.stderr)
        for chunk_ms in chunks:
            for L in lookaheads:
                geom = StreamGeometry(chunk_ms=chunk_ms, lookahead_ms=L,
                                      lookback_ms=a.lookback_ms)
                timer = StageTimer()
                row = run_condition(enc, geom, n_chunks=1, reps=reps, timer=timer)
                rows.append(row)
                flag = "OK " if row["feasible"] else "!! "
                print(f"{flag}C={chunk_ms:>3}ms L={L:>3}ms  "
                      f"t_algo={row['t_algorithmic_ms']:>6.1f}  "
                      f"t_cmp_p50={row['t_compute_p50_ms']:>7.2f}  "
                      f"p95={row['t_compute_p95_ms']:>7.2f}  "
                      f"RTF={row['rtf_p50']:>6.3f}  "
                      f"{row['achieved_gflops']:>6.1f} GF/s", file=sys.stderr)

    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
    write_csv(a.out_prefix + ".csv", rows)
    write_jsonl(a.out_prefix + ".jsonl", rows)
    print(json.dumps({"n_rows": len(rows), "csv": a.out_prefix + ".csv"}, indent=2))


if __name__ == "__main__":
    main()
