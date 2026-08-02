# ARM64 pilot results — measured 2 August 2026

> **Superseded as the headline results by [`RESULTS_M4.md`](RESULTS_M4.md)**,
> which was measured on the actual target hardware (Apple M4) and additionally
> contains the encoder-causality proof. This document is retained because it is
> the **independent replication** on a second architecture with a different BLAS
> (Neoverse-N1 / OpenBLAS vs Apple M4 / Accelerate), which is what lets Findings
> 1 and 2 be stated as architecture-independent rather than as one machine's
> quirk. Read it second.

Real measurements, not projections. Every number below came out of the scripts
in `bench/` on the machine described in `results/raw/hw_sandbox_aarch64.json`.

## Machine

| | |
|---|---|
| class | `cpu-arm64` (**not** `embedded-pi` — see caveat) |
| arch | aarch64, Neoverse-N1, 4 cores |
| RAM | 3.8 GB |
| BLAS | OpenBLAS 0.3.29, DYNAMIC_ARCH, neoversen1 |
| sgemm 1024³ | **403 GFLOP/s** |
| memcpy | 53.5 GB/s |
| onnxruntime | 1.23.2, CPUExecutionProvider |

**Caveat that must not be lost:** this is an ARM *server* core, not a
Raspberry Pi. A Pi 5 (Cortex-A76 @ 2.4 GHz) will land around 30–60 GFLOP/s —
roughly **7–13× slower**. Treat everything here as an ARM64 upper bound.
`bench/hardware_probe.py` enforces the distinction: it only returns
`embedded-pi` when `/proc/device-tree/model` names a Pi.

---

## Finding 1 — per-chunk compute is nearly flat in lookahead

`bench/bench_encoder_scaling.py --preset all --reps 25`, 63 conditions,
lookback fixed at 2 s (TVTSyn-style).

L = 0 vs L = 640 ms, per-chunk compute p50:

| preset | chunk | L=0 | L=640 | Δ compute | Δ t_algorithmic |
|---|---:|---:|---:|---:|---:|
| base | 20 ms | 35.16 ms | 37.17 ms | **+5.7%** | +3200% |
| base | 40 ms | 34.94 | 37.60 | **+7.6%** | +1600% |
| base | 80 ms | 40.42 | 43.14 | **+6.7%** | +800% |
| small | 20 ms | 3.94 | 4.41 | **+11.9%** | +3200% |
| small | 40 ms | 4.31 | 4.75 | **+10.2%** | +1600% |
| small | 80 ms | 4.83 | 5.23 | **+8.3%** | +800% |
| tiny | 20 ms | 1.31 | 1.47 | **+12.2%** | +3200% |
| tiny | 40 ms | 1.70 | 1.87 | **+10.0%** | +1600% |
| tiny | 80 ms | 1.68 | 1.83 | **+8.9%** | +800% |

**Going from 0 ms to 640 ms of lookahead costs 6–12% more compute. It costs
8–32× more algorithmic latency.** The pattern is consistent across three model
scales and three chunk sizes.

*Measurement note:* an earlier pass showed a 50% outlier at the first condition
of each matrix-shape family. That was OpenBLAS kernel selection on first use of
a new shape, not a real effect — each `(chunk, L)` pair is a new shape. Fixed
by warming three times per condition rather than once. Worth knowing before you
believe any single-warm benchmark, including other people's.

Why: lookahead widens the attention mask, but the feed-forward and convolution
stacks — which dominate FLOPs — process the same number of query frames
regardless. With a 2 s lookback already in the KV cache, adding 32 more key
frames is noise.

**Consequence for the paper.** The two latency terms are not merely separable,
they are *independently controllable*, and the trade is asymmetric:

- Cutting **lookahead** buys algorithmic latency at ~zero compute saving, and
  costs quality (magnitude unknown — that is RQ1).
- Cutting **chunk size** buys algorithmic latency and *costs* compute, because
  small matmuls waste vector units.

Reporting a single fused latency number makes this invisible. This is the
concrete payoff of the decomposition and it is measurable before any model is
trained.

## Finding 2 — chunk size, not lookahead, decides feasibility

`base` preset (d=768, 12 heads, ffn=3072, 12 layers — HuBERT/WavLM-base scale):

| chunk | t_compute p50 (L=0) | RTF p50 | deployable? |
|---:|---:|---:|---|
| 20 ms | 35.16 ms | **1.76** | **no — falls behind real time** |
| 40 ms | 34.94 ms | **0.87** | marginal, no headroom |
| 80 ms | 40.42 ms | 0.51 | yes |

Quadrupling the chunk raises per-chunk cost by only 15% while cutting RTF by
3.5×.
On this ARM64 core a base-scale encoder at 20 ms chunks **cannot run in real
time at any lookahead**, and no amount of lookahead reduction fixes it.

Scaled to a Pi 5 (7–13× slower), `base` is infeasible at every chunk size and
even `small` at 20 ms chunks (RTF 0.200 here → ~1.4–2.6 on a Pi) is likely
infeasible. **The Pi condition will be where the feasibility frontier actually
appears** — which is the argument for buying the board.

Achieved throughput was 85–110 GFLOP/s against a 403 GFLOP/s sgemm ceiling,
i.e. ~21–27% BLAS efficiency. That is normal for small-matrix transformer
workloads and means an optimised ONNX/ggml implementation could plausibly gain
2–3×. Absolute numbers here are an upper bound on latency; the *scaling* is the
result.

## Finding 3 — the cascade baseline (S1) loses structurally, not computationally

`bench/bench_cascade_onnx.py`, sherpa-onnx streaming zipformer (int8) + Kokoro
int8 TTS, the exact stack in `../accentbridge`.

**ASR compute is negligible:**

| threads | active encoder step | p95 | duty cycle | amortised RTF |
|---:|---:|---:|---:|---:|
| 1 | 9.08 ms | 9.50 ms | 0.31 | **0.029** |
| 2 | 7.25 ms | 7.43 ms | 0.31 | **0.024** |
| 4 | 12.26 ms | 18.57 ms | 0.32 | 0.040 |

Two threads beat four — on a 4-core box, ORT's intra-op pool oversubscribes
against the feed loop. Worth a footnote: naive `num_threads = cpu_count` makes
this pipeline **69% slower at p95**.

**But the ASR's own geometry imposes a fixed algorithmic cost:**
`decode_chunk_len = 320 ms`, right context `= 70 ms`, read from the model's ONNX
metadata. **390 ms of algorithmic latency before a single word is emitted** —
already 1.6× the entire PHONOS end-to-end budget, with 9 ms of compute.

**And TTS is where it dies:**

| | Kokoro int8, 1 thread |
|---|---|
| utterance synthesis p50 | **2425 ms** |
| RTF | 0.833 |
| **single word** | **903 ms** |

Plus `accentbridge`'s `COMMIT_TIMEOUT = 0.7 s` before an unstable word is
released to the synthesiser.

Rough cascade budget on this hardware:

```
t_algorithmic  = 390 (ASR chunk+right ctx) + 700 (commit) ≈ 1090 ms
t_compute      = 9 (ASR) + 903 (TTS, one word)            ≈  912 ms
t_buffer       = audio I/O, not yet measured              ≈   30 ms
                                                          ─────────
t_end_to_end                                              ≈ 2030 ms
```

**~2.0 s against PHONOS's ≤241 ms — an order of magnitude.** And 54% of it is
algorithmic: a faster chip does not fix a cascade. That is a clean, quotable
result for the paper's S1 row, and it is the first properly decomposed cascade
latency budget we are aware of.

*Caveat, now resolved on the M4:* Kokoro is a full-utterance TTS, so this row
originally could not distinguish "cascades are slow" from "Kokoro is slow". The
M4 run answered it by benchmarking Piper/VITS on identical inputs: **21 ms per
word vs Kokoro's 973 ms, a 46× difference.** So the TTS term is a model choice,
not a structural one — and removing it makes the conclusion *stronger*, because
what remains (390 ms of ASR geometry + 700 ms commit delay) is entirely
algorithmic. See `RESULTS_M4.md` Finding 5: with a fast vocoder the cascade is
~1.15 s and **95% algorithmic**.

---

## What these results do and do not establish

**Do:**
- t_compute and t_algorithmic are independently controllable, measured (F1).
- Chunk size dominates deployment feasibility; lookahead does not (F2).
- The cascade's latency is majority-algorithmic and ~2 s on ARM64 CPU (F3).
- Thread oversubscription is a real, unreported ~69% p95 penalty (F3).
- The tooling runs end-to-end on a fresh machine with no dataset downloads.

**Do not:**
- Say anything about **quality**. No FAC model is trained. RQ1/RQ2/RQ3 are
  untouched. These are t_compute results only.
- Represent a Raspberry Pi. Server ARM64 ≠ Cortex-A76.
- Represent an optimised implementation. NumPy at ~24% BLAS efficiency is an
  upper bound on latency.

## Reproduce

```bash
python3 bench/hardware_probe.py --out results/raw/hw_<name>.json
python3 bench/bench_encoder_scaling.py --preset all --reps 20 \
        --out-prefix results/raw/encoder_scaling_<name>
python3 bench/bench_cascade_onnx.py --threads 1 2 4 --skip-fp32 \
        --out-prefix results/raw/cascade_<name>
python3 bench/make_figures.py
```

Raw CSV/JSONL in `results/raw/`, figures in `results/figures/`.
