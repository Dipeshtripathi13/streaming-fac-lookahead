# Streaming Accent Conversion — Lookahead / Latency Study

**How much future do you need?** A characterisation of the lookahead
requirement for streaming foreign accent conversion, on hardware people
actually own.

Target: Interspeech 2027 (São Paulo). Fallback: SSW14, 20 April 2027.

---

## Explicitly out of scope

**We are not trying to beat PHONOS or TVTSyn on quality.** This project
characterises an axis those systems parameterise but do not study. Any PR,
issue or branch whose goal is "also beat SOTA" is out of scope. This is written
here on purpose — it is the highest-likelihood, highest-impact risk in the
proposal's own risk table.

---

## Start here

| Read | For |
|---|---|
| [`docs/PROPOSAL_v2.md`](docs/PROPOSAL_v2.md) | The plan. Supersedes the v1 markdown in the parent folder. |
| [`docs/LITERATURE.md`](docs/LITERATURE.md) | Verified citations + **four corrections to v1**, one strategically significant |
| [`docs/RESULTS_M4.md`](docs/RESULTS_M4.md) | **Measured on the Apple M4**, 2 Aug 2026 — incl. the causality proof |
| [`docs/RESULTS_PILOT.md`](docs/RESULTS_PILOT.md) | Measured on ARM64 (replication), 2 Aug 2026 |
| `setup/SETUP_*.md` | Per-platform install and benchmarking protocol |

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/cpu.txt          # or embedded.txt on a Pi

# these five must pass before you trust any number from this repo
python3 tests/test_causal.py
python3 tests/test_pipeline_invariants.py
python3 eval/metrics.py
python3 eval/phoneme_analysis.py
python3 bench/hardware_probe.py --out results/raw/hw_$(hostname).json

# real benchmarks, no dataset download needed
python3 bench/bench_encoder_scaling.py --preset all --reps 20 \
        --out-prefix results/raw/encoder_scaling_$(hostname)
python3 bench/bench_cascade_onnx.py --threads 1 2 4 --skip-fp32 \
        --out-prefix results/raw/cascade_$(hostname)
python3 bench/make_figures.py
```

Nothing above needs L2-ARCTIC, a GPU, or a network connection (beyond install).

---

## What's here

```
src/sfac/
  causal.py      lookahead masks, frame arithmetic, streaming buffer   <- load-bearing
  latency.py     t_algorithmic / t_compute / t_buffer decomposition
  pipeline.py    configurable-lookahead harness (torch, lazy import)
bench/
  hardware_probe.py         machine identity + calibration + thermal state
  bench_encoder_scaling.py  compute vs (lookahead, chunk, model size) -- numpy, runs anywhere
  bench_cascade_onnx.py     S1 cascade, real sherpa-onnx models
  bench_commit_delay.py     how long until a streaming ASR stops revising a word
  bench_content_degradation.py  RQ1 lower bound + the causality proof
  make_figures.py
eval/
  metrics.py            WER / accent probe / ECAPA / NISQA + degradation control
  phoneme_analysis.py   RQ2 per-phone attribution, knee detection, pre-registered H2
data/
  download.sh                CMU ARCTIC + VCTK; L2-ARCTIC subset is on HF (gated)
  verify_prompt_overlap.py   gate: is the parallel-prompt assumption actually true?
train/
  train_translator.py       RQ1+RQ3 sweep: causal phone translator, CTC on g2p vs ipa
setup/     macOS M4 / Windows / GPU+Colab / Raspberry Pi
notebooks/ colab_streaming_fac.ipynb -- GPU sweep + training
tests/     the invariant guards
results/   raw/ (CSV + JSONL + host JSON), figures/
run_m4.command        full pipeline (double-click on macOS)
run_selftest.command  causality proof only, ~1 min
```

### The second invariant: prove causality, don't assert it

`bench/bench_content_degradation.py --selftest` deletes the audio after frame
*t* and checks whether an earlier frame's output moved. Anything above 1e-4
means the encoder is not causal and every lookahead label is wrong.
`train/train_translator.py` runs it and **aborts training** if it fails.

---

## The one invariant everything depends on

Lookahead `L` must be the **only** thing that varies across the sweep. Same
weights, same seed, same data order, same chunking, same step count. If
anything else co-varies, RQ1 is unanswerable and the paper is unsalvageable.

This is enforced in code, not by discipline:

```python
from sfac.pipeline import sweep_configs, assert_only_L_varies
assert_only_L_varies(configs)   # raises with a field-level diff
```

`tests/test_causal.py` additionally proves the streaming buffer reproduces the
offline training mask exactly — otherwise the model you benchmark is not the
model you trained.

---

## Measured findings (Apple M4 + ARM64, 2 Aug 2026)

0. **Masking attention does not make WavLM causal.** Two leaks sit *under* the
   transformer: `pos_conv_embed` (kernel 128 → **1.28 s** of future) and the
   feature-encoder **GroupNorm** (normalises over the *whole utterance* →
   unbounded). Truncation proof, relative L2: mask only 1.14e-2 → +pos-conv
   6.00e-3 → **both patches 6.14e-6 (causal)**. Neither fix works alone, and
   patching GroupNorm alone makes it *worse*. `-large` checkpoints are
   causal-safe; base ones are not.
1. **Per-chunk compute is flat in lookahead.** 0 → 640 ms costs **−1.5% to +14%**
   compute and up to **+3200%** algorithmic latency. Replicated across Apple
   Silicon/Accelerate and ARM64/OpenBLAS.
2. **Chunk size decides feasibility, lookahead doesn't.** A base-scale encoder
   at 20 ms chunks: **RTF 1.60 on an M4** (infeasible at every L). At 80 ms: 0.54.
3. **Peak FLOPS overpredicts streaming inference ~10×.** The M4 hits 74–164
   GFLOP/s against its own 1632 GFLOP/s sgemm peak, and is only ~1.4× faster
   end-to-end than a box with a quarter of its peak.
4. **`num_threads = cpu_count` is the wrong default on big.LITTLE.** 8 threads
   makes the ASR encoder **3.2× slower** than 1 and pushes Kokoro past RTF 1.0.
   Single-threaded wins at every metric on the M4.
5. **The cascade loses structurally.** Piper synthesises a word in **21 ms** vs
   Kokoro's **973 ms** (46×) — so with a fast vocoder the cascade is ~1.15 s and
   **95% of it algorithmic**. It fails because it waits for the recogniser to
   stop revising words, not because synthesis is slow.
6. **That commit delay is now measured, not asserted.** 28.8% of words get
   revised after first appearing. The trade-off has a **cliff at the model's
   320 ms decode chunk**: 27% released-unstable at 300 ms → **1.5% at 400 ms**
   → 0% at 700 ms. So `accentbridge`'s 700 ms is ~300 ms over-provisioned; at
   400 ms the cascade is ~853 ms and **still ~93% algorithmic**. The conclusion
   survives its own best-case repair.

Compute-side and correctness only. No accent-conversion model is trained yet —
finding 0 was the prerequisite for the lookahead labels meaning anything.

---

## Three corrections worth knowing before you read the v1 proposal

1. **PHONOS uses ≤40 ms lookahead**, stated in its abstract, and was posted
   27 March 2026. v1 left this blank and dated it July.
2. **PHONOS, TVTSyn and DarkStream are all from the same lab** (TAMU PSI).
   That concentrates the competitive risk and makes emailing them the highest-
   value action in the whole plan.
3. **L2-ARCTIC no longer needs a signed form** for the annotated subset — it is
   on Hugging Face as `KoelLabs/L2Arctic` (gated, CC-BY-NC-4.0, 3599 utterances
   with both `g2p` and `ipa`). The non-commercial licence settles the
   model-weights question. The full 26,867-utterance corpus still needs the
   TAMU request, so start that anyway.

Full detail in [`docs/LITERATURE.md`](docs/LITERATURE.md).

---

## Relationship to `../accentbridge`

`accentbridge/` is the working real-time ASR→TTS prototype (sherpa-onnx
streaming zipformer + Kokoro/Piper + Silero VAD). It is the **S1 baseline**.
`bench/bench_cascade_onnx.py` benchmarks exactly its model stack, so the S1 row
of the paper and the live demo share an implementation.
