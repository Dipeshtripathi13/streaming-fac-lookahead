# Findings log

The running record of what this project measured, in the order it was
measured, including the entries that were later superseded. Superseded
findings are struck through and annotated rather than deleted, because how a
number moved is part of the evidence.

**Where this log disagrees with the paper, the paper is right.**
See [`paper/taslp/taslp.pdf`](../paper/taslp/taslp.pdf).

---

## Measured findings (Apple M4 + ARM64, 2 Aug 2026)

0. **Masking attention does not make WavLM causal.** Two leaks sit *under* the
   transformer: `pos_conv_embed` (kernel 128 → **1.28 s** of future) and the
   feature-encoder **GroupNorm** (normalises over the *whole utterance* →
   unbounded). Truncation proof, relative L2: mask only 1.14e-2 → +pos-conv
   6.00e-3 → **both patches 4.58e-6 (causal)**. Neither fix works alone, and
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
   Kokoro's **973 ms** (46×), so with a fast vocoder the cascade is ~1.15 s and
   **95% of it algorithmic**. It fails because it waits for the recogniser to
   stop revising words, not because synthesis is slow.
6. **That commit delay is now measured, not asserted.** 28.8% of words get
   revised after first appearing. The trade-off has a **cliff at the model's
   320 ms decode chunk**: 27% released-unstable at 300 ms → **1.5% at 400 ms**
   → 0% at 700 ms. So `accentbridge`'s 700 ms is ~300 ms over-provisioned; at
   400 ms the cascade is ~853 ms and **still ~93% algorithmic**. The conclusion
   survives its own best-case repair.

7. ~~**RQ1, first real answer (198 utts, Tesla T4).** Broad optimum in the
   exchange rate; gain per doubling peaks at +0.081 for 100→200 ms.~~
   **SUPERSEDED** by the 16-point dense sweep. See the paper §RQ1 and
   [`docs/DENSE_SWEEP_RQ1_ANSWERED.md`](docs/DENSE_SWEEP_RQ1_ANSWERED.md).
   The per-doubling profile was too noisy at 7 points to locate an optimum;
   the curve is smooth with **no locatable knee**, an exchange rate of
   **−0.045 PER per doubling (R² 0.983)** and **saturation at ≈340 ms**
   against a measured 2σ = 0.0029 repeatability floor.

8. **The trained translator (H3 supported).** 14 conditions on a T4, 2.5 h.
   Conversion (CTC on canonical `g2p`) gains more from lookahead than
   transcription (`ipa`), on arms that differ in one label tensor,
   **1.55×** (−0.045 against −0.029 PER per doubling). The claim is carried by
   the *preference margin*, which grows monotonically in 3/3 seeds
   (**+0.032 at L=0 → +0.099 at 640 ms**) while the transcription control is
   0/3 monotone and drifts down: more right context makes the model *more
   converting*, not merely more accurate.
   *(The 1.48× and +.036 → +.103 originally reported here predate the padding
   fix; see [`PADDING_FIX_RESOLVED.md`](docs/PADDING_FIX_RESOLVED.md).)*

9. **The exchange rate does not transfer across encoders.** Repeating
   the whole sweep on causalised wav2vec 2.0 base, same corpus, split, head,
   optimiser, budget, seed and grid held fixed, gives
   **+3.9%** relative gain from 640 ms of lookahead against WavLM's **+63.3%**,
   from a nearly identical starting point (0.4359 vs 0.4256 at L=0). The
   direction replicates; the magnitude does not. See
   [`docs/SECOND_ENCODER.md`](docs/SECOND_ENCODER.md).

10. **Match encoder layers by function, not by index.** Tapping wav2vec 2.0 at
    WavLM's layer 9 produces a degenerate run (PER *rises* to 0.93). Phone
    decodability falls monotonically with depth in wav2vec 2.0, 0.402 at
    layer 2 up to 0.971 at layer 10, so layer 9 is seven layers past its
    usable representation.

11. **RQ2's uncertainty belongs at the speaker level.** Bootstrapping the
    manner-class rates over the six test speakers rather than 42,255 phone
    tokens puts the between-group difference at +0.040 with a 95% CI of
    **[−0.010, +0.101]**, which includes zero. Token-level intervals would
    have been ~10× too narrow. Each L1 in the test split has **exactly one
    speaker**, so the by-L1 breakdown is fully confounded with speaker
    identity. See [`docs/RQ2_HIERARCHICAL.md`](docs/RQ2_HIERARCHICAL.md).

Correctness, latency, encoder-level and task-level results. **No audio is
synthesised and no listening test has been run.** That is now a deliberate
conclusion rather than a gap: phone-string outputs lack the lexical and
prosodic structure needed to judge accentedness, so no listening-test budget
would have measured what the paper claims. The paper's Limitations says so.

**Two errors worth reading about** (`docs/RESULTS_M4.md` F8): a 10 ms sampling
grid was invalid because lookahead is quantised to the 20 ms frame rate (half
the points were duplicates), and the knee test was deciding on an arbitrary R²
threshold. Both are fixed; both are documented rather than quietly corrected.

---

