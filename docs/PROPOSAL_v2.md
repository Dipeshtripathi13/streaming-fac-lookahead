# How Much Future Do You Need? — Proposal v2

### Latency–Quality Trade-offs in Streaming Accent Conversion on Commodity Hardware

**Target:** Interspeech 2027, São Paulo (29 Aug – 2 Sep 2027)
**Deadline:** not yet announced; plan for **1 March 2027** (historical 25 Feb – 21 Mar)
**Fallback:** SSW14, deadline **20 April 2027** (verified)
**Status:** v2.1, 2 August 2026 — supersedes v1. Driven by
`docs/LITERATURE.md` (citation verification), `docs/RESULTS_M4.md` (Apple M4)
and `docs/RESULTS_PILOT.md` (ARM64). All §7 numbers are measured.

---

## What changed from v1

| | v1 | v2 | Why |
|---|---|---|---|
| PHONOS lookahead | blank / "inherited from TVTSyn" | **≤40 ms**, stated in the abstract | v1 error |
| PHONOS date | July 2026 | **27 March 2026** | v1 error |
| PHONOS provenance | treated as outside work | **same lab** as TVTSyn + DarkStream | v1 error; changes the risk model |
| H1 | "knee is above 80–140 ms; budgets are under-provisioned" | restated as a two-sided question | a 40 ms AC system exists and works |
| RQ4 | open question | **partially answered by pilot data** | measured |
| New RQ5 | — | chunk-size/lookahead asymmetry | emerged from the pilot |
| Cascade (S1) | "nobody has published these numbers" | **measured: ~1.15 s with a fast vocoder, 95% algorithmic** | done |
| Pi via simulator | proposed | **rejected — buy the board** | simulators cannot measure microarchitecture |
| Encoder causality | assumed to follow from attention masking | **two leaks found and fixed; now contribution 5** | measured, §7 F0 |
| RQ1/RQ3 route | wait for the full synthesis pipeline | **trainable phone-translator sweep, ~1 GPU-day** | L2-ARCTIC's `g2p`/`ipa` columns make it possible now |

---

## 1. Pitch

Streaming accent conversion exists and works: PHONOS (Quamer et al., Mar 2026)
runs at ≤40 ms lookahead and cuts non-native accent confidence by 81%. What
does not exist is any account of **why 40 ms**, what it costs in quality, or
what any of it does on hardware people own.

We sweep lookahead from 0 to 640 ms with everything else held byte-identical,
decompose the resulting degradation by phoneme class, run the same pipeline in
accent-conversion and voice-conversion-only modes to isolate what is
accent-specific, and measure all of it across four hardware classes including
CPU-only laptops and a Raspberry Pi. We report algorithmic and computational
latency separately throughout — a distinction the field currently collapses,
and one our measurements show is not cosmetic: **over the full 0–640 ms
lookahead range, per-chunk compute changes by −1.5% to +14%, while algorithmic
latency changes by up to 32×** (replicated on Apple Silicon and ARM64, two
different BLAS libraries).

We also found that the standard way of building a "causal" SSL encoder does not
produce one: masking self-attention in wav2vec2/WavLM-base leaves a 1.28 s
positional-convolution leak and an utterance-global GroupNorm dependency. We
supply patches and a truncation proof (§7 F0).

The contribution is a characterisation and a public evaluation harness, not a
new architecture. That is what makes it tractable for a small team and hard to
scoop.

---

## 2. The gap, restated honestly after verification

PHONOS, TVTSyn (arXiv:2602.09389, <80 ms GPU) and DarkStream (arXiv:2509.04667)
are three papers from one group at Texas A&M. StreamVC (ICASSP 2024) runs
streaming VC "even on a mobile platform" without naming the device, core count,
or thread count.

| Gap | Status after verification |
|---|---|
| **CPU / embedded numbers** | **Strongest.** All GPU, or unfalsifiable. Nobody has published a CPU latency–quality curve for streaming AC. |
| **Swept lookahead budgets** | **Narrowed but real.** PHONOS *states* 40 ms; it publishes no ablation showing 40 ms is necessary or sufficient. The gap is "nobody swept it", not "nobody chose one". |
| **Phoneme-class decomposition** | **Holds.** All results are aggregate MOS/WER/accentedness. |
| **Algorithmic vs computational latency** | **Holds, and cheap to fix.** "≤241 ms on single GPU" fuses a 40 ms modelling choice with unstated inference and buffer costs. A reader cannot tell whether a faster chip helps. |
| **AC vs VC comparison** | **Holds.** No matched-capacity comparison exists. |

**Defence against a reviewer:** we do not claim to beat PHONOS on quality. We
characterise an axis it parameterises but does not study, on hardware it does
not test. Both positive and negative results are publishable — see §11.

---

## 3. Research questions

**RQ1.** How does accent-conversion quality degrade as lookahead falls from
640 ms to 0?

> **H1 (restated).** The degradation curve has a knee. v1 predicted the knee
> sits above current budgets. PHONOS's working 40 ms system makes the
> one-sided version untenable, so H1 is now two-sided and the interesting
> outcomes are symmetric:
>
> - **Knee ≫ 40 ms** → current budgets are under-provisioned; PHONOS's 81%
>   accent reduction is leaving quality on the table that a modest latency
>   increase would recover.
> - **Knee ≤ 40 ms** → the field converged on roughly the right budget by
>   engineering intuition, and *this paper is the evidence for that*, which is
>   worth publishing precisely because it is currently assumed rather than shown.
> - **No knee (linear)** → there is no natural operating point and every
>   deployment is a bare product decision. Also publishable, and the most
>   actionable of the three.
>
> Pre-register the knee-location estimate before running. Bootstrap CI on a
> knee from 7 points is wide; say so.
>
> **PROVISIONAL ANSWER (§7 F8, 198 L2-ARCTIC utterances on a T4,
> encoder-level): no cliff, but a broad optimum in the exchange rate.**
> R² = 0.9942 against log₂(L) vs 0.7267 against L — overwhelmingly log-linear —
> yet BIC also prefers a two-segment fit (ΔBIC = −22.9). Both are true: strong
> log-linearity *plus* small real curvature. The model-free reading settles it:
>
> | doubling | Δ divergence |
> |---|---:|
> | 20 → 40 ms | +0.050 |
> | 80 → 160 ms | +0.079 |
> | **100 → 200 ms** | **+0.081** ← peak |
> | 320 → 640 ms | +0.053 |
>
> **Nothing distinguishes 40 ms — it sits on the least productive doubling
> measured.** Best marginal return is **100–200 ms**, 2.5–5× PHONOS's budget.
> So "current budgets are under-provisioned" is defensible as
> *diminishing-returns geometry*, not as a knee. Report the exchange rate
> (~0.08 per doubling, peaking at 100–200 ms), not an operating point.
> Caveat: representation drift, not conversion quality — the trained sweep is
> the real test.
>
> **THE TRAINED SWEEP NOW ANSWERS IT (§7 F14, 42 conditions, padding-fixed):
> curvature is real in the conversion arm, and its location is not
> identifiable at 7 points.** BIC prefers two segments (native ΔBIC = −10.1,
> produced −7.4) where the pre-fix data preferred log-linear (+5.3). So H1's
> third branch — "no knee" — is **withdrawn**: it was an artefact of the
> zero-pad bug, which damaged large-lookahead conditions most and so flattened
> the curve.
>
> **But do not print a knee location.** It moves with an arbitrary modelling
> choice — where L=0 sits on a log axis:
>
> | treatment of L=0 | native | produced |
> |---|---|---|
> | all 7 points, log₂(L+1) | 40 ms (ΔBIC −10.1) | 40 ms (−7.4) |
> | L=0 dropped | 160 ms (−9.2) | **no knee** (−4.0) |
> | L=0 placed at 10 ms | 160 ms (−10.8) | **no knee** (−3.4) |
>
> On log₂(L+1) the L=0→20 ms gap is 4.39 units while every other gap is ~1.0,
> so a piecewise fit puts its breakpoint just past that gap whatever the data
> does. The power check confirms mislocalisation: a planted 0.05 cliff at 81 ms
> is "detected" at 40 ms. Right conclusion, wrong place.
>
> **RQ1 IS NOW ANSWERED — and the answer is none of H1's three branches
> (§7 F15, `docs/DENSE_SWEEP_RQ1_ANSWERED.md`).** The dense sweep — 16
> lookaheads × 2 seeds, 0–200 ms in 20 ms steps plus 240/280/320/480/640 —
> settles it:
>
> **There is curvature but no locatable knee, and a saturation point at
> ~240 ms.** All three axis treatments prefer piecewise decisively (ΔBIC −22,
> −31, −37) yet the breakpoint moves (40, 180, 180 ms) and every bootstrap
> interval spans ~3 octaves. **Sixteen points did not rescue a knee**, which is
> the finding: the earlier "7 points are underpowered" diagnosis was right about
> the power and wrong about the cause. The curve is smoothly curved, not
> piecewise, and a breakpoint estimator asked for a breakpoint always returns
> one.
>
> What to report instead, both stable and citable:
>
> - **Exchange rate: −0.0452 PER per doubling of lookahead, R² = 0.983**
>   (n = 15, L = 20–640 ms).
> - **Saturation at ~240 ms**: marginal gain per added 20 ms frame stays above
>   the measured 2σ = 0.0044 floor until 240 ms, and is below it for every step
>   thereafter. Read as 200–280 ms, not a sharp figure — the 140→160 and
>   180→200 steps are already only 1.0–1.5× the floor.
> - **The cost of the field's 40 ms budget: ~40% of the achievable PER
>   reduction** (0.3275 at 40 ms → 0.1973 at 240 ms, a 39.7% relative
>   reduction forgone; 52.3% out to 640 ms).
>
> So "current budgets are under-provisioned" is now supported on the **trained
> conversion curve**, not merely on encoder drift — and it converges with F8's
> independent encoder-level estimate of best marginal return at 100–200 ms.
>
> H1's branches (knee ≫ 40 / knee ≤ 40 / no knee) were the wrong trichotomy:
> the outcome is *smooth diminishing returns with a measurable saturation
> point*. Record that as a pre-registration lesson rather than quietly
> reclassifying it.

*Two methodological notes that belong in the paper, both from getting this
wrong first (§7 F8):* **(a)** lookahead is quantised to `ceil(L/frame_ms)`
frames, so a knee narrower than the 20 ms frame rate is not merely unmeasured
but *unrepresentable* — and sampling finer than that silently creates duplicate
conditions that reweight the fit. **(b)** A 7-point geometric grid is
**underpowered** for knee detection: a synthetic planted cliff returns
ΔBIC = −0.2 on n = 7. Any "no knee" claim needs ≥ ~16 distinct conditions.

**RQ2.** Is degradation uniform across phoneme classes?

> **H2 (unchanged).** No. Consonant substitutions (th-stopping, /v/–/w/,
> retroflex stops) are realised over 20–80 ms and largely determined by the
> local gesture; they should survive short lookahead. Vowel quality and
> diphthong trajectories are defined by formant movement over 80–250 ms and
> shifted by the following consonant; they should break first. Approximants
> pattern with vowels. Pre-registered in `eval/phoneme_analysis.py:H2_PREDICTION`.
>
> **First evidence (§7 F7): right direction, very small.** On a voicing ×
> spectral-flux proxy, the sonorant/steady bucket gains **+0.0129** more from
> lookahead than obstruent/transient (paired, n = 48, bootstrap CI
> [+0.0026, +0.0230]). Excludes zero, but 1.3 points on a crude proxy is a
> reason to look harder, not a phoneme-class result. Needs forced alignment
> against L2-ARCTIC's phone annotations.

**RQ3.** How much of the lookahead requirement is accent-specific?

> **H3 (unchanged, but now with a cleaner and cheaper test).** Accent
> conversion needs strictly more lookahead than surface-level speech processing
> at matched capacity.
>
> Original test: identical pipeline in AC and VC-only modes (`sfac.pipeline.Mode`).
> Kept for the full synthesis study.
>
> **ANSWERED (§7 F12/F13, 42 conditions = 3 seeds on a T4): H3 SUPPORTED.**
> The conversion-trained model's preference for canonical over produced phones
> grows **monotonically in 3/3 seeds** (+.032 at L=0 → +.099 at L=640; 18/18
> adjacent step × seed comparisons positive, sign p=7.6e-6), while the
> transcription control is 0/3 monotone and drifts *down*. Paired by seed the
> gap is t(2)=+41. At matched own-target PER the conversion margin is 3.46× the
> transcription margin for L≥160 ms and 0.88× at L=0, so lookahead — not
> accuracy — creates the divergence.
>
> **SUPERSEDED by the padding fix (§7 F14, `docs/PADDING_FIX_RESOLVED.md`).**
> The paragraph that stood here said PER was too seed-noisy to carry
> per-condition claims, that only 0→20 ms resolved at α=.05, and that the
> mid-range plateau was noise. All three statements were consequences of the
> MaskedBlock zero-pad bug, not of PER. On the corrected sweep:
>
> - **All six adjacent steps resolve**, in both arms, unanimous across 3/3
>   seeds (native: −.058, −.042, −.050, −.054, −.036, −.028).
> - The apparent 160→320 ms plateau is gone: **−.036 PER, 3/3 seeds**. The
>   earlier −0.0009 with 1/3 seeds improving was an artefact.
> - Endpoint 0→640 ms strengthens to **−0.270 ± 0.0007, 63.2% relative**
>   (was −0.212 ± 0.018, 47.6%). The seed SD fell 26×.
>
> PER is *not* too seed-noisy. It looked that way because the bug was itself a
> seed-varying noise source. The margin remains the right statistic for H3 —
> but for the reason given in F14, not because PER is unusable.
>
> **The test, available now:** the same causal phone translator trained
> against the CANONICAL phone sequence (`g2p`) versus the PRODUCED one (`ipa`).
> Conversion must decide what the speaker *should* have said — lexically and
> coarticulatorily conditioned. Transcription need only report the local
> gesture. Identical architecture, capacity, seed, data order and steps; the two
> arms differ **in one label tensor**, which is a tighter control than AC-vs-VC
> and costs ~1 GPU-day rather than 200 GPU-hours (`sfac.translator.Target`).

**RQ4.** What is the achievable latency–quality Pareto frontier on CPU, and
what dominates the CPU/GPU gap?

> **H4 (unchanged).** The gap is dominated by the vocoder and by non-causal
> operations that resist quantisation, not by parameter count. Tested by
> per-stage timing on identical inputs across hardware classes; if
> `gap_vocoder ≫ gap_encoder` despite fewer parameters, H4 holds.

**RQ5 (new, from the pilot).** Do lookahead and chunk size trade against
compute in the same direction?

> **H5.** No, and this is why fused latency numbers mislead. **Already
> measured on two architectures:** across 0–640 ms of lookahead, per-chunk
> compute moves −1.5% to +14%; across 20→80 ms of chunk size, per-chunk compute
> rises ~15–34% while RTF *falls* 3–3.5×. Cutting lookahead is nearly free in compute and costs quality.
> Cutting chunk size costs compute and does not cost quality. A practitioner
> optimising "latency" without separating the two will make the wrong trade.
> Full test: confirm the asymmetry holds once quality is in the loop.

---

## 4. Method

Not a new architecture — a *configurable* one. Implemented in
`src/sfac/pipeline.py`.

```
        ┌────────────────────────────────────────────────┐
  audio │  causal SSL content encoder (WavLM-base, L-masked)
  ──────┼─▶ conversion module   MODE ∈ {AC, VC_ONLY}     │
        │  ─▶ causal vocoder (HiFi-GAN causal)           │
        └────────────────────────────────────────────────┘
           instrumented: t_algorithmic | t_compute | t_buffer
```

**The invariant.** `L` controls the attention-mask width and nothing else.
Same weights, same seed, same data order, same step count across all 14 runs.
This is enforced in code, not by convention: `assert_only_L_varies()` compares
config fingerprints and refuses a confounded sweep with a field-level diff
(`tests/test_pipeline_invariants.py`).

**Three design details that are easy to get wrong and would invalidate the sweep:**

1. **Chunkwise masks, not per-frame masks.** In chunkwise streaming, frames
   early in a chunk get free intra-chunk right context. A per-frame training
   mask therefore gives the model *less* context at train time than at test
   time, appearing as an unexplained quality gain at large chunks and muddying
   the L sweep. `causal.chunked_lookahead_mask` handles this; the streaming
   buffer is tested to reproduce it exactly.

2. **Report the irreducible frontend lookahead.** A wav2vec2/HuBERT conv
   frontend has a 25 ms receptive field at a 20 ms frame rate, so "L = 0 ms"
   still sees 5 ms of future. `StreamGeometry.algorithmic_ms_honest` reports
   chunk + L + 5 ms. A paper claiming 0 ms lookahead over a 25 ms conv window
   is misreporting, and a reviewer who knows the frontend will notice.

   **And a much larger one: masking attention does not make WavLM causal.**
   `WavLMPositionalConvEmbedding` is a depthwise Conv1d with a wide kernel and
   *symmetric* padding, applied before the transformer stack. At a 20 ms frame
   rate a kernel of 128 leaks ~64 frames — **~1.28 s of future** — into every
   frame. Any "streaming" system that masks self-attention and leaves the
   positional convolution untouched has an unreported second of lookahead, and
   its L = 0 condition is not L = 0.

   We patch it to left-only padding, and we do not take the patch on trust:
   `bench/bench_content_degradation.py --selftest` runs a **truncation proof** —
   delete the audio after frame *t*, and assert that the output at frame
   *t − L − 1* is bit-identical — with the fix and without. The unfixed run is
   expected to fail. Training aborts if the fixed run does not pass
   (`train/train_translator.py --verify-causality`, on by default). Every
   lookahead label in the paper depends on this, so it is a gate rather than a
   check.

3. **Ceil, not round, when converting ms to frames.** Rounding down gives the
   model less context than the condition label claims. Ceil makes the reported
   L an upper bound on the true one — the safe direction for a latency claim.

**Backbone.** Adapt an existing open reference-free FAC system to causal
operation; prefer whatever PHONOS or the TAMU lab releases. Fall back to
reimplementing Zhao et al. (2021) with a causal encoder.

**Training targets must not depend on L.** Use PHONOS's golden-target recipe:
generate native-segmental targets **offline and non-causally** (silence-aware
DTW between the L2-ARCTIC utterance and the CMU ARCTIC rendition of the same
prompt, then zero-shot VC onto the source timbre), freeze them, hash them,
record the hash in every checkpoint. If the target were produced by a streaming
teacher it would vary with L and RQ1 would be unanswerable.

### Latency decomposition (a contribution in itself)

```
t_algorithmic = chunk + lookahead (+ frontend receptive field)   hardware-independent
t_compute     = encoder + conversion + vocoder                   hardware-dependent
t_buffer      = I/O + jitter buffer                              system-dependent
```

Every table reports all three. `sfac.latency.LatencyBudget` computes the
percentile of the per-chunk *sum* rather than summing per-stage percentiles —
the former is what a user experiences; the latter is conservative and assumes
worst-case co-occurrence.

---

## 5. Experimental matrix

| ID | System | Role | Status |
|---|---|---|---|
| S0 | Passthrough | Floor; probe sanity check | ready |
| S1 | Streaming ASR → TTS cascade | Accent-perfect upper bound | **measured, §7** |
| S2 | Offline reference-free FAC | Quality ceiling | Aug/Sep |
| S3 | PHONOS *(if released)* | Streaming SOTA | contingent |
| S4a | **Ours: causal phone translator, L swept** | RQ1 + RQ3, cheap route | **built, smoke-tested; blocked on HF login** |
| S4b | **Ours: configurable-lookahead FAC (synthesis)** | The full sweep | harness built, untrained |
| S5 | kNN-VC / Seed-VC, VC-only | Control: would a generic VC have done this? | Sep |

**Sweep:** L ∈ {0, 20, 40, 80, 160, 320, 640} × mode {AC, VC-only} × hardware
{M4 CPU, x86 CPU, Pi 5, GPU} × accent pair {Hindi, Mandarin, Spanish,
Arabic}→GA = **224 conditions** objective.
Secondary chunk sweep {20, 40, 80} ms at L = 80.
Human eval subset: L ∈ {0, 40, 80, 160, 640} × 2 modes × 2 pairs = **20 conditions**.

**Held out:** Korean and Vietnamese L2-ARCTIC speakers, as an unseen-L1
generalisation test. Costs inference only and pre-empts "is the lookahead
requirement L1-specific?"

---

## 6. Evaluation

| Metric | Tool | Role |
|---|---|---|
| Accentedness | frozen accent probe over SSL features | headline |
| Intelligibility | WER, **faster-whisper large-v3, third-party** | **guardrail only** |
| Naturalness | NISQA-MOS | artefacts |
| Speaker similarity | ECAPA-TDNN cosine | identity |
| Latency | instrumented, p50/p95, all three components | independent variable |
| RTF | compute ÷ audio duration | feasibility gate (p95 < 0.8) |

Three commitments that make the numbers believable:

- **WER never ranks systems.** Huang & Toda showed intelligibility correlates
  poorly with subjective accentedness, and a system can improve WER by
  flattening prosody into neutral robot speech. Guardrail with an explicit
  failure band.
- **The scoring ASR is never a pipeline component.** Using the cascade's own
  recogniser to score the cascade is circular.
- **The degradation control is mandatory.** The accent probe will happily
  report "less accented" for audio that is merely more degraded. We score
  unconverted audio degraded to matched NISQA (`metrics.degradation_control`).
  If the probe drops there too, the headline claim is discounted accordingly.
  This is the cheapest possible defence against the single most damaging
  reviewer question.

**Subjective:** *not funded for now* (decision 2026-08-09, see
`docs/DECISIONS.md`). The design and tooling are complete — `listening_test.py`
builds condition-blind stimuli from the hypothesis dumps via a single fixed
Piper voice, with ceiling/floor attention checks and a Bradley–Terry scorer, and
74 hyps files are already on disk. Only the raters were cut, so this is a budget
decision rather than a capability gap.

> **Wording discipline that follows.** Every headline number in this paper is a
> phone-sequence measurement. The saturation point (~240 ms) says marginal *PER*
> gain falls below the noise floor; it does **not** say listeners stop noticing
> improvement. The "40% forgone at 40 ms" figure is PER-relative, not
> quality-relative. No claim about perceived naturalness or accentedness is
> supported. H3 is unaffected — the preference margin is objective and needs no
> listeners.
>
> The honest risk: for an accent-conversion paper, accentedness is definitionally
> perceptual, so this is the most likely reviewer objection and a fair one. Two
> zero-cost mitigations remain available (build the stimuli and listen
> informally; NISQA-MOS as an automatic naturalness proxy with
> `metrics.degradation_control` as its guard) — see `docs/DECISIONS.md`.

**Statistics:** mixed-effects model with speaker, rater and utterance as random
effects. Report CIs, not means. Pre-register H1's knee and H2's class ordering.

---

## 7. Results already in hand

Two machines, 2 Aug 2026. Detail in `docs/RESULTS_M4.md` (Apple M4,
`cpu-apple-silicon`) and `docs/RESULTS_PILOT.md` (Neoverse-N1, `cpu-arm64`).
Everything below is measured, not projected.

### F0 — masking attention does not make WavLM causal. **This is now a contribution.**

A *truncation proof* (delete audio after frame *t*; check whether an earlier
frame's output changed) over four ablations on `wavlm-base-plus`, tolerance 1e-4:

| configuration | relative L2 change | causal? |
|---|---:|---|
| attention mask only | 1.14e-2 | no |
| + causal positional conv | 6.00e-3 | no |
| + cumulative GroupNorm | 2.60e-2 | no |
| **both patches** | **6.14e-6** | **yes** |

Two leaks sit *underneath* the transformer stack:

1. `pos_conv_embed` — depthwise Conv1d, **kernel 128**, symmetric padding:
   **1.28 s of future** into every frame at a 20 ms frame rate.
2. The feature-encoder **GroupNorm** — base checkpoints use
   `feat_extract_norm="group"`, normalising each channel over the *entire*
   utterance. Every output frame depends on every input frame: an unbounded
   dependency, not merely a long one. (The `-large` checkpoints use
   `feat_extract_norm="layer"` and are causal-safe — so this is specific to the
   base-sized encoders a CPU system would actually pick.)

Neither fix is sufficient alone, and **patching GroupNorm alone makes the
measured leak worse (2.6e-2)** — a partial causality fix is not a partial
improvement. We have not found either leak discussed in the streaming AC/VC
literature we have read.

**Why this changes the paper.** It elevates "report your lookahead honestly"
from a pedantic framing device to a substantive claim: any base-SSL streaming
system that masks self-attention and stops there has an unreported second of
lookahead plus an utterance-global dependency, and its *L* = 0 condition is not
*L* = 0. This becomes contribution 5 (§11) and probably the methods section.
The proof is a gate in the code, not a footnote: training aborts if it fails.

### F1 — per-chunk compute is flat in lookahead (replicated on two architectures)

0 → 640 ms of lookahead: **−1.5% to +13.9%** compute on M4 (63 conditions),
**+5.7% to +12.2%** on ARM64. Two of nine M4 cells are slightly negative, i.e.
within noise of zero. Meanwhile algorithmic latency changes by up to **32×**.
Different chip, different BLAS (Accelerate vs OpenBLAS), same conclusion.

### F2 — chunk size, not lookahead, decides feasibility

A base-scale (768-dim, 12-layer) encoder at 20 ms chunks: **RTF 1.599 on an
M4** — behind real time at *every* lookahead. At 80 ms chunks, RTF 0.535. No
reduction in *L* rescues the 20 ms configuration; a chunk-size change does.

### F3 — peak FLOPS overpredicts streaming inference by ~10×

The M4's sgemm peak is **4.0× the ARM64 box** (1632 vs 403 GFLOP/s). On this
workload it achieves **74–164 GFLOP/s — 5–10% of its own peak** — and is only
~**1.4×** faster end to end. Streaming transformer inference is a sequence of
small matmuls, bound by per-call overhead and memory movement, not arithmetic.
Sizing a deployment from a spec-sheet TFLOPS figure is wrong by an order of
magnitude, optimistically. Rescaling the Pi 5 estimate by *achieved* throughput
gives ~4–8× slower than M4, not the 7–13× a peak-FLOPS estimate implies.

### F4 — `num_threads = cpu_count` is the wrong default on big.LITTLE

M4 (4 P-cores + 6 E-cores), real sherpa-onnx models:

| threads | ASR active step p50 | Kokoro RTF | Piper 1-word |
|---:|---:|---:|---:|
| 1 | **11.8 ms** | **0.78** | **21.0 ms** |
| 2 | 12.6 | 0.85 | 23.4 |
| 4 | 17.9 | 0.92 | 25.6 |
| 8 | 37.7 | **1.10** | 33.4 |

Single-threaded is optimal at every metric. Eight threads makes the ASR encoder
**3.2× slower** and pushes Kokoro past RTF 1.0 — from real-time to infeasible,
purely by asking for more parallelism. Once the pool exceeds the P-core count,
work lands on E-cores and the op waits on its slowest thread. ARM64 (4
homogeneous cores) showed a milder form: 2 threads best, 4 threads 69% worse at
p95. A one-line default that nearly everyone ships wrong.

### F5 — the cascade (S1) loses structurally, and more so with a fast vocoder

Kokoro vs Piper/VITS on identical inputs: **973 ms vs 21.0 ms** to synthesise a
single word — **46×**. So the TTS bottleneck is a model choice. Substituting the
fast vocoder does not save the cascade; it clarifies why it fails:

```
                                       Kokoro     Piper/VITS
t_algorithmic  ASR chunk 320 + rctx 70   390 ms       390 ms
               + commit timeout          700 ms       700 ms
t_compute      ASR active step            12 ms        12 ms
               TTS, one word             973 ms        21 ms
t_buffer       (not yet measured)        ~30 ms       ~30 ms
                                       ────────     ────────
t_end_to_end                            ~2105 ms     ~1153 ms
algorithmic share                           52%          95%
```

**~1.15 s, 95% of it algorithmic.** The cascade does not lose because synthesis
is slow — it loses because it must wait for the recogniser to stop revising
words. A faster chip cannot help.

### F6 — and that commit delay is now measured, not asserted

`bench/bench_commit_delay.py` streams real speech in 20 ms increments and
tracks when each word stops being revised. **28.8% of words are revised after
first appearing**, so the delay is genuinely needed. The trade:

| commit timeout | words released while unstable |
|---:|---:|
| 0–300 ms | 27–29% |
| **400 ms** | **1.5%** |
| **700 ms** | **0.0%** |

**The cliff sits at the model's own 320 ms decode chunk** — the recogniser's
chunk size sets the granularity at which any cascade on it can commit, and no
tuning goes below it. Independently, the stabilisation latency (sound ends →
word stable) is p95 **260 ms**, max **340 ms**.

So 700 ms is over-provisioned by ~300 ms. Taking the most favourable
correction — commit at 400 ms, accept 1.5% instability — the cascade becomes
**~853 ms, still ~93% algorithmic** and still 3.5× PHONOS's ≤241 ms. **The
conclusion survives its own best-case repair**, which is the version to publish.
Remaining caveat: 66 words of clean read speech sets the shape, not the number;
re-run over L2-ARCTIC, where accented speech should revise more.

### F7 — RQ1 on real speech: no knee, and an estimator that nearly lied

48 L2-ARCTIC utterances, 8 per L1 across six L1s, provably-causal WavLM.
Divergence from the bidirectional representation falls **log-linearly** in
lookahead: R² = **0.9955** against log₂(L), 0.7746 against L, a constant
**−0.081 per doubling** from 20 to 640 ms, 47/48 utterances individually
monotone. No inflection anywhere. This is H1's pre-registered third branch.

**The methodological part is as important as the number.** The first pass used
maximum-distance-to-chord and reported *"knee at 160 ms, bootstrap 95% CI
[160, 160]"* — a tight interval around a plausible, headline-shaped value, four
times PHONOS's budget. It was an artefact: the chord estimator always returns a
point, and on a log-smooth curve it returns the middle of the sampled range;
the CI was tight *because* every bootstrap resample of a log-linear curve is
also log-linear. `find_knee` now fits log₂(L) first and refuses to name a knee
when that fit is good, reporting slope-per-doubling instead, with a regression
test on a synthetic log-linear curve.

Three caveats that must travel with this: it measures representation drift, not
conversion quality (the trained sweep is the real test); 7 geometric points
cannot see a knee narrower than an octave, and denser sampling costs 37 s; and
the 1−CKA variant does register a knee at R² = 0.987, so the two metrics
disagree at the margin.

### F14 — a padding bug was flattening the curve, and fixing it changed RQ1's answer

Full write-up: `docs/PADDING_FIX_RESOLVED.md`. This supersedes the open question
in `docs/CACHING_CHANGED_THE_NUMBERS.md`.

`MaskedBlock` handed a zero-padded hidden-state tensor to a depthwise
convolution and feed-forward that do not respect the key-padding mask. The
42-condition sweep was re-run on the fixed code with everything else held —
same seeds, steps, batch, chunk, device class, feature-cache path — and the
comparability guard confirmed no hyperparameter drift.

**It was not a constant offset, which is the only case that would have been
harmless.** The correction grows with lookahead: −0.014 PER at L=0 rising to
−0.086 at L=320, a spread of 0.072 against a measured noise floor of 0.0044
(**16.4×**; slope −3.98 SE from zero; 42/42 cells improved). The bug did most
of its damage where lookahead was longest, so the old curve **understated the
benefit of lookahead** — biasing RQ1's central quantity toward "lookahead does
not help much."

Consequences: all six adjacent steps now resolve in both arms and are unanimous
across seeds; the endpoint gain strengthens to −0.270 ± 0.0007 (63.2% relative,
from 47.6%) with seed SD down 26×; the L=160→320 ms "plateau" was an artefact
and must not be explained in the paper; and BIC now prefers a two-segment fit,
so H1's "no knee" branch is withdrawn — with the location caveat in §3.

**H3 is untouched, and by construction rather than luck.** It lives on the
preference margin (cross − own), and the bug moved both PERs together, so it
cancels in the difference: margin growth 0→640 ms is +0.0665 against +0.0690
before, still 3/3 monotone and 18/18 positive. The claim the paper leans on
hardest is the one this bug could never have manufactured — an argument for
building headline claims on differences rather than levels.

**A fixed-seed non-determinism floor, measured by accident.** Two sweep
processes briefly ran concurrently (operator error), repeating 6 conditions at
identical seed and configuration — the only direct estimate of run-to-run noise
available, and one a 3-seed design cannot produce. Differences ranged 0.00015 to
0.00649 PER, giving σ = **0.0022** and a 2σ resolvable threshold of **0.0044**.
Pairing within seed does not remove this, because it is present *at* fixed seed.
It belongs in methods, and it retires the inherited "resolution floor ≈ 0.01
PER" framing: the paired-by-seed SDs are now 0.001–0.006.

*Process note worth one line in the paper's reproducibility section:* the
summary JSON was written only after all 42 runs, so a runtime failure late in
the sweep destroyed every earlier result — which happened once, at run 41 of 42.
The trainer now appends each condition to a JSONL as it completes, flushed and
fsynced. Long sweeps on pre-emptible hardware need per-unit durability, not
end-of-run durability.

### F15 — sixteen points, and still no knee: RQ1 answered on the trained curve

Full write-up: `docs/DENSE_SWEEP_RQ1_ANSWERED.md`. Figure:
`results/figures/fig_dense_no_knee.png`. This supersedes F14's knee claim.

32 runs (16 lookaheads × 2 seeds, native arm, padding-fixed, 1200 steps, T4).
Grid is 20 ms, not 10: lookahead quantises to `ceil(L/frame_ms)` frames at a
20 ms frame rate, so sub-frame steps duplicate conditions and reweight the fit.
Verified: zero `t_algorithmic` collisions across all 16 conditions, so every
point is a genuinely distinct budget.

**The knee is withdrawn.** ΔBIC prefers a two-segment fit under all three
treatments of L=0 (−22.2, −31.2, −36.8) but the breakpoint lands at 40, 180 and
180 ms respectively, and every bootstrap 90% interval spans ~3 octaves. None is
identifiable. Denser sampling was the obvious remedy and it did not work,
because the curve is smoothly curved rather than piecewise.

*Methodological finding worth its own sentence in the paper:* **a breakpoint
estimator asked to find a breakpoint returns one whether or not one exists.**
ΔBIC alone cannot distinguish "there is a knee" from "piecewise fits a curve
better than a line does". Identifiability has to be tested by perturbing the
x-axis and bootstrapping the location. Our own synthetic check reproduces the
artefact exactly: on a curve that is log-linear for L ≥ 20 with L=0 off the
extrapolation, including L=0 yields ΔBIC −39 and a confident "knee at 40 ms",
while dropping it yields ΔBIC +5 and no knee at all.

**What replaces it.** Exchange rate −0.0452 PER per doubling (R² = 0.983,
L = 20–640). Saturation at ~240 ms, defined as the lookahead beyond which every
marginal 20 ms frame buys less than the measured 2σ = 0.0044 floor — a
floor-relative, axis-independent quantity. And the headline for the gap
argument: **a 40 ms budget forgoes ~40% of the achievable PER reduction**
(0.3275 → 0.1973 at 240 ms).

Caveat that must travel with it: this sweep has **zero repeated cells**, so the
noise floor is imported from the 3-seed run's accidental repeats rather than
re-measured. Median seed spread here is 0.0039 (max 0.0074), consistent with the
imported 0.0044 but not independent confirmation. Deliberately repeating 3–4
conditions would close that gap for ~15 minutes of GPU time.

### Scope of all of the above

**Compute-side and correctness only.** No accent-conversion model is trained
yet, so RQ1/RQ2/RQ3 are open. F0 is the prerequisite for answering them: without
both patches every *L* label in the sweep would have been wrong.

### What runs next, and what it will answer

Both are built, smoke-tested, and blocked only on a Hugging Face login —
`KoelLabs/L2Arctic` is gated (CC-BY-NC-4.0, 3599 phoneme-annotated utterances,
24 speakers, 6 L1s).

1. **`bench/bench_content_degradation.py` — the RQ1 lower bound, no training.**
   Push real L2-ARCTIC speech through the now-provably-causal WavLM at each *L*
   and measure how far layer-9 features drift from the bidirectional reference
   (per-frame cosine, plus CKA to separate "rotated" from "collapsed"). If the
   *encoder* has already lost the /iy/–/ih/ distinction at *L* = 0, no
   downstream converter can recover it — so the encoder's own knee is a lower
   bound on the whole system's requirement. Runs in minutes.

2. **`train/train_translator.py` — the trainable RQ1/RQ3 sweep.** The HF release
   ships `g2p` (canonical phones) and `ipa` (produced phones) per utterance, so
   PHONOS's supervised core is available directly: *causal accent translator,
   non-native audio → **native** phone sequence*, CTC against `g2p`.

   **The RQ3 control is one tensor.** Identical architecture, capacity, seed,
   data order and step count — swap only the target:

   | arm | target | task |
   |---|---|---|
   | `native` | `g2p` | accent **conversion**: decide what the speaker *should* have said |
   | `produced` | `ipa` | accent-faithful **transcription**: report the local gesture |

   H3 predicts the conversion arm gains more from lookahead. Two arms differing
   in a label tensor is a cleaner control than AC-vs-VC-only, and the gap
   between them *is* the accent-conversion signal — a model scoring well on
   `ipa` and badly on `g2p` has learned to transcribe, not convert.

   Speaker-disjoint splits stratified by L1 (a random utterance split would make
   this speaker memorisation). Sanity gate: if mean PER between `g2p` and `ipa`
   is near zero the two arms are the same task and RQ3 is unanswerable — the
   loader checks and warns.

   **Cost: a CTC head over a frozen encoder, ~30–60 min per condition on a T4.**
   The full 7 × 2 sweep is a single day, not the 200 GPU-hours §8 budgets for
   the synthesis pipeline. That buys the RQ1 and RQ3 curves months early, and
   lets the full pipeline be aimed at the question the curves raise instead of
   discovering it.

## 8. Hardware and budget

| Item | Cost | Note |
|---|---|---|
| Apple Silicon M4 | owned | reference platform |
| x86 laptop | borrow / ~$0.18/hr cloud | `c7i.xlarge` if none available |
| **Raspberry Pi 5 (8 GB) + active cooler** | **$80–120** | **buy it — see below** |
| Rented GPU, ~378 hrs @ ~$0.34 (RTX 4090) | ~$130 | includes 40% retry margin |
| Storage / egress | $20–60 | network volume beats re-downloading VCTK |
| Human evaluation, 3 rounds | $450–1,200 | |
| **Total excl. travel** | **~$680–1,730** | |
| Interspeech registration + travel if accepted | $1,200–3,500 | |

**On the Pi simulator plan: it cannot work.** Browser Raspberry Pi simulators
emulate GPIO and a shell. They do not emulate the Cortex-A76 microarchitecture,
NEON throughput, the cache hierarchy, or thermal behaviour. A latency measured
there is a measurement of someone else's server. $80 against a $130 compute
budget is not a real saving. `setup/SETUP_RASPBERRY_PI.md` documents the two
honest interim substitutes (ARM64 cloud for development — labelled `cpu-arm64`,
never `embedded-pi`; QEMU for ARM-only bug hunting, never for timings) and the
mandatory thermal protocol (`vcgencmd get_throttled` must read `0x0` both
before *and* after every sweep, or the run is discarded).

**The real cost is still ~7 months of part-time attention.** Compute is a
rounding error.

---

## 9. Timeline

| Month | Milestone | Gate |
|---|---|---|
| **Aug 2026** | Read the survey (2604.27281), then PHONOS, TVTSyn, DarkStream in full. **Email Quamer / Gutierrez-Osuna.** Request L2-ARCTIC **today** — it is on the critical path. Buy the Pi. Reproduce one offline FAC baseline. | Baseline runs end-to-end |
| **Sep 2026** | Harness first: metrics, accent probe, `verify_prompt_overlap.py`. Run S0/S2/S5. | Harness emits numbers for S0, S2, S5 |
| **Oct 2026** | Train S4 at one accent pair, all L. ~~Pilot listening test~~ → *build stimuli only (free); rater study unfunded, see `docs/DECISIONS.md`*. Re-run S1 with a *streaming* TTS. | One full L sweep, GPU only |
| **Nov 2026** | Full sweep: 224 conditions across 4 hardware classes. Randomise condition order (thermal). | Objective metrics complete |
| **Dec 2026** | ~~Human listening tests (20 conditions)~~ → *unfunded; automatic proxies only (NISQA + accent probe)*. Phoneme-class analysis. | H2 confirmed or refuted; H1 already answered (§7 F15) |
| **Jan 2027** | Ablations, first draft, **arXiv preprint** | Circulated to 2+ outside readers |
| **Feb 2027** | Revise to 4+2 pages, internal review | Submission-ready 2 weeks early |
| **~1 Mar 2027** | **Submit** | |
| **20 Apr 2027** | SSW14 fallback deadline | |

Two dependencies with long lead times, both starting **this week**: the
L2-ARCTIC request form, and the email to TAMU.

---

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **TAMU publishes the lookahead study first** | **Medium-High → Higher than v1 rated it** | High | Three papers from one group in 12 months, and PHONOS predates our scoping by 4 months. Scope to CPU/embedded + phoneme-class analysis, furthest from their anonymisation-on-GPU focus. **Email them in August — knowing their roadmap beats guessing, and collaboration beats racing.** |
| PHONOS code not released | Medium | Medium | Drop S3; compare against reimplemented Zhao 2021, state it in limitations |
| Causal adaptation degrades quality at all L | Medium | High | That *is* a result: "streaming FAC has a hard floor". Negative results with clean methodology are publishable. |
| **The knee lands at ≤40 ms** | Medium | **Low — v1 treated this as failure** | Then we are the evidence for a budget the field currently assumes. Restated H1 (§3) makes this a finding, not a null. |
| Annotated L2-ARCTIC subset underpowers H2 | **Medium — now partly quantified** | High | The HF release is 3599 annotated utterances / 6 L1s ≈ **600 per L1**: enough for class-level slopes, thin for per-phone slopes. Also: it ships IPA *strings*, not TextGrids, so per-phone attribution needs forced alignment. `bench_content_degradation.py` therefore uses an explicitly-labelled acoustic proxy (voicing × spectral flux) until an aligner is in place. |
| Base SSL encoders cannot be made causal (third leak) | Low — two found and fixed | High | The truncation proof is a gate: training aborts rather than producing mislabelled conditions. Fallback is a `-large` checkpoint (`feat_extract_norm="layer"`, causal-safe by construction) at higher compute cost. |
| **No perceptual evaluation at all** (rater study unfunded, 2026-08-09) | **High** | **Medium–High** | Most likely reviewer objection for an accent-conversion paper. Mitigations: scope every claim to PER explicitly; build stimuli and listen informally (free); NISQA-MOS + accent probe as automatic proxies with `metrics.degradation_control`. Fund the study if a reviewer requires it. |
| Solo first-time author, no affiliation | High | Medium | Find a co-author with a record. The harness release is the credibility. |
| Scope creep into "also beat SOTA" | High | High | **Explicitly out of scope.** Written into the repo README. |

---

## 11. Contribution claims

1. First systematic characterisation of the **lookahead requirement** for
   streaming accent conversion, with phoneme-class decomposition — including,
   on present evidence, that the requirement is **log-linear with no knee**, so
   the field's implicit search for a correct budget is looking for something
   that may not exist. Plus a knee estimator that tests for a knee before
   locating one; the standard one reports a confident artefact on this data.
2. Quantified evidence on whether accent conversion's context requirement
   differs from voice conversion's — bearing directly on whether streaming
   budgets inherited from VC/anonymisation are correctly specified for AC.
3. First **CPU and embedded** latency–quality characterisation of streaming AC,
   with algorithmic and computational latency reported separately — including
   the measured findings that they are **independently controllable and trade
   asymmetrically**, that peak FLOPS overpredicts streaming throughput ~10×, and
   that `num_threads = cpu_count` costs 3.2× on a big.LITTLE CPU.
4. A public, reproducible **benchmark harness**: objective metrics, latency
   instrumentation, a properly decomposed streaming ASR→TTS cascade reference,
   and machine-checked guards against a confounded sweep.

5. **A causality proof for masked SSL encoders, and evidence the field needs
   one.** Attention masking leaves two leaks in base-sized wav2vec2/WavLM — a
   1.28 s positional convolution and an utterance-global feature-encoder
   GroupNorm — so a "causal" system built the obvious way has an unreported
   second of lookahead. We supply the patches, the truncation proof, and the
   ablation attributing the residual to each component.

Claims 4 and 5 get cited even if 1–3 are superseded.

---

## 12. Deliverables

- Paper, 4+2 pages, Interspeech format
- arXiv preprint, January 2027
- Public repo: harness, configs, latency instrumentation, evaluation scripts — permissive licence
- Demo page with audio at each lookahead level
- **Not** model weights until the FTO position is clear

---

## 13. Immediate next actions

1. [x] ~~Request L2-ARCTIC~~ — **obtained** via `KoelLabs/L2Arctic` on Hugging Face
   (3599 phoneme-annotated utterances with `g2p` + `ipa`, 24 speakers, 6 L1s,
   CC-BY-NC-4.0). Note the licence: **non-commercial**, which decides the
   model-weights question in §12. Still worth requesting the full 26,867-utterance
   corpus from TAMU for the synthesis study — the HF release is the annotated
   subset only.
1b. [ ] **`huggingface-cli login`** — the repo is gated; this is the only thing
   blocking the RQ1 pilot and the training sweep from running today.
2. [ ] **Email Waris Quamer / Ricardo Gutierrez-Osuna.** Short, specific, shows you read PHONOS. Ask about the 40 ms choice directly.
3. [ ] **Buy the Raspberry Pi 5 (8 GB) + active cooler**
4. [ ] Read arXiv:2604.27281 (survey), then 2603.27001 (PHONOS) in full
5. [ ] Run `data/verify_prompt_overlap.py` the day L2-ARCTIC arrives
6. [ ] Verify the three `[unverified]` load-bearing numbers in `docs/LITERATURE.md`: TVTSyn's token count, DarkStream's 140 ms, LLVC's 20 ms
7. [ ] Re-check `interspeech2027.org` monthly for the deadline
8. [ ] Get one offline FAC baseline running before writing any novel code
