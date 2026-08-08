# Your "Causal" Encoder Is Not: Hidden Lookahead in Streaming Speech Models, and What Lookahead Actually Buys

**Draft v0.1 — 2 August 2026.** Target: Interspeech 2027 (4 pages + 2 references).
Status markers used throughout: **[M]** measured, **[P]** projected, **[TBD]** pending.

---

## Abstract

Streaming accent conversion (AC) and voice conversion (VC) systems are built on
self-supervised speech encoders made "causal" by masking self-attention. We show
this is not sufficient. In wav2vec2/WavLM-base checkpoints, two components below
the transformer stack leak future context: the positional convolution (kernel
128 — **1.28 s** of future at a 20 ms frame rate) and the feature-encoder
GroupNorm, which normalises each channel over the **entire utterance** and is
therefore unbounded. Under a truncation proof, attention masking alone leaves a
relative deviation of 1.14e-2; patching the positional convolution alone leaves
6.0e-3; patching GroupNorm alone makes it *worse* (2.6e-2); only both together
reach causality (6.1e-6 on Apple M4, exactly 0 on a Tesla T4). A system that
masks attention and stops there has an unreported second of lookahead, and its
*L*=0 condition is not *L*=0.

With a provably causal encoder we then ask what lookahead buys. On 198
L2-ARCTIC utterances, representation drift from the bidirectional reference is
strongly log-linear in lookahead (R²=0.994 vs 0.727 linear) with no cliff: every
doubling buys ≈0.081, and the marginal return *peaks* at 100–200 ms — 2.5–5×
the 40 ms budget used by current streaming AC. Training a causal phone translator over
that encoder, in which accent conversion and accent-faithful transcription
differ by a single label tensor, the model's preference for canonical over
produced phones grows **monotonically in all three seeds** — 18/18 comparisons,
tripling from *L*=0 to *L*=640 ms — while the transcription control drifts the
other way: more right context makes the model more *converting*, not merely more
accurate. Three seeds per condition also show that this margin, not phone error
rate, is the measurement that survives: PER's seed spread swamps every
individual doubling of lookahead, so curves of this kind cannot locate a knee. We further
separate algorithmic from computational latency and find they are independently
controllable: over 0–640 ms of lookahead per-chunk compute changes by −1.5% to
+14%, while algorithmic latency changes by up to 32×. Chunk size, not lookahead,
determines deployability. We release the harness, the causality proof, and all
raw data.

---

## 1. Introduction

Real-time accent conversion is no longer an open problem in the "nobody has done
it" sense. PHONOS [1] demonstrates streaming foreign accent conversion at ≤40 ms
lookahead and ≤241 ms end-to-end with an 81% reduction in non-native accent
confidence; TVTSyn [2] and DarkStream [3] achieve comparable streaming voice
anonymisation. What does not exist is any account of **why 40 ms**, what it
costs, or what any of it does on hardware people own.

Answering that requires two things the field currently lacks.

**A causal encoder you can prove is causal.** Every result about lookahead is
conditional on the *L*=0 condition genuinely having no lookahead. We show
(§3) that the standard construction does not.

**A latency decomposition.** Reported figures fuse an inherent modelling
delay with a contingent hardware delay. "≤241 ms on a single GPU" does not tell
a reader whether a faster chip helps.

Contributions:

1. **Two unreported lookahead leaks** in base-sized SSL encoders, with patches
   and a truncation proof that gates the rest of the study (§3). **[M]**
2. **A latency decomposition** — algorithmic / computational / buffer — measured
   across Apple M4, Neoverse-N1 and Tesla T4, showing the terms are
   independently controllable and trade asymmetrically (§4). **[M]**
3. **The lookahead exchange rate** at the encoder level on real accented speech:
   log-linear, no cliff, marginal return peaking at 100–200 ms (§5). **[M]**
4. **A trained accent translator** whose conversion/transcription control differs
   in one label tensor, showing that the conversion signal grows monotonically
   with lookahead while the transcription control does not — and, from three
   seeds per condition, that phone error rate is too seed-noisy to support the
   per-condition claims such curves are usually asked to carry (§5.2). **[M]**
5. **A properly decomposed streaming ASR→TTS cascade baseline**, showing it
   loses structurally (95% of its latency is algorithmic) rather than
   computationally (§4.3). **[M]**
6. **A public harness** with machine-checked guards against a confounded sweep,
   the causality proof, and all raw data.

---

## 2. Related work

| Work | Task | Lookahead | Latency | Hardware reported |
|---|---|---|---|---|
| PHONOS [1] | FAC + anonymisation | ≤40 ms | ≤241 ms | single GPU |
| TVTSyn [2] | streaming VC/anon | ~80 ms | <80 ms | GPU |
| DarkStream [3] | real-time anonymisation | short buffer | low | not stated |
| StreamVC [4] | streaming VC | causal | low | "mobile", unspecified |
| **This work** | characterisation | **0–640 ms swept** | **decomposed** | **M4 / ARM64 / T4** |

[1–3] are three papers from one group. None reports a CPU latency–quality curve;
none reports a lookahead ablation; none separates algorithmic from computational
latency. To our knowledge neither encoder-causality leak in §3 has been noted.

---

## 3. A causal SSL encoder, and how to prove it

### 3.1 Two leaks

**Positional convolution.** `WavLMPositionalConvEmbedding` is a depthwise Conv1d
with kernel 128 and symmetric padding, applied *before* the transformer stack.
At a 20 ms frame rate that is ~64 frames — **1.28 s** — of future entering every
frame. Fixed by left-only padding.

**Feature-encoder GroupNorm.** Base checkpoints use `feat_extract_norm="group"`:
`GroupNorm(num_groups=C, num_channels=C)` over a (B,C,T) tensor normalises each
channel by statistics computed over the **entire utterance**. Every output frame
depends on every input frame — unbounded, not merely long. Fixed by cumulative
(running) normalisation, applied identically in the bidirectional reference so
the sweep stays unconfounded. The *-large* checkpoints use
`feat_extract_norm="layer"` and are causal-safe; the leak is specific to the
base-sized encoders a CPU deployment would choose.

### 3.2 The truncation proof

Delete all audio after frame *t*; a genuinely causal encoder's output at an
earlier frame cannot change. Relative L2 deviation, tolerance 1e-4: **[M]**

| configuration | Apple M4 (torch 2.13) | Tesla T4 (torch 2.11) | causal? |
|---|---:|---:|---|
| attention mask only | 1.1438e-2 | 1.1438e-2 | no |
| + causal positional conv | 5.9956e-3 | 5.9956e-3 | no |
| + cumulative GroupNorm | 2.5983e-2 | 2.5983e-2 | no |
| **both patches** | **6.14e-6** | **0.0** | **yes** |

Three things worth stating. Neither fix suffices alone. **Patching GroupNorm
alone makes the measurement worse** — a partial causality fix is not a partial
improvement. And the numbers replicate to five significant figures across two
architectures and two torch versions, so this is a property of the checkpoint,
not of an environment.

This proof is a gate, not a footnote: training aborts if it fails.

---

## 4. Latency decomposition

```
t_algorithmic = chunk + lookahead (+ frontend receptive field)   hardware-independent
t_compute     = encoder + conversion + vocoder                   hardware-dependent
t_buffer      = I/O + jitter                                     system-dependent
```

### 4.1 The two terms are independently controllable **[M]**

63 conditions × 3 model scales × 3 chunk sizes, replicated on Apple M4
(Accelerate) and Neoverse-N1 (OpenBLAS). Going from *L*=0 to *L*=640 ms:

| | Δ t_compute | Δ t_algorithmic |
|---|---:|---:|
| Apple M4 | **−1.5% to +14%** | up to **+3200%** |
| Neoverse-N1 | +5.7% to +12.2% | up to +3200% |

Lookahead widens the attention mask but does not change the number of query
frames pushed through the feed-forward and convolution stacks. **Cutting
lookahead is nearly free in compute and costs quality; cutting chunk size costs
compute and does not cost quality.** A practitioner optimising a single fused
"latency" number will make the wrong trade.

### 4.2 Chunk size, not lookahead, decides feasibility **[M]**

A base-scale (768-dim, 12-layer) encoder at 20 ms chunks reaches **RTF 1.60 on
an M4** — behind real time at *every* lookahead. At 80 ms chunks, RTF 0.54. No
reduction in *L* rescues the 20 ms configuration.

Two secondary results with practical bite:

- **Peak FLOPS overpredicts streaming inference by ~10×.** The M4 achieves
  74–164 GFLOP/s against its own 1632 GFLOP/s sgemm rate, and ends up only
  ~1.4× faster end-to-end than a machine with a quarter of its peak. On the
  base preset the *ranking even flips*: the Neoverse box is faster.
- **`num_threads = cpu_count` is the wrong default on big.LITTLE.** On the M4
  (4 P + 6 E cores), 8 threads makes the ASR encoder **3.2× slower** than 1 and
  pushes a TTS model past RTF 1.0. Single-threaded wins at every metric.

### 4.3 The cascade baseline loses structurally **[M]**

Streaming zipformer ASR + Piper/VITS on an M4:

```
t_algorithmic  ASR chunk 320 + right-context 70   390 ms
               + commit timeout                   700 ms
t_compute      ASR active step                     12 ms
               TTS, one word                       21 ms
t_buffer       [TBD — not yet measured]           ~30 ms
                                                 ────────
t_end_to_end                                     ~1153 ms   (95% algorithmic)
```

Vocoder choice matters enormously — Piper synthesises a word in 21 ms against
Kokoro's 973 ms, a 46× difference — but removing that term makes the conclusion
*stronger*: what remains is almost entirely algorithmic, and no chip fixes it.

We measured the commit delay rather than asserting it. 28.8% of words are
revised after first appearing; the release-while-unstable rate falls off a cliff
between 300 ms (27%) and 400 ms (1.5%), and that cliff sits exactly at the
model's 320 ms decode chunk. **The recogniser's chunk size sets the granularity
at which any cascade built on it can commit.** Taking the most favourable
correction — commit at 400 ms, accept 1.5% instability — gives ~853 ms, still
~93% algorithmic. *Caveat: 66 words of clean read speech; see §7.*

---

## 5. What lookahead buys, at the encoder level **[M]**

With causality established, we push 198 L2-ARCTIC utterances (33 per L1 × 6 L1s)
through the patched WavLM at each lookahead and measure drift from the
bidirectional reference.

**Lookahead is quantised.** The encoder represents lookahead as
`ceil(L / frame_ms)` frames, so at a 20 ms frame rate a knee narrower than 20 ms
is not merely unmeasured — it is *unrepresentable*. Sampling finer produces
byte-identical duplicate conditions. We report 16 distinct conditions.

| | value |
|---|---|
| R² log-linear | **0.9942** |
| R² linear | 0.7267 |
| mean slope | −0.081 per doubling |
| ΔBIC (2-segment − 1-segment) | −22.9 |

Both facts are true at once: the curve is overwhelmingly log-linear, *and* BIC
detects small real curvature (each point is a mean over 198 utterances, so
residuals are structure rather than noise). The model-free view resolves it:

| doubling | Δ divergence |
|---|---:|
| 20 → 40 ms | +0.050 |
| 40 → 80 ms | +0.063 |
| 80 → 160 ms | +0.079 |
| **100 → 200 ms** | **+0.081** |
| 160 → 320 ms | +0.070 |
| 320 → 640 ms | +0.053 |

**No cliff, but a broad optimum in the exchange rate at 100–200 ms.** Nothing
distinguishes 40 ms: it sits on the *least* productive doubling measured. The
actionable form is the exchange rate — ≈0.08 per doubling, peaking at
100–200 ms — not a recommended operating point.

*Estimator note.* A maximum-distance-to-chord ("Kneedle") estimator reported
"knee at 160 ms, bootstrap CI [160,160]" on this data. It is an artefact: the
estimator always returns a point, and on a log-smooth curve it returns the
middle of the sampled range; the CI was tight *because* there was no effect. We
select between one- and two-segment fits by BIC instead, and note that a
7-point geometric grid is **underpowered** — a synthetic planted cliff returns
ΔBIC = −0.2 at n=7.

### 5.1 Phoneme-class asymmetry (H2) **[M, weak]**

On a voicing × spectral-flux proxy, sonorant/steady frames gain **+0.013** more
from lookahead than obstruent/transient ones (paired, n=48, bootstrap 95% CI
[+0.003, +0.023]). Direction matches the hypothesis and excludes zero, but the
effect is small on a crude proxy; a forced-alignment test against L2-ARCTIC's
phone annotations is required before claiming a phoneme-class story.

### 5.2 Trained accent translator **[M]**

The encoder result bounds the information *available*; it does not show what the
*task* needs. We now train.

**Task.** `KoelLabs/L2Arctic` ships, per utterance, the **canonical** phone
sequence a native speaker would produce (`g2p`) and the **produced** sequence
this L2 speaker actually said (`ipa`). That makes PHONOS's supervised core
directly trainable: a causal translator from non-native audio to the *native*
phone sequence, via CTC against `g2p`.

**The control is one tensor.** Identical architecture, capacity, seed, data
order and step count; only the target changes — `g2p` (accent **conversion**:
decide what the speaker *should* have said) vs `ipa` (accent-faithful
**transcription**: report the local gesture). Mean PER between the two targets
is **0.175**, confirming they are genuinely different tasks.

**Setup.** 4-layer causal conversion stack (masked self-attention +
left-padded depthwise conv + FiLM) and a linear CTC head over a **frozen**
WavLM-base-plus with both causality patches of §3; 24.6 M trainable parameters,
vocabulary 37. Chunk 40 ms, look-back 2 s. AdamW, OneCycle, lr 3e-4, batch 8,
**1200 steps** per condition. 3599 utterances, **speaker-disjoint** splits
stratified by L1 (1800 train / 899 val / 900 test; 6 held-out speakers each for
val and test) — a random utterance split would make this speaker memorisation.
**Three seeds, not one.** Every condition is trained at seeds {1337, 7, 99}:
7 lookaheads × 2 targets × 3 seeds = **42 runs**. Caching the frozen encoder's
features once per condition and reusing them across seeds makes three seeds cost
about 1.2× one seed rather than 3×, so the whole design is ≈**2.3 h on one Tesla
T4**. The causality proof runs before training and aborts it on failure.

**Results.** Test PER, speaker-disjoint, mean ± SD over 3 seeds:

| *L* (ms) | *t*<sub>algo</sub> | conversion (`g2p`) | transcription (`ipa`) |
|---:|---:|---:|---:|
| 0 | 40 | 0.447 ± 0.005 | 0.451 ± 0.006 |
| 20 | 60 | 0.397 ± 0.011 | 0.398 ± 0.001 |
| 40 | 80 | 0.363 ± 0.027 | 0.380 ± 0.008 |
| 80 | 120 | 0.335 ± 0.030 | 0.379 ± 0.032 |
| 160 | 200 | 0.303 ± 0.011 | 0.345 ± 0.029 |
| 320 | 360 | 0.302 ± 0.052 | 0.321 ± 0.020 |
| 640 | 680 | **0.234 ± 0.021** | 0.280 ± 0.017 |
| **relative gain 0→640** | | **0.476 ± 0.043** | **0.380 ± 0.046** |

**Most of this curve is not resolved, and that is the finding.** Blocking on
seed — the correct analysis, since an unlucky seed raises PER at *every*
lookahead — only the 0→20 ms step is significant at α=.05 in the conversion arm
(*t*(2)=+11.1). The 160→320 ms step is **−0.0009 PER with 1 of 3 seeds
improving**: the mid-range plateau visible in a single-seed run is noise. What
*is* solid is the endpoint, −0.212 ± 0.018 PER, *t*(2)=+20.1, unanimous. **The
curve is real and no individual doubling on it is.** Any published claim of the
form "the knee is at *X* ms" cannot be supported by an experiment of this size,
including ours.

> **Figure 6** (`results/figures/fig_3seed_margin_vs_per.pdf`) plots both panels
> on the same 42 runs with per-seed traces drawn under the ±SD ribbons. Panel A's
> ribbons overlap almost everywhere; panel B's do not. That contrast, not either
> curve alone, is the argument for changing instruments.

**A more stable instrument.** Scoring each model against *both* label sets gives
a **preference margin** (PER against the other target minus PER against its own).
Its seed SD is **0.0023** against **0.0223** for PER — roughly **10× more
stable**, because the margin is a within-model difference and the seed's effect
on overall accuracy cancels:

| *L* (ms) | 0 | 20 | 40 | 80 | 160 | 320 | 640 |
|---|---:|---:|---:|---:|---:|---:|---:|
| conversion margin | **+.032** | +.043 | +.052 | +.068 | +.084 | +.094 | **+.101** |
| transcription margin | +.037 | +.039 | +.038 | +.032 | +.027 | +.027 | +.026 |

The conversion margin is **strictly monotone increasing in all 3 seeds**, with
**18/18** adjacent step × seed comparisons positive (exact sign test
*p*=7.6×10⁻⁶), tripling from *L*=0 to *L*=640. The transcription arm — same
architecture, same budget, one different label tensor — does the opposite:
**0/3** seeds monotone, 7/18 positive (*p*=0.48), drifting *down* by 0.010. More
right context does not merely make the conversion model more accurate; it makes
it **more converting and less transcribing**.

**H3 is supported on both readings, but they differ in strength.** By relative
PER gain, conversion 0.476 vs transcription 0.380, paired difference
+0.096 ± 0.006, *t*(2)=+25.9. By margin growth, +0.069 vs −0.010, paired
difference +0.079 ± 0.003, *t*(2)=+41.4. The margin is the stronger and more
stable statement and is the one we lead with.

**The obvious objection, controlled.** Any model that gets better at its own
target must widen its margin, and the conversion arm does improve more in
absolute terms (0.212 vs 0.171 PER). We therefore compare the two arms **at
matched own-target PER**. At *L*≥160 ms the conversion margin is **3.46×** the
PER-matched transcription margin; at *L*=0 it is **0.88×**, i.e. slightly
*smaller*. The divergence is created by lookahead, not by accuracy.

**Task vs representation.** The trained curve is shallower than the encoder one
(R²<sub>log</sub> 0.96 vs 0.99). The representation loses information faster
than the task can exploit it, which is what one expects if the encoder is a
bound and not a bottleneck — consistent with, but not proof of, the §5 reading.

**Knee, with a stated detection limit.** At *n*=21 per arm, BIC prefers a single
log-linear regime (ΔBIC = +5.27 conversion, +4.25 transcription; the best
breakpoint is pinned at the grid edge, itself a signature of no interior knee).
Crucially we now calibrate the test: **planting** a 0.08 PER cliff at 160 ms is
*still missed* in the conversion arm (ΔBIC −2.93, short of the −6 threshold).
So this experiment excludes discontinuities larger than roughly **0.08 PER
(≈18 % relative)** and says nothing about smaller ones. The earlier 7-point
result carried no such bound and should not have been read as evidence of
absence.

**Limits.** 1200 steps, a frozen encoder, and three seeds — enough for the
endpoint and the margin, not for adjacent doublings. Absolute PER is high
because training is deliberately short; every comparison above is relative and
all conditions share a budget. A **zero-padding bug** was present in both this
run and the earlier single-seed run: the causal depthwise conv and the
feed-forward ignored `key_padding_mask`, so batch composition leaked into the
last *k*−1 real frames of every shorter utterance. It is now fixed, but a
bug-free replication is still pending, so absolute PERs must not be compared
across runs — only within one. All claims in this section are within-run
comparisons, which a constant offset does not affect.

---

## 6. Embedded feasibility **[P — PROJECTED, NOT MEASURED]**

> **These numbers are projections.** No Raspberry Pi was measured. Browser Pi
> simulators model GPIO, not microarchitecture; QEMU models no timing at all.
> Both would produce fabricated latencies. This section states what can be
> concluded *without* the board, and what cannot.

First, a negative result that constrains any projection: **a single scalar does
not describe the difference between two machines.** Across 63 matched
conditions, the per-preset M4→Neoverse ratio ranges 0.93–1.68 (1.82× spread) —
the ranking even reverses by model size. "The Pi is *N*× slower" is therefore
not a well-defined statement, and the projection must be per configuration.

Projecting RTF over a slowdown axis (2–13× relative to the M4), the conclusions
that survive the whole range:

| verdict across 2–13× | configurations |
|---|---|
| **infeasible regardless** | base @ 20/40/80 ms |
| **feasible regardless** | small @ 80 ms; tiny @ 20/40/80 ms |
| **depends on the factor** | small @ 20 ms, small @ 40 ms |

We do not need the Pi's exact slowdown to state that a base-scale encoder cannot
stream on it and a tiny one can. Only two boundary configurations require the
board — and those we decline to guess, because a peak-FLOPS derivation puts a
Pi 5 at 37–56× the M4 while an achieved-throughput derivation puts it far
closer, a disagreement of roughly an order of magnitude. That disagreement *is*
the finding: **embedded latency cannot be projected reliably from
specifications**, which is precisely the argument for measuring it.

---

## 7. Limitations

- **No converted audio, no listening test.** §5 and §5.2 measure representation
  drift and phone error, not perceptual accentedness. Synthesis and subjective
  evaluation are future work.
- **Three seeds, 1200 steps, frozen encoder** in §5.2. Adjacent doublings of
  lookahead are not resolved in PER, so the per-condition ordering must not be
  read in detail; only the endpoint effect and the margin are supported. Knee
  detection is bounded at ~0.08 PER by the planted-cliff calibration, not
  claimed absent.
- **A zero-padding bug was present in both §5.2 runs** (conv and FFN ignored
  `key_padding_mask`, letting batch composition leak into the last 30 frames of
  shorter utterances). Fixed; replication on the fixed code is pending, so
  absolute PERs are comparable only within a run. Every claim in §5.2 is a
  within-run comparison.
- **The Pi row is projected.** §6, flagged throughout.
- **Commit delay** measured on 66 words of clean read speech; accented speech
  should revise more, which would strengthen the argument, but is unmeasured.
- **`t_buffer` is unmeasured** and enters the cascade budget as a ~30 ms
  estimate.
- **H4 partially tested.** int8/fp32 speedup across ASR stages spanning 265× in
  parameter count is 1.55–2.09× with *negative* correlation to size
  (ρ = −0.51), supporting an operator-mix rather than parameter-count account.
  The vocoder clause is untested: the model set ships int8 only.
- **The commit-delay measurement did not reproduce on macOS**, returning empty
  hypotheses for every input with identical code, weights and audio that yield
  66 words on Linux. Reported numbers are from Linux; the discrepancy is
  unresolved.

---

## 8. Conclusion

Masking self-attention does not make a base-sized SSL encoder causal, and the
residual leak is over a second. Once that is fixed and proven, the lookahead
question has no cliff to find: the exchange rate is roughly constant per
doubling, with the best marginal return well above the budgets currently in use.
Meanwhile the term that actually decides whether a system runs on a CPU is chunk
size, not lookahead — and the two trade in opposite directions, which a single
fused latency number hides.

---

## References

[1] W. Quamer, M.-R. Tseng, G. Nasrallah, R. Gutierrez-Osuna, "PHONOS: Phonetic Neutralization for Online Streaming Applications," arXiv:2603.27001, 2026.
[2] W. Quamer et al., "TVTSyn: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization," arXiv:2602.09389, 2026.
[3] W. Quamer, R. Gutierrez-Osuna, "DarkStream: real-time speech anonymization with low latency," arXiv:2509.04667, 2025.
[4] Y. Yang et al., "StreamVC: Real-Time Low-Latency Voice Conversion," ICASSP 2024, arXiv:2401.03078.
[5] G. Zhao, S. Ding, R. Gutierrez-Osuna, "Converting foreign accent speech without a reference," IEEE/ACM TASLP 29, 2021.
[6] G. Zhao et al., "L2-ARCTIC: A Non-Native English Speech Corpus," Interspeech 2018.
[7] "Accent Conversion: A Problem-Driven Survey of Sociolinguistic and Technical Constraints," arXiv:2604.27281, 2026.

---

## Appendix — reproducibility

All results from `github.com/Dipeshtripathi13/streaming-fac-lookahead`.
Raw CSV/JSONL and host metadata for every number are in `results/raw/`.
Hardware: Apple M4 (Mac16,1, 4P+6E, macOS 26.5, Accelerate);
Neoverse-N1 (4 cores, OpenBLAS 0.3.29); Tesla T4 (14.6 GB, torch 2.11.0+cu128).
Data: L2-ARCTIC via `KoelLabs/L2Arctic` (CC-BY-NC-4.0) — non-commercial, which
constrains any model-weight release.
