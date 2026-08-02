# Measured results — Apple M4, 2 August 2026

Everything here came out of `run_m4.command` on Dipesh's machine. Raw data in
`results/raw/`, host metadata in `hw_m4.json`, full logs in
`run_m4_*.log` / `selftest_*.log`.

| | |
|---|---|
| class | `cpu-apple-silicon` |
| chip | Apple M4 (Mac16,1), **4 P-cores + 6 E-cores**, 16 GB |
| OS / Python | macOS 26.5 (Darwin 25.5.0) / 3.11.15 |
| BLAS | **Accelerate** (not OpenBLAS — recorded because they differ ~2×) |
| torch / MPS | 2.13.0, MPS available |
| sgemm 512³ / 1024³ | **1632 / 1055 GFLOP/s** |
| memcpy | 25.4 GB/s |

Comparison machine: `cpu-arm64` Neoverse-N1, 4 cores, 403 GFLOP/s sgemm.

---

## Finding 0 — masking attention does **not** make WavLM causal

This is the most important result of the day, and it is a correctness result
rather than a performance one: **every lookahead number in this literature
depends on it, and it is not what people assume.**

`bench/bench_content_degradation.py --selftest` runs a *truncation proof*:
delete all audio after frame *t*, then check whether the encoder's output at an
earlier frame changed. A genuinely causal encoder cannot notice the deletion.
Four ablations, relative L2 change, tolerance 1e-4:

| configuration | pos-conv patched | GroupNorm patched | relative L2 | causal? |
|---|---|---|---:|---|
| attention mask only | no | no | 1.14e-2 | **no** |
| + causal positional conv | yes | no | 6.00e-3 | **no** |
| + cumulative GroupNorm | no | yes | **2.60e-2** | **no** |
| **both** | yes | yes | **6.14e-6** | **yes** |

Two leaks, both underneath the transformer stack:

1. **`pos_conv_embed`** — depthwise Conv1d, **kernel 128**, symmetric padding.
   At a 20 ms frame rate that is **1.28 s of future** entering every frame.
2. **The feature-encoder GroupNorm** — wav2vec2/WavLM *base* checkpoints use
   `feat_extract_norm="group"`, i.e. `GroupNorm(num_groups=C, num_channels=C)`
   over a (B, C, T) tensor. Each channel is normalised by statistics taken over
   the **entire utterance**, so every output frame depends on every input frame.
   Unbounded, not merely long.

Three things worth stating carefully:

- **Neither fix is sufficient alone.** Attention masking plus the positional
  conv fix still leaks 6e-3 — a plausible-looking number that a project not
  running this test would never see.
- **Fixing GroupNorm alone makes the measurement *worse* (2.6e-2).** A partial
  causality fix is not a partial improvement. That is a good reason to treat
  "we made it causal" as a claim requiring proof rather than description.
- **The `-large` checkpoints use `feat_extract_norm="layer"`**, which is
  per-frame and causal-safe. So the GroupNorm leak is specific to base-sized
  encoders — which is exactly what a CPU/embedded system would choose.

We patch both (left-only padding; cumulative running normalisation) and apply
the *same* patches in the bidirectional reference condition, so the only thing
varying across the sweep remains the attention mask. Training aborts if the
proof fails (`train/train_translator.py --verify-causality`, on by default).

We have not found either leak discussed in the streaming AC/VC papers we have
read. If it holds up, this alone justifies a methods section, and it makes the
"report your lookahead honestly" argument concrete rather than pedantic.

---

## Finding 1 — per-chunk compute is flat in lookahead (confirms ARM64)

63 conditions, 30 reps, 2 s lookback. L = 0 vs L = 640 ms:

| preset | chunk | L=0 | L=640 | Δ compute | RTF @ L=0 | achieved |
|---|---:|---:|---:|---:|---:|---:|
| base | 20 ms | 31.99 ms | 31.50 ms | **−1.5%** | **1.599** | 94 GF/s |
| base | 40 ms | 42.98 | 43.53 | +1.3% | 1.075 | 74 GF/s |
| base | 80 ms | 42.79 | 45.69 | +6.8% | 0.535 | 82 GF/s |
| small | 20 ms | 2.30 | 2.62 | +13.9% | 0.115 | 164 GF/s |
| small | 40 ms | 2.91 | 3.28 | +12.7% | 0.073 | 137 GF/s |
| small | 80 ms | 3.21 | 3.48 | +8.4% | 0.040 | 138 GF/s |
| tiny | 20 ms | 0.99 | 0.98 | −1.0% | 0.049 | 113 GF/s |
| tiny | 40 ms | 1.05 | 1.04 | −1.0% | 0.026 | 113 GF/s |
| tiny | 80 ms | 1.09 | 1.17 | +7.3% | 0.014 | 121 GF/s |

**−1.5% to +13.9% compute for a 32× change in algorithmic latency**, on a second
architecture with a different BLAS. Two of the nine cells are slightly negative,
i.e. within noise of zero. The ARM64 run gave +5.7% to +12.2%. The claim
survives replication.

## Finding 2 — peak FLOPS badly overpredicts streaming inference

The M4's sgemm peak is **4.0× the Neoverse box** (1632 vs 403 GFLOP/s). On this
workload it achieves only **74–164 GFLOP/s — 5–10% of its own peak**, and ends up
roughly **1.4×** faster than the ARM64 machine, not 4×.

Streaming transformer inference is a sequence of small matmuls; it is bound by
per-call overhead and memory movement, not by peak arithmetic. Anyone sizing a
deployment from a spec-sheet TFLOPS number will be wrong by ~10×, in the
optimistic direction.

This also sharpens the Raspberry Pi estimate: scaling by *achieved* rather than
peak throughput, a Pi 5 should land ~4–8× slower than the M4 rather than the
7–13× a peak-FLOPS estimate suggests. Still enough to put `base` firmly out of
reach and to make `small` at 20 ms chunks marginal.

## Finding 3 — a base-scale encoder cannot stream at 20 ms chunks, even on an M4

RTF **1.599** at 20 ms chunks. It falls behind real time at *every* lookahead;
no reduction in L rescues it. At 80 ms chunks the same model runs at RTF 0.535.
Chunk size, not lookahead, is the feasibility lever — replicated.

## Finding 4 — more threads is monotonically worse on Apple Silicon

Sweeping ONNX Runtime `num_threads` on the real sherpa-onnx models:

| threads | ASR active step p50 | ASR p95 | Kokoro RTF | Kokoro 1-word | Piper/VITS 1-word |
|---:|---:|---:|---:|---:|---:|
| 1 | **11.8 ms** | **13.3 ms** | **0.78** | **973 ms** | **21.0 ms** |
| 2 | 12.6 | 14.0 | 0.85 | 976 | 23.4 |
| 4 | 17.9 | 29.8 | 0.92 | 1135 | 25.6 |
| 8 | 37.7 | 50.5 | **1.10** | 1548 | 33.4 |

**Single-threaded is optimal for every model at every metric.** Eight threads
makes the ASR encoder **3.2× slower** and pushes Kokoro past RTF 1.0 — from
comfortably real-time to structurally infeasible, purely by asking for more
parallelism.

Almost certainly heterogeneous-core scheduling: the M4 has 4 P-cores and 6
E-cores, and once the thread pool exceeds the P-core count the runtime starts
placing work on cores that are several times slower, while the whole op waits
on the slowest thread. The ARM64 box (4 homogeneous cores) showed a milder
version of the same thing: 2 threads best, 4 threads 69% worse at p95.

The practical form: **`num_threads = os.cpu_count()` is the wrong default for
streaming speech on a big.LITTLE CPU, and it is the default nearly everyone
ships.** `accentbridge.py` currently uses `num_threads=2`; on this machine
1 would be better. Cheap to state, easy to verify, and directly useful.

## Finding 5 — the cascade's TTS bottleneck is a model choice, not structural

| | Kokoro int8 | Piper/VITS int8 | ratio |
|---|---:|---:|---:|
| single-word synthesis | 972.6 ms | **21.0 ms** | **46×** |
| full utterance | 2429 ms | 381 ms | 6.4× |
| RTF | 0.78 | 0.14 | 5.7× |

The ARM64 pilot concluded the cascade dies at ~2.0 s with TTS dominating. On the
M4 with Piper instead of Kokoro, the compute term nearly vanishes and the
picture changes qualitatively:

```
                                  Kokoro          Piper/VITS
t_algorithmic  ASR chunk 320 + right-ctx 70   390 ms          390 ms
               + commit timeout (accentbridge) 700 ms          700 ms
t_compute      ASR active step                  12 ms           12 ms
               TTS, one word                   973 ms           21 ms
t_buffer       (not yet measured)              ~30 ms          ~30 ms
                                             ────────        ────────
t_end_to_end                                  ~2105 ms        ~1153 ms
algorithmic share                                52%             95%
```

**With a fast vocoder the cascade is ~1.15 s and 95% of it is algorithmic.**
That is a stronger version of the original claim, not a weaker one: the cascade
does not lose because synthesis is slow, it loses because it must wait for the
recogniser to stop revising words. A faster chip cannot help; only abandoning
the text bottleneck can.

Two caveats to carry into the paper. The 700 ms commit timeout is
`accentbridge`'s tuning parameter, not a law — but reducing it trades directly
against word-revision errors, and that trade needs measuring rather than
asserting. And Kokoro is a full-utterance model; a streaming Kokoro would
narrow the gap. The 390 ms of ASR geometry and the commit delay are the
structural terms and neither moves.

## Finding 6 — the commit delay, measured instead of asserted

Finding 5 says the cascade is ~1.15 s and 95% algorithmic, with the largest
single term being `accentbridge`'s `COMMIT_TIMEOUT = 0.7 s`. Until now that
700 ms was a tuning parameter we *asserted* traded against word-revision
errors. `bench/bench_commit_delay.py` measures the trade.

Method: stream real speech through the zipformer in 20 ms increments,
snapshot the full partial hypothesis after each one, and for every word in the
final hypothesis find the earliest time after which it never changed again.
66 words, 2 utterances, clean read speech.

**28.8% of words are revised after they first appear.** So the problem is real
— you cannot simply emit on first sight.

| | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| revision span (first emission → stable) | **0 ms** | 320 | 320 | 320 | 640 |
| stabilisation latency (sound ends → stable) | **60 ms** | 220 | 260 | 300 | **340** |

The latency-vs-correctness curve, i.e. what fraction of words a given commit
timeout releases while they are still unstable:

| commit timeout | words released unstable |
|---:|---:|
| 0–200 ms | 28.8% |
| 300 ms | 27.3% |
| **400 ms** | **1.5%** |
| 500 ms | 1.5% |
| **700 ms** | **0.0%** |

**There is a cliff between 300 ms and 400 ms**, and it sits exactly at the
model's 320 ms decode chunk. That is not a coincidence and it is the useful
part: *the recogniser's own chunk size sets the granularity at which any
cascade built on it can commit.* You cannot tune your way below it.

Three things follow.

1. **700 ms is over-provisioned by ~300 ms.** 400 ms buys 98.5% stability. That
   takes the cascade from ~1153 ms to **~853 ms** — still 3.5× PHONOS's
   ≤241 ms, and still **~93% algorithmic**. The conclusion survives its own
   most favourable correction, which is the version worth publishing.
2. **The irreducible floor is ~340 ms**, not 700. Even ignoring the commit
   heuristic entirely, the recogniser needs up to 340 ms after the sound ends
   before it stops changing its mind (p95 260 ms). Add the 390 ms of ASR chunk
   geometry and a cascade cannot beat ~600 ms on this model however it is tuned.
3. **`accentbridge.py` should drop `COMMIT_TIMEOUT` to 0.4** and take the 1.5%.

Caveats, which matter here more than usual: 66 words of clean read speech
establishes the shape, not a publishable number. Everything is quantised to the
320 ms decode chunk — a property, not error. And "stable" is measured against
the final hypothesis, so a word that is wrong from the start and never
corrected counts as stable; this measures instability, not accuracy. Re-run
over L2-ARCTIC with `--wavs`: accented speech should revise *more*, and if it
does, the cascade argument gets stronger.


## Finding 7 — RQ1 on real speech, first pass *(superseded by Finding 8)*

> **Read Finding 8 first.** This section's headline — "there is no knee" — was
> based on a 7-point grid that BIC later showed is underpowered to detect one.
> The measurement is sound; the conclusion was over-confident. Kept because the
> estimator failure it documents is the point.

The first result with actual accented speech in it. 48 L2-ARCTIC utterances,
8 per L1 across all six L1s, WavLM-base-plus layer 9, both causality patches
applied to the masked *and* the bidirectional reference condition so the only
difference is the attention mask. 37 s to run.

| L (ms) | divergence from bidirectional | CKA | sonorant | obstruent |
|---:|---:|---:|---:|---:|
| 0 | 0.5008 | 0.768 | 0.5010 | 0.4999 |
| 20 | 0.4294 | 0.810 | 0.4275 | 0.4206 |
| 40 | 0.3744 | 0.836 | 0.3705 | 0.3668 |
| 80 | 0.3091 | 0.869 | 0.3052 | 0.3015 |
| 160 | 0.2306 | 0.907 | 0.2201 | 0.2218 |
| 320 | 0.1639 | 0.934 | 0.1477 | 0.1532 |
| 640 | 0.1073 | 0.956 | 0.0900 | 0.0961 |

**The curve is log-linear: R² = 0.9955 against log₂(L), versus 0.7746 against
L.** Every doubling of lookahead buys a constant 0.081 reduction in divergence,
from 20 ms to 640 ms, with no inflection anywhere. 47 of 48 utterances are
individually monotone.

This is H1's **third branch**, which we pre-registered: *"No knee (linear) →
there is no natural operating point and every deployment is a bare product
decision. Also publishable, and the most actionable of the three."*

### The estimator nearly fooled us, and that is worth reporting

The first pass used maximum-distance-to-chord (Kneedle without smoothing) and
reported **"knee at 160 ms, bootstrap 95% CI [160, 160]"**. Tight CI, plausible
number, four times PHONOS's 40 ms budget — a headline.

It was an artefact. The chord method *always* returns a point, and on a curve
that is smooth in log-space plotted against a linear axis it returns the middle
of the sampled range. The bootstrap CI was tight precisely *because* the effect
is not there: every resample of a log-linear curve is also log-linear, so every
resample puts the "knee" in the same place. **A tight CI around an artefact is
the most dangerous possible output.**

`find_knee` now fits `y ~ log₂(L)` first and refuses to name a knee when that
fit is good (R² > 0.99 by default), reporting a slope-per-doubling instead. It
keeps `chord_argmax_ms` so the naive answer stays visible for contrast, and
`tests` include a synthetic log-linear curve that must come back `has_knee =
False`.

### What this means for the paper

If the finding holds once a converter is trained, the framing changes from
*"find the right budget"* to *"there is no right budget"*:

- **Nothing special happens at 40 ms.** PHONOS's choice is not wrong, but it is
  also not a discovered optimum — it is a point on a smooth curve.
- **Every doubling costs the same and buys the same.** 40→80 ms buys as much
  representational fidelity as 320→640 ms. That is a clean thing to tell a
  practitioner, and it is the opposite of what a knee would imply.
- **The trade is therefore purely a product decision**, and the paper's job is
  to give the exchange rate rather than a recommended operating point.

### Caveats, and they are load-bearing

1. This measures **drift from the bidirectional representation**, not
   accent-conversion quality. A converter may not need the full bidirectional
   representation. The trained sweep (`train/train_translator.py`) is what tests
   whether the *task* has a knee where the *representation* does not.
2. 7 points, geometrically spaced. A knee narrower than one octave would be
   invisible. Denser sampling between 20 and 160 ms is cheap (37 s per run) and
   should be done before this is written up.
3. The 1−CKA curve does register `has_knee = True` (R² = 0.987, just under the
   0.99 threshold). The two metrics disagree at the margin, which is itself a
   reason not to over-claim either way.

### H2 proxy: real but very small

Paired per utterance, the sonorant/steady bucket gains **+0.0129** more from
lookahead than the obstruent/transient bucket (bootstrap 95% CI
[+0.0026, +0.0230], n = 48). The direction matches H2 — formant-defined sounds
benefit more from right context — and it excludes zero, but a 1.3-point
difference on a crude voicing × spectral-flux proxy is not evidence for a
phoneme-class story. The real test needs forced alignment against L2-ARCTIC's
phone annotations; this only shows the effect is worth looking for.


## Finding 8 — RQ1 on GPU, 198 utterances: the answer, and two of my own errors

Tesla T4, torch 2.11.0+cu128, 198 L2-ARCTIC utterances (33 per L1 × 6 L1s),
165 s. This supersedes Finding 7's conclusion.

### First: the causality proof replicates exactly, and improves

| configuration | M4 (torch 2.13) | T4 (torch 2.11) |
|---|---:|---:|
| attention mask only | 1.1438e-2 | **1.1438e-2** |
| + causal positional conv | 5.9956e-3 | **5.9956e-3** |
| + cumulative GroupNorm | 2.5983e-2 | **2.5983e-2** |
| **both patches** | 6.14e-6 | **0.0 exactly** |

Bit-identical across two architectures and two torch versions, and on the T4
the patched encoder is causal to *machine zero* — `max_abs_delta = 0.0`.
Finding 0 is solid.

### Then: my "dense" grid was invalid

I asked for L every 10 ms. **Lookahead is quantised to `ceil(L / frame_ms)`
frames, and the frame rate is 20 ms**, so half the grid was byte-identical
duplicates:

```
L(ms)  frames  divergence
  10     1     0.42209
  20     1     0.42209   <- same condition
  30     2     0.37166
  40     2     0.37166   <- same condition
```

26 nominal points → **16 distinct conditions**. The 10 duplicates all sat in
0–200 ms, double-weighting that region and dragging R² from 0.994 to 0.989.

**A knee narrower than 20 ms is not unmeasured — it is unrepresentable.** The
encoder cannot express it. That is a real constraint on the whole question and
it should have been obvious from `StreamGeometry.lookahead_frames`, which I
wrote. `--dense` now steps by `frame_ms`, and the pilot deduplicates and warns.

### And my knee test was deciding on an arbitrary number

Those duplicates pushed R² across the 0.99 threshold I had chosen, flipping
`has_knee` from False to True and printing *"Knee at 170 ms"*. A scientific
conclusion turned on a magic constant. Replaced with BIC comparison between a
one- and two-segment fit on the log₂ axis.

That change immediately exposed a second problem: **on the original 7-point
grid, BIC cannot detect a knee at all.** A synthetic curve with a planted cliff
at 80 ms returns ΔBIC = −0.2 on 7 points — the `k·log n` penalty swamps it. So
Finding 7's "no knee" was never evidence of absence; it was an underpowered
grid. `find_knee` now returns `underpowered_for_bic` and says so.

### The actual result

16 distinct conditions, 198 utterances:

| | |
|---|---|
| R² log-linear | **0.9942** |
| R² linear | 0.7267 |
| ΔBIC (piecewise − log-linear) | **−22.9** → two-segment preferred |
| best breakpoint | 280 ms |
| mean slope | −0.081 per doubling |

These pull in opposite directions, and the honest reading is that **both are
true**: the curve is *overwhelmingly* log-linear (0.994 vs 0.727), and it also
has a small, real, systematic curvature that BIC detects because the residuals
are genuine structure rather than sampling noise (each point is a mean over 198
utterances, so noise is tiny).

The model-free view settles it — gain per doubling of lookahead:

| doubling | Δ divergence |
|---|---:|
| 20 → 40 ms | +0.0504 |
| 40 → 80 ms | +0.0625 |
| 80 → 160 ms | +0.0787 |
| **100 → 200 ms** | **+0.0811** ← peak |
| 160 → 320 ms | +0.0697 |
| 320 → 640 ms | +0.0534 |

**There is no cliff. There is a broad, shallow maximum in the exchange rate
around 100–200 ms.** Every doubling buys something, and doublings in the
100–200 ms band buy about 60% more than doublings at 20–40 ms. The curve is
closer to "no natural operating point" than to "a knee", but the flat statement
*"no knee"* from Finding 7 was over-confident and is withdrawn.

### What this means for the paper

- **Nothing distinguishes 40 ms.** PHONOS's budget is on the cheap part of the
  curve: the 20→40 ms doubling is the *least* productive one measured.
- **The best marginal return is 100–200 ms**, i.e. 2.5–5× PHONOS's budget. If
  the trained sweep reproduces this, "current budgets are under-provisioned"
  becomes defensible — but as *diminishing-returns geometry*, not a knee.
- **Report the exchange rate, not an operating point.** ~0.08 divergence per
  doubling, peaking around 100–200 ms.

### Caveats

Still representation drift, not conversion quality — the trained sweep remains
the real test. The 1−CKA curve agrees in shape. And "two-segment preferred at
280 ms" should not be quoted as a knee: with n = 16 and near-zero noise, BIC
will detect any curvature, and a 0.994 log-linear fit is not a cliff.


---

## Scope

**These are compute-side and correctness results.** No accent-conversion model
has been trained yet, so RQ1/RQ2/RQ3 remain open. Finding 0 is a prerequisite
for answering them at all: without both patches, every L label in the sweep
would have been wrong.

**Now unblocked:** HF login done; Finding 7 is real accented speech. The
corpus check also confirms the RQ3 control is well posed — mean PER between
`g2p` (canonical) and `ipa` (produced) is **0.175**, far above the 0.02 floor
below which the two training arms would be the same task.

**Known issue, unresolved:** the commit-delay measurement (Finding 6) returns an
empty hypothesis for every input on this macOS/arm64 machine — same sherpa-onnx
1.13.4, same model files, same audio, int8 *and* fp32, with and without
endpointing, whole-clip and chunked. The identical code produces 66 words on
Linux/aarch64, and `bench_cascade_onnx.py` decodes fine on this same Mac. The
Finding 6 numbers therefore come from the Linux run. That is defensible —
revision behaviour is a deterministic property of the decoder, not of the host —
but it should be reproduced on a second machine before publication.
`bench_commit_delay.py --tag ...` prints a full diagnostic table rather than
reporting zero.

## Reproduce

```bash
./run_m4.command          # full pipeline, ~6 min after first-run downloads
./run_selftest.command    # causality proof only, ~1 min
python3 bench/bench_commit_delay.py    # Finding 6, ~1 min, no dataset needed
```

*Finding 6 note:* commit delay is a property of the model's decoding, not of
the host CPU — the measurement is deterministic and portable, so the result
above is tagged `zipformer` rather than `m4`.
