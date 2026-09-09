# The exchange rate is a property of the encoder: not of speech

**Status: answers the reviewer question "is this a fact about speech or about
WavLM?".** Qualifies the headline of
[`DENSE_SWEEP_RQ1_ANSWERED.md`](DENSE_SWEEP_RQ1_ANSWERED.md) and is written
into the paper as §"The exchange rate is a property of the encoder".

Evidence: `results/raw/translator_w2v2_layer2.jsonl` (the replication),
`translator_w2v2_layer9.jsonl` (the first, invalid attempt),
`w2v2_layer_probe.json`, `w2v2_depth_sensitivity.json`. Analysed by
`eval/analyse_second_encoder.py` (7 self-tests) into
`results/analysis_second_encoder.json`.

Everything is held fixed except the checkpoint: corpus, speaker-disjoint
split, CTC head, optimiser, 1200 steps, batch 8, chunk 40 ms, lookback 2000 ms,
seed 1337, native arm, the 7-point grid. Both encoders are causalised by the
same two patches and both pass the same truncation proof.

## The answer

| encoder | L=0 | L=640 | rel. gain | PER/doubling | R² |
|---|---:|---:|---:|---:|---:|
| WavLM base+ (layer 9) | 0.4256 | 0.1561 | **+63.3%** | −0.0446 | 0.989 |
| wav2vec 2.0 base (layer 2) | 0.4359 | 0.4189 | **+3.9%** | −0.0015 | 0.099 |
| wav2vec 2.0 base, seed 7 | 0.4376 | 0.4265 | **+2.5%** | -- | -- |

**Replicated at a second seed.** Seed 7 lands within 0.008 PER of seed 1337 at
every shared budget (0.4376/0.4359 at L=0, 0.4219/0.4169 at 40, 0.4043/0.4004
at 160, 0.4265/0.4189 at 640; seed 7 adds 0.4155 at 340). Both curves reach
their minimum at **the same L=160 ms** and both rise again by 640 ms, so the
non-monotonicity is a reproducible property of this encoder rather than one
seed's noise, which matters, because that shape is the evidence that the
curve shape does not transfer.

The two have **similar zero-lookahead PER**, 0.4359 against 0.4256, within
0.01 PER, and then differ by more than an order of magnitude in what they do with
640 ms of it. Direction replicates; magnitude does not. The log-linear fit that
describes WavLM almost perfectly (R² 0.989) is meaningless here (R² 0.099), and
the wav2vec 2.0 curve bottoms out at L=160 ms before rising again.

The gain is small but **real, not noise**: 0.017 PER absolute is 5.9× the
2σ = 0.0029 repeatability floor.

## Three things that had to be ruled out first

### 1. Layer 9 is the wrong tap for wav2vec 2.0

The first attempt matched layers by *index* and produced a degenerate run:
PER **rose** from 0.772 at L=0 to 0.928, with training loss still falling
steeply at step 1200. Reporting that as a failed replication would have been
wrong. Probing the stack at L=160 ms, everything else fixed:

| layer | 2 | 3 | 4 | 6 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| test PER | **0.402** | 0.411 | 0.437 | 0.586 | 0.880 | 0.928 | 0.971 |

Monotone in depth. wav2vec 2.0 base carries phonetic information low in the
stack; WavLM base+ carries it at 9. **Matching two encoders by layer index is
not matching them.**

### 2. "It's the 1200-step budget": tested, and false

A second seed cannot rule out an optimisation effect that both seeds share, so
wav2vec 2.0 gets the same convergence check WavLM has, at L ∈ {0, 160, 640} ms
to 6000 steps:

| L (ms) | 1200 steps | 6000 steps |
|---:|---:|---:|
| 0 | 0.4359 | 0.3782 |
| 160 | 0.4004 | **0.3361** |
| 640 | 0.4189 | 0.3714 |

Every condition improves, but **they improve together**. The minimum stays at
L=160 ms and L=640 stays worse than it, so the non-monotonic shape is
unchanged. The endpoint gain does not grow with training, it **shrinks**,
from +3.9% to +1.8%, while WavLM at the same 6000 steps gains +65.8%. Five
times the budget widens the gap rather than closing it.

### 3. "Layer 2 is too shallow to need context": tested, and false

The natural objection to the result above: layer 2 has had almost no
contextual mixing, so there was little context to withhold, and sensitivity
should recover deeper in the stack. Measured directly, relative gain
L=0 → L=640 by depth:

| layer | 2 | 4 | 6 | 8 | WavLM 9 |
|---|---:|---:|---:|---:|---:|
| rel. gain | +3.9% | +5.4% | +5.2% | **−14.1%** | +63.3% |

**The hypothesis is refuted.** Sensitivity is flat across every layer where
this encoder can still decode phones, then goes negative. There is no depth at
which wav2vec 2.0 base both decodes phones and exploits lookahead the way
WavLM does. We looked for the trade-off and did not find it.

## What this costs the paper

- **−0.045 PER per doubling**, the **≈340 ms** saturation point, and the cost
  attributed to a 40 ms budget are properties of **WavLM base+**, not facts
  about how much future context English phones require. They must be
  re-measured on whatever backbone a system actually deploys.
- **What survives the swap** is the qualitative shape: returns are positive,
  smooth and diminishing, with no locatable knee.

## What this does not establish

Two pretrained checkpoints. The wav2vec 2.0 side is replicated at a second
training seed and re-checked at 6000 steps; the WavLM curve rests on the seeds
of the main sweep. This is enough to show the exchange rate does not transfer,
and not enough to say *why*. The gap could trace to pretraining scale
(960 h against 94k h), to WavLM's denoising and speaker-conditioned objectives,
or to architecture. Each encoder is also tapped at a single layer chosen by a
probe at one lookahead, so layer and sensitivity are selected on partly
overlapping evidence. A third backbone (HuBERT base) would turn two points into
a trend.

## A bug this work exposed

`MaskInjector` has two strategies. WavLM takes the `position_bias` path, which
adds to an existing tensor and leaves the model's padding mask alone. **Every
other encoder** takes the `layer_attention_mask` path, which *replaced* the
layer's attention mask, discarding the padding mask that
`Wav2Vec2Encoder.forward` builds, so frames near the end of a short utterance
attended into the zero-padded tail of the batch.

It was worse than a constant bias: a strictly causal mask cannot reach the
trailing pad at all, a wide one can, so **the corruption grows with lookahead**
, 3.6e-4 at L=0 rising to 1.8e-2 at L=640 on a 3-layer probe, riding directly
on the swept variable. `padding_mask_selftest()` makes it permanent, runs
offline on a randomly initialised wav2vec 2.0, and the trainer aborts on
failure. WavLM results are unaffected: they never took this path.
