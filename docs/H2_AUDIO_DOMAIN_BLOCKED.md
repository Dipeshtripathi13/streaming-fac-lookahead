# H2 in the audio domain: the cheap route is invalid, and why

**Status: item (i) of `paper/main.tex` §Remaining work is NOT closed.** A
TTS-counterfactual substitute for forced alignment was built, tested, and
rejected on evidence. The rejection is the result; the numbers it would have
produced are not reportable.

Code: `eval/h2_audio_domain.py` (13 self-tests passing). Retained rather than
deleted, because the distortion measure, the determinism guard and the
counterfactual construction are all correct and reusable, only the conclusion
they were meant to support is unreachable this way.

## What was attempted

Sequence-level PER refuted H2 (`H2_NOT_SUPPORTED.md`), but PER charges every
error exactly 1.0 regardless of how it sounds. That leaves one escape: maybe the
manner grouping is fine and PER is simply blind to how much each error costs
acoustically. The plan was to re-run the identical H2 criterion with each error
weighted by its acoustic consequence instead of by 1.0, reusing
`h2_from_rates(..., metric=...)` so only the weighting changed.

Acoustic consequence was to be measured counterfactually: rebuild the canonical
phone string with **one** error applied, synthesise it, and compare against the
clean reference. No forced aligner required, which is what made it attractive.

## Why it does not work

**1. Piper is stochastic by default, and nobody had checked.**
`noise_scale=0.667`, `noise_w=0.8`. The same phone string synthesised twice:

| | value |
|---|---|
| waveform lengths | 41472 vs 39424 samples |
| bitwise identical | no |
| **self-distortion** | **244.8** |
| a genuine one-phone change | 277.6 |

**About 88% of the apparent "cost of an error" was synthesis noise.** Zeroing
`noise_scale` and `noise_w` makes synthesis bitwise deterministic (self-distortion
exactly 0.0000). `_assert_deterministic()` now refuses to measure otherwise, and
that guard is self-tested.

**2. Even deterministic, VITS is globally coupled, and this is the fatal one.**
Changing a single phone perturbs the entire utterance, not its own region:

| probe | share of total distance in worst 10% of frames |
|---|---|
| ɪ→u at phone 1 | 16.5% |
| b→s at phone 17 | 23.4% |
| m→k at phone 33 | 22.1% |

Uniform would be 10%. So the change is spread almost evenly across the whole
signal. Every single-phone substitution costs a near-constant floor of ~210
whatever the phone, with only ~208–282 of spread on top, and the correlation
with duration change is just +0.28, so this is genuine global coupling in the
vocoder, not a duration artefact.

**The second-order consequence is worse than the first.** With a near-constant
cost per error, `acoustic_cost_rate` collapses to `(constant × error_rate)`. The
audio-weighted H2 test would therefore have reproduced the sequence-level test
**by construction**, and the agreement would have read as independent
confirmation that acoustic weighting does not rescue H2. For this specific
question that is the most dangerous failure mode available: a result that is
both wrong and exactly what the authors expected.

## What item (i) actually requires

Frame-level forced alignment against real converted audio, as
`paper/main.tex` already states, a phoneme-CTC acoustic model (e.g. an
espeak-IPA wav2vec2) aligning the reference phone string to the audio, then
per-phone distortion read off the alignment. The blocker is mundane: the
sandbox running this work has no network, and the model is not in the local HF
cache. It is straightforward to run where network exists (the Colab notebook
already clones this repo and has both network and a GPU).

## Knock-on: the listening-test stimuli

`synth/phones_to_audio.py` takes `noise_scale`/`noise_w` from the model card, so
**every clip built by `eval/listening_test.py build` carries random synthesis
variation**. For a forced-choice test that is not a bias, the variation is
independent of condition, but it is added variance in exactly the comparison
raters are asked to make, and paired A/B clips are not minimal pairs. Rebuilding
the stimuli with `make_deterministic()` would remove it at no cost. Worth doing
before any rater study is funded.
