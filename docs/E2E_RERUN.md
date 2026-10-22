# End-to-end lookahead re-run: the lookahead benefit was mostly leaked context

**Status: overturns RQ1, most of RQ3 and the cross-encoder result of the first
arXiv version.** Data: `results/raw/e2e/` (Colab T4, 10-11 Oct 2026,
`notebooks/colab_rerun_end_to_end.ipynb`, `--lookahead-scope end_to_end`).

The first version applied the lookahead mask in every encoder layer and probe
block, so right context compounded with depth (`eval/effective_lookahead.py`:
labelled 40 ms reached at least 220 ms, 340 ms at least 1.5 s). The re-run uses
a strictly causal encoder and puts the lookahead in the first probe block only,
so the future context equals the label exactly (verified on Colab for every
budget before training).

## 1. WavLM base+, canonical-phone arm, 21 budgets x 2 seeds

| L (ms) | per-layer (old) | end-to-end (new) |
|---:|---:|---:|
| 0 | 0.4256 | 0.4517 |
| 40 | 0.3275 | 0.4446 |
| 80 | 0.2754 | 0.4335 |
| 160 | 0.2206 | 0.4301 |
| 320 | 0.1823 | **0.4273** (minimum) |
| 640 | 0.1561 | 0.4352 |

| | per-layer | end-to-end |
|---|---:|---:|
| slope, PER per doubling | -0.0453 (R² 0.985) | -0.0033 (R² 0.60) |
| relative gain 0 -> 640 ms | 63.3% | 3.7% |
| best relative gain | 63.3% at 640 | 5.4% at 320 |

Almost all of the gain arrives by 80 ms (0.4517 -> 0.4335, 0.018 PER, 3.6x the
new floor). Beyond 80 ms no 20 ms step exceeds the floor and the curve is flat
to within noise, rising slightly at 360-640 ms.

## 2. Repeatability floor (4 budgets x 4 repeats, fixed seed)

sigma = 0.00251, **2 sigma = 0.0050** (12 df), larger than the per-layer
estimate of 0.0029. On this floor the saturation onset is about **80 ms**.

## 3. Matched-capacity control (RQ3)

| | canonical | realized |
|---|---:|---:|
| relative gain 0 -> 640 | 3.7% | 2.4% |
| best relative gain | 5.4% (320 ms) | 3.9% (180 ms) |
| slope per doubling | -0.0033 | -0.0018 |

Three-seed sweep: endpoint gains per seed 0.020 / 0.022 / 0.013 (canonical)
against 0.016 / 0.038 / 0.009 (realized), so the arms are not separable.
Preference margin: canonical 0.031 -> 0.034, realized 0.040 -> 0.038; both
changes are within the floor. **The 1.55x difference and the monotone margin
growth of the first version do not survive.**

## 4. Convergence (6000 steps, test PER)

0 ms 0.4183, 40 ms 0.4078, 160 ms 0.3941, **240 ms 0.3892**, 640 ms 0.4053.
Longer training enlarges the benefit somewhat (7.0% relative at 240 ms) but
the curve stays shallow and 640 ms is worse than 240 ms.

## 5. Second encoder (wav2vec 2.0 base)

Layer selection at L = 0 again picks layer 2 (0.438; layers 4/6/8/9: 0.465,
0.564, 0.699, 0.775).

| L (ms) | 0 | 40 | 80 | 160 | 320 | 640 |
|---|---:|---:|---:|---:|---:|---:|
| seed 1337 | 0.4357 | 0.4199 | **0.4183** | 0.4267 | 0.4402 | 0.4544 |
| seed 7 | 0.4387 | 0.4267 | | 0.4295 | (340) 0.4358 | 0.4429 |
| 6000 steps | 0.3797 | | | **0.3593** | | 0.3953 |

Best gain 4.0% at 80 ms (5.4% at 160 ms after 6000 steps), then it degrades.
**Under end-to-end lookahead the two encoders behave alike.** The 63% vs 3.9%
contrast of the first version came from tap depth: nine compounding layers
against two.

## 6. What this means

1. With exact end-to-end lookahead, this probe gains at most about 4-7%
   relative PER from future context, and essentially all of it within
   80-240 ms.
2. The headline curve of the first version measured leaked context.
3. Amount of context is not the whole story. The per-layer model at a
   labelled 20 ms has at most about 540 ms of future (13 layers x 2 frames)
   and reaches 0.370; the end-to-end model with 660 ms reaches only 0.435.
   Future context entering *inside* the pretrained encoder is far more useful
   than the same amount added at the head over causal features.

## Open question this leaves

Point 3 has two readings: the encoder needs future context to build good
representations, or the deep per-layer model simply has more effective
context than the threshold-based bound shows. The test that separates them is
an encoder-side lookahead that does not compound: exact lookahead in the first
encoder layer only, all later layers chunk-causal. That needs a code change
and one more Colab sweep.

## 7. Encoder-side lookahead (`encoder_first`, 22 Oct 2026)

Exact lookahead in encoder layer 0 only; every later encoder layer and every
probe block chunk-causal. Verified exact for every budget on Colab before
training. Data: `results/raw/e2e/translator_encfirst.csv`,
`w2v2_encfirst_l2.csv`.

WavLM base+, canonical-phone arm, 2 seeds:

| L (ms) | per-layer | end-to-end | encoder-first |
|---:|---:|---:|---:|
| 0 | 0.4256 | 0.4517 | 0.4238 |
| 20 | 0.3700 | 0.4486 | 0.4082 |
| 40 | 0.3275 | 0.4446 | 0.3979 |
| 80 | 0.2754 | 0.4335 | 0.3883 |
| 160 | 0.2206 | 0.4301 | 0.3854 |
| 320 | 0.1823 | 0.4273 | 0.3776 |
| 640 | 0.1561 | 0.4352 | 0.3707 |
| gain 0 -> 640 | 63.3% | 3.7% | **12.5%** |
| slope / doubling | -0.0446 | -0.0035 | -0.0072 |

wav2vec 2.0 base, layer 2, 1 seed: 0.4295 at 0 ms, 0.4226 at 320 ms, 0.4288 at
640 ms; best gain 1.6%, effectively flat.

Reading:

1. Context inside the encoder is worth more than the same context at the head
   (12.5% against 3.7% at 640 ms, and the encoder-first curve is still falling
   at 640 ms rather than flat).
2. It does not come close to the per-layer result. Per-layer at a labelled
   20 ms (at most ~520 ms of real future) reaches 0.370; encoder-first with a
   full 660 ms reaches 0.371. The 63% of the first version was mainly the
   amount of leaked context, not where it entered.
3. A small in-encoder context matters disproportionately: at L = 0 the only
   difference between end-to-end (0.4517) and encoder-first (0.4238) is that
   encoder frames may see the rest of their 40 ms chunk, one extra 20 ms frame
   for half of them, and that alone is worth 6%.
4. The encoder-first seeds disagree by up to 0.024 PER (160 ms), about 5x the
   repeatability floor, so per-budget differences above 80 ms are not well
   resolved with two seeds.
5. The wav2vec 2.0 tap at layer 2 has only two encoder layers above the
   lookahead layer, and it gains nothing.
