# RQ1, answered: there is no knee to publish — there is a saturation point at ~240 ms

**Status: RQ1 answered on the trained curve.** Supersedes the provisional
answers in `PROPOSAL_v2.md` §3 (encoder-level F8) and the knee claim in
`PADDING_FIX_RESOLVED.md` §3.

Evidence: `results/raw/translator_dense.jsonl` — 16 lookaheads × 2 seeds = 32
runs, native arm, 1200 steps, batch 8, chunk 40 ms, T4, padding-fixed code.
Analysed by `eval/analyse_dense_knee.py` (7 self-tests), output in
`results/analysis_dense_knee.json`. All 32 cells present, single arm, no frame
quantisation collisions.

## The answer

**Curvature is real. A knee location is not.** All three axis treatments prefer
a two-segment fit — decisively, ΔBIC −22 to −37 — but the breakpoint moves and
none of them can localise it:

| treatment of L=0 | ΔBIC | breakpoint | bootstrap 90% CI | identifiable |
|---|---:|---:|---|---|
| log₂(L+1), all 16 points | −22.2 | 40 ms | [40, 320] ms | **no** |
| L=0 dropped (15 points) | −31.2 | 180 ms | [60, 320] ms | **no** |
| L=0 at half a frame | −36.8 | 180 ms | [40, 320] ms | **no** |

The bootstrap intervals span 3 octaves. **Sixteen points is not the fix**, which
is the thing worth reporting: the earlier diagnosis that 7 points were
underpowered was correct but incomplete. Denser sampling did not rescue a knee,
because the curve is smoothly curved rather than piecewise. A breakpoint
estimator asked to find a breakpoint will always return one; that is a property
of the estimator, not of the speech.

So H1's three pre-registered branches all mis-frame the outcome. The honest
statement is the fourth: **smooth diminishing returns, no operating point, but a
measurable point where returns stop being measurable at all.**

## What replaces the knee: the saturation point

Marginal PER gain per added 20 ms frame, against the measured 2σ = 0.0044 floor:

| step (ms) | gain / frame | × floor |
|---|---:|---:|
| 0 → 20 | +0.0556 | 12.7 |
| 20 → 40 | +0.0426 | 9.8 |
| 40 → 60 | +0.0312 | 7.2 |
| 60 → 80 | +0.0209 | 4.8 |
| 80 → 100 | +0.0190 | 4.4 |
| 100 → 120 | +0.0118 | 2.7 |
| 120 → 140 | +0.0176 | 4.0 |
| 140 → 160 | +0.0064 | 1.5 |
| 160 → 180 | +0.0099 | 2.3 |
| 180 → 200 | +0.0045 | 1.0 |
| 200 → 240 | +0.0044 | 1.0 |
| 240 → 280 | +0.0041 | **0.9** |
| 280 → 320 | +0.0034 | **0.8** |
| 320 → 480 | +0.0022 | **0.5** |
| 480 → 640 | +0.0011 | **0.3** |

**Saturation at L ≈ 240 ms**: beyond it, every subsequent step is below the
noise floor, so further lookahead is not measurably useful at this sample size.
This is a deployment-relevant number that needs no breakpoint, and unlike a knee
it is stable — it depends only on the floor, not on an axis choice.

Note the honest weakness: the 140→160 and 180→200 steps are already at 1.0–1.5×
the floor while 120→140 is 4.0×. The per-step series is noisy at that scale, so
240 ms should be read as "somewhere around 200–280 ms", not as a sharp figure.

## The exchange rate, which is the citable result

Fitting L ≥ 20 ms on log₂(L):

> **−0.0452 PER per doubling of lookahead, R² = 0.983 (n = 15, 20–640 ms).**

Largest residual is L=640 (+0.018), i.e. the curve flattens slightly faster than
log-linear at the extreme — consistent with saturation, and the reason ΔBIC
prefers piecewise even though no single breakpoint fits.

## What this says about the field's 40 ms budget

PHONOS runs at ≤40 ms. On this curve:

| from 40 ms to | PER | relative reduction forgone at 40 ms |
|---|---:|---:|
| 200 ms | 0.3275 → 0.2062 | **37.1%** |
| 240 ms | 0.3275 → 0.1973 | **39.7%** |
| 320 ms | 0.3275 → 0.1823 | 44.3% |
| 640 ms | 0.3275 → 0.1561 | 52.3% |

**A 40 ms budget forgoes ~40% of the achievable PER reduction, and the returns
remain measurable out to ~240 ms — six times that budget.** This is the
strongest form of the "current budgets are under-provisioned" claim the project
has produced, and it now rests on the trained conversion curve rather than on
encoder representation drift.

It also converges with the independent encoder-level result (F8), which put the
best marginal return at 100–200 ms. Two different measurements, same region.

## Caveats that must travel with this

1. **The noise floor is imported, not re-measured.** This sweep has zero
   repeated cells, so the 0.0044 threshold comes from the 3-seed run's
   accidental repeats. Median seed spread here is 0.0039 with a max of 0.0074,
   which is consistent with that floor but does not independently confirm it. A
   deliberate repeat of 3–4 conditions would fix this cheaply.
2. **Two seeds, native arm only.** Adequate for curve shape given seed spreads
   of 0.002–0.007, but the produced arm was not densely sampled, so nothing here
   speaks to H3.
3. **PER, not perceptual quality.** The listening test remains the only route to
   a claim about what listeners hear; 240 ms of *measurable* PER return does not
   mean 240 ms of *audible* improvement.
4. **1200 steps, not converged training.** The curve is a comparison at fixed
   budget, not an asymptotic one.

## Changes to make in the paper

1. **Withdraw the knee entirely** — including the 40 ms figure from
   `PADDING_FIX_RESOLVED.md` §3, which this supersedes. Report ΔBIC prefers
   curvature *and* that the location is unidentifiable at 16 points.
2. **Lead with the exchange rate** (−0.045 PER/doubling, R² 0.98) and the
   **saturation point** (~240 ms, floor-relative).
3. **State the 40 ms cost**: ~40% of achievable reduction forgone.
4. Add the methodological finding: **a breakpoint estimator returns a breakpoint
   whether or not one exists**; identifiability must be tested by perturbing the
   axis and bootstrapping the location, not by ΔBIC alone. Sixteen points with
   ΔBIC −37 still failed to localise.
5. Keep H1's restated framing but record that all three of its branches were the
   wrong trichotomy.
