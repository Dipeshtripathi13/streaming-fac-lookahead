# The padding fix changed the shape of the curve, not just its height

**Status: resolved.** This closes the open question in
`CACHING_CHANGED_THE_NUMBERS.md`. The 3-seed numbers can go in the paper, but
**not** the ones already there — several conclusions change.

Evidence: `results/raw/translator_sweep_3seed.jsonl` (42 conditions, 48 lines,
6 accidental repeats), analysed by `eval/compare_padding_fix.py`, output in
`results/compare_padding_fix.json`. Same seeds, lookaheads, targets, steps
(1200), batch size (8), chunk (40 ms), T4 and feature-cache path as the earlier
sweep. Only the MaskedBlock zero-pad fix differs, and the comparability guard
reported no hyperparameter drift.

## 1. It is a shape change, and that was the thing that mattered

The question was whether fixing the bug shifted every PER by roughly the same
amount — in which case every *within-run* comparison survives untouched, because
a constant cancels in every difference — or whether the shift depended on
lookahead, in which case the curve's shape was partly an artefact.

It depends on lookahead, by a wide margin:

| L (ms) | 0 | 20 | 40 | 80 | 160 | 320 | 640 |
|---|---|---|---|---|---|---|---|
| mean ΔPER (new − old) | −0.014 | −0.019 | −0.027 | −0.054 | −0.068 | **−0.086** | −0.052 |

Spread across lookahead is 0.072 PER against a measured noise floor of 0.0044 —
**16.4×**. Slope on log₂(L+1) is −3.98 standard errors from zero. All 42 cells
improved.

So the bug did most of its damage at *large* lookahead. The old curve therefore
**understated the benefit of lookahead**, which is the opposite of a harmless
offset: it biased the central quantity of RQ1 toward "lookahead does not help
much".

## 2. What the corrected curve looks like

Native arm, own-target test PER, mean over 3 seeds:

| L (ms) | 0 | 20 | 40 | 80 | 160 | 320 | 640 |
|---|---|---|---|---|---|---|---|
| new | 0.427 | 0.369 | 0.327 | 0.276 | 0.222 | 0.185 | 0.157 |
| old | 0.447 | 0.396 | 0.362 | 0.335 | 0.303 | 0.302 | 0.234 |

Three things improve at once:

- **Monotone.** Every adjacent step is negative, unanimous across all 3 seeds,
  and significant. The old curve was not monotone: seed 1337 went *up* from
  0.3148 at L=160 to 0.3514 at L=320. That non-monotonicity was a bug artefact,
  not a finding, and it should not be explained in the paper.
- **Endpoint gain is much larger.** Native L0→640 improves by 0.270 absolute /
  **63.2% relative**, against 0.212 / 47.6% before.
- **Seed spread collapses.** SD of the endpoint gain across seeds falls from
  0.0183 to **0.0007** — a 26× reduction. The zero-pad contamination was itself
  a major source of apparent seed variance, which is why the earlier analysis
  had to work so hard to resolve anything.

## 3. RQ1's answer flips direction, but the operating point is still unknown

> **SUPERSEDED (9 Aug 2026) by `docs/DENSE_SWEEP_RQ1_ANSWERED.md`.** The
> dense 16-point sweep shows there is *no locatable knee at all* -- the
> breakpoint remains unidentifiable even at 16 points, and the 40 ms and
> 160 ms figures below are both fitting artefacts. Cite the exchange rate
> (-0.0452 PER/doubling) and the saturation point (~240 ms) instead. The
> section is kept because the axis-sensitivity table is the evidence that
> motivated the dense sweep.

Previously: no knee, log-linear, ΔBIC +5.27 → *"publish the exchange rate,
there is no operating point."*

Now a knee **is** detected (native ΔBIC −10.11, produced −7.41). That is a
different answer to RQ1.

**But the location is not identified, and this must not be reported as though it
were.** The knee position depends entirely on an arbitrary choice — where L=0
sits on a log axis:

| treatment of L=0 | native | produced |
|---|---|---|
| all 7 points, log₂(L+1) | knee at **40 ms** (ΔBIC −10.11) | knee at 40 ms (−7.41) |
| L=0 dropped (6 points) | knee at **160 ms** (ΔBIC −9.16) | **no knee** (−3.95) |
| L=0 placed at 10 ms | knee at **160 ms** (ΔBIC −10.83) | **no knee** (−3.36) |

The cause is mechanical: on log₂(L+1) the gap from L=0 to L=20 is 4.39 units
while every other gap is ~1.0. A piecewise fit will place a breakpoint just
after that gap whatever the data does. The power calibration confirms the
estimator cannot localise: planting a 0.05 cliff at 81 ms yields a "detected"
knee at 40 ms — right conclusion, wrong place.

Defensible claim: **the native curve has real curvature; its location cannot be
resolved by a 7-point geometric grid.** The produced arm's detection is fragile
and should not be claimed at all.

This promotes a **dense-L sweep to required**. Two caveats on what that means,
because the obvious cheap option does not answer it:

- **Not 10 ms steps.** Lookahead is quantised to `ceil(L/frame_ms)` frames at a
  20 ms frame rate, so sub-frame steps silently duplicate conditions and
  reweight the fit (`bench_content_degradation.py --dense` documents this). The
  finest meaningful grid is 20 ms: 0–200 in 20 ms steps plus 240/280/320/480/640
  = 16 conditions.
- **Not the encoder pilot.** `--dense` on `bench_content_degradation.py` costs
  ~5 min but measures *encoder representation drift*, which is a lower bound on
  the system requirement, not the trained conversion curve. The knee that is
  unresolved here is in the **trained translator PER curve**, so locating it
  requires a dense *trained* sweep: 16 conditions x seeds at ~4 min each
  (~1 h at 1 seed, ~2 h at 2 seeds, ~3.2 h at 3).

Both are worth having and they answer different questions. The encoder pilot
also upgrades F8, whose 7-point grid the proposal already flags as underpowered
(a planted cliff returns dBIC = -0.2 at n = 7).

## 4. H3 is untouched — and that is not luck

The conversion-vs-transcription claim survives intact:

| | old | new |
|---|---|---|
| paired diff in endpoint gain | +0.0964 (t 25.9) | +0.0802 (t 39.7) |
| margin growth 0→640, native | +0.0690 | +0.0665 |
| margin growth 0→640, produced | −0.0103 | −0.0103 |

H3 lives on the **preference margin** (cross − own), and the padding bug moved
`own` and `cross` PER together, so it cancels in the difference. The one claim
the paper leans on hardest is the one the bug could never have manufactured.
That is a point in favour of the earlier decision to build H3 on the margin
rather than on raw PER.

*Coincidence, flagged so nobody reads meaning into it:* produced margin growth
is −0.010311 new against −0.010300 old — agreement to four decimals is chance.
The per-seed values differ (new −0.0071/−0.0098/−0.0140, old
−0.0089/−0.0119/−0.0101).

## 5. A GPU non-determinism floor, measured by accident

Two sweep processes ran concurrently for about an hour (operator error), which
repeated 6 conditions at **identical seed and configuration**. Those repeats are
the only direct measurement of run-to-run noise available, and a 3-seed design
cannot produce it:

| cell | run A | run B | Δ |
|---|---|---|---|
| native L0 s1337 | 0.4301 | 0.4236 | 0.00649 |
| native L20 s99 | 0.3700 | 0.3722 | 0.00222 |
| native L0 s7 | 0.4239 | 0.4259 | 0.00204 |
| native L0 s99 | 0.4277 | 0.4295 | 0.00177 |
| native L20 s1337 | 0.3645 | 0.3628 | 0.00168 |
| native L20 s7 | 0.3714 | 0.3715 | 0.00015 |

σ per measurement = **0.0022**; resolvable threshold (2σ) = **0.0044**.

This matters beyond bookkeeping. `analyse_3seed.py` cancels *seed* by pairing
within seed, but this residual is present **at fixed seed**, so no amount of
seed averaging removes it. Any claimed effect below ~0.004 PER is not
distinguishable from kernel non-determinism. The methods section should state
this floor and the analysis should gate on it, which
`compare_padding_fix.py` now does (`effect_resolvable`, `shape_spread_resolvable`).

All effects claimed above clear it comfortably: the overall offset is 10.4× the
floor, the shape spread 16.4×.

## 6. What to change in the paper

1. Replace every absolute PER in the 3-seed table with the new values.
2. Delete any discussion of the L=160→320 non-monotonicity. It was a bug.
3. Restate RQ1: curvature exists in the conversion arm; the operating point is
   unresolved at 7 points; the dense-L pilot decides it.
4. Keep H3 as written; note explicitly that it is offset-invariant by
   construction.
5. Add the non-determinism floor to methods, with the 6 repeat pairs as
   evidence.
6. Drop the "resolution floor ≈ 0.01 PER" framing inherited from the pooled-SD
   analysis; the measured fixed-seed floor is 0.0044 and the paired-by-seed SDs
   are now 0.001–0.006, so several previously "unresolvable" adjacent steps are
   resolvable — indeed all six adjacent steps are now unanimous and significant
   in both arms.
