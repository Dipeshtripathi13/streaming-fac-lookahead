"""Did fixing the zero-pad contamination change the science, or just the offset?

Why this file exists
--------------------
`docs/CACHING_CHANGED_THE_NUMBERS.md` leaves one question open, and it blocks
the 3-seed numbers from going in the paper: the sweep those numbers came from
ran with the MaskedBlock zero-pad bug present. Two things could follow from
fixing it.

  (a) Every PER moves by roughly the same amount. A constant offset cannot
      change any *within-run* comparison -- adjacent-step deltas, endpoint
      gains, margin growth, H3 -- because it cancels in every difference. The
      prior conclusions survive intact and only the absolute numbers get
      restated.

  (b) The offset varies with lookahead. Then the *shape* of the PER curve was
      an artefact, and the conclusions that live on that shape have to be
      re-derived rather than restated.

Distinguishing (a) from (b) is the entire job here, and it is not answered by
looking at whether the numbers "got better". They will get better; that is
uninformative. The question is whether the improvement is flat in L.

WHAT MAKES THIS COMPARISON LEGITIMATE
-------------------------------------
The old and new sweeps share seeds {1337, 7, 99}, lookaheads, targets, steps
(1200), batch size (8), chunk (40 ms), device class (T4) and the feature-cache
path. The *only* intended difference is the padding fix. So conditions can be
matched exactly on (target, lookahead, seed) and compared pairwise, which
removes seed -- a shared nuisance factor -- from the error term.

Two traps this script is built to avoid:

  1. `test_per` and `test_per_cross` are different quantities. The old sweep's
     cross-PER at L=0 native is 0.4838 while its own-PER is 0.4495. Comparing
     one against the other would manufacture a ~0.035 effect out of nothing.
     Both tables are matched by name here, never by position.

  2. Partial results. If the new sweep died at run 30/42, silently averaging
     over whatever landed would compare a full curve against a truncated one.
     Coverage is asserted before any statistic is computed.

Usage
-----
    python eval/compare_padding_fix.py --self-test
    python eval/compare_padding_fix.py --new results/raw/translator_sweep_3seed_summary.json
    python eval/compare_padding_fix.py --new <path> --out results/compare_padding_fix.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the prior analysis's data and estimators rather than re-typing either.
# Re-transcribing the old tables here would create a second source of truth
# that could silently drift from the one the paper already cites.
from analyse_3seed import (  # noqa: E402
    # Explicitly the PRE-FIX tables. analyse_3seed.OWN/CROSS now point at the
    # corrected sweep, so importing those here would compare the new data
    # against itself and report a delta of exactly zero -- a silent failure
    # that would look like "the padding fix changed nothing". Guarded by
    # self-test 12.
    CROSS_PREFIX_BUG as OLD_CROSS,
    L as LOOKAHEADS,
    OWN_PREFIX_BUG as OLD_OWN,
    SEEDS,
    adjacent_report,
    exact_sign_p,
    knee_bic,
    knee_power,
    mean,
    paired_t,
    sd,
)

ARMS = ("native", "produced")


# ==========================================================================
# Ingest
# ==========================================================================

def load_new(path: str) -> Tuple[Dict, Dict, Dict]:
    """Parse a sweep summary into {arm: {seed: [per per lookahead]}}.

    Returns (own, cross, provenance). Raises on any missing cell -- see trap 2.
    """
    # Accept either the end-of-sweep summary or the per-run JSONL. The JSONL is
    # the recovery path: it is appended after every condition, so if the runtime
    # dies before the summary is written it is the only surviving record of the
    # completed runs.
    if path.endswith(".jsonl"):
        rows = []
        with open(path) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A kill mid-write can truncate the final line. Drop it and
                    # say so, rather than failing on an otherwise usable file.
                    print(f"  note: {path} line {ln} is truncated; ignoring",
                          file=sys.stderr)
        d = {}
    else:
        with open(path) as f:
            d = json.load(f)
        rows = d.get("results")
    if not rows:
        raise SystemExit(f"{path}: no results found")

    own: Dict[str, Dict[int, List[Optional[float]]]] = {
        a: {s: [None] * len(LOOKAHEADS) for s in SEEDS} for a in ARMS
    }
    cross = {a: {s: [None] * len(LOOKAHEADS) for s in SEEDS} for a in ARMS}

    idx = {float(v): i for i, v in enumerate(LOOKAHEADS)}
    unexpected: List[str] = []

    # Collect ALL observations per cell rather than letting a later row
    # overwrite an earlier one. A cell can legitimately appear twice -- e.g. two
    # sweep processes ran concurrently against the same output file -- and those
    # repeats are the only direct measurement of run-to-run (fixed-seed) noise
    # available. Overwriting would discard exactly the information needed to
    # decide whether an effect is resolvable.
    obs_own: Dict[tuple, List[float]] = collections.defaultdict(list)
    obs_cross: Dict[tuple, List[float]] = collections.defaultdict(list)
    for r in rows:
        arm = r.get("target")
        seed = r.get("seed")
        lm = r.get("lookahead_ms")
        if arm not in ARMS or seed not in SEEDS or float(lm) not in idx:
            unexpected.append(f"target={arm} seed={seed} L={lm}")
            continue
        i = idx[float(lm)]
        obs_own[(arm, seed, i)].append(float(r["test_per"]))
        obs_cross[(arm, seed, i)].append(float(r["test_per_cross"]))

    for (arm, seed, i), vals in obs_own.items():
        own[arm][seed][i] = mean(vals)          # average repeats
    for (arm, seed, i), vals in obs_cross.items():
        cross[arm][seed][i] = mean(vals)

    missing = [
        f"{a}/seed{s}/L={LOOKAHEADS[i]}"
        for a in ARMS
        for s in SEEDS
        for i in range(len(LOOKAHEADS))
        if own[a][s][i] is None
    ]
    if missing:
        raise SystemExit(
            "new sweep is incomplete -- refusing to compare a truncated curve "
            f"against a full one.\n  missing {len(missing)} of "
            f"{len(ARMS) * len(SEEDS) * len(LOOKAHEADS)} cells:\n    "
            + "\n    ".join(missing[:12])
            + ("\n    ..." if len(missing) > 12 else "")
        )

    prov = {
        k: d.get(k)
        for k in ("steps", "batch_size", "chunk_ms", "device", "gpu", "encoder")
    }
    if unexpected:
        prov["unexpected_rows"] = unexpected[:10]
    prov["repeat_noise"] = repeat_noise(obs_own)
    return own, cross, prov


def repeat_noise(obs: Dict[tuple, List[float]]) -> Dict[str, object]:
    """Estimate fixed-seed run-to-run noise from repeated cells.

    Why this exists
    ---------------
    `analyse_3seed.py` treats seed as the only nuisance factor and cancels it by
    comparing conditions *within* a seed. That is the right move for seed
    effects, but it silently assumes a rerun at the same seed reproduces
    exactly. On a GPU it does not: non-deterministic reductions and kernel
    autotuning leave a residual. Observed here at up to 0.0065 PER on identical
    (target, lookahead, seed) -- the same order as the effects being claimed.

    So any |effect| below this floor is not resolvable no matter how many seeds
    are averaged, because the floor is present *at fixed seed*. Reporting a
    paired t without it would overstate confidence.

    Estimator: for repeats a, b of the same cell, d = a - b has variance 2*sigma^2,
    so sigma = sqrt(mean(d^2) / 2). This uses the magnitudes directly. An earlier
    version used sd(|d|)/sqrt(2), which is wrong in a way that fails silently:
    if the observed |d| happen to be equal, sd(|d|) is 0 and the noise floor
    collapses to zero -- reporting perfect precision precisely when there is no
    evidence for it. Caught by self-test 11.
    """
    diffs: List[float] = []
    per_cell: Dict[str, float] = {}
    for k, vals in obs.items():
        if len(vals) > 1:
            d = max(vals) - min(vals)
            diffs.append(d)
            per_cell[f"{k[0]}/L{LOOKAHEADS[k[2]]}/s{k[1]}"] = round(d, 5)
    if not diffs:
        return {
            "n_repeated_cells": 0,
            "available": False,
            "note": "no repeated cells; run-to-run noise is UNMEASURED and the "
                    "significance below assumes reruns are exact, which on GPU "
                    "they are not",
        }
    sd_single = math.sqrt(sum(d * d for d in diffs) / len(diffs) / 2.0)
    return {
        "n_repeated_cells": len(diffs),
        "available": True,
        "mean_abs_diff": mean(diffs),
        "max_abs_diff": max(diffs),
        "sd_per_measurement": sd_single,
        "per_cell": per_cell,
        "resolvable_threshold": 2 * sd_single,
        "note": "fixed-seed reruns of identical configs; effects below "
                "resolvable_threshold are not distinguishable from GPU "
                "non-determinism",
    }


def check_comparability(prov: Dict) -> List[str]:
    """Flag anything that would make the pairing invalid.

    A difference here does not stop the run, but it must be reported: if steps
    or batch size changed, the delta is not attributable to the padding fix
    alone and the verdict below is not entitled to name a cause.
    """
    expected = {"steps": 1200, "batch_size": 8, "chunk_ms": 40}
    warn = []
    for k, v in expected.items():
        got = prov.get(k)
        if got is not None and got != v:
            warn.append(
                f"{k}={got} but the old sweep used {v}; the delta is NOT "
                "attributable to the padding fix alone"
            )
    return warn


# ==========================================================================
# Statistics
# ==========================================================================

def bootstrap_ci(
    xs: Sequence[float], reps: int = 20000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean. Fixed seed: a CI that moves
    between runs of the same script is not a CI anyone can cite."""
    rng = random.Random(seed)
    n = len(xs)
    if n < 2:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(reps):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * reps)]
    hi = means[min(reps - 1, int((1 - alpha / 2) * reps))]
    return (lo, hi)


def ols(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    """Slope, intercept, and the slope's standard error."""
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0 or n < 3:
        return (0.0, my, float("inf"))
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / (n - 2)
    return (b, a, math.sqrt(s2 / sxx))


def offset_or_shape(
    deltas_by_L: Dict[int, List[float]]
) -> Dict[str, object]:
    """Is the old->new change a constant offset, or does it depend on L?

    Regresses the per-condition delta on log2(L+1). A slope
    indistinguishable from zero means case (a): the fix shifted the whole
    curve and every within-run conclusion is untouched. A slope that clears
    its own standard error by ~2x means case (b).

    Also reports the spread of per-L mean deltas against the pooled
    within-L seed SD, because a formally significant but tiny slope is still
    an offset for practical purposes.
    """
    xs, ys = [], []
    for lm, ds in deltas_by_L.items():
        for d in ds:
            xs.append(math.log2(lm + 1.0))
            ys.append(d)
    slope, intercept, se = ols(xs, ys)
    t = slope / se if se > 0 and math.isfinite(se) else 0.0

    per_L_means = {lm: mean(ds) for lm, ds in deltas_by_L.items()}
    within_sds = [sd(ds) for ds in deltas_by_L.values() if len(ds) > 1]
    spread = max(per_L_means.values()) - min(per_L_means.values())
    pooled_within = (
        math.sqrt(sum(s * s for s in within_sds) / len(within_sds))
        if within_sds
        else float("nan")
    )

    verdict = (
        "shape changed (delta depends on lookahead)"
        if abs(t) >= 2.0
        else "constant offset (within-run conclusions unaffected)"
    )
    return {
        "slope_per_log2L": slope,
        "slope_se": se,
        "slope_t": t,
        "intercept": intercept,
        "per_L_mean_delta": per_L_means,
        "range_of_per_L_means": spread,
        "pooled_within_L_seed_sd": pooled_within,
        "spread_over_noise": spread / pooled_within
        if pooled_within and math.isfinite(pooled_within) and pooled_within > 0
        else float("nan"),
        "verdict": verdict,
    }


def paired_block(old: Dict, new: Dict, label: str) -> Dict[str, object]:
    """Paired old->new comparison over all (arm, seed, L) cells."""
    deltas: List[float] = []
    by_L: Dict[int, List[float]] = {lm: [] for lm in LOOKAHEADS}
    for arm in ARMS:
        for s in SEEDS:
            for i, lm in enumerate(LOOKAHEADS):
                d = new[arm][s][i] - old[arm][s][i]
                deltas.append(d)
                by_L[lm].append(d)

    t, dof, n = paired_t(deltas)
    n_neg = sum(1 for d in deltas if d < 0)
    lo, hi = bootstrap_ci(deltas)
    return {
        "quantity": label,
        "n_pairs": n,
        "mean_delta": mean(deltas),
        "sd_delta": sd(deltas),
        "paired_t": t,
        "dof": dof,
        "n_improved": n_neg,
        "sign_p": exact_sign_p(n_neg, n),
        "ci95_mean_delta": [lo, hi],
        "offset_or_shape": offset_or_shape(by_L),
    }


def rederive_claims(own: Dict, cross: Dict) -> Dict[str, object]:
    """Re-run the conclusions that the paper actually states, on the new data.

    A constant offset predicts these come out materially unchanged. Reporting
    them is what turns "the offset is flat" from an assertion into a check.
    """
    out: Dict[str, object] = {}
    for arm in ARMS:
        per = {s: own[arm][s] for s in SEEDS}
        marg = {
            s: [c - o for c, o in zip(cross[arm][s], own[arm][s])] for s in SEEDS
        }

        endpoint = [per[s][-1] - per[s][0] for s in SEEDS]
        t, _, n = paired_t(endpoint)
        out[f"per_endpoint_{arm}"] = {
            "delta_mean": mean(endpoint),
            "delta_sd": sd(endpoint),
            "t": t,
            "relative_gain": -mean(endpoint) / mean([per[s][0] for s in SEEDS]),
        }
        out[f"per_adjacent_{arm}"] = adjacent_report(per, lower_is_better=True)

        mg = [marg[s][-1] - marg[s][0] for s in SEEDS]
        out[f"margin_growth_{arm}"] = {
            "delta_0_to_640_mean": mean(mg),
            "delta_0_to_640_sd": sd(mg),
            "monotone_seeds": sum(
                1
                for s in SEEDS
                if all(b >= a for a, b in zip(marg[s], marg[s][1:]))
            ),
            "ratio_640_over_0": (
                mean([marg[s][-1] for s in SEEDS])
                / mean([marg[s][0] for s in SEEDS])
                if mean([marg[s][0] for s in SEEDS])
                else float("nan")
            ),
        }

        xs = [math.log2(lm + 1.0) for lm in LOOKAHEADS]
        ys = [mean([per[s][i] for s in SEEDS]) for i in range(len(LOOKAHEADS))]
        out[f"knee_{arm}"] = knee_bic(xs, ys)
        out[f"knee_power_{arm}"] = knee_power(xs, ys, cliff=0.05, at_x=math.log2(81))

    # H3: is the conversion arm's gain larger than the transcription arm's?
    conv = [own["native"][s][-1] - own["native"][s][0] for s in SEEDS]
    trans = [own["produced"][s][-1] - own["produced"][s][0] for s in SEEDS]
    diff = [t_ - c for c, t_ in zip(conv, trans)]
    t, _, _ = paired_t(diff)
    out["H3_per_gain"] = {
        "paired_diff_mean": mean(diff),
        "paired_diff_sd": sd(diff),
        "t": t,
        "resolved": abs(t) > 4.0,
    }
    return out


# ==========================================================================
# Self-test
# ==========================================================================

def self_test() -> int:
    print("compare_padding_fix self-test")
    fails = 0

    def mk(fn):
        return {
            a: {s: [fn(a, s, i, lm) for i, lm in enumerate(LOOKAHEADS)] for s in SEEDS}
            for a in ARMS
        }

    # --- 1. a pure constant offset must be called an offset ---
    off = -0.05
    new = mk(lambda a, s, i, lm: OLD_OWN[a][s][i] + off)
    r = paired_block(OLD_OWN, new, "own")
    ok = (
        abs(r["mean_delta"] - off) < 1e-9
        and r["offset_or_shape"]["verdict"].startswith("constant offset")
        and abs(r["offset_or_shape"]["slope_per_log2L"]) < 1e-9
        and r["n_improved"] == 42
    )
    print(f"  constant offset {off}: mean={r['mean_delta']:.4f} "
          f"slope={r['offset_or_shape']['slope_per_log2L']:.2e} "
          f"-> {r['offset_or_shape']['verdict'][:28]}  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 2. an L-dependent change must be called a shape change ---
    new = mk(lambda a, s, i, lm: OLD_OWN[a][s][i] - 0.01 * math.log2(lm + 1.0))
    r = paired_block(OLD_OWN, new, "own")
    ok = r["offset_or_shape"]["verdict"].startswith("shape changed") and abs(
        r["offset_or_shape"]["slope_per_log2L"] + 0.01
    ) < 1e-9
    print(f"  L-dependent: slope={r['offset_or_shape']['slope_per_log2L']:.4f} "
          f"t={r['offset_or_shape']['slope_t']:.1f} "
          f"-> {r['offset_or_shape']['verdict'][:28]}  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 3. identical input => zero delta, and no false shape claim ---
    r = paired_block(OLD_OWN, OLD_OWN, "own")
    ok = abs(r["mean_delta"]) < 1e-12 and r["offset_or_shape"]["verdict"].startswith(
        "constant offset"
    )
    print(f"  identical: mean={r['mean_delta']:.2e} "
          f"-> {r['offset_or_shape']['verdict'][:28]}  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 4. offset + seed noise: still an offset, CI must cover the truth ---
    rng = random.Random(7)
    new = mk(lambda a, s, i, lm: OLD_OWN[a][s][i] + off + rng.gauss(0, 0.004))
    r = paired_block(OLD_OWN, new, "own")
    lo, hi = r["ci95_mean_delta"]
    ok = lo <= off <= hi and r["offset_or_shape"]["verdict"].startswith("constant")
    print(f"  offset+noise: CI95=[{lo:.4f},{hi:.4f}] covers {off}  "
          f"{'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 5. a constant offset must leave the re-derived claims unchanged ---
    base = rederive_claims(OLD_OWN, OLD_CROSS)
    shifted_own = mk(lambda a, s, i, lm: OLD_OWN[a][s][i] + off)
    shifted_cross = mk(lambda a, s, i, lm: OLD_CROSS[a][s][i] + off)
    shift = rederive_claims(shifted_own, shifted_cross)
    d_end = abs(
        base["per_endpoint_native"]["delta_mean"]
        - shift["per_endpoint_native"]["delta_mean"]
    )
    d_h3 = abs(base["H3_per_gain"]["paired_diff_mean"]
               - shift["H3_per_gain"]["paired_diff_mean"])
    ok = d_end < 1e-9 and d_h3 < 1e-9
    print(f"  offset-invariance of claims: endpoint drift={d_end:.2e} "
          f"H3 drift={d_h3:.2e}  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 6. incomplete input must be refused, not averaged ---
    import tempfile
    partial = {
        "results": [
            {"target": "native", "seed": 1337, "lookahead_ms": 0,
             "test_per": 0.4, "test_per_cross": 0.45}
        ],
        "steps": 1200, "batch_size": 8, "chunk_ms": 40,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(partial, f)
        p = f.name
    try:
        load_new(p)
        print("  partial input: accepted  FAIL")
        fails += 1
    except SystemExit as e:
        ok = "incomplete" in str(e)
        print(f"  partial input: refused ({'correct message' if ok else 'wrong message'})"
              f"  {'ok' if ok else 'FAIL'}")
        fails += 0 if ok else 1
    finally:
        os.unlink(p)

    # --- 7. comparability guard notices a changed hyperparameter ---
    w = check_comparability({"steps": 2000, "batch_size": 8, "chunk_ms": 40})
    ok = len(w) == 1 and "steps" in w[0]
    print(f"  comparability guard: {len(w)} warning(s)  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # ---- noise-floor machinery ----
    import tempfile as _tf

    def write_jsonl(extra_dup=None, bump=0.0):
        """Full 42-cell file, optionally with duplicated cells offset by a
        known amount so the recovered noise floor has a known answer."""
        fh = _tf.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for a in ARMS:
            for s in SEEDS:
                for i, lm in enumerate(LOOKAHEADS):
                    fh.write(json.dumps({
                        "target": a, "seed": s, "lookahead_ms": lm,
                        "test_per": OLD_OWN[a][s][i] + bump,
                        "test_per_cross": OLD_CROSS[a][s][i] + bump}) + "\n")
        for (a, s, i), delta in (extra_dup or {}).items():
            fh.write(json.dumps({
                "target": a, "seed": s, "lookahead_ms": LOOKAHEADS[i],
                "test_per": OLD_OWN[a][s][i] + bump + delta,
                "test_per_cross": OLD_CROSS[a][s][i] + bump + delta}) + "\n")
        fh.close()
        return fh.name

    # --- 8. repeats are averaged, not overwritten ---
    p = write_jsonl({("native", 1337, 0): 0.02})
    o2, _, prov2 = load_new(p)
    expect = OLD_OWN["native"][1337][0] + 0.01  # mean of x and x+0.02
    ok = abs(o2["native"][1337][0] - expect) < 1e-12
    print(f"  duplicate averaged: got {o2['native'][1337][0]:.5f} want {expect:.5f}"
          f"  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1
    os.unlink(p)

    # --- 9. noise floor recovered from planted repeats ---
    planted = {("native", 1337, 0): 0.004, ("native", 7, 1): 0.002,
               ("produced", 99, 2): 0.006}
    p = write_jsonl(planted)
    _, _, prov3 = load_new(p)
    rn = prov3["repeat_noise"]
    ok = (rn["available"] and rn["n_repeated_cells"] == 3
          and abs(rn["max_abs_diff"] - 0.006) < 1e-9)
    print(f"  noise floor: n={rn['n_repeated_cells']} max={rn['max_abs_diff']:.4f} "
          f"sd={rn['sd_per_measurement']:.5f} thr={rn['resolvable_threshold']:.5f}"
          f"  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1
    os.unlink(p)

    # --- 10. no repeats => declares noise UNMEASURED rather than assuming zero ---
    p = write_jsonl()
    _, _, prov4 = load_new(p)
    ok = prov4["repeat_noise"]["available"] is False
    print(f"  no repeats -> unmeasured: {not prov4['repeat_noise']['available']}"
          f"  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1
    os.unlink(p)

    # --- 11. an effect below the floor must be called unresolvable ---
    # bump every cell by 0.001 (a "real" offset) but plant 0.008 repeat noise.
    p = write_jsonl({("native", 1337, 0): 0.008, ("native", 7, 0): 0.008,
                     ("produced", 99, 3): 0.008}, bump=0.001)
    own_n, cross_n, prov5 = load_new(p)
    blk = paired_block(OLD_OWN, own_n, "own")
    thr = prov5["repeat_noise"]["resolvable_threshold"]
    eff = abs(blk["mean_delta"])
    ok = eff < thr  # 0.001-ish effect vs a much larger floor
    print(f"  sub-noise effect: |eff|={eff:.5f} < thr={thr:.5f} -> unresolvable"
          f"  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1
    os.unlink(p)

    # --- 12. the baseline must be the PRE-FIX table, not the corrected sweep ---
    # If analyse_3seed.OWN (now the fixed data) were imported as the baseline,
    # every delta would be 0.0000 and the script would confidently report that
    # the padding fix changed nothing. Assert the imported baseline still
    # matches the known pre-fix value and differs from the current data.
    try:
        import analyse_3seed as _a3
        pre_ok = abs(OLD_OWN["native"][1337][0] - 0.4495) < 1e-9
        differs = OLD_OWN["native"][1337] != _a3.OWN["native"][1337]
        cur_ok = _a3.SWEEP_IS_PADDING_FIXED
        ok = pre_ok and (differs or not cur_ok)
        print(f"  baseline is PRE-FIX: L0/s1337={OLD_OWN['native'][1337][0]:.4f} "
              f"(want 0.4495), differs from current={differs}, "
              f"current_is_fixed={cur_ok}  {'ok' if ok else 'FAIL'}")
        fails += 0 if ok else 1
    except Exception as e:
        print(f"  baseline provenance check errored: {e}  FAIL")
        fails += 1

    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="results/raw/translator_sweep_3seed_summary.json")
    ap.add_argument("--out", default="results/compare_padding_fix.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(self_test())

    if not os.path.exists(a.new):
        raise SystemExit(
            f"{a.new} not found.\nThe re-run writes it as "
            "<out>_summary.json; copy it off /content before the Colab "
            "runtime is reclaimed."
        )

    new_own, new_cross, prov = load_new(a.new)
    warns = check_comparability(prov)

    res: Dict[str, object] = {
        "provenance": {
            "old": "results/analysis_3seed.json (cached path, padding bug present)",
            "new": a.new,
            "new_config": prov,
            "comparability_warnings": warns,
        },
        "own_per": paired_block(OLD_OWN, new_own, "test_per (own target)"),
        "cross_per": paired_block(OLD_CROSS, new_cross, "test_per_cross"),
        "claims_on_new_data": rederive_claims(new_own, new_cross),
    }

    # ---- headline ----
    o = res["own_per"]
    shape = o["offset_or_shape"]
    l0_old = mean([OLD_OWN["native"][s][0] for s in SEEDS])
    l0_new = mean([new_own["native"][s][0] for s in SEEDS])
    noise = prov.get("repeat_noise", {}) or {}
    thr = float(noise.get("resolvable_threshold", 0.0) or 0.0)
    eff = abs(float(o["mean_delta"]))
    per_L = shape["per_L_mean_delta"]
    spread = float(shape["range_of_per_L_means"])

    res["verdict"] = {
        "L0_native_mean_old": round(l0_old, 4),
        "L0_native_mean_new": round(l0_new, 4),
        "L0_native_delta": round(l0_new - l0_old, 4),
        "overall_mean_delta": round(float(o["mean_delta"]), 4),
        "ci95": [round(float(x), 4) for x in o["ci95_mean_delta"]],
        "improved_cells": f"{o['n_improved']}/{o['n_pairs']}",
        "shape_verdict": shape["verdict"],
        "slope_t": round(float(shape["slope_t"]), 2),
        # --- the honesty gate -------------------------------------------------
        # A paired t over 42 cells will look impressive even for an effect that
        # a rerun at the same seed could produce by itself. Gate the claim on
        # the measured floor, not on the p-value.
        "repeat_noise_available": bool(noise.get("available")),
        "repeat_noise_sd": round(float(noise.get("sd_per_measurement", 0.0)), 5)
        if noise.get("available") else None,
        "resolvable_threshold": round(thr, 5) if noise.get("available") else None,
        "effect_over_noise": round(eff / thr, 2) if thr > 0 else None,
        "effect_resolvable": (eff > thr) if thr > 0 else None,
        "shape_spread_over_noise": round(spread / thr, 2) if thr > 0 else None,
        "shape_spread_resolvable": (spread > thr) if thr > 0 else None,
    }

    if not noise.get("available"):
        impl = ("Run-to-run noise UNMEASURED -- no repeated cells in this file. "
                "Treat the significance above as an upper bound on confidence.")
    elif thr > 0 and eff <= thr:
        impl = (f"Mean delta ({eff:.4f}) is WITHIN the fixed-seed noise floor "
                f"({thr:.4f}). The padding fix cannot be shown to have changed "
                "PER at all by this evidence -- do not report it as an effect.")
    elif shape["verdict"].startswith("constant offset") and spread <= thr:
        impl = ("Offset is real and flat, and its variation across lookahead is "
                "within noise. Prior within-run conclusions stand; restate "
                "absolute PERs only.")
    elif shape["verdict"].startswith("constant offset"):
        impl = (f"Offset is real, and the regression calls it flat, but per-L "
                f"means span {spread:.4f} vs a {thr:.4f} noise floor -- the "
                "flatness is weaker than the slope test alone implies. Re-check "
                "shape-dependent claims rather than assuming they carry over.")
    else:
        impl = ("Curve shape moved by more than the noise floor -- re-derive "
                "every shape-dependent claim before citing the old ones.")
    res["verdict"]["implication"] = impl

    if warns:
        print("!! COMPARABILITY WARNINGS")
        for w in warns:
            print("   -", w)
        print()

    print(json.dumps(res["verdict"], indent=2))
    print("\n-- shape detail --")
    print(json.dumps(shape["per_L_mean_delta"], indent=2))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nwrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
