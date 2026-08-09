"""Does the dense lookahead curve have a locatable knee, or only curvature?

Why this file exists
--------------------
The 7-point geometric grid said "knee" but could not say *where*: the estimate
moved 40 ms -> 160 ms depending only on where L=0 was placed on a log axis, and
a planted cliff at 81 ms was reported at 40 ms
(`docs/PADDING_FIX_RESOLVED.md` §3). That is an identifiability failure, not a
finding, and no amount of re-fitting the same 7 points fixes it.

This script analyses the dense sweep -- 16 lookaheads, 0-200 ms in 20 ms steps
plus 240/280/320/480/640 -- and is built to *resist* the specific ways the
sparse analysis fooled itself:

1. **The x-axis is not a free parameter.** Every fit is repeated under three
   defensible treatments of L=0 (drop it; place it at half a frame; keep
   log2(L+1)). A knee that moves between them is not a knee, it is an artefact
   of the axis. Reported explicitly rather than choosing the flattering one.

2. **A breakpoint is not reported unless it is stable under resampling.** The
   breakpoint is bootstrapped; if its 90% interval spans more than an octave the
   location is declared unidentifiable, whatever BIC prefers.

3. **Effects are gated on the measured noise floor** (sigma = 0.0022,
   2-sigma = 0.0044 from fixed-seed repeats). A per-step gain smaller than that
   is not evidence of flattening -- it is unmeasurable.

4. **"Diminishing returns" is quantified without a breakpoint at all**, via the
   lookahead at which marginal PER gain per added 20 ms frame drops below the
   noise floor. That is an operationally meaningful number even when no knee
   exists, and it is what a deployment engineer actually needs.

Usage
-----
    python eval/analyse_dense_knee.py --self-test
    python eval/analyse_dense_knee.py --jsonl results/raw/translator_dense.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

NOISE_SIGMA = 0.00218          # measured, fixed-seed repeats (PADDING_FIX_RESOLVED §5)
NOISE_2SIGMA = 2 * NOISE_SIGMA
FRAME_MS = 20.0                # lookahead quantises to ceil(L / FRAME_MS) frames


# ==========================================================================
# Ingest
# ==========================================================================

def load_dense(path: str, target: Optional[str] = None
               ) -> Tuple[List[float], Dict[float, List[float]], Dict]:
    """Return (sorted lookaheads, {L: [per per seed]}, provenance).

    Repeats of the same (target, L, seed) are averaged, and their spread is
    reported -- the same treatment used in compare_padding_fix.py, for the same
    reason: a repeat is a measurement of noise, not a nuisance to discard.

    A file containing more than one target arm is REFUSED unless `target` is
    given. An earlier version keyed cells on (L, seed) only, which silently
    pooled the native and produced arms and averaged them as if they were
    repeats -- turning the entire conversion-vs-transcription difference into
    apparent measurement noise. Caught by self-test 7 on real data.
    """
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  note: {path} line {ln} truncated; ignoring", file=sys.stderr)
    if not rows:
        raise SystemExit(f"{path}: no rows")

    present = sorted({r.get("target") for r in rows})
    if target is not None:
        rows = [r for r in rows if r.get("target") == target]
        if not rows:
            raise SystemExit(f"{path}: no rows with target={target!r} "
                             f"(present: {present})")
    elif len(present) > 1:
        raise SystemExit(
            f"{path} contains {len(present)} target arms {present}. Pass "
            "--target to pick one. Refusing to pool them: averaging arms as if "
            "they were repeats would hide the arm difference inside the noise "
            "estimate and flatten the curve toward their mean."
        )

    cells: Dict[Tuple[object, float, object], List[float]] = collections.defaultdict(list)
    algo: Dict[float, float] = {}
    for r in rows:
        L = float(r["lookahead_ms"])
        cells[(r.get("target"), L, r["seed"])].append(float(r["test_per"]))
        if r.get("t_algorithmic_ms") is not None:
            algo[L] = float(r["t_algorithmic_ms"])

    repeat_diffs = [max(v) - min(v) for v in cells.values() if len(v) > 1]
    per_L: Dict[float, List[float]] = collections.defaultdict(list)
    for (_t, L, _seed), vals in cells.items():
        per_L[L].append(sum(vals) / len(vals))

    Ls = sorted(per_L)

    # Frame-quantisation check. Distinct L that map to the same algorithmic
    # budget are duplicate conditions; averaging them as if independent
    # reweights the fit. This is the trap documented in
    # bench_content_degradation.py --dense.
    collisions: Dict[float, List[float]] = collections.defaultdict(list)
    for L in Ls:
        if L in algo:
            collisions[algo[L]].append(L)
    dupes = {a: v for a, v in collisions.items() if len(v) > 1}

    prov = {
        "path": path,
        "target": target or (present[0] if present else None),
        "targets_in_file": present,
        "n_rows": len(rows),
        "n_lookaheads": len(Ls),
        "n_cells": len(cells),
        "seeds": sorted({r["seed"] for r in rows}, key=str),
        "repeated_cells": len(repeat_diffs),
        "repeat_max_diff": max(repeat_diffs) if repeat_diffs else None,
        "t_algorithmic_collisions": {str(k): v for k, v in dupes.items()} or "none",
        "steps": sorted({r.get("steps") for r in rows}),
        "targets": sorted({r.get("target") for r in rows}),
    }
    return Ls, dict(per_L), prov


# ==========================================================================
# Fitting
# ==========================================================================

def _fit_line(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my, sum((y - my) ** 2 for y in ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return b, a, ss


def _bic(ss: float, n: int, k: int) -> float:
    return n * math.log(max(ss, 1e-12) / n) + k * math.log(n)


def knee_bic(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, object]:
    """1-segment vs 2-segment fit. Returns best breakpoint and delta-BIC.

    Negative delta_bic favours the piecewise model.
    """
    n = len(xs)
    _, _, ss1 = _fit_line(xs, ys)
    bic1 = _bic(ss1, n, 2)
    best = None
    for i in range(2, n - 2):           # need >=2 points each side
        _, _, ssa = _fit_line(xs[: i + 1], ys[: i + 1])
        _, _, ssb = _fit_line(xs[i:], ys[i:])
        bic2 = _bic(ssa + ssb, n, 4)
        if best is None or bic2 < best[1]:
            best = (xs[i], bic2, i)
    if best is None:
        return {"knee_detected": False, "delta_bic": 0.0, "knee_x": None,
                "reason": "too few points"}
    return {
        "knee_x": best[0],
        "knee_index": best[2],
        "delta_bic": best[1] - bic1,
        "bic_loglinear": bic1,
        "bic_piecewise": best[1],
        "knee_detected": (best[1] - bic1) < -6.0,   # strong evidence, Kass & Raftery
    }


def bootstrap_knee(
    xs: Sequence[float], ys: Sequence[float], reps: int = 2000, seed: int = 0
) -> Dict[str, object]:
    """Residual bootstrap on the breakpoint location.

    A breakpoint whose interval spans more than an octave is not a location.
    The sparse-grid failure was precisely a *tight-looking* estimate that moved
    wholesale when the axis changed, so stability under resampling is necessary
    but not sufficient -- the axis sweep below is the other half.
    """
    n = len(xs)
    base = knee_bic(xs, ys)
    if base.get("knee_x") is None:
        return {"available": False}
    b, a, _ = _fit_line(xs, ys)
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    rng = random.Random(seed)
    locs = []
    for _ in range(reps):
        ry = [(a + b * x) + resid[rng.randrange(n)] for x in xs]
        r = knee_bic(xs, ry)
        if r.get("knee_x") is not None:
            locs.append(r["knee_x"])
    if not locs:
        return {"available": False}
    locs.sort()
    lo = locs[int(0.05 * len(locs))]
    hi = locs[min(len(locs) - 1, int(0.95 * len(locs)))]
    return {
        "available": True,
        "point": base["knee_x"],
        "ci90": [lo, hi],
        "width_octaves": hi - lo,          # xs are log2, so difference is octaves
        "identifiable": (hi - lo) <= 1.0,
    }


def axis_sweep(Ls: Sequence[float], means: Sequence[float]) -> List[Dict[str, object]]:
    """Refit under three defensible treatments of L=0.

    log2(L+1) puts L=0 at 0.0 and L=20 at 4.39 -- a 4.39-unit gap where every
    other gap is ~1.0. Any piecewise fit will want a breakpoint just past that
    gap regardless of the data, which is exactly what happened at n=7.
    """
    out = []
    variants = [
        ("log2(L+1), all points", [math.log2(l + 1) for l in Ls], list(means)),
        ("L=0 dropped", [math.log2(l + 1) for l in Ls[1:]], list(means[1:])),
        ("L=0 at half a frame",
         [math.log2((FRAME_MS / 2) + 1)] + [math.log2(l + 1) for l in Ls[1:]],
         list(means)),
    ]
    for name, xs, ys in variants:
        if Ls and Ls[0] != 0 and name != "log2(L+1), all points":
            continue
        k = knee_bic(xs, ys)
        bs = bootstrap_knee(xs, ys)
        out.append({
            "axis": name,
            "n": len(xs),
            "delta_bic": round(k["delta_bic"], 2),
            "knee_detected": k["knee_detected"],
            "knee_ms": round(2 ** k["knee_x"] - 1, 1) if k.get("knee_x") else None,
            "bootstrap_ci90_ms": [round(2 ** v - 1, 1) for v in bs["ci90"]]
            if bs.get("available") else None,
            "identifiable": bs.get("identifiable") if bs.get("available") else None,
        })
    return out


def marginal_returns(Ls: Sequence[float], means: Sequence[float]) -> Dict[str, object]:
    """Where does an extra frame of lookahead stop paying, in noise units?

    This is the deployment-relevant quantity and it needs no breakpoint. For
    each adjacent pair, the PER gain per added 20 ms frame is compared with the
    measured 2-sigma floor. The first lookahead beyond which every subsequent
    step is below the floor is the practical saturation point.
    """
    steps = []
    for (l0, y0), (l1, y1) in zip(zip(Ls, means), list(zip(Ls, means))[1:]):
        frames = max(1.0, (l1 - l0) / FRAME_MS)
        gain = y0 - y1                       # positive = PER improved
        steps.append({
            "from_ms": l0, "to_ms": l1,
            "gain": gain,
            "gain_per_frame": gain / frames,
            "over_noise": (gain / frames) / NOISE_2SIGMA,
            "resolvable": (gain / frames) > NOISE_2SIGMA,
        })
    sat = None
    for i in range(len(steps)):
        if all(not s["resolvable"] for s in steps[i:]):
            sat = steps[i]["from_ms"]
            break
    return {
        "noise_2sigma": NOISE_2SIGMA,
        "steps": steps,
        "saturation_ms": sat,
        "reading": (
            f"marginal gain per 20 ms frame stays above the {NOISE_2SIGMA:.4f} "
            f"noise floor until L={sat} ms, beyond which further lookahead is "
            "not measurably useful at this sample size"
            if sat is not None else
            "every step remains above the noise floor -- no saturation within "
            "the measured range"
        ),
    }


# ==========================================================================
# Self-test
# ==========================================================================

def self_test() -> int:
    print("analyse_dense_knee self-test")
    fails = 0
    Ls = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 480, 640]
    xs = [math.log2(l + 1) for l in Ls]

    # --- 1. a genuinely log-linear curve must NOT yield a knee ---
    ys = [0.45 - 0.03 * x for x in xs]
    k = knee_bic(xs, ys)
    ok = not k["knee_detected"]
    print(f"  log-linear -> knee={k['knee_detected']} dBIC={k['delta_bic']:+.1f}"
          f"  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 2. a planted sharp breakpoint must be found AT the planted place ---
    bp = math.log2(120 + 1)
    ys = [0.45 - 0.06 * x if x <= bp else 0.45 - 0.06 * bp - 0.005 * (x - bp)
          for x in xs]
    k = knee_bic(xs, ys)
    got = 2 ** k["knee_x"] - 1
    ok = k["knee_detected"] and abs(got - 120) < 45
    print(f"  planted knee at 120 ms -> found {got:.0f} ms, dBIC={k['delta_bic']:+.1f}"
          f"  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 3. bootstrap must call a noisy log-linear curve unidentifiable ---
    rng = random.Random(1)
    ys = [0.45 - 0.03 * x + rng.gauss(0, 0.004) for x in xs]
    bs = bootstrap_knee(xs, ys)
    ok = (not bs.get("available")) or (not bs["identifiable"])
    print(f"  noisy log-linear -> identifiable={bs.get('identifiable')} "
          f"width={bs.get('width_octaves', float('nan')):.2f} octaves"
          f"  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 4. saturation detector finds a planted plateau ---
    ys = [max(0.20, 0.43 - 0.0012 * l) for l in Ls]   # flat beyond ~190 ms
    m = marginal_returns(Ls, ys)
    ok = m["saturation_ms"] is not None and 120 <= m["saturation_ms"] <= 240
    print(f"  planted plateau -> saturation at {m['saturation_ms']} ms"
          f"  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 5. a curve that never saturates must report None, not a number ---
    ys = [0.45 - 0.0004 * l for l in Ls]
    m = marginal_returns(Ls, ys)
    ok = m["saturation_ms"] is None
    print(f"  never saturates -> {m['saturation_ms']}  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 6. axis sweep must expose an L=0-driven knee as axis-dependent ---
    # The real sparse-grid failure mode: the curve is exactly log-linear for
    # L >= 20, but L=0 does not lie on that extrapolation (in the actual data
    # L=0 is only ~0.055 worse than L=20, whereas the log-linear extension to
    # x=0 would predict far worse). Including L=0 therefore forces a spurious
    # breakpoint just past the 4.39-unit axis gap; dropping it removes the knee
    # entirely. If the sweep reports the same knee under both, it is blind to
    # the exact artefact it exists to catch.
    #
    # NB an earlier version of this test used a curve linear in L and expected
    # instability. That was wrong: such a curve is genuinely convex in
    # log2(L+1), so a stable breakpoint there is correct behaviour, not an
    # artefact. Testing the wrong premise would have hidden a real blind spot.
    slope = 0.031
    ys = [0.0] * len(Ls)
    for i, l in enumerate(Ls):
        if l == 0:
            continue
        ys[i] = 0.55 - slope * math.log2(l + 1)
    ys[0] = ys[1] + 0.055                     # L=0 deliberately off the line

    sweep = axis_sweep(Ls, ys)
    with_zero = next(s for s in sweep if s["axis"] == "log2(L+1), all points")
    without_zero = next(s for s in sweep if s["axis"] == "L=0 dropped")
    ok = without_zero["delta_bic"] > with_zero["delta_bic"] and \
        not without_zero["knee_detected"]
    print(f"  L=0-driven artefact: with L=0 dBIC={with_zero['delta_bic']:+.1f} "
          f"knee={with_zero['knee_ms']} ms | without L=0 "
          f"dBIC={without_zero['delta_bic']:+.1f} knee_detected="
          f"{without_zero['knee_detected']}  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 7. a multi-arm file must be REFUSED, not silently pooled ---
    real = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "results", "raw", "translator_sweep_3seed.jsonl")
    if os.path.exists(real):
        refused = False
        try:
            load_dense(real)
        except SystemExit as e:
            refused = "target arms" in str(e)
        Ls2, per2, prov2 = load_dense(real, target="native")
        cells_ok = prov2["n_cells"] == 21 and len(per2[0.0]) == 3
        ok = refused and cells_ok
        print(f"  multi-arm file: refused={refused}, with --target native "
              f"n_cells={prov2['n_cells']} seeds@L0={len(per2[0.0])}"
              f"  {'ok' if ok else 'FAIL'}")
        fails += 0 if ok else 1
    else:
        print("  multi-arm file: skipped (3seed jsonl not present)")

    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="results/raw/translator_dense.jsonl")
    ap.add_argument("--target", default=None,
                    help="which arm to analyse; required if the file has more "
                         "than one (pooling arms is refused)")
    ap.add_argument("--out", default="results/analysis_dense_knee.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(self_test())

    if not os.path.exists(a.jsonl):
        raise SystemExit(f"{a.jsonl} not found -- copy it off Drive first")

    Ls, per_L, prov = load_dense(a.jsonl, target=a.target)
    means = [sum(per_L[L]) / len(per_L[L]) for L in Ls]
    spreads = [(max(per_L[L]) - min(per_L[L])) if len(per_L[L]) > 1 else 0.0
               for L in Ls]

    res = {
        "provenance": prov,
        "curve": [{"L_ms": L, "mean_per": round(m, 5),
                   "seed_spread": round(s, 5), "n_seeds": len(per_L[L])}
                  for L, m, s in zip(Ls, means, spreads)],
        "axis_sweep": axis_sweep(Ls, means),
        "marginal_returns": marginal_returns(Ls, means),
    }

    # ---- headline ----
    sweep = res["axis_sweep"]
    detected = [s for s in sweep if s["knee_detected"]]
    locs = [s["knee_ms"] for s in sweep if s["knee_ms"] is not None]
    ident = [s for s in sweep if s.get("identifiable")]
    stable = len(locs) >= 2 and (max(locs) / max(1e-9, min(locs))) <= 1.5

    if len(detected) == len(sweep) and stable and ident:
        verdict = (f"KNEE at ~{statistics.median(locs):.0f} ms -- detected under "
                   "every axis treatment, bootstrap-identifiable, and stable "
                   "across treatments.")
    elif detected and not stable:
        verdict = (f"CURVATURE but NO LOCATABLE KNEE -- BIC prefers piecewise, but "
                   f"the breakpoint moves across axis treatments ({locs} ms). "
                   "Report the exchange rate and the saturation point, not a knee.")
    elif not detected:
        verdict = ("NO KNEE -- log-linear is preferred, or the piecewise gain is "
                   "below the delta-BIC threshold, under every axis treatment. "
                   "Report the exchange rate.")
    else:
        verdict = ("AMBIGUOUS -- detection depends on axis treatment. Not "
                   "reportable as a knee.")

    res["verdict"] = {
        "statement": verdict,
        "knee_locations_by_axis_ms": locs,
        "detected_under_n_axes": f"{len(detected)}/{len(sweep)}",
        "saturation_ms": res["marginal_returns"]["saturation_ms"],
        "practical_reading": res["marginal_returns"]["reading"],
    }

    print(json.dumps(res["provenance"], indent=2))
    print("\n-- curve --")
    for c in res["curve"]:
        print(f"  L={c['L_ms']:>5.0f}  PER={c['mean_per']:.4f}  "
              f"spread={c['seed_spread']:.4f}")
    print("\n-- marginal return per 20 ms frame, in units of the 2-sigma floor --")
    for s in res["marginal_returns"]["steps"]:
        flag = "" if s["resolvable"] else "   <- below floor"
        print(f"  {s['from_ms']:>5.0f} -> {s['to_ms']:>5.0f}  "
              f"{s['gain_per_frame']:+.4f}  ({s['over_noise']:.1f}x){flag}")
    print("\n-- axis sweep --")
    for s in res["axis_sweep"]:
        print(f"  {s['axis']:<24} n={s['n']:<3} dBIC={s['delta_bic']:+7.2f}  "
              f"knee={s['knee_ms']} ms  CI90={s['bootstrap_ci90_ms']}  "
              f"identifiable={s['identifiable']}")
    print("\n== VERDICT ==")
    print(json.dumps(res["verdict"], indent=2))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nwrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
