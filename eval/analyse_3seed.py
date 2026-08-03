"""Paired-by-seed analysis of the 3-seed lookahead sweep.

Why this file exists
--------------------
The sweep summary reports, per condition, a mean and an SD over three seeds and
then a `pooled_sd`-derived "resolution floor". That framing is the wrong test.
Seed is a *shared nuisance factor*: an unlucky seed raises PER at every
lookahead, so seed-to-seed spread inside one condition contains variance that
cancels when two conditions are compared within the same seed. The right
analysis blocks on seed.

It is also silent about *which* quantity to trust. Raw PER and the own-target
preference margin have very different seed stability, and the paper's H3 claim
lives on the margin, not on PER. One pooled SD for "the sweep" hides that.

So this script does three things the summary does not:

  1. Adjacent-step comparisons *within* seed (paired), with an exact sign test
     and a paired t, instead of thresholding on a pooled SD.
  2. The same for the preference margin, which is the H3 quantity.
  3. A power calibration for the knee test at n=21: plant a cliff of known size
     into the observed curve and ask whether BIC finds it. Without this, "no
     knee" is again absence of evidence rather than evidence of absence.

Numbers transcribed from {RES}/translator_sweep_3seed_summary.json
(42 runs = 7 lookaheads x 2 targets x 3 seeds; 1200 steps, batch 8, T4,
chunk 40 ms, frozen WavLM-base with both causality patches).

PROVENANCE WARNING: this sweep ran on the *cached* feature path with the
zero-pad contamination bug in MaskedBlock still present (see
docs/CACHING_CHANGED_THE_NUMBERS.md). Absolute PERs are therefore not
comparable with the earlier uncached single-seed run. Everything below is a
within-run comparison, which a constant offset does not touch.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Sequence, Tuple

L = [0, 20, 40, 80, 160, 320, 640]
SEEDS = [1337, 7, 99]

# test_per: PER against the arm's own target labels
OWN: Dict[str, Dict[int, List[float]]] = {
    "native": {
        1337: [.4495, .3961, .3928, .3613, .3148, .3514, .2245],
        7:    [.4490, .4079, .3542, .3411, .3013, .3068, .2576],
        99:   [.4410, .3854, .3406, .3029, .2930, .2482, .2201],
    },
    "produced": {
        1337: [.4536, .3977, .3892, .3959, .3691, .3437, .2721],
        7:    [.4438, .3967, .3760, .3993, .3528, .3061, .2989],
        99:   [.4549, .3980, .3744, .3416, .3135, .3132, .2675],
    },
}
# test_per_cross: same model, scored against the *other* label set
CROSS: Dict[str, Dict[int, List[float]]] = {
    "native": {
        1337: [.4838, .4409, .4433, .4311, .4018, .4488, .3261],
        7:    [.4795, .4481, .4089, .4125, .3839, .4001, .3590],
        99:   [.4734, .4284, .3927, .3669, .3741, .3403, .3213],
    },
    "produced": {
        1337: [.4901, .4359, .4252, .4265, .3947, .3719, .2997],
        7:    [.4807, .4353, .4168, .4317, .3785, .3357, .3239],
        99:   [.4917, .4375, .4101, .3757, .3443, .3368, .2942],
    },
}


def margin(arm: str, seed: int) -> List[float]:
    """cross - own. Positive => model matches its own target better.

    For the native arm this is *canonical preference*: how much more the output
    looks like the L1-native pronunciation than like what the speaker actually
    said. That is the operational definition of accent conversion available
    without a listener.
    """
    return [c - o for c, o in zip(CROSS[arm][seed], OWN[arm][seed])]


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_t(d: Sequence[float]) -> Tuple[float, int, int]:
    n = len(d)
    s = sd(d)
    if s == 0 or n < 2:
        return (float("inf") if mean(d) else 0.0), n - 1, n
    return mean(d) / (s / math.sqrt(n)), n - 1, n


def exact_sign_p(k: int, n: int) -> float:
    """Two-sided exact binomial p for k successes of n at p=0.5."""
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


T_CRIT_2DOF = {0.05: 4.303, 0.10: 2.920}


def adjacent_report(series: Dict[int, List[float]],
                    lower_is_better: bool) -> List[dict]:
    rows = []
    for i in range(len(L) - 1):
        d = [series[s][i + 1] - series[s][i] for s in SEEDS]
        imp = [(-x if lower_is_better else x) for x in d]
        k = sum(1 for x in imp if x > 0)
        t, _, n = paired_t(imp)
        rows.append({
            "from_ms": L[i], "to_ms": L[i + 1],
            "delta_mean": mean(d), "delta_sd": sd(d),
            "n_seeds_improving": k,
            "sign_p": exact_sign_p(max(k, n - k), n),
            "t": t,
            "sig_05": abs(t) > T_CRIT_2DOF[0.05],
            "unanimous": k == n or k == 0,
        })
    return rows


def _fit_seg(xs, ys):
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0, sum((y - my) ** 2 for y in ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, ss


def _bic(ss: float, n: int, k: int) -> float:
    return n * math.log(max(ss, 1e-12) / n) + k * math.log(n)


def knee_bic(xs, ys) -> dict:
    """1-segment vs 2-segment linear fit on a log2 x-axis.

    Model selection, not an R^2 threshold: an R^2 cutoff is a magic constant
    that silently decides a scientific conclusion. BIC at least states its
    penalty. Delta = BIC(piecewise) - BIC(loglinear); negative favours a knee,
    and Kass-Raftery call < -6 strong.
    """
    n = len(xs)
    ss1 = _fit_seg(xs, ys)[2]
    bic1 = _bic(ss1, n, 2)
    best = None
    for bp in sorted(set(xs))[1:-1]:
        lo = [(x, y) for x, y in zip(xs, ys) if x <= bp]
        hi = [(x, y) for x, y in zip(xs, ys) if x >= bp]
        if len(lo) < 3 or len(hi) < 3:
            continue
        ss = (_fit_seg([p[0] for p in lo], [p[1] for p in lo])[2]
              + _fit_seg([p[0] for p in hi], [p[1] for p in hi])[2])
        b = _bic(ss, n, 4)
        if best is None or b < best[1]:
            best = (bp, b)
    if best is None:
        return {"knee_x": None, "delta_bic": float("nan"), "bic_loglinear": bic1}
    return {"knee_x": best[0], "delta_bic": best[1] - bic1,
            "bic_loglinear": bic1, "bic_piecewise": best[1],
            "knee_detected": (best[1] - bic1) < -6.0}


def knee_power(xs, ys, cliff: float, at_x: float) -> dict:
    """Plant a cliff of size `cliff` at `at_x` and see whether BIC finds it.

    This is the check that was missing at n=7: a test that cannot detect a
    cliff you put there by hand cannot be used to claim there isn't one.
    """
    yy = [y - (cliff if x > at_x else 0.0) for x, y in zip(xs, ys)]
    r = knee_bic(xs, yy)
    r["planted_cliff"] = cliff
    r["planted_at_ms"] = 2 ** at_x
    return r


def main() -> None:
    out: dict = {"provenance": {
        "runs": 42, "lookaheads_ms": L, "seeds": SEEDS, "steps": 1200,
        "batch_size": 8, "chunk_ms": 40, "device": "cuda (T4)",
        "feature_cache": True,
        "known_bug": "MaskedBlock zero-pad contamination not yet fixed; "
                     "absolute PER not comparable to the uncached run",
    }}

    print("=" * 78)
    print("1. PER: is each doubling of lookahead actually resolved?")
    print("=" * 78)
    print("Blocking on seed. `seeds` = how many of 3 improved.")
    for arm in ("native", "produced"):
        print(f"\n  {arm} (own-target PER, lower better)")
        print(f"  {'step':>14} {'dPER':>8} {'sd':>7} {'seeds':>6} {'signp':>7} "
              f"{'t(2)':>7} {'sig.05':>7}")
        rows = adjacent_report(OWN[arm], lower_is_better=True)
        out[f"per_adjacent_{arm}"] = rows
        for r in rows:
            print(f"  {r['from_ms']:5d}->{r['to_ms']:<8d} {r['delta_mean']:+8.4f} "
                  f"{r['delta_sd']:7.4f} {r['n_seeds_improving']}/3    "
                  f"{r['sign_p']:7.3f} {r['t']:+7.2f} "
                  f"{'YES' if r['sig_05'] else '.':>7}")
        d = [OWN[arm][s][-1] - OWN[arm][s][0] for s in SEEDS]
        t, _, _ = paired_t([-x for x in d])
        print(f"  {'0->640 total':>14} {mean(d):+8.4f} {sd(d):7.4f} 3/3    "
              f"{exact_sign_p(3,3):7.3f} {t:+7.2f} "
              f"{'YES' if abs(t) > T_CRIT_2DOF[0.05] else '.':>7}")
        out[f"per_endpoint_{arm}"] = {
            "delta_mean": mean(d), "delta_sd": sd(d), "t": t,
            "relative_gain": -mean(d) / mean([OWN[arm][s][0] for s in SEEDS])}

    print()
    print("=" * 78)
    print("2. Own-target preference margin (cross - own): the H3 quantity")
    print("=" * 78)
    for arm in ("native", "produced"):
        series = {s: margin(arm, s) for s in SEEDS}
        print(f"\n  {arm}")
        print(f"  {'L(ms)':>6} " + " ".join(f"{'s'+str(s):>9}" for s in SEEDS)
              + f" {'mean':>9} {'sd':>7}")
        cond = []
        for i, x in enumerate(L):
            vals = [series[s][i] for s in SEEDS]
            cond.append({"lookahead_ms": x, "mean": mean(vals), "sd": sd(vals),
                         "per_seed": vals})
            print(f"  {x:6d} " + " ".join(f"{v:+9.4f}" for v in vals)
                  + f" {mean(vals):+9.4f} {sd(vals):7.4f}")
        out[f"margin_{arm}"] = cond
        rows = adjacent_report(series, lower_is_better=False)
        out[f"margin_adjacent_{arm}"] = rows
        nmono = sum(1 for s in SEEDS
                    if all(series[s][i + 1] > series[s][i]
                           for i in range(len(L) - 1)))
        unan = sum(1 for r in rows if r["n_seeds_improving"] == 3)
        k = sum(r["n_seeds_improving"] for r in rows)
        ntot = 3 * len(rows)
        print(f"    strictly monotone increasing in {nmono}/3 seeds; "
              f"{unan}/{len(rows)} steps unanimous")
        print(f"    sign test over all steps x seeds: {k}/{ntot} positive, "
              f"p = {exact_sign_p(max(k, ntot-k), ntot):.2e}")
        print(f"    mean seed sd across conditions: "
              f"{mean([c['sd'] for c in cond]):.4f}  "
              f"(cf. PER seed sd "
              f"{mean([sd([OWN[arm][s][i] for s in SEEDS]) for i in range(len(L))]):.4f})")
        out[f"margin_growth_{arm}"] = {
            "monotone_seeds": nmono, "positive": k, "total": ntot,
            "sign_p": exact_sign_p(max(k, ntot - k), ntot),
            "delta_0_to_640_mean": cond[-1]["mean"] - cond[0]["mean"],
            "delta_0_to_640_sd": sd([series[s][-1] - series[s][0] for s in SEEDS]),
            "ratio_640_over_0": cond[-1]["mean"] / cond[0]["mean"],
            "mean_seed_sd": mean([c["sd"] for c in cond]),
            "mean_per_seed_sd": mean([sd([OWN[arm][s][i] for s in SEEDS])
                                      for i in range(len(L))]),
        }

    print()
    print("=" * 78)
    print("3. H3, paired by seed")
    print("=" * 78)
    print("H3: lookahead buys more for conversion than for transcription.")
    print("Two readings of 'buys more'; they do not agree, and that matters.\n")
    gc = [(OWN['native'][s][0] - OWN['native'][s][-1]) / OWN['native'][s][0]
          for s in SEEDS]
    gt = [(OWN['produced'][s][0] - OWN['produced'][s][-1]) / OWN['produced'][s][0]
          for s in SEEDS]
    d = [a - b for a, b in zip(gc, gt)]
    t, _, _ = paired_t(d)
    print("  (a) relative PER gain 0->640 ms")
    print(f"      conversion    {mean(gc):.4f} +- {sd(gc):.4f}  "
          f"{['%.4f' % x for x in gc]}")
    print(f"      transcription {mean(gt):.4f} +- {sd(gt):.4f}  "
          f"{['%.4f' % x for x in gt]}")
    print(f"      paired diff   {mean(d):+.4f} +- {sd(d):.4f}, t(2)={t:+.2f}, "
          f"{sum(1 for x in d if x > 0)}/3 positive -> "
          f"{'SUPPORTED' if abs(t) > T_CRIT_2DOF[0.05] else 'NOT RESOLVED at n=3'}")
    out["H3_per_gain"] = {"conversion": gc, "transcription": gt,
                          "paired_diff_mean": mean(d), "paired_diff_sd": sd(d),
                          "t": t, "resolved": abs(t) > T_CRIT_2DOF[0.05]}

    mc = [margin("native", s) for s in SEEDS]
    mp = [margin("produced", s) for s in SEEDS]
    dg = [(mc[i][-1] - mc[i][0]) - (mp[i][-1] - mp[i][0]) for i in range(3)]
    t2, _, _ = paired_t(dg)
    print("\n  (b) growth of own-target preference 0->640 ms")
    print(f"      conversion    {mean([m[-1]-m[0] for m in mc]):+.4f} "
          f"+- {sd([m[-1]-m[0] for m in mc]):.4f}")
    print(f"      transcription {mean([m[-1]-m[0] for m in mp]):+.4f} "
          f"+- {sd([m[-1]-m[0] for m in mp]):.4f}")
    print(f"      paired diff   {mean(dg):+.4f} +- {sd(dg):.4f}, t(2)={t2:+.2f}, "
          f"{sum(1 for x in dg if x > 0)}/3 positive -> "
          f"{'SUPPORTED' if abs(t2) > T_CRIT_2DOF[0.05] else 'NOT RESOLVED'}")
    out["H3_margin_growth"] = {
        "conversion": [m[-1] - m[0] for m in mc],
        "transcription": [m[-1] - m[0] for m in mp],
        "paired_diff_mean": mean(dg), "paired_diff_sd": sd(dg), "t": t2,
        "resolved": abs(t2) > T_CRIT_2DOF[0.05]}

    # ---- (c) the control a reviewer will demand -----------------------
    # Objection: the margin has to grow as a model gets better at its own
    # target, so (b) might be an artefact of the native arm simply improving
    # more (0.212 PER vs 0.171). The transcription arm is the control, and it
    # improves substantially while its margin *falls* -- but the improvements
    # are not matched in size, so compare at matched PER instead.
    print("\n  (c) PER-matched control: margin at comparable accuracy")
    print("      If the margin merely tracked 'better at your own target',")
    print("      two models with the same own-PER would have the same margin.")
    matched = []
    for i, x in enumerate(L):
        pn = mean([OWN["native"][s][i] for s in SEEDS])
        mn = mean([margin("native", s)[i] for s in SEEDS])
        # nearest produced condition by own-PER
        j = min(range(len(L)),
                key=lambda q: abs(mean([OWN["produced"][s][q] for s in SEEDS]) - pn))
        pp = mean([OWN["produced"][s][j] for s in SEEDS])
        mp_ = mean([margin("produced", s)[j] for s in SEEDS])
        matched.append({"native_ms": x, "native_per": pn, "native_margin": mn,
                        "produced_ms": L[j], "produced_per": pp,
                        "produced_margin": mp_,
                        "per_gap": pn - pp, "margin_ratio": mn / mp_})
        print(f"      native L={x:>3} ms PER {pn:.3f} margin {mn:+.4f}  vs  "
              f"produced L={L[j]:>3} ms PER {pp:.3f} margin {mp_:+.4f}  "
              f"(PER gap {pn-pp:+.3f}, margin x{mn/mp_:.2f})")
    out["H3_per_matched"] = matched
    hi = [m for m in matched if m["native_ms"] >= 160]
    print(f"      at L>=160 ms the native margin is "
          f"{mean([m['margin_ratio'] for m in hi]):.2f}x the PER-matched "
          f"transcription margin -> not an accuracy artefact")

    print()
    print("=" * 78)
    print("4. Knee, and whether the test could have found one")
    print("=" * 78)
    for arm in ("native", "produced"):
        xs, ys = [], []
        for s in SEEDS:
            for i, x in enumerate(L):
                xs.append(math.log2(max(x, 10)))   # 0 ms placed at 10 ms
                ys.append(OWN[arm][s][i])
        r = knee_bic(xs, ys)
        print(f"\n  {arm}: n={len(xs)}  dBIC = {r['delta_bic']:+.2f} at "
              f"{2 ** r['knee_x']:.0f} ms -> "
              f"{'KNEE' if r.get('knee_detected') else 'no knee; single log-linear regime'}")
        out[f"knee_{arm}"] = r
        print("      power check -- plant a cliff at 160 ms and re-test:")
        for cliff in (0.02, 0.04, 0.08):
            p = knee_power(xs, ys, cliff, math.log2(160))
            print(f"        cliff {cliff:.2f} PER -> dBIC {p['delta_bic']:+7.2f} "
                  f"{'DETECTED' if p.get('knee_detected') else 'missed'}")
            out.setdefault(f"knee_power_{arm}", []).append(p)

    print()
    print("=" * 78)
    print("5. Takeaway")
    print("=" * 78)
    print("""  - PER falls monotonically over 0-640 ms and the endpoint effect is
    large and unanimous across seeds.
  - Individual doublings are mostly NOT resolved at 3 seeds. Any sentence of
    the form "the knee is at X ms" is unsupported by this experiment; the BIC
    power check states how large a cliff would have had to be to show up.
  - The own-target preference margin is an order of magnitude more
    seed-stable than PER and separates the two arms cleanly. That is the
    measurement the H3 claim should rest on.""")

    with open("results/analysis_3seed.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/analysis_3seed.json")


if __name__ == "__main__":
    main()
