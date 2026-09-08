"""Is the lookahead x canonical-deviation interaction real, or composition?

The objection this answers
--------------------------
`rq3_accent_changing.py` reports that absolute PER falls 1.32x faster at
positions where the speaker deviated from canonical. A reviewer's objection is
that those positions are not a random subset: they start much harder (0.590 vs
0.354), they contain a different mix of phones, and they are unevenly spread
over speakers. Any of those could produce the slope difference without
lookahead interacting with accent at all.

Three controls, each attacking a different alternative explanation:

1. **Speaker-clustered bootstrap.** The independent unit is the speaker, not
   the phone token. Tens of thousands of phones nested in six speakers do not
   give tens of thousands of degrees of freedom, and treating them as if they
   did is how a null result becomes "significant".

2. **Phone-matched comparison.** Compute the slope difference *within each
   reference phone type* and pool. If /i/ in the deviating set is compared only
   against /i/ in the non-deviating set, phone composition cannot explain the
   gap.

3. **Both absolute and relative effects**, because they disagree here and the
   disagreement is a property of the baselines, not of the measurement.

    python3 eval/rq3_interaction.py --self-test
    python3 eval/rq3_interaction.py
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyse_h2_sequences import align
from rq3_accent_changing import changed_mask, correct_mask


def positions(hyps: Sequence[dict]) -> List[dict]:
    """One record per reference position: speaker, phone, deviation, error."""
    out = []
    for h in hyps:
        g, i, p = list(h["g2p"]), list(h["ipa"]), list(h["pred"])
        ch, ok = changed_mask(g, i), correct_mask(g, p)
        n = min(len(ch), len(ok), len(g))
        for k in range(n):
            out.append({"speaker": h.get("speaker"), "phone": g[k],
                        "dev": bool(ch[k]), "err": (not ok[k])})
    return out


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """OLS slope of error rate against log2 lookahead."""
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    m = ~np.isnan(y)
    if m.sum() < 3: return float("nan")
    x, y = x[m], y[m]
    return float(np.polyfit(x, y, 1)[0])


def slopes_by_group(by_L: Dict[float, List[dict]], keep=None) -> Tuple[float, float]:
    """(deviating slope, non-deviating slope) over lookaheads with L>=20."""
    Ls = sorted(L for L in by_L if L >= 20)
    xs = [np.log2(L) for L in Ls]
    dev, und = [], []
    for L in Ls:
        rows = by_L[L] if keep is None else [r for r in by_L[L] if keep(r)]
        d = [r["err"] for r in rows if r["dev"]]
        u = [r["err"] for r in rows if not r["dev"]]
        dev.append(np.mean(d) if d else np.nan)
        und.append(np.mean(u) if u else np.nan)
    return slope(xs, dev), slope(xs, und)


def _self_test() -> None:
    print("rq3_interaction self-test")
    ok_all = True
    def check(n, c, extra=""):
        nonlocal ok_all; ok_all &= bool(c)
        print(f"  {n:<52s} {'ok' if c else 'FAIL'} {extra}")

    recs = positions([{"speaker":"S1","g2p":["k","æ","t"],
                       "ipa":["k","ɪ","t"],"pred":["k","u","t"]}])
    check("one record per reference position", len(recs) == 3, f"{len(recs)}")
    check("deviation flagged only where speaker differed",
          [r["dev"] for r in recs] == [False,True,False])
    check("error flagged only where model was wrong",
          [r["err"] for r in recs] == [False,True,False])

    # a planted interaction: deviating positions improve, unchanged do not
    by_L = {}
    for L,(pd_,pu) in {20:(0.8,0.4),40:(0.6,0.4),80:(0.4,0.4),160:(0.2,0.4)}.items():
        rows=[]
        for k in range(100):
            rows.append({"speaker":"S1","phone":"a","dev":True,"err":k < pd_*100})
            rows.append({"speaker":"S1","phone":"a","dev":False,"err":k < pu*100})
        by_L[L]=rows
    sd, su = slopes_by_group(by_L)
    check("planted interaction recovered (dev steeper)", sd < su - 0.1,
          f"dev={sd:.3f} und={su:.3f}")

    flat = {L:[{"speaker":"S1","phone":"a","dev":d,"err":k<50}
               for d in (True,False) for k in range(100)] for L in (20,40,80,160)}
    sd2, su2 = slopes_by_group(flat)
    check("no interaction when both groups flat",
          abs(sd2-su2) < 1e-9, f"{sd2:.3f} vs {su2:.3f}")
    print("\nALL PASS" if ok_all else "\nFAILURES ABOVE")
    if not ok_all: sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", default="results/raw/hyps")
    ap.add_argument("--target", default="native")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results/analysis_rq3_interaction.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test: _self_test(); return

    by_L: Dict[float, List[dict]] = {}
    for f in sorted(glob.glob(os.path.join(a.hyp_dir, f"hyps_L*_{a.target}_s*.json"))):
        d = json.load(open(f))
        by_L.setdefault(float(d["lookahead_ms"]), []).extend(positions(d["hyps"]))
    Ls = sorted(by_L)
    speakers = sorted({r["speaker"] for r in by_L[Ls[0]]})
    print(f"{len(Ls)} lookaheads, {len(speakers)} speakers, "
          f"{len(by_L[Ls[0]])} positions per condition")

    sd, su = slopes_by_group(by_L)
    point = sd - su
    print(f"\npoint estimate")
    print(f"  slope, canonical-deviation : {sd:+.4f} PER/doubling")
    print(f"  slope, unchanged           : {su:+.4f} PER/doubling")
    print(f"  difference                 : {point:+.4f}   ratio {sd/su:.2f}x")

    # --- 1. speaker-clustered bootstrap: the unit is the speaker ---
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(a.boot):
        pick = rng.choice(speakers, size=len(speakers), replace=True)
        keep = {s: int((pick == s).sum()) for s in speakers}
        rs = {L: [r for r in by_L[L] for _ in range(keep.get(r["speaker"], 0))]
              for L in by_L}
        d_, u_ = slopes_by_group(rs)
        if not (np.isnan(d_) or np.isnan(u_)): diffs.append(d_ - u_)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    frac = float(np.mean(np.array(diffs) < 0))
    print(f"\nspeaker-clustered bootstrap ({len(diffs)} resamples of {len(speakers)} speakers)")
    print(f"  95% CI on slope difference : [{lo:+.4f}, {hi:+.4f}]")
    print(f"  fraction of resamples with deviation steeper: {frac:.3f}")
    print(f"  -> {'EXCLUDES zero' if hi < 0 else 'INCLUDES zero (not resolved)'}")

    # --- 1b. leave-one-speaker-out: the non-asymptotic companion ---
    # A cluster bootstrap over six clusters relies on asymptotics six clusters
    # do not supply. If dropping any single talker flips the sign, the pooled
    # slope difference is one speaker's result wearing a group's clothes.
    loo = []
    for s_ in speakers:
        rs = {L: [r for r in by_L[L] if r["speaker"] != s_] for L in by_L}
        d_, u_ = slopes_by_group(rs)
        if not (np.isnan(d_) or np.isnan(u_)):
            loo.append({"dropped": s_, "slope_difference": d_ - u_})
    print("\nleave-one-speaker-out on the slope difference")
    for r in loo:
        print(f"  drop {r['dropped']:<8s} {r['slope_difference']:+.4f}")
    sp = [r["slope_difference"] for r in loo]
    print(f"  range [{min(sp):+.4f}, {max(sp):+.4f}]"
          f"  -> sign {'STABLE' if min(sp) * max(sp) > 0 else 'FLIPS'}")

    # --- 2. phone-matched: same phone type on both sides ---
    phones = [p for p, n in
              sorted(((p, sum(1 for r in by_L[Ls[0]] if r["phone"] == p))
                      for p in {r["phone"] for r in by_L[Ls[0]]}),
                     key=lambda t: -t[1])]
    per_phone = []
    for ph in phones:
        d_, u_ = slopes_by_group(by_L, keep=lambda r, ph=ph: r["phone"] == ph)
        if not (np.isnan(d_) or np.isnan(u_)):
            per_phone.append((ph, d_ - u_))
    matched = float(np.mean([v for _, v in per_phone]))
    n_neg = sum(1 for _, v in per_phone if v < 0)
    print(f"\nphone-matched (slope difference computed within each reference phone)")
    print(f"  phones with both groups present : {len(per_phone)}")
    print(f"  mean within-phone difference    : {matched:+.4f}")
    print(f"  phones where deviation steeper  : {n_neg}/{len(per_phone)}")

    json.dump({"slope_deviation": sd, "slope_unchanged": su,
               "difference": point, "ratio": sd/su if su else None,
               "leave_one_speaker_out": loo,
               "speaker_bootstrap": {"n": len(diffs), "ci95": [lo, hi],
                                     "frac_deviation_steeper": frac,
                                     "excludes_zero": bool(hi < 0)},
               "phone_matched": {"n_phones": len(per_phone),
                                 "mean_difference": matched,
                                 "n_deviation_steeper": n_neg}},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
