"""RQ2 uncertainty at the right level: speakers, not phone tokens.

The problem
-----------
The class analysis pools 42,255 reference-phone tokens per condition, but those
tokens are nested in six speakers (and, below that, in utterances). Treating
them as 42,255 independent samples makes every interval far too narrow: the
independent replicate for a claim about accented speech is a speaker, not a
phone. Class token counts also differ by an order of magnitude -- 1,098
affricates against 13,008 monophthongs -- so a single global repeatability
floor is not an equally good yardstick for each class.

This module recomputes the H2 quantities with a bootstrap over SPEAKERS, so the
intervals reflect the number of talkers actually observed.

A second, harder limitation surfaces on inspection: in the speaker-disjoint
test split each L1 is represented by exactly one speaker. The by-L1 breakdown
is therefore completely confounded with per-speaker variation and cannot
separate a language effect from an individual one. That is reported, not
adjusted away.

    python3 eval/rq2_hierarchical.py --self-test
    python3 eval/rq2_hierarchical.py
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyse_h2_sequences import H2_PREDICTION, align, ipa_classes


def class_positions(hyps: Sequence[dict]) -> List[dict]:
    """One record per reference phone: speaker, class, error."""
    out = []
    for h in hyps:
        ref = list(h["g2p"])
        cls = ipa_classes(ref)
        ri = 0
        for op, _rp, _hp in align(ref, list(h["pred"])):
            if op == "ins":
                continue
            if ri < len(cls):
                out.append({"speaker": h.get("speaker"), "cls": cls[ri],
                            "err": op != "ok"})
            ri += 1
    return out


def class_rates(recs: Sequence[dict], keep=None) -> Dict[str, float]:
    e, n = defaultdict(int), defaultdict(int)
    for r in recs:
        if keep is not None and not keep(r):
            continue
        n[r["cls"]] += 1
        e[r["cls"]] += r["err"]
    return {c: e[c] / n[c] for c in n if n[c]}


def rel_gain(lo: Dict[str, float], hi: Dict[str, float]) -> Dict[str, float]:
    """Relative error reduction from the L=0 rates `lo` to the L=max rates `hi`."""
    return {c: (lo[c] - hi[c]) / lo[c] for c in lo if c in hi and lo[c] > 0}


def _self_test() -> None:
    print("rq2_hierarchical self-test")
    ok = True
    def chk(n, c, x=""):
        nonlocal ok; ok &= bool(c); print(f"  {n:<50s} {'ok' if c else 'FAIL'} {x}")

    recs = class_positions([{"speaker":"S1","g2p":["k","æ","t"],
                             "pred":["k","u","t"]}])
    chk("one record per reference phone", len(recs)==3, f"{len(recs)}")
    chk("class assigned from the reference phone",
        [r["cls"] for r in recs]==["stop","vowel_mono","stop"],
        f'{[r["cls"] for r in recs]}')
    chk("error flagged only where the model was wrong",
        [r["err"] for r in recs]==[False,True,False])

    r = class_rates(recs)
    chk("rates computed per class", abs(r["vowel_mono"]-1.0)<1e-9 and abs(r["stop"])<1e-9, f"{r}")
    g = rel_gain({"a":0.5,"b":0.2}, {"a":0.25,"b":0.2})
    chk("relative gain", abs(g["a"]-0.5)<1e-9 and abs(g["b"])<1e-9, f"{g}")

    recs2 = class_positions([{"speaker":"S2","g2p":["k"],"pred":["k","t"]}])
    chk("hypothesis insertion consumes no reference slot", len(recs2)==1)
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    if not ok: sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", default="results/raw/hyps")
    ap.add_argument("--target", default="native")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results/analysis_rq2_hierarchical.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test: _self_test(); return

    by_L: Dict[float, List[dict]] = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.hyp_dir, f"hyps_L*_{a.target}_s*.json"))):
        d = json.load(open(f))
        by_L[float(d["lookahead_ms"])].extend(class_positions(d["hyps"]))
    Ls = sorted(by_L)
    lo_L, hi_L = Ls[0], Ls[-1]
    speakers = sorted({r["speaker"] for r in by_L[lo_L]})
    print(f"{len(Ls)} lookaheads, {len(speakers)} speakers, "
          f"{len(by_L[lo_L])} phone tokens per condition")

    point = rel_gain(class_rates(by_L[lo_L]), class_rates(by_L[hi_L]))
    brk = [c for c in H2_PREDICTION["breaks_first"] if c in point]
    srv = [c for c in H2_PREDICTION["survives_short_lookahead"] if c in point]

    rng = np.random.default_rng(0)
    samples: Dict[str, List[float]] = defaultdict(list)
    diffs = []
    for _ in range(a.boot):
        pick = rng.choice(speakers, size=len(speakers), replace=True)
        cnt = {s: int((pick == s).sum()) for s in speakers}
        keep = lambda r: cnt.get(r["speaker"], 0) > 0
        w = lambda recs: [r for r in recs for _ in range(cnt.get(r["speaker"], 0))]
        g = rel_gain(class_rates(w(by_L[lo_L])), class_rates(w(by_L[hi_L])))
        for c, v in g.items(): samples[c].append(v)
        if all(c in g for c in brk+srv):
            diffs.append(np.mean([g[c] for c in brk]) - np.mean([g[c] for c in srv]))

    print(f"\nrelative error reduction L={lo_L:.0f}->{hi_L:.0f} ms, "
          f"speaker-clustered 95% CI ({len(speakers)} speakers)")
    print(f"  {'':2}{'class':<14}{'gain':>7}   95% CI")
    out_cls = {}
    for c in sorted(point, key=lambda x: -point[x]):
        lo, hi = np.percentile(samples[c], [2.5, 97.5])
        tag = "B" if c in brk else ("S" if c in srv else " ")
        out_cls[c] = {"gain": point[c], "ci95": [lo, hi]}
        print(f"  {tag} {c:<14}{point[c]:7.3f}   [{lo:+.3f}, {hi:+.3f}]")

    d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
    d_pt = np.mean([point[c] for c in brk]) - np.mean([point[c] for c in srv])
    print(f"\n  between-group difference (breaks-first minus survives)")
    print(f"    point {d_pt:+.4f}   95% CI [{d_lo:+.4f}, {d_hi:+.4f}]"
          f"   -> {'excludes' if (d_lo>0 or d_hi<0) else 'INCLUDES'} zero")

    json.dump({"n_speakers": len(speakers), "lookaheads_ms": [lo_L, hi_L],
               "per_class": out_cls,
               "between_group": {"point": d_pt, "ci95": [d_lo, d_hi],
                                 "excludes_zero": bool(d_lo > 0 or d_hi < 0)},
               "note": "bootstrap over speakers; each L1 in the test split has "
                       "exactly one speaker, so by-L1 and by-speaker are "
                       "confounded"},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
