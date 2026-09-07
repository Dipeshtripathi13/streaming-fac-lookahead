"""RQ3, sharpened: is the lookahead benefit concentrated where accent actually is?

The reviewer objection this answers
-----------------------------------
The conversion and transcription arms differ only in their target tensor, but
that tensor differs in more than "conversion vs transcription": annotation
noise, class distribution, sequence length and label entropy all move with it.
So "conversion needs more lookahead" is not established by the arm comparison
alone.

This test removes the confound by staying INSIDE the conversion arm. Split the
reference (g2p) positions by whether the speaker actually deviated there:

  accent-changing : the speaker produced something else (g2p[i] != ipa[i])
  unchanged       : the speaker already produced the canonical phone

Both groups come from the same model, the same run, the same target tensor and
the same utterances -- only the position type differs. If future context is
doing accent conversion rather than generic acoustic modelling, its benefit
must be concentrated on the accent-changing positions.

    python3 eval/rq3_accent_changing.py --self-test
    python3 eval/rq3_accent_changing.py
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from analyse_h2_sequences import align


def changed_mask(g2p: Sequence[str], ipa: Sequence[str]) -> List[bool]:
    """Per g2p position: did the speaker deviate from canonical here?

    Insertions in the ipa stream consume no g2p position, matching how
    analyse_h2_sequences charges errors to the reference phone.
    """
    out: List[bool] = []
    for op, _rp, _hp in align(list(g2p), list(ipa)):
        if op == "ins":
            continue
        out.append(op != "ok")
    return out


def correct_mask(g2p: Sequence[str], pred: Sequence[str]) -> List[bool]:
    out: List[bool] = []
    for op, _rp, _hp in align(list(g2p), list(pred)):
        if op == "ins":
            continue
        out.append(op == "ok")
    return out


def rates(hyps: Sequence[dict]) -> Dict[str, Tuple[float, int]]:
    """Error rate on accent-changing vs unchanged reference positions."""
    err = defaultdict(int); opp = defaultdict(int)
    for h in hyps:
        g, i, p = list(h["g2p"]), list(h["ipa"]), list(h["pred"])
        ch, ok = changed_mask(g, i), correct_mask(g, p)
        n = min(len(ch), len(ok))
        for k in range(n):
            grp = "accent_changing" if ch[k] else "unchanged"
            opp[grp] += 1
            if not ok[k]:
                err[grp] += 1
    return {g: (err[g] / opp[g] if opp[g] else float("nan"), opp[g])
            for g in ("accent_changing", "unchanged")}


def _self_test() -> None:
    print("rq3_accent_changing self-test")
    ok_all = True
    def check(n, c, extra=""):
        nonlocal ok_all; ok_all &= bool(c)
        print(f"  {n:<50s} {'ok' if c else 'FAIL'} {extra}")

    m = changed_mask(["k","æ","t"], ["k","ɪ","t"])
    check("substitution marks that position accent-changing", m == [False,True,False], f"{m}")
    m = changed_mask(["k","æ","t"], ["k","æ","t"])
    check("identical production -> nothing changing", m == [False]*3, f"{m}")
    m = changed_mask(["k","æ","t"], ["k","æ","s","t"])
    check("speaker insertion consumes no reference slot", len(m) == 3, f"{m}")
    c = correct_mask(["k","æ","t"], ["k","ɪ","t"])
    check("model error marks that position wrong", c == [True,False,True], f"{c}")

    # a model that is perfect on unchanged positions and wrong on changed ones
    hyps=[{"g2p":["k","æ","t"], "ipa":["k","ɪ","t"], "pred":["k","u","t"]}]
    r = rates(hyps)
    check("rates separate the two groups",
          r["accent_changing"][0] == 1.0 and r["unchanged"][0] == 0.0, f"{r}")
    print("\nALL PASS" if ok_all else "\nFAILURES ABOVE")
    if not ok_all: sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", default="results/raw/hyps")
    ap.add_argument("--target", default="native")
    ap.add_argument("--out", default="results/analysis_rq3_accent_changing.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test(); return

    by_L: Dict[float, List[dict]] = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.hyp_dir, f"hyps_L*_{a.target}_s*.json"))):
        d = json.load(open(f))
        by_L[float(d["lookahead_ms"])].extend(d["hyps"])
    if not by_L:
        sys.exit(f"no hyps for target={a.target} in {a.hyp_dir}")

    Ls = sorted(by_L)
    curve = {}
    print(f"{'L(ms)':>6} {'accent-changing':>16} {'unchanged':>12}   ratio")
    for L in Ls:
        r = rates(by_L[L])
        curve[L] = r
        ac, un = r["accent_changing"][0], r["unchanged"][0]
        print(f"{L:6.0f} {ac:16.4f} {un:12.4f}   {ac/un if un else float('nan'):5.2f}")

    def rel(g):
        a0, a1 = curve[Ls[0]][g][0], curve[Ls[-1]][g][0]
        return (a0 - a1) / a0 if a0 else float("nan")
    ra, ru = rel("accent_changing"), rel("unchanged")
    print(f"\nrelative error reduction L={Ls[0]:.0f} -> {Ls[-1]:.0f} ms")
    print(f"  accent-changing positions : {ra:.1%}  (n={curve[Ls[0]]['accent_changing'][1]})")
    print(f"  unchanged positions       : {ru:.1%}  (n={curve[Ls[0]]['unchanged'][1]})")
    print(f"  ratio                     : {ra/ru if ru else float('nan'):.2f}x")

    json.dump({"lookaheads_ms": Ls,
               "curve": {str(k): {g: {"error_rate": v[0], "n": v[1]}
                                  for g, v in c.items()} for k, c in curve.items()},
               "relative_reduction": {"accent_changing": ra, "unchanged": ru,
                                      "ratio": ra/ru if ru else None}},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
