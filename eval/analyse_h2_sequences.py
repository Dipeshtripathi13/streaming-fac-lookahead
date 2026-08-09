"""RQ2 / H2 from phone sequences: which sound classes benefit most from lookahead?

Why this file exists alongside phoneme_analysis.py
--------------------------------------------------
`phoneme_analysis.py` tests H2 the way the proposal originally specified: force
align converted audio against a native rendition and compute mel-cepstral
distortion per phone. That needs synthesised audio and TextGrids, neither of
which exists yet.

But the sweep already dumped, for every condition and every test utterance, the
canonical target sequence (`g2p`), what the speaker actually produced (`ipa`),
and the model's prediction (`pred`). That supports a *sequence-level* form of the
same hypothesis, and given the paper is now scoped to phone error rate (no
listening test, see docs/DECISIONS.md) it is arguably the form that matches the
rest of the claims.

What is tested, and what is NOT
-------------------------------
H2 predicts degradation is not uniform: consonant substitutions are
coarticulatorily local and should survive short lookahead, while vowels and
approximants are formant-defined over 80-250 ms and should break first.

Here that becomes: **does per-class error rate fall more steeply with lookahead
for the breaks-first classes than for the survives classes?**

Three things this deliberately does not do:

* **It does not re-derive the hypothesis.** `H2_PREDICTION` is imported from
  `phoneme_analysis.py` unchanged -- it was pre-registered before any of this
  data existed, and redefining it here would make the test unfalsifiable. The
  *symbol inventory* is translated ARPAbet -> IPA (see IPA_CLASS_OF below),
  because the pre-registered map is ARPAbet and the sweep emits IPA. The class
  definitions and the group memberships are untouched; only the keys change.
* **It does not claim perceptual relevance.** A phone-class error rate is not
  audibility. See docs/DECISIONS.md.
* **It does not treat utterances as independent for the confirmatory test.**
  Slopes here are descriptive OLS on condition means. The nesting (utterances
  within speakers within L1s) is reported via a by-L1 split and a
  speaker-clustered bootstrap, but a proper mixed-effects model belongs in
  eval/stats.py.

Alignment
---------
Errors are attributed by Levenshtein alignment with backtrace, charging each
operation to the class of the *reference* phone (for substitutions and
deletions) so that the denominator is "opportunities to get this class right".
Insertions have no reference phone and are counted separately rather than
silently attributed to a neighbour's class -- attributing them would inflate
whichever class happens to precede them.

Usage
-----
    python eval/analyse_h2_sequences.py --self-test
    python eval/analyse_h2_sequences.py --hyp-dir results/raw/hyps
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phoneme_analysis import (  # noqa: E402  pre-registered, do not redefine
    H2_PREDICTION,
)

NOISE_2SIGMA = 0.00436     # measured fixed-seed floor, PADDING_FIX_RESOLVED §5


# ==========================================================================
# IPA -> class, because the pre-registered map is ARPAbet and the data is not
# ==========================================================================
#
# phoneme_analysis.CLASS_OF is keyed on ARPAbet ("IY", "TH", "NG"). The sweep's
# g2p/ipa/pred sequences are IPA. Applying the ARPAbet map to IPA silently sent
# 8094 of 14085 reference tokens (57%) to class "other" -- including EVERY vowel
# and /r/ -- which reduced H2's "breaks_first" group to l and w alone and made
# the hypothesis untestable while still producing plausible-looking output.
#
# The class *definitions* and H2_PREDICTION are unchanged; only the symbol
# inventory is translated. Two features of this transcription force context
# sensitivity, both verified empirically against the data:
#
#   * Diphthongs are decomposed into two symbols: a+ɪ, a+ʊ, e+ɪ, o+ʊ, ɔ+ɪ.
#     'a' is followed by ɪ or ʊ 100% of the time, 'e' by ɪ 100%, 'o' by ʊ 100%.
#     But the offglides also stand alone: ɪ follows a diphthong onset only 38.6%
#     of the time and ʊ 78.5%. A context-free map would therefore mislabel
#     thousands of genuine monophthongs as diphthongs.
#   * Affricates are decomposed too: t+ʃ and d+ʒ. Without detecting those
#     sequences the pre-registered "affricate" class is empty, so H2's
#     survives-group would silently lose a member.

IPA_STOPS = {"p", "b", "t", "d", "k", "ɡ"}
IPA_FRICATIVES = {"f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h"}
IPA_NASALS = {"m", "n", "ŋ"}
IPA_APPROXIMANTS = {"l", "ɹ", "w", "j"}
IPA_VOWELS = {"i", "ɪ", "ɛ", "æ", "ɑ", "ɔ", "ʊ", "u", "ʌ", "ɝ", "a", "e", "o"}

DIPHTHONGS = {("a", "ɪ"), ("a", "ʊ"), ("e", "ɪ"), ("o", "ʊ"), ("ɔ", "ɪ")}
AFFRICATES_SEQ = {("t", "ʃ"), ("d", "ʒ")}


def _base_class(p: str) -> str:
    if p in IPA_STOPS:
        return "stop"
    if p in IPA_FRICATIVES:
        return "fricative"
    if p in IPA_NASALS:
        return "nasal"
    if p in IPA_APPROXIMANTS:
        return "approximant"
    if p in IPA_VOWELS:
        return "vowel_mono"
    return "other"


def ipa_classes(seq: Sequence[str]) -> List[str]:
    """Class per position, resolving decomposed diphthongs and affricates.

    Both members of a detected two-symbol unit get the unit's class, which keeps
    the mapping 1:1 with the sequence so alignment indices stay valid.
    """
    out = [_base_class(p) for p in seq]
    i = 0
    while i < len(seq) - 1:
        pair = (seq[i], seq[i + 1])
        if pair in DIPHTHONGS:
            out[i] = out[i + 1] = "vowel_diph"
            i += 2
            continue
        if pair in AFFRICATES_SEQ:
            out[i] = out[i + 1] = "affricate"
            i += 2
            continue
        i += 1
    return out


# ==========================================================================
# Alignment
# ==========================================================================

def align(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Levenshtein alignment with backtrace.

    Returns a list of (op, ref_phone, hyp_phone) where op is one of
    "ok" / "sub" / "del" / "ins". ref_phone is None only for insertions.
    """
    n, m = len(ref), len(hyp)
    # d[i][j] = cost of aligning ref[:i] with hyp[:j]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost,   # match / sub
                          d[i - 1][j] + 1,          # deletion
                          d[i][j - 1] + 1)          # insertion
    out: List[Tuple[str, Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                out.append(("ok" if cost == 0 else "sub", ref[i - 1], hyp[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append(("del", ref[i - 1], None))
            i -= 1
            continue
        out.append(("ins", None, hyp[j - 1]))
        j -= 1
    out.reverse()
    return out


def class_error_rates(pairs: Iterable[Tuple[Sequence[str], Sequence[str]]]
                      ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    """Per-class error rate over many (ref, hyp) pairs.

    error_rate(class) = (subs + dels charged to that class) / (opportunities)
    where opportunities = number of reference phones of that class. Insertions
    are counted globally, not per class -- see the module docstring.
    """
    opp: Dict[str, int] = collections.Counter()
    err: Dict[str, int] = collections.Counter()
    sub: Dict[str, int] = collections.Counter()
    dele: Dict[str, int] = collections.Counter()
    ins_total = 0
    for ref, hyp in pairs:
        ref = list(ref)
        cls = ipa_classes(ref)
        ri = 0                      # index into ref, advanced on non-insertions
        for op, rp, _hp in align(ref, list(hyp)):
            if op == "ins":
                ins_total += 1
                continue
            c = cls[ri]             # contextual class, not a per-symbol lookup
            ri += 1
            opp[c] += 1
            if op == "sub":
                err[c] += 1
                sub[c] += 1
            elif op == "del":
                err[c] += 1
                dele[c] += 1
    rates = {
        c: {
            "error_rate": err[c] / opp[c] if opp[c] else float("nan"),
            "sub_rate": sub[c] / opp[c] if opp[c] else float("nan"),
            "del_rate": dele[c] / opp[c] if opp[c] else float("nan"),
            "opportunities": float(opp[c]),
        }
        for c in sorted(opp)
    }
    return rates, {"insertions_total": ins_total}


# ==========================================================================
# Ingest
# ==========================================================================

def load_hyps(hyp_dir: str, target: str = "native"
              ) -> Tuple[Dict[float, List[dict]], Dict]:
    """Group hypothesis entries by lookahead, pooling seeds.

    Seeds are pooled deliberately: H2 is about *which classes* respond to
    lookahead, and per-seed class rates over 400 utterances are far noisier than
    the pooled estimate. The by-seed split is reported separately as a
    robustness check rather than used as the primary estimate.
    """
    files = sorted(glob.glob(os.path.join(hyp_dir, f"hyps_L*_{target}_s*.json")))
    if not files:
        raise SystemExit(
            f"no hyps_L*_{target}_s*.json in {hyp_dir}. "
            "Re-run the sweep with --save-hyps.")
    by_L: Dict[float, List[dict]] = collections.defaultdict(list)
    seeds: Dict[float, set] = collections.defaultdict(set)
    for f in files:
        d = json.load(open(f))
        L = float(d["lookahead_ms"])
        by_L[L].extend(d.get("hyps") or [])
        seeds[L].add(d.get("seed"))
    prov = {
        "hyp_dir": hyp_dir,
        "target": target,
        "n_files": len(files),
        "lookaheads": sorted(by_L),
        "seeds_per_lookahead": {str(k): sorted(map(str, v)) for k, v in sorted(seeds.items())},
        "utterances_per_lookahead": {str(k): len(v) for k, v in sorted(by_L.items())},
        "l1s": sorted({h.get("l1") for L in by_L for h in by_L[L] if h.get("l1")}),
    }
    return dict(by_L), prov


# ==========================================================================
# H2 test
# ==========================================================================

def _ols(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my, float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sst = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum(r * r for r in resid) / sst if sst > 0 else float("nan")
    return b, a, r2


def h2_from_rates(per_L: Dict[float, Dict[str, Dict[str, float]]],
                  metric: str = "error_rate") -> Dict[str, object]:
    """Fit class-wise error rate against log2 lookahead and compare groups.

    Uses log2(L) because RQ1 established the response is log-linear in
    lookahead; fitting against raw ms would make the vowel classes look
    artificially flat at the top end simply because the x-spacing is uneven.
    L=0 is placed at half a frame so it can sit on a log axis, consistent with
    analyse_dense_knee.py.
    """
    Ls = sorted(per_L)
    xs_all = [math.log2(10.0 if L == 0 else L) for L in Ls]
    classes = sorted({c for d in per_L.values() for c in d})
    fits: Dict[str, Dict[str, float]] = {}
    for c in classes:
        pts = [(x, per_L[L][c][metric]) for x, L in zip(xs_all, Ls)
               if c in per_L[L] and not math.isnan(per_L[L][c][metric])]
        if len(pts) < 4:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        slope, icpt, r2 = _ols(xs, ys)
        fits[c] = {
            "slope_per_doubling": slope,
            "r2": r2,
            "rate_at_Lmin": ys[0],
            "rate_at_Lmax": ys[-1],
            "absolute_gain": ys[0] - ys[-1],
            "relative_gain": (ys[0] - ys[-1]) / ys[0] if ys[0] else float("nan"),
            "opportunities": per_L[Ls[0]][c]["opportunities"],
            "resolvable": (ys[0] - ys[-1]) > NOISE_2SIGMA,
        }
    brk = [c for c in H2_PREDICTION["breaks_first"] if c in fits]
    srv = [c for c in H2_PREDICTION["survives_short_lookahead"] if c in fits]

    def mean(vals): return sum(vals) / len(vals) if vals else float("nan")

    gb = mean([fits[c]["relative_gain"] for c in brk])
    gs = mean([fits[c]["relative_gain"] for c in srv])
    sb = mean([fits[c]["slope_per_doubling"] for c in brk])
    ss_ = mean([fits[c]["slope_per_doubling"] for c in srv])

    # Direction of support: breaks-first classes should improve MORE, i.e. have
    # a more negative slope and a larger relative gain.
    direction = (not math.isnan(gb) and not math.isnan(gs)
                 and gb > gs and sb < ss_)

    # Direction alone is far too weak a criterion. A 6% difference in group means
    # can satisfy it while the grouping explains almost none of the variance --
    # which is what the real data does. Two harder checks:
    #
    #   effect_size   between-group difference over the pooled SD across classes.
    #   n_violations  how many (breaks_first, survives) pairs are ordered the
    #                 WRONG way. H2 predicts zero; every violation is a class
    #                 the hypothesis put in the wrong group.
    #
    # H2 is only reported as supported if the direction holds AND the grouping
    # beats the within-group spread AND violations are a minority.
    all_g = [fits[c]["relative_gain"] for c in brk + srv]
    pooled_sd = (sum((g - sum(all_g) / len(all_g)) ** 2 for g in all_g)
                 / len(all_g)) ** 0.5 if len(all_g) > 1 else float("nan")
    between = gb - gs
    effect = between / pooled_sd if pooled_sd else float("nan")
    violations = [(s_, b) for b in brk for s_ in srv
                  if fits[s_]["relative_gain"] > fits[b]["relative_gain"]]
    n_pairs = len(brk) * len(srv)
    within_srv = (max(fits[c]["relative_gain"] for c in srv)
                  - min(fits[c]["relative_gain"] for c in srv)) if srv else float("nan")
    within_brk = (max(fits[c]["relative_gain"] for c in brk)
                  - min(fits[c]["relative_gain"] for c in brk)) if brk else float("nan")

    supported = (direction and not math.isnan(effect) and effect >= 1.0
                 and len(violations) < n_pairs / 2)
    return {
        "direction_only": direction,
        "effect_size_between_over_pooled_sd": effect,
        "between_group_difference": between,
        "within_group_range_breaks_first": within_brk,
        "within_group_range_survives": within_srv,
        "ordering_violations": [f"{a} > {b}" for a, b in violations],
        "n_ordering_violations": len(violations),
        "n_ordering_pairs": n_pairs,
        "observed_rank_by_gain": [c for c, _ in sorted(
            fits.items(), key=lambda kv: -kv[1]["relative_gain"])],
        "metric": metric,
        "x_axis": "log2(lookahead_ms), L=0 at 10 ms",
        "lookaheads_ms": Ls,
        "per_class_fit": fits,
        "breaks_first_classes": brk,
        "survives_classes": srv,
        "mean_relative_gain_breaks_first": gb,
        "mean_relative_gain_survives": gs,
        "mean_slope_breaks_first": sb,
        "mean_slope_survives": ss_,
        "H2_supported": supported,
        "H2_direction_supported": direction,   # kept: weaker, direction only
        "gain_ratio": gb / gs if gs and not math.isnan(gs) and gs != 0 else float("nan"),
    }


def speaker_clustered_bootstrap(by_L: Dict[float, List[dict]], reps: int = 300,
                                seed: int = 0) -> Dict[str, object]:
    """Resample SPEAKERS, not utterances, and re-run the H2 direction test.

    Utterances from one speaker are not independent -- one speaker's L1 and
    idiolect drive many of their phone errors. Resampling utterances would
    understate the uncertainty; resampling speakers is the cheap correction that
    respects the nesting.
    """
    speakers = sorted({h.get("speaker") for L in by_L for h in by_L[L]
                       if h.get("speaker")})
    if len(speakers) < 3:
        return {"available": False, "reason": f"only {len(speakers)} speakers"}
    rng = random.Random(seed)
    supported = 0
    ratios: List[float] = []
    for _ in range(reps):
        pick = collections.Counter(rng.choice(speakers) for _ in speakers)
        per_L = {}
        for L, hs in by_L.items():
            pairs = []
            for h in hs:
                k = pick.get(h.get("speaker"), 0)
                if k:
                    pairs.extend([(h["g2p"], h["pred"])] * k)
            if pairs:
                rates, _ = class_error_rates(pairs)
                per_L[L] = rates
        r = h2_from_rates(per_L)
        if r["H2_supported"]:
            supported += 1
        if not math.isnan(r["gain_ratio"]):
            ratios.append(r["gain_ratio"])
    ratios.sort()
    return {
        "available": True,
        "reps": reps,
        "n_speakers": len(speakers),
        "fraction_supporting_H2": supported / reps,
        "gain_ratio_ci90": [ratios[int(0.05 * len(ratios))],
                            ratios[min(len(ratios) - 1, int(0.95 * len(ratios)))]]
        if ratios else None,
    }


# ==========================================================================
# Self-test
# ==========================================================================

def self_test() -> int:
    print("analyse_h2_sequences self-test")
    fails = 0

    # --- 1. alignment ops on hand-checked cases ---
    cases = [
        (list("abc"), list("abc"), {"ok": 3}),
        (list("abc"), list("axc"), {"ok": 2, "sub": 1}),
        (list("abc"), list("ac"),  {"ok": 2, "del": 1}),
        (list("ac"),  list("abc"), {"ok": 2, "ins": 1}),
        ([], list("ab"), {"ins": 2}),
        (list("ab"), [], {"del": 2}),
    ]
    for ref, hyp, want in cases:
        got = collections.Counter(op for op, _, _ in align(ref, hyp))
        ok = dict(got) == want
        print(f"  align {ref!s:>12} -> {hyp!s:<12} {dict(got)}  {'ok' if ok else 'FAIL'}")
        fails += 0 if ok else 1

    # --- 2. errors are charged to the REFERENCE class, not the hypothesis ---
    # Reference /i/ (vowel) replaced by /t/ (stop) must count against vowels.
    rates, _ = class_error_rates([(["i"], ["t"])])
    ok = (rates.get("vowel_mono", {}).get("error_rate") == 1.0
          and "stop" not in rates)
    print(f"  substitution charged to reference class 'vowel_mono' not 'stop'"
          f"  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 3. insertions are NOT attributed to any class ---
    rates, extra = class_error_rates([(["t"], ["t", "i"])])
    ok = extra["insertions_total"] == 1 and rates["stop"]["error_rate"] == 0.0
    print(f"  insertion counted globally ({extra['insertions_total']}), no class "
          f"charged  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 3b. contextual classes: decomposed diphthongs and affricates ---
    ctx = [
        (["a", "ɪ"],           ["vowel_diph", "vowel_diph"], "aɪ diphthong"),
        (["e", "ɪ"],           ["vowel_diph", "vowel_diph"], "eɪ diphthong"),
        (["o", "ʊ"],           ["vowel_diph", "vowel_diph"], "oʊ diphthong"),
        (["ɪ", "n"],           ["vowel_mono", "nasal"],      "bare ɪ is mono"),
        (["ʊ", "t"],           ["vowel_mono", "stop"],       "bare ʊ is mono"),
        (["ɔ", "ɹ"],           ["vowel_mono", "approximant"], "ɔɹ not a diphthong"),
        (["t", "ʃ"],           ["affricate", "affricate"],   "tʃ affricate"),
        (["d", "ʒ"],           ["affricate", "affricate"],   "dʒ affricate"),
        (["t", "ɹ"],           ["stop", "approximant"],      "tɹ is not affricate"),
        (["ɹ", "ɛ", "d"],      ["approximant", "vowel_mono", "stop"], "ɹ is approximant"),
    ]
    for seq, want, label in ctx:
        got = ipa_classes(seq)
        ok = got == want
        print(f"  ctx {label:<22} {got}  {'ok' if ok else 'FAIL want ' + str(want)}")
        fails += 0 if ok else 1

    # --- 3c. the bug that started this: no reference phone may be "other" ---
    # The original run applied the ARPAbet map to IPA and sent 57% of tokens,
    # including every vowel, to "other" -- while still printing a plausible
    # H2 verdict. A coverage assertion is the cheap guard against that.
    inventory = sorted(IPA_STOPS | IPA_FRICATIVES | IPA_NASALS
                       | IPA_APPROXIMANTS | IPA_VOWELS)
    unmapped = [p for p in inventory if _base_class(p) == "other"]
    ok = not unmapped
    print(f"  every phone in the declared inventory maps to a class "
          f"(unmapped: {unmapped})  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 4. planted class difference must be detected in the right direction ---
    # Vowels improve a lot with L, stops barely. Build synthetic per_L.
    Ls = [0, 20, 40, 80, 160, 320, 640]
    v, s = "vowel_mono", "stop"
    per_L = {}
    for i, L in enumerate(Ls):
        frac = i / (len(Ls) - 1)
        per_L[L] = {
            v: {"error_rate": 0.50 - 0.40 * frac, "opportunities": 1000.0},
            s: {"error_rate": 0.30 - 0.02 * frac, "opportunities": 1000.0},
        }
    r = h2_from_rates(per_L)
    ok = r["H2_supported"] and r["gain_ratio"] > 5
    print(f"  planted: vowels gain {r['mean_relative_gain_breaks_first']:.3f} vs "
          f"stops {r['mean_relative_gain_survives']:.3f}, ratio "
          f"{r['gain_ratio']:.1f}, effect {r['effect_size_between_over_pooled_sd']:.2f} "
          f"-> supported={r['H2_supported']}  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 5. the REVERSE pattern must NOT be reported as support ---
    per_L2 = {}
    for i, L in enumerate(Ls):
        frac = i / (len(Ls) - 1)
        per_L2[L] = {
            v: {"error_rate": 0.50 - 0.02 * frac, "opportunities": 1000.0},
            s: {"error_rate": 0.30 - 0.24 * frac, "opportunities": 1000.0},
        }
    r2 = h2_from_rates(per_L2)
    ok = not r2["H2_supported"]
    print(f"  reversed pattern -> supported={r2['H2_supported']} "
          f"(want False)  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 6. the pre-registered hypothesis is imported, not redefined here ---
    import phoneme_analysis as pa
    ok = (H2_PREDICTION is pa.H2_PREDICTION
          and set(H2_PREDICTION["breaks_first"]) ==
          {"vowel_diph", "vowel_mono", "approximant"}
          and set(H2_PREDICTION["survives_short_lookahead"]) ==
          {"stop", "fricative", "affricate", "nasal"})
    print(f"  H2_PREDICTION imported unchanged from phoneme_analysis: {ok}"
          f"  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 7. all seven pre-registered classes must be reachable ---
    reachable = set()
    for seq in (["p"], ["f"], ["m"], ["l"], ["i"], ["a", "ɪ"], ["t", "ʃ"]):
        reachable.update(ipa_classes(seq))
    want = set(H2_PREDICTION["breaks_first"]) | set(
        H2_PREDICTION["survives_short_lookahead"])
    ok = want <= reachable
    print(f"  all pre-registered classes reachable: {sorted(want - reachable) or 'yes'}"
          f"  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", default="results/raw/hyps")
    ap.add_argument("--target", default="native")
    ap.add_argument("--out", default="results/analysis_h2_sequences.json")
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(self_test())

    by_L, prov = load_hyps(a.hyp_dir, a.target)

    per_L: Dict[float, Dict[str, Dict[str, float]]] = {}
    extras: Dict[str, object] = {}
    for L, hs in sorted(by_L.items()):
        rates, ex = class_error_rates([(h["g2p"], h["pred"]) for h in hs])
        per_L[L] = rates
        extras[str(L)] = ex

    res: Dict[str, object] = {
        "provenance": prov,
        "noise_2sigma": NOISE_2SIGMA,
        "per_lookahead_class_rates": {str(k): v for k, v in per_L.items()},
        "insertions": extras,
        "h2": h2_from_rates(per_L),
    }

    # by-L1 split: same test, one L1 at a time
    by_l1: Dict[str, object] = {}
    for l1 in prov["l1s"]:
        sub = {}
        for L, hs in by_L.items():
            pairs = [(h["g2p"], h["pred"]) for h in hs if h.get("l1") == l1]
            if pairs:
                sub[L], _ = class_error_rates(pairs)
        if len(sub) >= 4:
            r = h2_from_rates(sub)
            by_l1[l1] = {
                "H2_supported": r["H2_supported"],
                "H2_direction_supported": r["H2_direction_supported"],
                "gain_ratio": r["gain_ratio"],
                "mean_relative_gain_breaks_first": r["mean_relative_gain_breaks_first"],
                "mean_relative_gain_survives": r["mean_relative_gain_survives"],
            }
    res["by_l1"] = by_l1

    if a.bootstrap:
        res["speaker_bootstrap"] = speaker_clustered_bootstrap(by_L, a.bootstrap)

    h = res["h2"]
    print(json.dumps(prov, indent=2))
    print("\n-- per-class error rate vs lookahead --")
    Ls = h["lookaheads_ms"]
    classes = sorted(h["per_class_fit"])
    hdr = "  class            " + "".join(f"{int(L):>7}" for L in Ls)
    print(hdr)
    for c in classes:
        row = "".join(f"{per_L[L][c]['error_rate']:>7.3f}" if c in per_L[L] else "      -"
                      for L in Ls)
        tag = "B" if c in h["breaks_first_classes"] else (
            "S" if c in h["survives_classes"] else " ")
        print(f"  {tag} {c:<14}{row}")
    print("\n-- fits (B = predicted to break first, S = predicted to survive) --")
    for c in classes:
        f = h["per_class_fit"][c]
        tag = "B" if c in h["breaks_first_classes"] else (
            "S" if c in h["survives_classes"] else " ")
        print(f"  {tag} {c:<14} slope/doubling {f['slope_per_doubling']:+.4f}  "
              f"rel.gain {f['relative_gain']:+.3f}  R2 {f['r2']:.3f}  "
              f"n_ref {int(f['opportunities']):>6}  "
              f"{'resolvable' if f['resolvable'] else 'BELOW FLOOR'}")
    print("\n== H2 ==")
    print(json.dumps({k: v for k, v in h.items()
                      if k not in ("per_class_fit", "lookaheads_ms")}, indent=2))
    if "speaker_bootstrap" in res:
        print("\n-- speaker-clustered bootstrap --")
        print(json.dumps(res["speaker_bootstrap"], indent=2))
    if by_l1:
        print("\n-- by L1 --")
        for l1, v in sorted(by_l1.items()):
            print(f"  {l1:<12} supported={str(v['H2_supported']):<5} "
                  f"ratio={v['gain_ratio']:.2f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\nwrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
