"""Verify empirically that L2-ARCTIC and CMU ARCTIC share prompts.

Why this is action item #4 in the proposal and not an assumption
---------------------------------------------------------------
The whole data plan depends on it. If L2-ARCTIC speaker HKK reads the same
1132 sentences as CMU ARCTIC speaker BDL, then for every accented utterance
there is a native rendition of the identical text -- which gives you, free:

  * parallel training pairs for the accent translator,
  * a reference waveform for per-phone MCD (eval/phoneme_analysis.py),
  * a ground-truth transcript for WER without transcribing anything.

That is the free equivalent of the curated parallel script a commercial
system would pay to record. Documented sources say the overlap is exact
(L2-ARCTIC speakers each read the 1132 phonetically balanced CMU ARCTIC
prompts; 24 speakers, 6 L1s, ~26,867 utterances total). Documented is not
verified. Utterance IDs drift, some speakers skip prompts, and a handful of
L2-ARCTIC transcripts were corrected against what the speaker actually said
rather than what was on the card. Those corrections are exactly the
utterances where a WER-from-prompt assumption would silently break.

This script reports, per speaker:
  - how many prompt IDs are shared with CMU ARCTIC
  - how many shared IDs have *textually identical* prompts
  - the diffs for the ones that do not

Run it before writing any training code.

Usage:
    python3 data/verify_prompt_overlap.py \
        --l2-arctic /path/to/l2arctic_release_v5 \
        --cmu-arctic /path/to/cmu_arctic
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

L1_OF_SPEAKER = {
    # L2-ARCTIC v5, 24 speakers, 6 L1s, gender balanced
    "ABA": "Arabic",   "SKA": "Arabic",   "YBAA": "Arabic",  "ZHAA": "Arabic",
    "BWC": "Mandarin", "LXC": "Mandarin", "NCC": "Mandarin", "TXHC": "Mandarin",
    "ASI": "Hindi",    "RRBI": "Hindi",   "SVBI": "Hindi",   "TNI": "Hindi",
    "HJK": "Korean",   "HKK": "Korean",   "YDCK": "Korean",  "YKWK": "Korean",
    "EBVS": "Spanish", "ERMS": "Spanish", "MBMPS": "Spanish", "NJS": "Spanish",
    "HQTV": "Vietnamese", "PNV": "Vietnamese", "THV": "Vietnamese", "TLV": "Vietnamese",
}

# The four pairs in the experimental matrix. Korean and Vietnamese are held
# out as an unseen-L1 generalisation test rather than being dropped -- a
# reviewer will ask whether the lookahead requirement is L1-specific, and
# holding out two L1s answers it for the price of inference only.
MATRIX_L1 = ["Hindi", "Mandarin", "Spanish", "Arabic"]
HELDOUT_L1 = ["Korean", "Vietnamese"]

_NORM = re.compile(r"[^a-z0-9' ]")


def norm(s: str) -> str:
    return " ".join(_NORM.sub(" ", s.lower()).split())


def read_cmu_arctic(root: str) -> Dict[str, str]:
    """Read etc/txt.done.data (festival format):  ( arctic_a0001 "text" )"""
    cands = glob.glob(os.path.join(root, "**", "txt.done.data"), recursive=True)
    cands += glob.glob(os.path.join(root, "**", "cmuarctic.data"), recursive=True)
    if not cands:
        raise FileNotFoundError(
            f"no txt.done.data / cmuarctic.data under {root}. "
            "Download from http://festvox.org/cmu_arctic/")
    out: Dict[str, str] = {}
    pat = re.compile(r'\(\s*(\S+)\s+"(.*)"\s*\)')
    for c in cands:
        with open(c, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = pat.match(line.strip())
                if m:
                    out.setdefault(m.group(1), m.group(2))
    return out


def read_l2_arctic(root: str) -> Dict[str, Dict[str, str]]:
    """Per speaker: {utt_id: transcript} from <SPK>/transcript/*.txt"""
    out: Dict[str, Dict[str, str]] = {}
    for spk_dir in sorted(glob.glob(os.path.join(root, "*"))):
        spk = os.path.basename(spk_dir)
        tdir = os.path.join(spk_dir, "transcript")
        if not os.path.isdir(tdir):
            continue
        d: Dict[str, str] = {}
        for p in sorted(glob.glob(os.path.join(tdir, "*.txt"))):
            uid = os.path.splitext(os.path.basename(p))[0]
            with open(p, encoding="utf-8", errors="replace") as f:
                d[uid] = f.read().strip()
        if d:
            out[spk] = d
    if not out:
        raise FileNotFoundError(
            f"no <SPK>/transcript/*.txt under {root}. "
            "Request L2-ARCTIC at https://psi.engr.tamu.edu/l2-arctic-corpus/")
    return out


def count_annotations(root: str) -> Dict[str, int]:
    """L2-ARCTIC ships human phoneme-error TextGrids for a subset per speaker.

    That subset is the ground truth for the H2 analysis, so knowing its size
    per speaker determines whether the phoneme-class analysis is powered at
    all. If it is ~150 utterances/speaker, four accent pairs gives roughly
    600 annotated utterances per L1 -- enough for class-level slopes, not
    enough for per-phone slopes. Better to learn that now than in December.
    """
    out: Dict[str, int] = {}
    for spk_dir in sorted(glob.glob(os.path.join(root, "*"))):
        spk = os.path.basename(spk_dir)
        n = len(glob.glob(os.path.join(spk_dir, "annotation", "*.TextGrid")))
        if n:
            out[spk] = n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2-arctic", required=True)
    ap.add_argument("--cmu-arctic", required=True)
    ap.add_argument("--out", default="results/raw/prompt_overlap.json")
    ap.add_argument("--show-diffs", type=int, default=8)
    a = ap.parse_args()

    cmu = read_cmu_arctic(a.cmu_arctic)
    l2 = read_l2_arctic(a.l2_arctic)
    print(f"CMU ARCTIC prompts: {len(cmu)}")
    print(f"L2-ARCTIC speakers: {len(l2)}\n")

    ann = count_annotations(a.l2_arctic)
    report: Dict[str, object] = {
        "cmu_prompt_count": len(cmu),
        "l2_speakers": len(l2),
        "annotated_utts_per_speaker": ann,
        "per_speaker": {},
    }

    print(f"{'SPK':<7}{'L1':<12}{'utts':>6}{'shared':>8}{'exact':>8}"
          f"{'textdiff':>10}{'ann':>6}")
    print("-" * 60)
    all_shared, all_exact = 0, 0
    diffs_shown = 0
    for spk, d in l2.items():
        shared = [u for u in d if u in cmu]
        exact = [u for u in shared if norm(d[u]) == norm(cmu[u])]
        difftext = [u for u in shared if u not in set(exact)]
        all_shared += len(shared)
        all_exact += len(exact)
        report["per_speaker"][spk] = {
            "l1": L1_OF_SPEAKER.get(spk, "?"),
            "n_utts": len(d),
            "n_shared_ids": len(shared),
            "n_exact_text": len(exact),
            "n_text_differs": len(difftext),
            "text_diff_ids": difftext[:50],
            "n_annotated": ann.get(spk, 0),
        }
        print(f"{spk:<7}{L1_OF_SPEAKER.get(spk,'?'):<12}{len(d):>6}"
              f"{len(shared):>8}{len(exact):>8}{len(difftext):>10}"
              f"{ann.get(spk,0):>6}")
        for u in difftext[:max(0, a.show_diffs - diffs_shown)]:
            print(f"        {u}\n          L2 : {d[u]}\n          CMU: {cmu[u]}")
            diffs_shown += 1

    frac = all_exact / max(1, all_shared)
    report["overall"] = {
        "shared_ids": all_shared,
        "exact_text": all_exact,
        "exact_fraction": round(frac, 5),
    }
    print("-" * 60)
    print(f"shared IDs {all_shared}, textually identical {all_exact} "
          f"({100*frac:.2f}%)")

    # verdict
    if frac > 0.99:
        v = ("PASS: the parallel-prompt assumption holds. Use CMU ARCTIC "
             "renditions as native references and prompts as WER ground truth.")
    elif frac > 0.90:
        v = (f"PARTIAL: {100*(1-frac):.1f}% of shared IDs have different text. "
             "Filter those out and state the filter in the paper -- do NOT use "
             "the prompt as a WER reference for them.")
    else:
        v = ("FAIL: the corpora are not reliably parallel. Re-plan: either "
             "transcribe the test set, or use a reference-free objective.")
    print("\n" + v)
    report["verdict"] = v

    for l1 in MATRIX_L1:
        spks = [s for s, m in report["per_speaker"].items() if m["l1"] == l1]
        n_ann = sum(report["per_speaker"][s]["n_annotated"] for s in spks)
        print(f"  {l1:<10} speakers={len(spks)} annotated_utts={n_ann}")
        if n_ann < 300:
            print(f"    WARNING: {n_ann} annotated utterances may underpower "
                  f"the per-phoneme-class H2 test for {l1}.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
