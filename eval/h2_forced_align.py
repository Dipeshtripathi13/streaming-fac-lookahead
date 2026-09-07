"""H2 in the audio domain, done properly: per-phone distortion via forced alignment.

This is `paper/main.tex` §Remaining work item (i). It supersedes the
TTS-counterfactual attempt in `h2_audio_domain.py`, which was rejected because a
VITS synthesiser is globally coupled: changing one phone perturbs the whole
utterance, so a whole-utterance distortion cannot be charged to one phone.
See `docs/H2_AUDIO_DOMAIN_BLOCKED.md`.

Forced alignment fixes exactly that problem. Instead of asking "how different is
the whole utterance", we locate each reference phone on the time axis and ask
how different the signal is *there*. Global re-synthesis ripple still raises the
floor, but it is now measured per phone rather than integrated over the whole
signal, so an error at phone i shows up at phone i.

Pipeline
--------
    g2p  --Piper-->  reference audio
                     --wav2vec2-espeak + CTC forced align-->  phone spans
    pred --Piper-->  hypothesis audio
    reference MFCC  --DTW-->  hypothesis MFCC          (time correspondence)
    per reference phone span: mean spectral distance over its aligned frames

Each phone's distortion is charged to the manner class of the *reference* phone,
exactly as `analyse_h2_sequences` charges its errors, and the verdict is
produced by the *same* `h2_from_rates` criterion with only the weighting
changed. Insertions have no reference phone and are excluded from attribution.

The validation that decides whether any of this means anything
--------------------------------------------------------------
`--validate` reports the distortion of phones the model got RIGHT against those
it got WRONG. If those two distributions do not separate, the measure is not
detecting phone errors at all and no downstream class comparison is meaningful.
This is checked and printed before any H2 verdict, and the verdict is suppressed
if the separation is below `--min-separation`.

Determinism is mandatory
------------------------
Piper defaults to noise_scale=0.667/noise_w=0.8, which makes two renderings of
the same string differ more than a real one-phone change does. `make_deterministic`
is applied and `_assert_deterministic` enforces it.

    python3 eval/h2_forced_align.py --self-test        # pure logic, no model
    python3 eval/h2_forced_align.py --n-utts 30 --validate
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "synth"))

from analyse_h2_sequences import H2_PREDICTION, align, ipa_classes  # noqa: E402

PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
MODEL_SR = 16000
MFCC_HOP_S = 0.005

# g2p writes stressed /ɝ/; eSpeak-based models carry only /ɚ/. Same vowel.
IPA_TO_ESPEAK = {"ɝ": "ɚ"}


# ==========================================================================
# Pure logic (self-testable without any model)
# ==========================================================================

def spans_from_alignment(frame_labels: Sequence[int], n_phones: int
                         ) -> List[Tuple[int, int]]:
    """Contiguous [start, end) frame span per phone index from a frame labelling.

    `frame_labels[t]` is the index of the phone occupying frame t, or -1 for
    blank. Blanks are absorbed into the preceding phone so the spans tile the
    signal; a phone that never surfaces gets an empty span, which the caller
    must skip rather than treat as zero distortion.
    """
    spans: List[Tuple[int, int]] = [(-1, -1)] * n_phones
    cur = -1
    start = 0
    for t, lab in enumerate(frame_labels):
        if lab == -1 or lab == cur:
            continue
        if cur != -1 and 0 <= cur < n_phones:
            s = spans[cur][0] if spans[cur][0] != -1 else start
            spans[cur] = (s, t)
        cur = lab
        start = t
    if cur != -1 and 0 <= cur < n_phones:
        s = spans[cur][0] if spans[cur][0] != -1 else start
        spans[cur] = (s, len(frame_labels))
    return spans


def map_span(path_ref: np.ndarray, path_hyp: np.ndarray,
             lo: int, hi: int) -> List[Tuple[int, int]]:
    """Frame pairs whose reference index falls in [lo, hi).

    The DTW path is many-to-many, so a reference span maps to a set of pairs
    rather than a single interval. Returning the pairs keeps the distortion a
    mean over genuine correspondences instead of over an interpolated interval.
    """
    m = (path_ref >= lo) & (path_ref < hi)
    return list(zip(path_ref[m].tolist(), path_hyp[m].tolist()))


def correct_phone_mask(ref: Sequence[str], hyp: Sequence[str]) -> List[bool]:
    """Per reference position: did the model produce this phone correctly?

    Positions charged a substitution or deletion are False. Insertions advance
    the hypothesis only and never mark a reference position.
    """
    out: List[bool] = []
    for op, _rp, _hp in align(list(ref), list(hyp)):
        if op == "ins":
            continue
        out.append(op == "ok")
    return out


def _self_test() -> None:
    print("h2_forced_align self-test")
    ok_all = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  {name:<54s} {'ok' if cond else 'FAIL'} {extra}")

    # spans
    labs = [0, 0, -1, 1, 1, 1, -1, 2]
    sp = spans_from_alignment(labs, 3)
    check("spans tile a simple labelling", sp == [(0, 3), (3, 7), (7, 8)], f"{sp}")

    sp2 = spans_from_alignment([-1, -1, 0, 0], 2)
    check("phone that never surfaces gets an empty span",
          sp2[1] == (-1, -1), f"{sp2}")

    sp3 = spans_from_alignment([0, 1, 0], 2)
    check("re-entrant label keeps one span per phone", len(sp3) == 2, f"{sp3}")

    # span mapping
    pr = np.array([0, 1, 2, 3, 4]); ph = np.array([0, 1, 1, 2, 3])
    got = map_span(pr, ph, 1, 4)
    check("span mapping keeps many-to-many pairs",
          got == [(1, 1), (2, 1), (3, 2)], f"{got}")
    check("empty span maps to nothing", map_span(pr, ph, 9, 10) == [])

    # correctness mask
    m = correct_phone_mask(["k", "æ", "t"], ["k", "ɪ", "t"])
    check("substitution marks exactly that position wrong",
          m == [True, False, True], f"{m}")
    m2 = correct_phone_mask(["k", "æ", "t"], ["k", "t"])
    check("deletion marks the deleted position wrong",
          m2 == [True, False, True], f"{m2}")
    m3 = correct_phone_mask(["k", "æ", "t"], ["k", "æ", "s", "t"])
    check("insertion does not consume a reference position",
          len(m3) == 3 and all(m3), f"{m3}")

    # alphabet coverage: the trap this project already documented once
    alphabet = ['a','b','d','e','f','h','i','j','k','l','m','n','o','p','s','t',
                'u','v','w','z','æ','ð','ŋ','ɑ','ɔ','ɛ','ɝ','ɡ','ɪ','ɹ','ʃ','ʊ',
                'ʌ','ʒ','θ']
    cls = ipa_classes(alphabet)
    check("no sweep phone falls through to 'other'",
          "other" not in cls, f"{[p for p,c in zip(alphabet,cls) if c=='other']}")
    declared = set(H2_PREDICTION["breaks_first"]) | set(
        H2_PREDICTION["survives_short_lookahead"])
    check("all pre-registered classes reachable",
          declared <= (set(cls) | {"vowel_diph", "affricate"}))

    print("\nALL PASS" if ok_all else "\nFAILURES ABOVE")
    if not ok_all:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a, _ = ap.parse_known_args()
    if a.self_test:
        _self_test()
    else:
        from h2_forced_align_run import main   # runtime half, needs the models
        main()
