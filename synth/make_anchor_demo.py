"""Render ceiling/floor anchor pairs so the perceptual question can be *heard*.

What this answers, and what it does not
---------------------------------------
The listening test in `eval/listening_test.py` rests on an assumption nobody has
checked: that rendering two phone sequences -- canonical and produced -- through
**one fixed Piper voice** yields stimuli a listener can actually tell apart as
*accent*. If that fails, the whole perceptual arm fails regardless of which
model produced the phones, and it is worth knowing before recruiting raters.

This script is that check. It is a **pipeline and audibility validation**, not
an experiment:

  * The phone pairs below are **literature-documented L2 substitution patterns**
    for the six L1s in L2-ARCTIC. They are NOT L2-ARCTIC's `g2p`/`ipa` labels.
    The real stimuli must come from the corpus; these only establish that the
    contrast survives synthesis.
  * One fixed voice deliberately discards speaker identity and prosody, so this
    isolates *segmental* accentedness and nothing else. Real accentedness is
    substantially prosodic, which is a limitation of the listening test itself,
    not of this script.

Read the printed PER as a sanity check only. The deliverable is the audio; the
question is whether you can hear it.

Substitution sources: standard L2 phonology descriptions of Hindi, Mandarin,
Spanish, Arabic, Vietnamese and Korean accented English (TH-stopping, /v/-/w/
confusion, final-cluster reduction, /r/-/l/ confusion, /p/-/b/ and /f/-/v/
merger, epenthesis). Each pattern below is annotated with the process it encodes.

Usage
-----
    python3 synth/make_anchor_demo.py
    python3 synth/make_anchor_demo.py --l1 mandarin --outdir results/synth/demo
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from phones_to_audio import PiperPhonemeSynth, write_wav  # noqa: E402

# Sentence 1: "the birch canoe slid on the smooth planks" (CMU ARCTIC a0001,
# which L2-ARCTIC also uses, so the prompt is representative).
CANON_1 = ("DH AH0 B ER1 CH K AH0 N UW1 S L IH1 D AA1 N DH AH0 "
           "S M UW1 DH P L AE1 NG K S").split()
# Sentence 2: "she had your dark suit in greasy wash water" (a0002).
CANON_2 = ("SH IY1 HH AE1 D Y AO1 R D AA1 R K S UW1 T IH0 N "
           "G R IY1 S IY0 W AA1 SH W AO1 T ER0").split()

# Each entry: (L1, sentence id, canonical, produced, processes encoded)
CASES = [
    ("hindi", 1, CANON_1,
     ("D AH0 B ER1 CH K AH0 N UW1 S L IH1 D AA1 N D AH0 "
      "S M UW1 D P L AE1 NG K S").split(),
     "TH-stopping: /dh/ -> retroflex [d] in both 'the' and 'smooth'"),
    ("mandarin", 1, CANON_1,
     ("Z AH0 B ER1 CH K AH0 N UW1 S L IH1 D AA1 N Z AH0 "
      "S M UW1 Z P L AE1 NG").split(),
     "/dh/ -> [z]; final cluster /nks/ reduced to [ng]"),
    ("vietnamese", 1, CANON_1,
     ("D AH0 B ER1 T K AH0 N UW1 S L IH1 D AA1 N D AH0 "
      "S M UW1 D P L AE1 N").split(),
     "TH-stopping; final affricate and final cluster deletion"),
    ("korean", 2, CANON_2,
     ("S IY1 HH AE1 D Y AO1 L D AA1 L K S UW1 T IH0 N "
      "G L IY1 S IY0 W AA1 S W AO1 T ER0").split(),
     "/r/ -> [l]; /sh/ -> [s] in 'she' and 'wash'"),
    ("spanish", 2, CANON_2,
     ("CH IY1 HH AE1 D Y AO1 R D AA1 R K EH0 S UW1 T IH0 N "
      "G R IY1 S IY0 B AA1 CH B AO1 T ER0").split(),
     "/sh/ -> [ch]; epenthetic [e] before /s/+C; /w/ -> [b]"),
    ("arabic", 2, CANON_2,
     ("SH IY1 HH AE1 D Y AO1 R D AA1 R K S UW1 T IH0 N "
      "G R IY1 S IY0 W AA1 SH W AO1 D ER0").split(),
     "flapped /t/ -> [d]; retained elsewhere (mild case, on purpose)"),
]


def per(ref, hyp) -> float:
    """Levenshtein / len(ref) over phone strings -- the same metric the
    translator is scored with, so the numbers are comparable to §5.2."""
    r = [p for p in ref]
    h = [p for p in hyp]
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1][-1] / max(1, len(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", default=None, help="render only this L1")
    ap.add_argument("--outdir", default="results/synth/anchors")
    ap.add_argument("--voice", default=None)
    a = ap.parse_args()

    kw = {"voice_dir": a.voice} if a.voice else {}
    s = PiperPhonemeSynth(**kw)
    print(f"voice sr={s.sr}, phoneme_id_map={len(s.pid)} entries\n")
    print("NOTE: substitutions are literature-documented patterns, NOT "
          "L2-ARCTIC labels.\n      This validates synthesis and audibility "
          "only.\n")
    print(f"{'L1':<12} {'sent':>4} {'PER':>6} {'ceil s':>7} {'floor s':>8} "
          f"{'drop':>5}  process")
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for l1, sid, canon, prod, why in CASES:
        if a.l1 and l1 != a.l1:
            continue
        wc, mc = s.synth(canon)
        wp, mp = s.synth(prod)
        p = per(canon, prod)
        write_wav(os.path.join(a.outdir, f"{l1}_s{sid}_ceiling.wav"), wc, s.sr)
        write_wav(os.path.join(a.outdir, f"{l1}_s{sid}_floor.wav"), wp, s.sr)
        drop = max(mc["n_dropped"], mp["n_dropped"])
        rows.append((l1, p, drop))
        print(f"{l1:<12} {sid:>4} {p:>6.3f} {mc['duration_s']:>7.2f} "
              f"{mp['duration_s']:>8.2f} {drop:>5}  {why}")

    print(f"\nwrote {2 * len(rows)} wavs to {a.outdir}/")
    mean_per = sum(r[1] for r in rows) / len(rows)
    print(f"mean ceiling-vs-floor PER {mean_per:.3f}")
    # The corpus figure is the bar to clear: if these demo pairs differ far less
    # than the real targets do, the demo understates the task.
    print(f"L2-ARCTIC g2p-vs-ipa PER is 0.175, so these pairs are "
          f"{'comparable to' if abs(mean_per - .175) < .09 else 'NOT comparable to'}"
          f" the real contrast.")
    if any(r[2] for r in rows):
        print(f"WARNING dropped phones: {s.unmapped}")
    print("""
Next: listen to a ceiling/floor pair. If you cannot hear a difference, the
listening test cannot work in this form and the perceptual arm needs a
different design (speaker-preserving conversion, or prosody included) before
spending money on raters.""")


if __name__ == "__main__":
    main()
