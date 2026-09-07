"""H2 in the audio domain: does acoustic weighting rescue the manner grouping?

The gap this closes
-------------------
`analyse_h2_sequences.py` refuted H2 at the sequence level: every manner class
improves 52--72% relative with lookahead, and the pre-registered grouping does
not order the data. But phone error rate charges every error exactly 1.0,
whatever it sounds like. A substituted vowel and a substituted stop cost the
same in PER and obviously do not cost the same acoustically. So the sequence
result leaves one escape open, and `paper/main.tex` names it as remaining work
item (i): is the grouping wrong, or is PER simply blind to how much each error
costs when you listen?

This module answers the first half of that: **re-run the H2 test with each error
weighted by its acoustic consequence instead of by 1.0.** If the class ordering
survives that reweighting, "PER mis-weights the classes" is dead as an
explanation and the grouping really is wrong.

STATUS: THIS METHOD DOES NOT WORK. READ THIS BEFORE USING IT.
-------------------------------------------------------------
Two measurements, both made before any result was reported, show the
counterfactual design below is invalid **with a VITS-class synthesiser**:

1. **Piper is stochastic by default.** `noise_scale=0.667`, `noise_w=0.8`. The
   same phone string synthesised twice differs in length (41472 vs 39424
   samples on the first utterance tested) and scores a self-distortion of
   **244.8**, against **277.6** for a genuine one-phone change. Roughly 88% of
   the apparent "cost of an error" was synthesis noise. Setting noise_scale and
   noise_w to 0 makes synthesis bitwise deterministic (self-distortion exactly
   0.0000), and `_assert_deterministic` below now enforces that before any
   measurement runs.

2. **Even deterministic, VITS is globally coupled, and that is fatal here.**
   Changing one phone perturbs the whole utterance rather than its own region:
   across three probes the worst 10% of frames carried only 16.5-23.4% of the
   total distance (uniform would be 10%). Every single-phone substitution costs
   a near-constant floor of ~210 whatever the phone, with only ~208-282 of
   spread on top. Two consequences, the second worse than the first:
     * the local phonetic difference is swamped by global re-synthesis ripple;
     * with a near-constant cost per error, acoustic_cost_rate collapses to
       (constant x error_rate), so the "audio-weighted" H2 test would have
       reproduced the sequence-level test **by construction** and looked like
       independent confirmation. That is the most dangerous possible failure
       mode for this particular question.

   Correlation with duration change is only +0.28, so this is not merely a
   duration artefact -- it is genuine global coupling in the vocoder.

The honest form of item (i) therefore needs a **frame-level forced aligner and
real converted audio**, not TTS counterfactuals: align synthesised output
against a native rendition and read per-phone distortion off the alignment.
That needs a phoneme-CTC acoustic model, which needs network access this
sandbox does not have.

What is retained below: the distortion measure, the determinism guard, the
counterfactual construction and its self-tests -- all correct and reusable --
plus this record, so the next person does not rediscover the dead end.

Method: counterfactual single-error attribution
-----------------------------------------------
For each utterance we synthesise the canonical (`g2p`) phone string once, then
for every error the model made we rebuild that same string with **only that one
error applied**, synthesise it, and measure the distortion against the clean
reference. The erroneous phone therefore sits in its correct phonetic context,
and the measured distortion is attributable to that error alone rather than to
a soup of co-occurring ones.

Errors are charged to the class of the *reference* phone, exactly as
`analyse_h2_sequences` does. Insertions have no reference phone and so have no
class; they are counted and reported separately rather than silently dropped.

What this does NOT measure, stated plainly
------------------------------------------
Both signals come from the same TTS driven by phone strings, so the synthesiser
renders canonical formant trajectories for whatever string it is given. This
measures **the acoustic cost of getting a phone's identity wrong**. It does not
measure whether a converter produces a wrong formant *trajectory* for a phone
whose identity is right -- that needs real converted audio aligned against a
native rendition, which is the full form of item (i). A negative result here
therefore closes the "PER mis-weights classes" escape but not the "PER is blind
to trajectories" one. Do not report it as though it closed both.

Distortion measure
------------------
MCD-style mel-cepstral distortion over MFCCs with c0 excluded, DTW-aligned so
the two utterances need not share a length:

    MCD = (10*sqrt(2)/ln 10) * mean_over_aligned_frames( ||c_ref - c_hyp|| )

This is the MFCC-based approximation, not a WORLD/pysptk MCEP implementation;
it is labelled that way everywhere it is reported.

    python3 eval/h2_audio_domain.py --self-test
    python3 eval/h2_audio_domain.py --hyp-dir results/raw/hyps --n-utts 40
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "synth"))

from analyse_h2_sequences import (  # noqa: E402  reuse, do not re-derive
    H2_PREDICTION,
    align,
    ipa_classes,
)

MCD_CONST = 10.0 * math.sqrt(2.0) / math.log(10.0)   # ~6.1418


# ==========================================================================
# Distortion
# ==========================================================================

def mfcc(wav: np.ndarray, sr: int, n_mfcc: int = 25) -> np.ndarray:
    """(frames, n_mfcc-1) MFCCs with c0 dropped.

    c0 is overall energy. Keeping it would let a loudness difference masquerade
    as a spectral one, and the synthesiser's per-utterance peak normalisation
    makes that difference arbitrary.
    """
    import librosa
    m = librosa.feature.mfcc(y=wav.astype(np.float32), sr=sr, n_mfcc=n_mfcc,
                             hop_length=int(0.005 * sr), n_fft=int(0.025 * sr))
    return m[1:].T


def mcd_dtw(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """MCD-style distortion between two waveforms, DTW-aligned."""
    import librosa
    ca, cb = mfcc(a, sr), mfcc(b, sr)
    if len(ca) == 0 or len(cb) == 0:
        return float("nan")
    # librosa wants (features, frames)
    _, path = librosa.sequence.dtw(X=ca.T, Y=cb.T, metric="euclidean")
    d = [np.linalg.norm(ca[i] - cb[j]) for i, j in path]
    return float(MCD_CONST * np.mean(d))



def _assert_deterministic(synth, phones: Sequence[str]) -> None:
    """Refuse to measure with a stochastic synthesiser.

    Piper/VITS defaults to noise_scale=0.667 and noise_w=0.8, which makes two
    renderings of the same phone string differ by more than a real one-phone
    change does. Any TTS-counterfactual measurement built on that is noise.
    """
    a1, _ = synth.synth(list(phones))
    a2, _ = synth.synth(list(phones))
    if len(a1) != len(a2) or not np.array_equal(a1, a2):
        raise SystemExit(
            "synthesiser is not deterministic: the same phone string produced "
            "two different waveforms. Set noise_scale and noise_w to 0 "
            "(synth.scales[0] and synth.scales[2]) before measuring.")


def make_deterministic(synth):
    """Zero the two stochastic scales, leaving length_scale untouched."""
    synth.scales = np.array([0.0, float(synth.scales[1]), 0.0], dtype=np.float32)
    return synth


# ==========================================================================
# Counterfactual construction
# ==========================================================================

def single_error_sequences(ref: Sequence[str], hyp: Sequence[str]
                           ) -> List[Tuple[int, str, str, List[str]]]:
    """One perturbed copy of `ref` per error, each containing that error alone.

    Returns (ref_index, op, ref_phone, perturbed_sequence). Insertions are
    excluded: they have no reference phone, so no manner class can be charged
    for them, and inventing one would be the same error this project already
    documented once (bucketing unmatched symbols and reporting anyway).
    """
    out: List[Tuple[int, str, str, List[str]]] = []
    ri = 0
    for op, rp, hp in align(list(ref), list(hyp)):
        if op == "ins":
            continue                      # no reference phone to charge
        if op == "sub":
            pert = list(ref)
            pert[ri] = hp
            out.append((ri, op, rp, pert))
        elif op == "del":
            pert = list(ref)
            del pert[ri]
            out.append((ri, op, rp, pert))
        ri += 1
    return out


def count_insertions(ref: Sequence[str], hyp: Sequence[str]) -> int:
    return sum(1 for op, _, _ in align(list(ref), list(hyp)) if op == "ins")


# ==========================================================================
# Per-utterance measurement
# ==========================================================================

def measure_utterance(synth, ref: Sequence[str], hyp: Sequence[str],
                      sr: int, max_errors: Optional[int] = None
                      ) -> Dict[str, object]:
    """Acoustic cost of each of this utterance's errors, charged to its class.

    Also returns the whole-hypothesis distortion so the additivity of the
    single-error costs can be checked rather than assumed.
    """
    classes = ipa_classes(list(ref))
    ref_wav, _ = synth.synth(list(ref))

    per_class_costs: Dict[str, List[float]] = defaultdict(list)
    errs = single_error_sequences(ref, hyp)
    if max_errors is not None:
        errs = errs[:max_errors]

    for ri, op, rp, pert in errs:
        if not pert:
            continue
        try:
            w, _ = synth.synth(pert)
        except ValueError:
            continue                      # nothing survived the phone mapping
        cost = mcd_dtw(ref_wav, w, sr)
        if not math.isnan(cost):
            per_class_costs[classes[ri]].append(cost)

    try:
        hyp_wav, _ = synth.synth(list(hyp))
        full = mcd_dtw(ref_wav, hyp_wav, sr)
    except ValueError:
        full = float("nan")

    return {
        "per_class_costs": {k: v for k, v in per_class_costs.items()},
        "full_hyp_mcd": full,
        "n_errors_measured": sum(len(v) for v in per_class_costs.values()),
        "n_insertions": count_insertions(ref, hyp),
    }


# ==========================================================================
# Self-tests
# ==========================================================================

def _self_test() -> None:
    print("h2_audio_domain self-test")
    ok_all = True

    def check(name: str, cond: bool, extra: str = "") -> None:
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  {name:<52s} {'ok' if cond else 'FAIL'} {extra}")

    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    a = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    b = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    c = np.sin(2 * np.pi * 880 * t).astype(np.float32)

    d_aa = mcd_dtw(a, a, sr)
    check("identical signals -> ~0 distortion", d_aa < 1e-6, f"({d_aa:.2e})")

    d_ab, d_ba = mcd_dtw(a, b, sr), mcd_dtw(b, a, sr)
    check("symmetric", abs(d_ab - d_ba) < 1e-6, f"({d_ab:.3f})")

    d_ac = mcd_dtw(a, c, sr)
    check("further apart -> larger distortion", d_ac > d_ab,
          f"(220v880={d_ac:.2f} > 220v440={d_ab:.2f})")

    short = a[: sr // 2]
    check("handles unequal lengths", not math.isnan(mcd_dtw(a, short, sr)))

    # --- counterfactual construction ---
    ref = ["k", "æ", "t"]
    subs = single_error_sequences(ref, ["k", "ɪ", "t"])
    check("substitution -> one perturbed seq, one position differs",
          len(subs) == 1 and subs[0][3] == ["k", "ɪ", "t"] and subs[0][1] == "sub")

    dels = single_error_sequences(ref, ["k", "t"])
    check("deletion -> sequence one shorter",
          len(dels) == 1 and len(dels[0][3]) == 2, f"({dels[0][3]})")

    ins = single_error_sequences(ref, ["k", "æ", "s", "t"])
    n_ins = count_insertions(ref, ["k", "æ", "s", "t"])
    check("insertions excluded from attribution but counted",
          all(op != "ins" for _, op, _, _ in ins) and n_ins == 1, f"(n_ins={n_ins})")

    # each perturbation must contain exactly ONE error against the reference
    multi = single_error_sequences(["k", "æ", "t"], ["p", "ɪ", "t"])
    # count error ops positively: align() labels a match "ok", and negating a
    # guessed label is how you get a test that passes for the wrong reason.
    each_one = all(
        sum(1 for op, _, _ in align(["k", "æ", "t"], p)
            if op in ("sub", "del", "ins")) == 1
        for _, _, _, p in multi)
    check("every perturbation contains exactly one error", each_one,
          f"({len(multi)} errors isolated)")

    # --- the guard this project learned the hard way ---
    alphabet = ["a", "b", "d", "e", "f", "h", "i", "j", "k", "l", "m", "n", "o",
                "p", "s", "t", "u", "v", "w", "z", "æ", "ð", "ŋ", "ɑ", "ɔ", "ɛ",
                "ɝ", "ɡ", "ɪ", "ɹ", "ʃ", "ʊ", "ʌ", "ʒ", "θ"]
    cls = ipa_classes(alphabet)
    unmapped = [p for p, k in zip(alphabet, cls) if k == "other"]
    check("no sweep phone falls through to 'other'", not unmapped, f"({unmapped})")

    declared = set(H2_PREDICTION["breaks_first"]) | set(
        H2_PREDICTION["survives_short_lookahead"])
    reachable = set(ipa_classes(alphabet)) | {"vowel_diph", "affricate"}
    check("all pre-registered classes reachable",
          declared <= reachable, f"(missing {declared - reachable})")

    # --- the guard that would have caught the stochastic-synthesis trap ---
    class _Stochastic:
        scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
        def synth(self, ph):
            return np.random.RandomState().randn(1000).astype(np.float32), {}
    class _Fixed:
        scales = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        def synth(self, ph):
            return np.ones(1000, dtype=np.float32), {}
    caught = False
    try:
        _assert_deterministic(_Stochastic(), ["k"])
    except SystemExit:
        caught = True
    check("stochastic synthesiser is refused", caught)
    passed = True
    try:
        _assert_deterministic(_Fixed(), ["k"])
    except SystemExit:
        passed = False
    check("deterministic synthesiser is accepted", passed)
    check("make_deterministic zeroes noise, keeps length_scale",
          list(make_deterministic(_Stochastic()).scales) == [0.0, 1.0, 0.0])

    print("\nALL PASS" if ok_all else "\nFAILURES ABOVE")
    if not ok_all:
        sys.exit(1)


# ==========================================================================
# Driver
# ==========================================================================

def _utt_key(h: dict) -> Tuple[str, Tuple[str, ...]]:
    """Stable identity for an utterance across conditions.

    `text` is empty in the dumps, so the canonical phone string plus the
    speaker is what identifies an utterance. g2p does not vary with lookahead,
    which is exactly the property needed here.
    """
    return (str(h.get("speaker")), tuple(h.get("g2p") or []))


def run(a) -> None:
    from analyse_h2_sequences import h2_from_rates, load_hyps
    from phones_to_audio import PiperPhonemeSynth

    by_L, prov = load_hyps(a.hyp_dir, a.target)
    Ls = sorted(by_L)

    # One entry per (L, utterance): the audio pass costs a synthesis per error,
    # so pooling three seeds would triple it for a quantity -- the acoustic cost
    # of a class of error -- that is a property of the phones, not the seed.
    # The sequence-level control below is computed on this same subset, so the
    # two metrics differ only in weighting.
    picked: Dict[float, Dict[Tuple[str, Tuple[str, ...]], dict]] = {}
    for L in Ls:
        best: Dict[Tuple[str, Tuple[str, ...]], dict] = {}
        for h in by_L[L]:
            k = _utt_key(h)
            if k not in best or str(h.get("seed")) < str(best[k].get("seed")):
                best[k] = h
        picked[L] = best

    common = set.intersection(*(set(picked[L]) for L in Ls))
    keys = sorted(common)[: a.n_utts]
    if not keys:
        raise SystemExit("no utterances common to every lookahead")
    print(f"{len(common)} utterances common to all {len(Ls)} lookaheads; "
          f"using {len(keys)}")

    synth = make_deterministic(PiperPhonemeSynth())
    _assert_deterministic(synth, list(picked[Ls[0]][keys[0]]["g2p"]))
    sr = synth.sr

    per_L: Dict[float, Dict[str, Dict[str, float]]] = {}
    detail: Dict[str, object] = {}
    for L in Ls:
        opp: Dict[str, int] = defaultdict(int)
        nerr: Dict[str, int] = defaultdict(int)
        cost: Dict[str, float] = defaultdict(float)
        costs_each: Dict[str, List[float]] = defaultdict(list)
        full_mcds: List[float] = []
        ins_total = 0

        for k in keys:
            h = picked[L][k]
            ref, hyp = list(h["g2p"]), list(h["pred"])
            for c in ipa_classes(ref):
                opp[c] += 1
            m = measure_utterance(synth, ref, hyp, sr, max_errors=a.max_errors)
            ins_total += int(m["n_insertions"])
            if not math.isnan(float(m["full_hyp_mcd"])):
                full_mcds.append(float(m["full_hyp_mcd"]))
            for c, vals in m["per_class_costs"].items():
                nerr[c] += len(vals)
                cost[c] += float(sum(vals))
                costs_each[c].extend(vals)

        per_L[L] = {
            c: {
                # sequence-level weighting: every error counts 1.0
                "error_rate": nerr[c] / opp[c] if opp[c] else float("nan"),
                # audio weighting: every error counts its own MCD
                "acoustic_cost_rate": cost[c] / opp[c] if opp[c] else float("nan"),
                "mean_cost_per_error": (cost[c] / nerr[c]) if nerr[c] else float("nan"),
                "n_errors_measured": float(nerr[c]),
                "opportunities": float(opp[c]),
            }
            for c in sorted(opp)
        }
        detail[str(L)] = {
            "insertions_total": ins_total,
            "mean_full_hyp_mcd": (sum(full_mcds) / len(full_mcds)) if full_mcds else None,
            "sum_of_single_error_costs": sum(cost.values()),
        }
        print(f"  L={L:>5.0f} ms  errors measured={sum(nerr.values()):>4d}  "
              f"mean cost/error="
              f"{(sum(cost.values())/max(1,sum(nerr.values()))):.3f}")

    verdict_seq = h2_from_rates(per_L, metric="error_rate")
    verdict_audio = h2_from_rates(per_L, metric="acoustic_cost_rate")

    out = {
        "provenance": {
            **prov,
            "n_utts_used": len(keys),
            "max_errors_per_utt": a.max_errors,
            "one_entry_per_utterance_per_L": True,
            "distortion": "MFCC-based MCD-style (c0 excluded, DTW-aligned); "
                          "not a WORLD/pysptk MCEP implementation",
            "caveat": "Both signals are TTS renderings of phone strings, so this "
                      "measures the acoustic cost of phone-IDENTITY errors. It "
                      "does not measure wrong formant trajectories for a "
                      "correctly-identified phone.",
        },
        "per_lookahead": {str(k): v for k, v in per_L.items()},
        "detail": detail,
        "verdict_sequence_weighting": verdict_seq,
        "verdict_acoustic_weighting": verdict_audio,
        "comparison": {
            "H2_supported_sequence": verdict_seq.get("H2_supported"),
            "H2_supported_acoustic": verdict_audio.get("H2_supported"),
            "effect_size_sequence": verdict_seq.get("effect_size_between_over_pooled_sd"),
            "effect_size_acoustic": verdict_audio.get("effect_size_between_over_pooled_sd"),
            "reading": "If the acoustic verdict matches the sequence verdict, "
                       "reweighting each error by how much it actually changes "
                       "the signal does not rescue the manner grouping, and "
                       "'PER mis-weights the classes' is no longer an available "
                       "explanation for the sequence-level refutation.",
        },
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}")
    print(f"  H2 (sequence weighting): supported={verdict_seq.get('H2_supported')}")
    print(f"  H2 (acoustic weighting): supported={verdict_audio.get('H2_supported')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", default="results/raw/hyps")
    ap.add_argument("--target", default="native")
    ap.add_argument("--n-utts", type=int, default=40)
    ap.add_argument("--max-errors", type=int, default=12,
                    help="cap errors measured per utterance; keeps cost bounded "
                         "and is applied identically at every lookahead")
    ap.add_argument("--out", default="results/analysis_h2_audio.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        _self_test()
        return
    run(a)


if __name__ == "__main__":
    main()
