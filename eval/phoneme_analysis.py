"""RQ2 / H2: which sounds break first under latency pressure?

Hypothesis H2 says degradation is not uniform: consonant substitutions
(th-stopping, /v/-/w/, retroflex stops) are coarticulatorily local and should
survive short lookahead, while vowel quality and rhythm depend on longer
context and should break first.

To test that you need a per-phoneme error attribution, not an utterance-level
MOS. The machinery here:

  1. Force-align the *converted* audio against the known prompt text.
  2. Force-align the *reference* native rendition of the same prompt.
     (L2-ARCTIC and CMU ARCTIC share all 1132 prompts, so a native rendition
     of every test utterance exists -- that is the whole reason this dataset
     pair was chosen.)
  3. For each phone, compute a local distortion score and bucket by class.
  4. Regress distortion on lookahead, per class, and test whether the slopes
     differ (the actual statistical form of H2).

L2-ARCTIC additionally ships human phoneme-level mispronunciation
annotations for a subset -- substitution / deletion / addition tags. Those
give a *ground-truth* accent-error inventory per utterance, which lets us ask
the sharper question: does short lookahead specifically fail to correct the
annotated errors, or does it damage phones that were already correct?
Those are very different failure modes and only the first is an accent-
conversion failure; the second is a vocoder failure.

Alignment backend: Montreal Forced Aligner if available, else the
charsiu/w2v2 frame classifier, else a simple DTW over MFCCs against the
reference (weakest, but dependency-free).
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Phoneme classes (ARPAbet, stress-stripped)
# --------------------------------------------------------------------------

VOWELS_MONO = {"IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "ER", "AX"}
VOWELS_DIPH = {"EY", "AY", "OW", "AW", "OY"}
STOPS = {"P", "B", "T", "D", "K", "G"}
FRICATIVES = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
AFFRICATES = {"CH", "JH"}
NASALS = {"M", "N", "NG"}
APPROXIMANTS = {"L", "R", "W", "Y"}

CLASS_OF: Dict[str, str] = {}
for s, name in [(VOWELS_MONO, "vowel_mono"), (VOWELS_DIPH, "vowel_diph"),
                (STOPS, "stop"), (FRICATIVES, "fricative"),
                (AFFRICATES, "affricate"), (NASALS, "nasal"),
                (APPROXIMANTS, "approximant")]:
    for p in s:
        CLASS_OF[p] = name

# The specific L1-transfer phenomena named in H2. These are the phones where
# a *targeted* claim can be made, rather than a broad class-level one.
TARGET_PHENOMENA = {
    "th_stopping":      {"L1": ["Hindi", "Mandarin", "Arabic"], "phones": ["TH", "DH"]},
    "v_w_merger":       {"L1": ["Hindi"],                       "phones": ["V", "W"]},
    "retroflex_stops":  {"L1": ["Hindi"],                       "phones": ["T", "D"]},
    "final_devoicing":  {"L1": ["Mandarin", "Spanish"],         "phones": ["B", "D", "G", "Z", "V"]},
    "vowel_epenthesis": {"L1": ["Spanish", "Arabic"],           "phones": ["AH", "IH"]},
    "tense_lax_merger": {"L1": ["Spanish", "Mandarin", "Arabic"], "phones": ["IY", "IH", "UW", "UH"]},
    "rhotic_quality":   {"L1": ["Mandarin", "Arabic"],          "phones": ["R", "ER"]},
    "cluster_reduction": {"L1": ["Mandarin"],                   "phones": ["S", "T", "K"]},
}

# H2's directional prediction, stated before running anything. Pre-registering
# this is the difference between a test and a fishing expedition.
H2_PREDICTION = {
    "survives_short_lookahead": ["stop", "fricative", "affricate", "nasal"],
    "breaks_first":             ["vowel_diph", "vowel_mono", "approximant"],
    "rationale": (
        "Segmental consonant substitutions are realised over 20-80 ms and are "
        "largely determined by the local articulatory gesture. Vowel quality "
        "and diphthong trajectories are defined by formant movement across "
        "80-250 ms, and their targets are shifted by the following consonant, "
        "so they require right context that a 40 ms budget does not provide. "
        "Approximants (/r/, /l/, /w/) pattern with vowels because they are "
        "formant-defined and heavily coarticulated."),
}


def phone_class(p: str) -> str:
    return CLASS_OF.get(strip_stress(p), "other")


def strip_stress(p: str) -> str:
    return "".join(c for c in p.upper() if not c.isdigit())


def is_vowel(p: str) -> bool:
    return phone_class(p).startswith("vowel")


# --------------------------------------------------------------------------
# Alignment container
# --------------------------------------------------------------------------

@dataclass
class Phone:
    label: str
    start_s: float
    end_s: float

    @property
    def dur_ms(self) -> float:
        return (self.end_s - self.start_s) * 1000

    @property
    def cls(self) -> str:
        return phone_class(self.label)


def load_textgrid(path: str, tier: str = "phones") -> List[Phone]:
    """Minimal TextGrid reader for MFA output (long or short format)."""
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    import re
    # long format intervals
    blocks = re.findall(
        r"intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = \"([^\"]*)\"",
        txt)
    if not blocks:
        raise ValueError(f"no intervals parsed from {path}")
    return [Phone(t.strip(), float(a), float(b))
            for a, b, t in blocks if t.strip() and t.strip() not in ("sil", "sp", "")]


# --------------------------------------------------------------------------
# Distortion per phone
# --------------------------------------------------------------------------

def mel_spectrogram(wav: np.ndarray, sr: int = 16_000, n_mels: int = 80,
                    hop: int = 160, win: int = 400) -> np.ndarray:
    """Dependency-light log-mel. Used for the per-phone distortion score."""
    n_fft = 512
    hann = np.hanning(win).astype(np.float32)
    n_frames = 1 + max(0, (len(wav) - win) // hop)
    frames = np.lib.stride_tricks.as_strided(
        wav, shape=(n_frames, win),
        strides=(wav.strides[0] * hop, wav.strides[0])).copy() * hann
    spec = np.abs(np.fft.rfft(frames, n_fft, axis=1)) ** 2

    # mel filterbank
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l: c = l + 1
        if r == c: r = c + 1
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(1, c - l)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(1, r - c)
    return np.log(spec @ fb.T + 1e-8)


def mel_cepstral_distortion(a: np.ndarray, b: np.ndarray) -> float:
    """MCD-style distance between two log-mel segments of possibly unequal length.

    Length mismatch is resolved by DTW rather than truncation, because at low
    lookahead the converter's *timing* changes too, and truncating would
    silently convert a duration error into a spectral one -- attributing
    rhythm damage to the wrong bucket.
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    na, nb = a.shape[0], b.shape[0]
    D = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    acc = np.full((na + 1, nb + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            acc[i, j] = D[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    path_len = na + nb
    return float(acc[na, nb] / path_len)


def per_phone_distortion(
    converted: np.ndarray,
    reference: np.ndarray,
    align_conv: Sequence[Phone],
    align_ref: Sequence[Phone],
    sr: int = 16_000,
) -> List[Dict[str, object]]:
    """Align phone-by-phone (by label sequence) and score each one."""
    mc = mel_spectrogram(converted, sr)
    mr = mel_spectrogram(reference, sr)
    hop_s = 160 / sr

    # match phone sequences by label; skip where the aligner disagrees on
    # identity, and count those separately -- a label mismatch IS the result
    # for a deletion/insertion, not an error to be swept up.
    out: List[Dict[str, object]] = []
    i = j = 0
    while i < len(align_conv) and j < len(align_ref):
        pc, pr = align_conv[i], align_ref[j]
        lc, lr = strip_stress(pc.label), strip_stress(pr.label)
        if lc != lr:
            out.append({"phone": lr, "cls": phone_class(lr),
                        "event": "mismatch", "produced": lc,
                        "mcd": float("nan"), "dur_err_ms": float("nan")})
            i += 1
            j += 1
            continue
        sc = mc[int(pc.start_s / hop_s):max(int(pc.end_s / hop_s), int(pc.start_s / hop_s) + 1)]
        sr_ = mr[int(pr.start_s / hop_s):max(int(pr.end_s / hop_s), int(pr.start_s / hop_s) + 1)]
        out.append({
            "phone": lr,
            "cls": phone_class(lr),
            "event": "match",
            "mcd": mel_cepstral_distortion(sc, sr_),
            "dur_err_ms": pc.dur_ms - pr.dur_ms,
            "ref_dur_ms": pr.dur_ms,
        })
        i += 1
        j += 1
    return out


# --------------------------------------------------------------------------
# Aggregation and the actual H2 test
# --------------------------------------------------------------------------

def aggregate_by_class(rows: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        buckets[str(r["cls"])].append(r)
    out = {}
    for cls, rs in buckets.items():
        mcds = [float(r["mcd"]) for r in rs if r["event"] == "match"
                and not np.isnan(float(r["mcd"]))]
        durs = [abs(float(r["dur_err_ms"])) for r in rs if r["event"] == "match"
                and not np.isnan(float(r["dur_err_ms"]))]
        mm = sum(1 for r in rs if r["event"] == "mismatch")
        out[cls] = {
            "n": len(rs),
            "mcd_mean": float(np.mean(mcds)) if mcds else float("nan"),
            "mcd_p95": float(np.percentile(mcds, 95)) if mcds else float("nan"),
            "dur_err_mean_ms": float(np.mean(durs)) if durs else float("nan"),
            "mismatch_rate": mm / max(1, len(rs)),
        }
    return out


def test_h2(per_L: Dict[float, Dict[str, Dict[str, float]]],
            metric: str = "mcd_mean") -> Dict[str, object]:
    """Fit distortion ~ lookahead per class and compare slopes.

    H2 is supported iff the classes in H2_PREDICTION["breaks_first"] have
    significantly steeper negative slopes (more improvement per ms of
    lookahead) than those in "survives_short_lookahead".

    This returns the descriptive fit. The confirmatory test belongs in
    eval/stats.py as a mixed-effects model with speaker and utterance as
    random effects -- an OLS slope here ignores the nesting and will
    understate the standard errors.
    """
    Ls = sorted(per_L)
    classes = sorted({c for d in per_L.values() for c in d})
    fits: Dict[str, Dict[str, float]] = {}
    for c in classes:
        ys = [per_L[L].get(c, {}).get(metric, np.nan) for L in Ls]
        xs = np.array([L for L, y in zip(Ls, ys) if not np.isnan(y)], float)
        yy = np.array([y for y in ys if not np.isnan(y)], float)
        if len(xs) < 3:
            continue
        A = np.vstack([xs, np.ones_like(xs)]).T
        slope, icpt = np.linalg.lstsq(A, yy, rcond=None)[0]
        resid = yy - (slope * xs + icpt)
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        fits[c] = {
            "slope_per_100ms": float(slope * 100),
            "intercept_at_L0": float(icpt),
            "r2": 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan"),
            "value_at_L0": float(yy[0]),
            "value_at_Lmax": float(yy[-1]),
            "relative_gain": float((yy[0] - yy[-1]) / yy[0]) if yy[0] else float("nan"),
        }
    brk = [c for c in H2_PREDICTION["breaks_first"] if c in fits]
    srv = [c for c in H2_PREDICTION["survives_short_lookahead"] if c in fits]
    gb = float(np.mean([fits[c]["relative_gain"] for c in brk])) if brk else float("nan")
    gs = float(np.mean([fits[c]["relative_gain"] for c in srv])) if srv else float("nan")
    return {
        "metric": metric,
        "lookaheads_ms": Ls,
        "per_class_fit": fits,
        "mean_relative_gain_breaks_first": gb,
        "mean_relative_gain_survives": gs,
        "h2_direction_supported": bool(gb > gs) if not (np.isnan(gb) or np.isnan(gs)) else None,
        "note": ("Descriptive only. Confirmatory test = mixed-effects model in "
                 "eval/stats.py; report CIs, not this point estimate."),
    }


def _lin_r2(x: np.ndarray, y: np.ndarray) -> float:
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - (m * x + c)
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / ss if ss > 0 else float("nan")


def _fit_ss(x: np.ndarray, y: np.ndarray) -> float:
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(((y - (m * x + c)) ** 2).sum())


def _bic(ss: float, n: int, k: int) -> float:
    return n * math.log(max(ss, 1e-300) / n) + k * math.log(n)


def find_knee(Ls: Sequence[float], ys: Sequence[float],
              frame_ms: float = 20.0) -> Dict[str, object]:
    """Is there a knee in the quality-vs-lookahead curve, and where?

    Three lessons are baked in here, each learned by getting it wrong first.

    **1. Test for a knee before locating one.** Maximum-distance-to-chord
    (Kneedle) *always* returns a point; on a curve that is smooth in log space
    it returns the middle of the sampled range and looks like a finding. Our
    first run reported "knee at 160 ms, bootstrap CI [160, 160]" on a curve
    that was log-linear at R2 = 0.9955 — the CI was tight *because* there was
    no effect, since every resample of a log-linear curve is log-linear.

    **2. Do not decide on an R2 threshold.** The second run, on a denser grid,
    gave R2 = 0.9887 and flipped `has_knee` purely by crossing an arbitrary
    0.99 cutoff. A scientific conclusion must not hinge on a magic number, so
    model choice is now BIC between a one-segment and a two-segment fit on the
    log2 axis.

    **3. Check for duplicate conditions.** Lookahead is quantised to
    ceil(L / frame_ms) frames. Sampling finer than the frame rate produces
    byte-identical duplicates that silently reweight the fit. We warn, and the
    caller should deduplicate.

    Because these curves are means over hundreds of utterances, residuals are
    real curvature rather than sampling noise, and BIC will happily prefer the
    two-segment model for a *gentle* bend. So the honest headline is neither
    "knee" nor "no knee" but the **gain-per-doubling profile**, which is
    model-free and is what a practitioner actually needs. It is always returned.
    """
    x_ms = np.asarray(Ls, float)
    y = np.asarray(ys, float)
    if len(x_ms) < 4:
        return {"knee_ms": float("nan"), "has_knee": None,
                "note": "need >= 4 points"}

    frames = np.ceil(x_ms / frame_ms - 1e-9).astype(int)
    n_dup = len(frames) - len(set(frames.tolist()))

    step = float(np.min(np.diff(np.unique(x_ms)))) if len(np.unique(x_ms)) > 1 else 1.0
    X = np.log2(x_ms + max(step, frame_ms))
    n = len(y)

    ss_log = _fit_ss(X, y)
    bic_log = _bic(ss_log, n, 3)
    best_bic, best_bp = float("inf"), float("nan")
    for i in range(2, n - 2):
        s2 = _fit_ss(X[:i + 1], y[:i + 1]) + _fit_ss(X[i:], y[i:])
        b = _bic(s2, n, 6)
        if b < best_bic:
            best_bic, best_bp = b, float(x_ms[i])
    d_bic = best_bic - bic_log
    has_knee = bool(d_bic < -6)      # "strong" evidence on the Kass-Raftery scale
    # BIC pays k*log(n) for the 3 extra parameters. Below ~10 points that
    # penalty swamps a real bend: a planted knee on a 7-point grid comes back
    # dBIC = -0.2, i.e. undetectable. So a "no knee" verdict from a sparse grid
    # means "underpowered", not "absent" -- say so rather than let the reader
    # assume evidence of absence.
    underpowered = n < 10

    # gain per doubling of lookahead -- the model-free headline
    # Pair each sampled L with its double wherever the grid contains one --
    # not just adjacent points, which on a linear grid are rarely doublings.
    profile = []
    for i, a in enumerate(x_ms):
        if a <= 0:
            continue
        j = int(np.argmin(abs(x_ms - 2 * a)))
        if abs(x_ms[j] - 2 * a) > 1e-6 or j == i:
            continue
        profile.append({"from_ms": float(a), "to_ms": float(x_ms[j]),
                        "delta": round(float(y[i] - y[j]), 5)})
    A = np.vstack([X, np.ones_like(X)]).T
    slope = float(np.linalg.lstsq(A, y, rcond=None)[0][0])

    xn = (x_ms - x_ms.min()) / (np.ptp(x_ms) or 1)
    yn = (y - y.min()) / (np.ptp(y) or 1)
    d = np.abs((yn[-1] - yn[0]) * xn - (xn[-1] - xn[0]) * yn
               + xn[-1] * yn[0] - yn[-1] * xn[0]) / \
        np.hypot(yn[-1] - yn[0], xn[-1] - xn[0])

    return {
        "knee_ms": best_bp if has_knee else float("nan"),
        "has_knee": has_knee,
        "delta_bic_piecewise_minus_loglinear": round(d_bic, 2),
        "breakpoint_if_piecewise_ms": best_bp,
        "r2_loglinear": round(1 - ss_log / float(((y - y.mean()) ** 2).sum()), 5),
        "r2_linear": round(_lin_r2(x_ms, y), 5),
        "slope_per_doubling": round(slope, 5),
        "gain_per_doubling_profile": profile,
        "n_duplicate_conditions": n_dup,
        "n_points": n,
        "underpowered_for_bic": underpowered,
        "chord_argmax_ms": float(x_ms[int(np.argmax(d))]),
        "monotone": bool(np.all(np.diff(y) <= 1e-9) or np.all(np.diff(y) >= -1e-9)),
        "interpretation": (
            (f"WARNING: {n_dup} duplicate conditions at the {frame_ms:g} ms frame "
             f"rate -- deduplicate before trusting this. " if n_dup else "")
            + (f"UNDERPOWERED: only {n} points; BIC cannot detect a knee below "
               f"~10 points, so a 'no knee' verdict here is absence of evidence, "
               f"not evidence of absence. " if underpowered else "")
            + (f"Two-segment fit preferred (dBIC={d_bic:.1f}), bend near "
               f"{best_bp:.0f} ms. But R2 of the single log-linear fit is "
               f"{1 - ss_log / float(((y - y.mean()) ** 2).sum()):.4f}, so this is a "
               f"gentle curvature, not a cliff -- report the gain-per-doubling "
               f"profile, not a knee location."
               if has_knee else
               f"No knee: log-linear in lookahead (dBIC={d_bic:+.1f} favours one "
               f"segment). Every doubling buys ~{abs(slope):.4f}. No natural "
               f"operating point; the budget is a product decision.")),
    }


if __name__ == "__main__":
    # self-test on synthetic data with a planted knee at 80 ms
    Ls = [0, 20, 40, 80, 160, 320, 640]
    q = [1.0, 0.95, 0.80, 0.40, 0.32, 0.30, 0.29]      # big drop up to 80 ms
    k = find_knee(Ls, q)
    assert k["underpowered_for_bic"] and k["has_knee"] is False, k
    print(f"ok  7-point grid reported as UNDERPOWERED, not as 'no knee' "
          f"(dBIC={k['delta_bic_piecewise_minus_loglinear']:+.1f} on n=7)")

    # Same shape, grid dense enough for BIC to have power.
    import numpy as _np
    Ld = [0,20,40,60,80,100,120,140,160,200,240,280,320,400,480,640]
    qd = [1.0 if L == 0 else (1.0 - 0.0075 * L if L <= 80 else 0.42 - 0.00018 * (L - 80))
          for L in Ld]
    kd = find_knee(Ld, qd)
    assert kd["has_knee"], kd
    assert 60 <= kd["breakpoint_if_piecewise_ms"] <= 120, kd["breakpoint_if_piecewise_ms"]
    print(f"ok  planted knee recovered on a 16-point grid: bend at "
          f"{kd['breakpoint_if_piecewise_ms']:.0f} ms "
          f"(dBIC={kd['delta_bic_piecewise_minus_loglinear']:.1f})")

    # The case that actually bit us: a smoothly log-linear curve has NO knee,
    # but the chord estimator will happily name one in the middle of the grid.
    import numpy as _np
    loglin = [1.0 - 0.08 * _np.log2(L + 20) for L in Ls]
    k2 = find_knee(Ls, loglin)
    assert k2["has_knee"] is False, k2
    assert _np.isnan(k2["knee_ms"]), k2
    assert k2["r2_loglinear"] > 0.999
    assert k2["chord_argmax_ms"] > 0        # the naive answer, kept for contrast
    print(f"ok  log-linear curve correctly reported as NO knee "
          f"(chord estimator would have said {k2['chord_argmax_ms']:.0f} ms)")

    # The duplicate-condition trap: sampling finer than the frame rate makes
    # byte-identical conditions that silently reweight the fit. Must be flagged.
    dense_L = list(range(0, 201, 10))
    dense_y = [1.0 - 0.08 * _np.log2(20 * _np.ceil(L / 20 - 1e-9) + 20) for L in dense_L]
    k3 = find_knee(dense_L, dense_y)
    assert k3["n_duplicate_conditions"] == 10, k3["n_duplicate_conditions"]
    assert "duplicate conditions" in k3["interpretation"]
    print(f"ok  duplicate conditions flagged "
          f"({k3['n_duplicate_conditions']} of {len(dense_L)} at the 20 ms frame rate)")

    # And the real GPU curve, deduplicated: gentle curvature, not a cliff
    real_L = [0,20,40,60,80,100,120,140,160,180,200,240,280,320,480,640]
    real_y = [0.49716,0.42209,0.37166,0.33439,0.30919,0.28621,0.26562,0.24664,
              0.23045,0.21639,0.20509,0.18769,0.17298,0.16078,0.12768,0.10737]
    k4 = find_knee(real_L, real_y)
    assert k4["n_duplicate_conditions"] == 0
    assert k4["r2_loglinear"] > 0.99
    print(f"ok  real GPU curve: R2_loglin={k4['r2_loglinear']} "
          f"dBIC={k4['delta_bic_piecewise_minus_loglinear']} "
          f"has_knee={k4['has_knee']}")
    print(f"    gain per doubling: "
          f"{[p['delta'] for p in k4['gain_per_doubling_profile']]}")

    per_L = {}
    for L in Ls:
        per_L[L] = {
            "stop":       {"mcd_mean": 3.0 - 0.0002 * L},   # barely improves
            "fricative":  {"mcd_mean": 3.1 - 0.0003 * L},
            "vowel_diph": {"mcd_mean": 5.0 - 0.0040 * L},   # improves a lot
            "vowel_mono": {"mcd_mean": 4.5 - 0.0030 * L},
            "approximant": {"mcd_mean": 4.2 - 0.0025 * L},
            "nasal":      {"mcd_mean": 3.0 - 0.0002 * L},
        }
    r = test_h2(per_L)
    assert r["h2_direction_supported"] is True
    assert r["per_class_fit"]["vowel_diph"]["slope_per_100ms"] < \
           r["per_class_fit"]["stop"]["slope_per_100ms"]
    print("ok  H2 slope comparison detects the planted class difference")
    print(f"    breaks-first gain {r['mean_relative_gain_breaks_first']:.3f} "
          f"vs survives {r['mean_relative_gain_survives']:.3f}")

    assert phone_class("AY1") == "vowel_diph" and phone_class("TH") == "fricative"
    print("ok  phone class map")
