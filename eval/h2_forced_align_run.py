"""Runtime half of the forced-alignment H2 test. Needs Piper + a phoneme CTC model.

Split from `h2_forced_align.py` so the pure logic there stays testable on a
machine with no model weights and no network. All the decisions live there; this
file is plumbing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "synth"))

from analyse_h2_sequences import align, ipa_classes, h2_from_rates, load_hyps
from h2_forced_align import (IPA_TO_ESPEAK, MFCC_HOP_S, MODEL_SR, PHONEME_MODEL,
                             correct_phone_mask, map_span, spans_from_alignment)


def make_deterministic(synth):
    synth.scales = np.array([0.0, float(synth.scales[1]), 0.0], dtype=np.float32)
    return synth


def assert_deterministic(synth, phones):
    a1, _ = synth.synth(list(phones))
    a2, _ = synth.synth(list(phones))
    if len(a1) != len(a2) or not np.array_equal(a1, a2):
        raise SystemExit("synthesiser is not deterministic; zero noise_scale/noise_w")


class Aligner:
    """espeak-IPA wav2vec2 CTC model + torchaudio forced alignment."""

    def __init__(self, device: str = "cpu"):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC
        self.torch = torch
        self.vocab = json.load(open(hf_hub_download(PHONEME_MODEL, "vocab.json")))
        self.model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL).eval().to(device)
        self.device = device
        self.blank = self.vocab.get("<pad>", 0)

    def ids(self, phones: Sequence[str]) -> Tuple[List[int], List[int]]:
        """(token ids, kept reference indices). Unmappable phones are reported."""
        out, keep = [], []
        for i, p in enumerate(phones):
            q = IPA_TO_ESPEAK.get(p, p)
            if q in self.vocab:
                out.append(self.vocab[q])
                keep.append(i)
        return out, keep

    def emissions(self, wav16: np.ndarray):
        t = self.torch
        with t.inference_mode():
            x = t.from_numpy(wav16).float().unsqueeze(0).to(self.device)
            x = (x - x.mean()) / (x.std() + 1e-7)
            logits = self.model(x).logits
            return t.log_softmax(logits, dim=-1)

    def frame_spans(self, wav16: np.ndarray, phones: Sequence[str]):
        """[start, end) model-frame span per KEPT phone, plus the kept indices."""
        import torchaudio.functional as AF
        t = self.torch
        tok, keep = self.ids(phones)
        if not tok:
            return [], []
        logp = self.emissions(wav16)
        targets = t.tensor([tok], dtype=t.int32, device=self.device)
        aligned, _scores = AF.forced_align(logp, targets, blank=self.blank)
        aligned = aligned[0].tolist()
        # frame -> index into `tok`, advancing whenever the next target surfaces
        labels, j = [], -1
        for fr in aligned:
            if fr == self.blank:
                labels.append(-1)
                continue
            if j + 1 < len(tok) and fr == tok[j + 1]:
                j += 1
            labels.append(j if j >= 0 else -1)
        return spans_from_alignment(labels, len(tok)), keep


def mfccs(wav: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    m = librosa.feature.mfcc(y=wav.astype(np.float32), sr=sr, n_mfcc=25,
                             hop_length=int(MFCC_HOP_S * sr),
                             n_fft=int(0.025 * sr))
    return m[1:].T


def dtw_path(a: np.ndarray, b: np.ndarray):
    import librosa
    _, p = librosa.sequence.dtw(X=a.T, Y=b.T, metric="euclidean")
    p = p[::-1]
    return p[:, 0], p[:, 1]


def resample(w: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    import librosa
    return w if sr_in == sr_out else librosa.resample(w, orig_sr=sr_in, target_sr=sr_out)


def measure(synth, aligner, ref: Sequence[str], hyp: Sequence[str]):
    """Per reference phone: local spectral distortion, class, and correctness."""
    sr = synth.sr
    ref_wav, _ = synth.synth(list(ref))
    try:
        hyp_wav, _ = synth.synth(list(hyp))
    except ValueError:
        return []
    spans, keep = aligner.frame_spans(resample(ref_wav, sr, MODEL_SR), ref)
    if not spans:
        return []

    ca, cb = mfccs(ref_wav, sr), mfccs(hyp_wav, sr)
    if len(ca) == 0 or len(cb) == 0:
        return []
    pr, ph = dtw_path(ca, cb)

    model_hop_s = len(resample(ref_wav, sr, MODEL_SR)) / MODEL_SR / max(1, len(spans) and 1)
    # wav2vec2 stride is 320 samples at 16 kHz = 20 ms, fixed.
    frame_s = 320.0 / MODEL_SR

    classes = ipa_classes(list(ref))
    correct = correct_phone_mask(ref, hyp)
    rows = []
    for k, (s, e) in zip(keep, spans):
        if s < 0 or e <= s:
            continue
        lo = int(round(s * frame_s / MFCC_HOP_S))
        hi = int(round(e * frame_s / MFCC_HOP_S))
        pairs = map_span(pr, ph, lo, min(hi, len(ca)))
        if not pairs:
            continue
        d = float(np.mean([np.linalg.norm(ca[i] - cb[j]) for i, j in pairs]))
        rows.append({"cls": classes[k], "dist": d,
                     "correct": bool(correct[k]) if k < len(correct) else True})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-dir", default="results/raw/hyps")
    ap.add_argument("--target", default="native")
    ap.add_argument("--n-utts", type=int, default=30)
    ap.add_argument("--voice", default="/content/piper/ryan")
    ap.add_argument("--min-separation", type=float, default=1.10,
                    help="require wrong/correct mean distortion ratio above this "
                         "before any H2 verdict is reported")
    ap.add_argument("--out", default="results/analysis_h2_forced_align.json")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    from phones_to_audio import PiperPhonemeSynth
    synth = make_deterministic(PiperPhonemeSynth(a.voice))
    aligner = Aligner()

    by_L, prov = load_hyps(a.hyp_dir, a.target)
    Ls = sorted(by_L)
    picked = {}
    for L in Ls:
        best = {}
        for h in by_L[L]:
            k = (str(h.get("speaker")), tuple(h.get("g2p") or []))
            if k not in best or str(h.get("seed")) < str(best[k].get("seed")):
                best[k] = h
        picked[L] = best
    keys = sorted(set.intersection(*(set(picked[L]) for L in Ls)))[: a.n_utts]
    assert_deterministic(synth, list(keys[0][1]))
    print(f"{len(keys)} utterances x {len(Ls)} lookaheads")

    per_L: Dict[float, Dict[str, Dict[str, float]]] = {}
    sep_all = {"correct": [], "wrong": []}
    for L in Ls:
        opp = defaultdict(int); tot = defaultdict(float); nerr = defaultdict(int)
        for k in keys:
            h = picked[L][k]
            rows = measure(synth, aligner, list(h["g2p"]), list(h["pred"]))
            for r in rows:
                opp[r["cls"]] += 1
                tot[r["cls"]] += r["dist"]
                if not r["correct"]:
                    nerr[r["cls"]] += 1
                sep_all["correct" if r["correct"] else "wrong"].append(r["dist"])
        per_L[L] = {c: {"acoustic_distortion": tot[c] / opp[c] if opp[c] else float("nan"),
                        "error_rate": nerr[c] / opp[c] if opp[c] else float("nan"),
                        "opportunities": float(opp[c])} for c in sorted(opp)}
        print(f"  L={L:>5.0f}  phones={sum(opp.values()):>5d}  "
              f"mean distortion={np.mean([v['acoustic_distortion'] for v in per_L[L].values()]):.2f}")

    mc = float(np.mean(sep_all["correct"])) if sep_all["correct"] else float("nan")
    mw = float(np.mean(sep_all["wrong"])) if sep_all["wrong"] else float("nan")
    ratio = mw / mc if mc else float("nan")
    print(f"\nVALIDATION  correct-phone distortion {mc:.2f} | wrong-phone {mw:.2f} "
          f"| ratio {ratio:.3f}  (need > {a.min_separation})")

    out = {"provenance": {**prov, "n_utts": len(keys), "model": PHONEME_MODEL},
           "validation": {"mean_distortion_correct": mc, "mean_distortion_wrong": mw,
                          "ratio": ratio, "min_separation": a.min_separation,
                          "passed": bool(ratio > a.min_separation)},
           "per_lookahead": {str(k): v for k, v in per_L.items()}}
    if ratio > a.min_separation:
        out["verdict_acoustic"] = h2_from_rates(per_L, metric="acoustic_distortion")
        out["verdict_sequence_same_subset"] = h2_from_rates(per_L, metric="error_rate")
        print(f"  H2 (acoustic weighting): {out['verdict_acoustic'].get('H2_supported')}")
        print(f"  H2 (sequence, same subset): {out['verdict_sequence_same_subset'].get('H2_supported')}")
    else:
        out["verdict_acoustic"] = None
        print("  VERDICT SUPPRESSED: the measure does not separate correct from "
              "incorrect phones, so class comparisons on it are meaningless.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
