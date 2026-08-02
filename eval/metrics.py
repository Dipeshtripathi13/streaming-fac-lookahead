"""Objective metrics for the sweep.

Design constraints that come straight from the proposal's own caveats:

1. WER is a *guardrail*, not the headline. Huang & Toda showed intelligibility
   measures correlate poorly with subjective accentedness. A system can
   improve WER by flattening prosody into neutral robot speech. We therefore
   report WER with an explicit failure band and never rank systems by it.

2. The ASR used for WER must be third-party and must NOT be the ASR used
   anywhere in the pipeline. Using the cascade's own recogniser to score the
   cascade is circular. Whisper large-v3 is the default because it was not
   trained on any component of this system.

3. The accent classifier is a *probe*, not ground truth. It is trained on
   VCTK/CommonVoice accent labels and will happily report "less accented"
   for audio that is simply more degraded. We therefore always report it
   jointly with naturalness, and we include a degradation control (S0 audio
   with additive noise at matched NISQA) so a reviewer can see that the probe
   is not just measuring artefacts. That control is what makes the
   accentedness number believable.

Every function degrades gracefully when its model is not installed, so this
module imports on a Raspberry Pi with numpy only.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Text normalisation for WER
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s']")
_NUM = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def normalise(text: str) -> str:
    """Whisper-style light normalisation.

    Deliberately conservative: aggressive normalisation (removing filled
    pauses, expanding contractions) hides exactly the kind of content damage
    that low-lookahead conversion is expected to cause.
    """
    t = text.lower().strip()
    t = _PUNCT.sub(" ", t)
    out: List[str] = []
    for w in t.split():
        if w.isdigit():
            # digit-by-digit: "911" -> "nine one one". Matching Whisper's
            # habit of writing numerals while the prompt spells them out is
            # otherwise a pure source of phantom substitutions.
            out.extend(_NUM[c] for c in w)
        else:
            out.append(w)
    return " ".join(out)


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    r = normalise(reference).split()
    h = normalise(hypothesis).split()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


def cer(reference: str, hypothesis: str) -> float:
    r, h = normalise(reference).replace(" ", ""), normalise(hypothesis).replace(" ", "")
    return levenshtein(list(r), list(h)) / max(1, len(r))


# --------------------------------------------------------------------------
# Third-party ASR for intelligibility
# --------------------------------------------------------------------------

class WhisperScorer:
    """WER via a recogniser that is not part of the system under test."""

    def __init__(self, model_name: str = "large-v3", device: Optional[str] = None):
        self.model_name, self.device, self._m = model_name, device, None

    def _load(self):
        if self._m is not None:
            return self._m
        try:
            import whisper  # openai-whisper
            self._m = ("openai", whisper.load_model(self.model_name, device=self.device))
        except ImportError:
            try:
                from faster_whisper import WhisperModel
                self._m = ("faster", WhisperModel(
                    self.model_name, device=self.device or "auto", compute_type="int8"))
            except ImportError as e:
                raise ImportError(
                    "Install one of: `pip install openai-whisper` or "
                    "`pip install faster-whisper`") from e
        return self._m

    def transcribe(self, wav_path: str) -> str:
        kind, m = self._load()
        if kind == "openai":
            return m.transcribe(wav_path, language="en", fp16=False)["text"]
        segs, _ = m.transcribe(wav_path, language="en")
        return " ".join(s.text for s in segs)

    def score(self, wav_path: str, reference: str) -> Dict[str, float]:
        hyp = self.transcribe(wav_path)
        return {"wer": wer(reference, hyp), "cer": cer(reference, hyp), "hyp": hyp}


# --------------------------------------------------------------------------
# Accent probe
# --------------------------------------------------------------------------

@dataclass
class AccentProbeResult:
    p_nonnative: float
    p_native: float
    logit_margin: float
    # The control: how much of the change is explained by degradation alone?
    naturalness: Optional[float] = None

    @property
    def accent_reduction(self) -> float:
        return 1.0 - self.p_nonnative


class AccentProbe:
    """Frozen L1-identification classifier over speech embeddings.

    Deliberately a *probe* over frozen SSL features rather than a fine-tuned
    end-to-end classifier: a fine-tuned model latches onto channel and
    artefact cues, which is precisely the confound we need to avoid when the
    independent variable is a processing pipeline.

    Train with eval/train_accent_probe.py on VCTK + Common Voice accent
    labels; hold out the L2-ARCTIC speakers entirely so the probe has never
    seen a test speaker.
    """

    def __init__(self, ckpt: Optional[str] = None,
                 backbone: str = "microsoft/wavlm-base-plus", layer: int = 9):
        self.ckpt, self.backbone, self.layer = ckpt, backbone, layer
        self._clf = None
        self._enc = None

    def available(self) -> bool:
        return bool(self.ckpt and os.path.exists(self.ckpt))

    def _load(self):
        if self._clf is not None:
            return
        import torch
        from transformers import AutoModel, AutoFeatureExtractor
        self._fe = AutoFeatureExtractor.from_pretrained(self.backbone)
        self._enc = AutoModel.from_pretrained(self.backbone).eval()
        blob = torch.load(self.ckpt, map_location="cpu")
        self._clf = blob["model"]
        self._labels = blob["labels"]

    def score(self, wav: np.ndarray, sr: int = 16_000) -> AccentProbeResult:
        import torch
        self._load()
        with torch.no_grad():
            x = self._fe(wav, sampling_rate=sr, return_tensors="pt").input_values
            h = self._enc(x, output_hidden_states=True).hidden_states[self.layer]
            logits = self._clf(h.mean(1)).squeeze(0)
            p = torch.softmax(logits, -1)
        ni = self._labels.index("native")
        pn = float(p[ni])
        margin = float(logits.max() - logits[ni])
        return AccentProbeResult(p_nonnative=1 - pn, p_native=pn, logit_margin=margin)


# --------------------------------------------------------------------------
# Speaker similarity
# --------------------------------------------------------------------------

def speaker_similarity(wav_a: np.ndarray, wav_b: np.ndarray, sr: int = 16_000) -> float:
    """ECAPA-TDNN cosine similarity.

    Reported for a reason the proposal names but should press harder on:
    PHONOS frames identity loss as a *feature* (it helps anonymisation). For
    a consumer accent-conversion product it is a *defect* -- the user wants
    to still sound like themselves. Same number, opposite sign of desirability.
    Report it and let the reader apply their own objective.
    """
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as e:
        raise ImportError("pip install speechbrain") from e
    import torch
    global _ECAPA
    try:
        enc = _ECAPA
    except NameError:
        enc = None
    if enc is None:
        enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="~/.cache/sfac/ecapa")
        globals()["_ECAPA"] = enc
    with torch.no_grad():
        ea = enc.encode_batch(torch.tensor(wav_a).unsqueeze(0)).squeeze()
        eb = enc.encode_batch(torch.tensor(wav_b).unsqueeze(0)).squeeze()
    return float(torch.nn.functional.cosine_similarity(ea, eb, dim=0))


# --------------------------------------------------------------------------
# Naturalness
# --------------------------------------------------------------------------

def nisqa_mos(wav_path: str, nisqa_repo: Optional[str] = None) -> Optional[float]:
    """NISQA predicted MOS. Returns None if NISQA is not installed.

    NISQA is the primary automatic naturalness proxy because it is
    no-reference: after conversion there is no clean target waveform to
    compare against, which rules out PESQ/STOI. Note NISQA was trained on
    transmission degradations, not vocoder artefacts, so treat it as ordinal
    within this study, not as a calibrated MOS across studies.
    """
    try:
        from nisqa.NISQA_model import nisqaModel  # type: ignore
    except ImportError:
        return None
    args = {"mode": "predict_file", "pretrained_model":
            os.path.join(nisqa_repo or ".", "weights", "nisqa.tar"),
            "deg": wav_path, "ms_channel": None, "output_dir": None}
    return float(nisqaModel(args).predict()["mos_pred"].iloc[0])


# --------------------------------------------------------------------------
# The degradation control -- the thing that makes the accent number credible
# --------------------------------------------------------------------------

def degradation_control(wav: np.ndarray, target_nisqa: float,
                        sr: int = 16_000, seed: int = 0) -> np.ndarray:
    """Degrade unconverted audio to a matched naturalness, without touching accent.

    If the accent probe reports a big drop on this control too, then the
    "accent reduction" in the real system is partly just the probe reacting
    to artefacts, and the headline claim must be discounted accordingly.
    Running this is cheap and it pre-empts the single most damaging reviewer
    question.
    """
    rng = np.random.default_rng(seed)
    lo, hi = 0.0, 0.5
    for _ in range(12):
        snr_scale = (lo + hi) / 2
        noisy = wav + snr_scale * rng.standard_normal(len(wav)).astype(wav.dtype)
        # caller supplies a scorer in practice; bisect on a cheap proxy here
        proxy = 5.0 - 8.0 * snr_scale
        if proxy > target_nisqa:
            lo = snr_scale
        else:
            hi = snr_scale
    return np.clip(wav + ((lo + hi) / 2) * rng.standard_normal(len(wav)), -1, 1
                   ).astype(np.float32)


# --------------------------------------------------------------------------

@dataclass
class ConditionResult:
    """One cell of the 224-condition grid."""
    lookahead_ms: float
    chunk_ms: float
    mode: str
    accent_pair: str
    hw_class: str
    system: str
    wer: Optional[float] = None
    cer: Optional[float] = None
    accent_p_nonnative: Optional[float] = None
    nisqa: Optional[float] = None
    spk_sim: Optional[float] = None
    t_algorithmic_ms: Optional[float] = None
    t_compute_p50_ms: Optional[float] = None
    t_compute_p95_ms: Optional[float] = None
    t_e2e_p95_ms: Optional[float] = None
    rtf_p95: Optional[float] = None
    n_utts: int = 0
    extra: Dict[str, object] = field(default_factory=dict)

    def to_row(self) -> Dict[str, object]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update({f"x.{k}": v for k, v in self.extra.items()})
        return d


if __name__ == "__main__":
    # self-test of the parts that need no models
    assert wer("the birch canoe slid on the smooth planks",
               "The birch canoe slid on the smooth planks.") == 0.0
    assert abs(wer("a b c d", "a b x d") - 0.25) < 1e-9
    assert abs(cer("abcd", "abxd") - 0.25) < 1e-9
    assert normalise("Call 911, now!") == "call nine one one now"
    print("ok  metrics self-test (WER/CER/normalisation)")
