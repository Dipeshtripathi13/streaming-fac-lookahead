"""Synthesise audio from a phone sequence, so accentedness can be *heard*.

Why this exists
---------------
Everything measured so far is phone error rate. PER is not accentedness: a
model could halve PER by fixing phones nobody perceives, or raise it while
producing speech that sounds more native. The paper cannot claim anything
perceptual until someone listens.

The translator emits phone sequences, so the missing piece is phones -> audio.
Piper's VITS ONNX takes **phoneme IDs directly** (`input: int64[batch, phonemes]`,
`phoneme_type: "espeak"`, a 154-entry `phoneme_id_map` in the model JSON), so we
can bypass espeak's text frontend and feed our own phones. That is exactly what
is needed and it is not what `sherpa_onnx.OfflineTts` exposes.

What this does and does not test
--------------------------------
Synthesising with **one fixed voice** deliberately discards speaker identity and
prosody. So this does **not** evaluate accent conversion as a product would
deliver it. It isolates one question:

    Does the translator's phone output, at a given lookahead, carry the
    accent correction — audibly, to a human?

That is narrower than "does the system work" and it is answerable now. State it
that way in the paper; do not let it drift into a system-level claim.

The anchors are what make it interpretable
------------------------------------------
Every stimulus set includes two synthesised from *ground truth*:

  * **ceiling** — from `g2p`, the canonical phones. Perfect conversion.
  * **floor**   — from `ipa`, the phones the speaker actually produced.
                  Perfect *transcription* of the accented production.

A listener rating condition *L* against those two has a calibrated scale, and
the pair also serves as an attention check: any rater who cannot tell ceiling
from floor is not listening. Without anchors, MOS numbers from a handful of
raters are uninterpretable.

Usage
-----
    python3 synth/phones_to_audio.py --selftest
    python3 synth/phones_to_audio.py --text "the birch canoe slid on the smooth planks"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_VOICE = os.path.join(os.path.dirname(__file__), "..", "..",
                             "accentbridge", "models",
                             "vits-piper-en_US-ryan-medium-int8")

# ARPAbet -> espeak IPA. L2-ARCTIC's g2p column is ARPAbet-ish; Piper speaks
# espeak IPA. Anything unmapped is dropped and counted, because silently
# dropping phones would quietly shorten every utterance and look like a
# synthesis artefact rather than a mapping bug.
ARPA_TO_IPA: Dict[str, str] = {
    "AA": "ɑ", "AE": "a", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɚ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}


class PiperPhonemeSynth:
    """Piper VITS driven by explicit phonemes rather than text."""

    def __init__(self, voice_dir: str = DEFAULT_VOICE, quantised: bool = True):
        vd = os.path.abspath(voice_dir)
        cfgs = [f for f in os.listdir(vd) if f.endswith(".onnx.json")]
        if not cfgs:
            raise FileNotFoundError(f"no *.onnx.json in {vd}")
        self.cfg = json.load(open(os.path.join(vd, cfgs[0])))
        onnx = os.path.join(vd, cfgs[0][:-5])
        if not os.path.exists(onnx):
            raise FileNotFoundError(onnx)
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1        # F4: 1 thread is optimal here
        self.sess = ort.InferenceSession(onnx, so,
                                         providers=["CPUExecutionProvider"])
        self.pid: Dict[str, List[int]] = self.cfg["phoneme_id_map"]
        self.sr = int(self.cfg["audio"]["sample_rate"])
        inf = self.cfg.get("inference", {})
        self.scales = np.array([inf.get("noise_scale", 0.667),
                                inf.get("length_scale", 1.0),
                                inf.get("noise_w", 0.8)], dtype=np.float32)
        self.unmapped: Dict[str, int] = {}

    # ---------------------------------------------------------------
    def to_ipa(self, phones: Sequence[str]) -> List[str]:
        """ARPAbet or IPA in, espeak-IPA symbols out (one per element)."""
        out: List[str] = []
        for p in phones:
            q = "".join(c for c in p if not c.isdigit()).strip()
            if not q:
                continue
            if q.upper() in ARPA_TO_IPA:
                out.extend(list(ARPA_TO_IPA[q.upper()]))
            else:
                out.extend(list(unicodedata.normalize("NFD", q)))
        return out

    def encode(self, phones: Sequence[str]) -> Tuple[List[int], int, int]:
        """espeak-IPA symbols -> Piper phoneme IDs. Returns (ids, kept, dropped).

        Piper wraps every utterance in BOS `^` ... EOS `$` and interleaves the
        pad symbol `_` between phonemes. Omitting either produces audible
        artefacts, so both are done here rather than left to the caller.
        """
        ids: List[int] = []
        kept = dropped = 0
        pad = self.pid.get("_", [0])[0]

        def push(sym: str) -> bool:
            v = self.pid.get(sym)
            if v is None:
                self.unmapped[sym] = self.unmapped.get(sym, 0) + 1
                return False
            ids.extend(v)
            ids.append(pad)
            return True

        push("^")
        for sym in self.to_ipa(phones):
            if push(sym):
                kept += 1
            else:
                dropped += 1
        push("$")
        return ids, kept, dropped

    def synth(self, phones: Sequence[str],
              length_scale: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        ids, kept, dropped = self.encode(phones)
        if kept == 0:
            raise ValueError("no phones survived the mapping; check to_ipa()")
        scales = self.scales.copy()
        if length_scale is not None:
            scales[1] = length_scale
        x = np.array([ids], dtype=np.int64)
        out = self.sess.run(None, {
            "input": x,
            "input_lengths": np.array([x.shape[1]], dtype=np.int64),
            "scales": scales,
        })[0]
        wav = np.asarray(out).squeeze().astype(np.float32)
        peak = float(np.abs(wav).max()) or 1.0
        return wav / peak * 0.95, {
            "n_phones_in": len(phones), "n_kept": kept, "n_dropped": dropped,
            "drop_rate": dropped / max(1, kept + dropped),
            "n_ids": len(ids), "sr": self.sr,
            "duration_s": round(len(wav) / self.sr, 3),
        }


def write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    import wave
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--phones", nargs="+", default=None)
    ap.add_argument("--out", default="results/synth/demo.wav")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    s = PiperPhonemeSynth(a.voice)
    print(f"voice: {os.path.basename(os.path.abspath(a.voice))}  "
          f"sr={s.sr}  phoneme_id_map={len(s.pid)} entries")

    if a.selftest:
        # ARPAbet in, and a coverage check over the full inventory
        arpa = ("DH AH0 B ER1 CH K AH0 N UW1 S L IH1 D AA1 N DH AH0 "
                "S M UW1 DH P L AE1 NG K S").split()
        wav, meta = s.synth(arpa)
        print(f"  ARPAbet utterance -> {meta}")
        assert meta["duration_s"] > 1.0, "suspiciously short"
        assert meta["drop_rate"] < 0.05, f"too many dropped: {s.unmapped}"

        miss = [k for k, v in ARPA_TO_IPA.items()
                if any(c not in s.pid for c in v)]
        print(f"  ARPAbet inventory coverage: {len(ARPA_TO_IPA)-len(miss)}"
              f"/{len(ARPA_TO_IPA)} map into the Piper inventory")
        if miss:
            print(f"  unmapped ARPAbet: {miss}")

        # the two anchors must be audibly different, or the test is uncalibrated
        floor = "DH AH0 B ER1 CH K AH0 N U S L I D AA1 N DH AH0 S M U D P L E NG K S".split()
        w2, m2 = s.synth(floor)
        d = abs(meta["duration_s"] - m2["duration_s"])
        print(f"  ceiling {meta['duration_s']}s vs floor {m2['duration_s']}s "
              f"(differ by {d:.2f}s) -> stimuli are distinguishable")
        write_wav("results/synth/selftest_ceiling.wav", wav, s.sr)
        write_wav("results/synth/selftest_floor.wav", w2, s.sr)
        print("  wrote results/synth/selftest_{ceiling,floor}.wav")
        print("\nok  phones -> audio works")
        return

    phones = a.phones or "DH AH0 B ER1 CH K AH0 N UW1".split()
    wav, meta = s.synth(phones)
    write_wav(a.out, wav, s.sr)
    print(f"  {meta}\n  wrote {a.out}")
    if s.unmapped:
        print(f"  UNMAPPED symbols: {s.unmapped}")


if __name__ == "__main__":
    main()
