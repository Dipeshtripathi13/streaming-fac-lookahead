"""S1 baseline: streaming ASR -> TTS cascade, measured properly.

The cascade is the accent-perfect upper bound: resynthesising from text
guarantees a native accent, at the cost of losing prosody, speaker identity,
and any word the recogniser gets wrong. Every accent-conversion paper waves
at it; nobody publishes its latency decomposition. This does.

The decomposition for a cascade is different from a frame-synchronous
converter, and that difference is itself worth a paragraph in the paper:

    t_algorithmic  = ASR endpoint/commit delay + TTS utterance granularity
    t_compute      = ASR encoder + TTS forward
    t_buffer       = audio I/O

The dominant term is NOT compute. It is the *commit delay*: the cascade
cannot synthesise a word until the recogniser stops revising it. Dipesh's
accentbridge prototype uses COMMIT_TIMEOUT = 0.7 s, which alone is 3x the
entire PHONOS end-to-end budget. That is the structural reason cascades lose,
and it is measurable here.

Models: reuses the sherpa-onnx models already downloaded under
../accentbridge/models (streaming zipformer ASR, Kokoro / Piper-VITS TTS,
Silero VAD). No GPU, no network, runs identically on macOS / Windows / Linux
/ Raspberry Pi.

Usage:
    python3 bench/bench_cascade_onnx.py --threads 1 2 4
    python3 bench/bench_cascade_onnx.py --models-dir ../accentbridge/models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfac.latency import StageTimer, write_csv, write_jsonl, _dist  # noqa: E402

DEFAULT_MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "accentbridge", "models")

ASR_DIR = "sherpa-onnx-streaming-zipformer-en-2023-06-21"
KOKORO_DIR = "kokoro-int8-en-v0_19"
VITS_DIR = "vits-piper-en_US-ryan-medium-int8"
VAD_PATH = "silero_vad_v4.onnx"

# Harvard-sentence-length prompts; matched roughly to L2-ARCTIC utterance length
TEST_TEXTS = [
    "The birch canoe slid on the smooth planks.",
    "He wrote a very interesting article about the three little pigs.",
    "Please call Stella and ask her to bring these things with her.",
    "We were able to take the boat out on the lake yesterday afternoon.",
]


def _p(models_dir: str, *parts: str) -> str:
    return os.path.join(models_dir, *parts)


# --------------------------------------------------------------------------
# ASR: real streaming zipformer, chunk by chunk
# --------------------------------------------------------------------------

def bench_asr(models_dir: str, threads: int, quantised: bool,
              audio: np.ndarray, sample_rate: int = 16_000) -> Dict[str, object]:
    import sherpa_onnx

    sfx = ".int8" if quantised else ""
    d = _p(models_dir, ASR_DIR)
    t_load = time.perf_counter_ns()
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{d}/tokens.txt",
        encoder=f"{d}/encoder-epoch-99-avg-1{sfx}.onnx",
        decoder=f"{d}/decoder-epoch-99-avg-1.onnx",
        joiner=f"{d}/joiner-epoch-99-avg-1{sfx}.onnx",
        num_threads=threads,
        sample_rate=sample_rate,
        provider="cpu",
        decoding_method="greedy_search",
        enable_endpoint_detection=True,
    )
    load_ms = (time.perf_counter_ns() - t_load) / 1e6

    # The zipformer's own geometry -- read it, do not assume it.
    import onnxruntime as ort
    meta = ort.InferenceSession(
        f"{d}/encoder-epoch-99-avg-1{sfx}.onnx",
        providers=["CPUExecutionProvider"],
    ).get_modelmeta().custom_metadata_map
    decode_chunk_len = int(meta.get("decode_chunk_len", 32))
    T_in = int(meta.get("T", 39))
    # fbank frames are 10 ms; the encoder consumes T frames to emit
    # decode_chunk_len worth of new context => right context = T - chunk
    asr_chunk_ms = decode_chunk_len * 10.0
    asr_right_ctx_ms = (T_in - decode_chunk_len) * 10.0

    timer = StageTimer()
    stream = rec.create_stream()
    block = int(0.1 * sample_rate)                 # 100 ms feed granularity
    n_blocks = 0
    for i in range(0, len(audio) - block, block):
        chunk = audio[i:i + block]
        t0 = time.perf_counter_ns()
        stream.accept_waveform(sample_rate, chunk)
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        timer.record("asr_per_100ms", (time.perf_counter_ns() - t0) / 1e6)
        n_blocks += 1

    s = timer.summary(drop_warmup=2).get("asr_per_100ms", {})

    # The zipformer only fires its encoder once it has accumulated
    # decode_chunk_len frames, so most 100 ms feed blocks cost almost nothing
    # and a minority cost a lot. A p50 over feed blocks therefore reports
    # ~0 ms and is actively misleading. Separate the two populations: the
    # "active" steps are the ones that actually ran the encoder.
    xs = sorted(timer.samples("asr_per_100ms")[2:])
    active = [x for x in xs if x > 1.0] or xs
    da = _dist(active)
    duty = len(active) / max(1, len(xs))

    return {
        "component": "asr_streaming_zipformer",
        "quantised": quantised,
        "threads": threads,
        "load_ms": round(load_ms, 1),
        "feed_block_ms": 100.0,
        "compute_p50_ms": round(s.get("p50", 0), 3),
        "compute_p95_ms": round(s.get("p95", 0), 3),
        "compute_p99_ms": round(s.get("p99", 0), 3),
        "rtf_p50": round(s.get("p50", 0) / 100.0, 4),
        "rtf_p95": round(s.get("p95", 0) / 100.0, 4),
        # what a user actually waits for when the encoder fires:
        "active_step_p50_ms": round(da["p50"], 3),
        "active_step_p95_ms": round(da["p95"], 3),
        "encoder_duty_cycle": round(duty, 3),
        # amortised: cost per second of audio, the honest throughput number
        "rtf_amortised": round(sum(xs) / (len(xs) * 100.0), 4),
        "model_decode_chunk_ms": asr_chunk_ms,
        "model_right_context_ms": asr_right_ctx_ms,
        "n_blocks": n_blocks,
        "transcript": rec.get_result(stream)[:120],
    }


# --------------------------------------------------------------------------
# TTS
# --------------------------------------------------------------------------

def bench_tts(models_dir: str, engine: str, threads: int) -> Dict[str, object]:
    import sherpa_onnx

    t_load = time.perf_counter_ns()
    if engine == "kokoro":
        d = _p(models_dir, KOKORO_DIR)
        mc = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=f"{d}/model.int8.onnx", voices=f"{d}/voices.bin",
                tokens=f"{d}/tokens.txt", data_dir=f"{d}/espeak-ng-data"),
            provider="cpu", num_threads=threads)
        sid = 5
    else:
        d = _p(models_dir, VITS_DIR)
        mc = sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=f"{d}/en_US-ryan-medium.onnx", lexicon="",
                tokens=f"{d}/tokens.txt", data_dir=f"{d}/espeak-ng-data"),
            provider="cpu", num_threads=threads)
        sid = 0
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=mc))
    load_ms = (time.perf_counter_ns() - t_load) / 1e6

    tts.generate("warming up", sid=sid, speed=1.0)   # first call pays alloc

    per_utt, rtfs, first_word_ms = [], [], []
    for text in TEST_TEXTS:
        for _ in range(3):
            t0 = time.perf_counter_ns()
            au = tts.generate(text, sid=sid, speed=1.0)
            dt = (time.perf_counter_ns() - t0) / 1e6
            dur_ms = len(au.samples) / au.sample_rate * 1000
            per_utt.append(dt)
            rtfs.append(dt / dur_ms)
        # single-word latency: the floor if you commit word-by-word
        w = text.split()[0]
        t0 = time.perf_counter_ns()
        tts.generate(w, sid=sid, speed=1.0)
        first_word_ms.append((time.perf_counter_ns() - t0) / 1e6)

    du, dr, dw = _dist(per_utt), _dist(rtfs), _dist(first_word_ms)
    return {
        "component": f"tts_{engine}",
        "threads": threads,
        "load_ms": round(load_ms, 1),
        "utt_synth_p50_ms": round(du["p50"], 2),
        "utt_synth_p95_ms": round(du["p95"], 2),
        "rtf_p50": round(dr["p50"], 4),
        "rtf_p95": round(dr["p95"], 4),
        "single_word_synth_p50_ms": round(dw["p50"], 2),
        "n": int(du["n"]),
    }


# --------------------------------------------------------------------------

def synth_probe_audio(seconds: float = 12.0, sr: int = 16_000) -> np.ndarray:
    """Deterministic speech-like probe signal.

    Real L2-ARCTIC audio is the right input for *quality*; for *timing* what
    matters is that the signal is non-silent (so the recogniser actually
    decodes rather than short-circuiting) and identical across machines.
    Formant-ish sum of harmonics with an amplitude envelope does that and
    needs no data download, so this benchmark runs on a fresh clone.
    """
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = sum((1.0 / h) * np.sin(h * ph) for h in range(1, 12))
    env = 0.5 * (1 + np.sin(2 * np.pi * 2.5 * t)) ** 2
    x = x * env + 0.01 * rng.standard_normal(len(t))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=DEFAULT_MODELS)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--out-prefix", default="results/raw/cascade_onnx")
    ap.add_argument("--skip-tts", action="store_true")
    ap.add_argument("--tts-engines", nargs="+", default=["kokoro", "vits"],
                    choices=["kokoro", "vits"])
    ap.add_argument("--skip-fp32", action="store_true",
                    help="fp32 ASR encoder is 354 MB; skip on memory-limited boxes")
    a = ap.parse_args()

    md = os.path.abspath(a.models_dir)
    if not os.path.isdir(md):
        sys.exit(f"models dir not found: {md}\n"
                 f"Run ../accentbridge/models/download.sh first.")

    from hardware_probe import probe  # noqa
    hw = probe(run_calibration=False)
    print(f"host: {hw['hw_class']}  {hw['machine']}  {hw['cpu_count_logical']} cores  "
          f"{hw['mem_total_gb']} GB", file=sys.stderr)

    audio = synth_probe_audio(a.seconds)
    rows: List[Dict[str, object]] = []

    for th in a.threads:
        for quant in ([True] if a.skip_fp32 else [True, False]):
            try:
                r = bench_asr(md, th, quant, audio)
            except Exception as e:  # OOM on fp32 is expected on a Pi
                print(f"  ASR threads={th} int8={quant} FAILED: {e}", file=sys.stderr)
                continue
            r["hw_class"] = hw["hw_class"]
            rows.append(r)
            print(f"ASR   th={th} int8={quant}  p50={r['compute_p50_ms']:>7.2f}ms/100ms  "
                  f"RTF={r['rtf_p50']:.3f}  p95RTF={r['rtf_p95']:.3f}", file=sys.stderr)

        if not a.skip_tts:
            for eng in a.tts_engines:
                try:
                    r = bench_tts(md, eng, th)
                except Exception as e:
                    print(f"  TTS {eng} threads={th} FAILED: {e}", file=sys.stderr)
                    continue
                r["hw_class"] = hw["hw_class"]
                rows.append(r)
                print(f"TTS   th={th} {eng:<7} utt_p50={r['utt_synth_p50_ms']:>8.1f}ms  "
                      f"RTF={r['rtf_p50']:.3f}  1word={r['single_word_synth_p50_ms']:.0f}ms",
                      file=sys.stderr)

    os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
    write_csv(a.out_prefix + ".csv", rows)
    write_jsonl(a.out_prefix + ".jsonl", rows)
    with open(a.out_prefix + "_host.json", "w") as f:
        json.dump(hw, f, indent=2, default=str)
    print(json.dumps({"n_rows": len(rows), "csv": a.out_prefix + ".csv"}, indent=2))


if __name__ == "__main__":
    main()
