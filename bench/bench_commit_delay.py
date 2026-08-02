"""How long must a cascade wait before a recognised word stops changing?

Why this experiment exists
--------------------------
The measured cascade budget (docs/RESULTS_M4.md, Finding 5) is ~1.15 s, and
**95% of it is algorithmic**. The single largest term is not the ASR encoder
(12 ms) and not the vocoder (21 ms with Piper) — it is the **commit delay**:
`accentbridge.py` holds a word for `COMMIT_TIMEOUT = 0.7 s` before handing it to
the synthesiser, because a streaming recogniser keeps revising its hypothesis.

Up to now the proposal *asserted* that shrinking that timeout trades against
word-revision errors. Asserting the dominant term of your headline result is
not good enough. This measures it.

Method
------
Feed real speech through the streaming zipformer in 20 ms increments. After
every increment, snapshot the full partial hypothesis (tokens + the model's own
per-token timestamps). Then, for each token in the final hypothesis:

    emit_time(i)     first wall position at which slot i held any token
    stable_time(i)   earliest time after which slot i never changed again
    acoustic_end(i)  when the sound for token i finished (model timestamps)

    revision_span(i)          = stable_time(i) - emit_time(i)
        -> how long the word stayed unstable after first appearing.
           This is exactly what COMMIT_TIMEOUT is trying to cover.

    stabilisation_latency(i)  = stable_time(i) - acoustic_end(i)
        -> delay between the speaker finishing the word and the recogniser
           committing to it. This is the cascade's irreducible floor: you
           cannot synthesise a word before this, no matter how fast the
           vocoder or the chip.

From those we derive the curve the proposal needs: **for a commit timeout T,
what fraction of words are released while still unstable?** That is the
latency-vs-correctness trade, measured rather than asserted.

Honest limits, stated up front
------------------------------
* The zipformer emits on a 320 ms decode chunk, so all of these quantities are
  quantised to ~320 ms. That quantisation is not measurement error — it is a
  real property of the model, and it is a floor on how finely any cascade built
  on it can commit.
* `stable_time` is defined against the *final* hypothesis of this utterance.
  A word that is wrong from first emission and never corrected counts as
  "stable" here. This measures instability, not accuracy.
* Default input is the 2 clean test wavs shipped with the sherpa model. That is
  enough to establish the shape and the order of magnitude; it is **not** enough
  for a paper. Re-run with `--wavs '<glob>'` over L2-ARCTIC once available —
  accented speech should revise *more*, which if true strengthens the argument.

Usage
-----
    python3 bench/bench_commit_delay.py
    python3 bench/bench_commit_delay.py --wavs '/path/to/l2arctic/**/*.wav' --tag l2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import wave
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfac.latency import write_csv, _dist  # noqa: E402

DEFAULT_MODELS = os.path.join(os.path.dirname(__file__), "..", "..",
                              "accentbridge", "models")
ASR_DIR = "sherpa-onnx-streaming-zipformer-en-2023-06-21"


def read_wav(path: str, target_sr: int = 16_000) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
        sw = w.getsampwidth()
    if sw != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {sw*8}-bit")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != target_sr:
        # linear resample is fine here: we are measuring decoder behaviour,
        # not audio quality, and the model expects 16 kHz.
        idx = np.linspace(0, len(x) - 1, int(len(x) * target_sr / sr))
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return x


def build_recognizer(models_dir: str, threads: int = 1,
                     endpointing: bool = True, quantised: bool = True):
    """Construct the recogniser with the EXACT kwargs known to decode.

    A leaner construction (fewer kwargs, `enable_endpoint_detection=False`)
    decoded fine on Linux/aarch64 and returned an empty hypothesis for every
    input on macOS/arm64 — same sherpa-onnx 1.13.4, same model files, same
    audio, both int8 and fp32. Rather than keep bisecting a platform
    difference remotely, mirror the configuration that `bench_cascade_onnx.py`
    already proves works on both machines.

    Endpointing is therefore ON, with the same rules. That is safe for this
    measurement as long as clips are short: rule1 needs 2.4 s of trailing
    silence and rule3 needs a 20 s utterance, neither of which fires inside a
    normal read sentence. `trace_utterance` additionally detects a reset (the
    hypothesis suddenly shrinking) and stops rather than silently mixing
    segments.
    """
    import sherpa_onnx
    d = os.path.join(models_dir, ASR_DIR)
    sfx = ".int8" if quantised else ""
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{d}/tokens.txt",
        encoder=f"{d}/encoder-epoch-99-avg-1{sfx}.onnx",
        decoder=f"{d}/decoder-epoch-99-avg-1.onnx",
        joiner=f"{d}/joiner-epoch-99-avg-1{sfx}.onnx",
        num_threads=threads,             # 1 is optimal on Apple Silicon (F4)
        sample_rate=16_000,
        enable_endpoint_detection=endpointing,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=0.8,
        rule3_min_utterance_length=20,
        decoding_method="greedy_search",
        provider="cpu",
    )


def tokens_to_words(tokens: Sequence[str],
                    times: Sequence[float]) -> List[Tuple[str, float]]:
    """BPE pieces -> (word, word_start_time).

    Grouping to words matters: a cascade commits *words*, not sub-word pieces,
    so a piece-level number would understate the delay.

    The boundary marker is model-dependent and getting it wrong is silent — it
    collapses the whole utterance into one "word" and every statistic becomes
    meaningless. This k2/icefall zipformer uses a **leading ASCII space**
    (`' AFTER'`, `' E'`, `'AR'`, `'LY'`); SentencePiece models use U+2581.
    Handle both, and let the caller assert the result is plausible.
    """
    words: List[Tuple[str, float]] = []
    for tok, t in zip(tokens, times):
        boundary = tok.startswith("▁") or tok.startswith(" ")
        piece = tok.lstrip("▁ ")
        if boundary or not words:
            words.append((piece, float(t)))
        else:
            words[-1] = (words[-1][0] + piece, words[-1][1])
    return [(w, t) for w, t in words if w]


def diagnose(models_dir: str, audio: np.ndarray, sr: int = 16_000) -> str:
    """Dump exactly what this sherpa-onnx build returns, and how.

    The token/timestamp API is not stable across sherpa-onnx releases: the same
    code produced 66 words on one machine (1.13.4) and 0 on another. Rather
    than guess across a slow edit-run loop, print the ground truth once.
    """
    import sherpa_onnx
    out = [f"  sherpa_onnx version : {getattr(sherpa_onnx, '__version__', 'unknown')}",
           f"  audio (wave module) : {len(audio)/sr:.2f}s, "
           f"dtype={audio.dtype}, contiguous={audio.flags['C_CONTIGUOUS']}, "
           f"peak={float(np.abs(audio).max()):.3f}, rms="
           f"{float(np.sqrt((audio**2).mean())):.4f}"]

    # Cross-check the audio itself against a second reader and against the
    # synthetic probe that IS known to decode on this machine. That separates
    # "the wav read is wrong" from "the recogniser config is wrong".
    probes = [("wav via wave module", audio)]
    try:
        import soundfile as sf  # noqa
        out.append("  soundfile available : yes")
    except Exception:
        out.append("  soundfile available : no")
    rng = np.random.default_rng(0)
    t = np.arange(int(4.0 * sr)) / sr
    ph = 2 * np.pi * np.cumsum(120 + 25 * np.sin(2 * np.pi * 0.7 * t)) / sr
    syn = sum((1.0 / h) * np.sin(h * ph) for h in range(1, 12))
    syn = (0.3 * syn / np.abs(syn).max()).astype(np.float32)
    probes.append(("synthetic probe (known-good)", syn))

    d = os.path.join(models_dir, ASR_DIR)
    for f in ("encoder-epoch-99-avg-1.int8.onnx", "encoder-epoch-99-avg-1.onnx",
              "joiner-epoch-99-avg-1.int8.onnx", "tokens.txt"):
        fp = os.path.join(d, f)
        out.append(f"  model {f:<34} "
                   f"{os.path.getsize(fp) if os.path.exists(fp) else 'MISSING'}")

    for aud_label, aud in probes:
        for q in (True, False):
            for label, chunked, finish in (
                    ("whole clip", False, True),
                    ("20 ms chunks", True, True)):
                tag = f"{aud_label} | int8={q} | {label}"
                try:
                    rec = build_recognizer(models_dir, 1, quantised=q)
                    s = rec.create_stream()
                    if chunked:
                        step = int(sr * 0.02)
                        for i in range(0, len(aud), step):
                            s.accept_waveform(sr, np.ascontiguousarray(aud[i:i + step]))
                            while rec.is_ready(s):
                                rec.decode_stream(s)
                    else:
                        s.accept_waveform(sr, aud)
                    s.accept_waveform(sr, np.zeros(int(0.5 * sr), np.float32))
                    if finish:
                        s.input_finished()
                    while rec.is_ready(s):
                        rec.decode_stream(s)
                    txt = rec.get_result(s)
                    try:
                        toks = list(rec.tokens(s))[:6]
                    except Exception as e:
                        toks = f"<{type(e).__name__}>"
                    out.append(f"  [{tag}]\n      result: {txt[:60]!r}  tokens: {toks}")
                except Exception as e:
                    out.append(f"  [{tag}] RAISED {type(e).__name__}: {e}")
    return "\n".join(out)


def _diagnose_old(models_dir: str, audio: np.ndarray, sr: int = 16_000) -> str:
    import sherpa_onnx
    out: List[str] = []
    for label, chunked, finish in (("whole clip, no input_finished", False, False),
                                   ("whole clip + input_finished", False, True),
                                   ("20 ms chunks + input_finished", True, True)):
        try:
            rec = build_recognizer(models_dir, 1)
            s = rec.create_stream()
            if chunked:
                step = int(sr * 0.02)
                for i in range(0, len(audio), step):
                    s.accept_waveform(sr, np.ascontiguousarray(audio[i:i + step]))
                    while rec.is_ready(s):
                        rec.decode_stream(s)
            else:
                s.accept_waveform(sr, audio)
            s.accept_waveform(sr, np.zeros(int(0.5 * sr), np.float32))
            if finish:
                s.input_finished()
            while rec.is_ready(s):
                rec.decode_stream(s)
            txt = rec.get_result(s)
            try:
                toks = list(rec.tokens(s))[:8]
            except Exception as e:
                toks = f"<{type(e).__name__}: {e}>"
            try:
                ts = list(rec.timestamps(s))[:4]
            except Exception as e:
                ts = f"<{type(e).__name__}: {e}>"
            out.append(f"  [{label}]\n"
                       f"      get_result : {txt[:70]!r}\n"
                       f"      tokens[:8] : {toks}\n"
                       f"      times[:4]  : {ts}")
        except Exception as e:
            out.append(f"  [{label}] RAISED {type(e).__name__}: {e}")
    return "\n".join(out)


def trace_utterance(rec, audio: np.ndarray, step_ms: float = 20.0,
                    sr: int = 16_000) -> Dict[str, object]:
    """Stream one utterance, snapshotting the hypothesis after every step."""
    stream = rec.create_stream()
    step = int(sr * step_ms / 1000)
    snaps: List[Tuple[float, List[Tuple[str, float]]]] = []
    warned: List[str] = []

    def snapshot(t_audio: float):
        """Read the current hypothesis as words.

        Two backends, because the token/timestamp API is not stable across
        sherpa-onnx builds. Preferred: `rec.tokens()` + `rec.timestamps()`,
        which gives acoustic times. Fallback: split `rec.get_result()` text on
        whitespace — no timestamps, so `stabilisation_latency` becomes
        unavailable, but `revision_span` (the number COMMIT_TIMEOUT actually
        covers) is still measurable.

        Earlier this swallowed the exception and returned an empty list, which
        produced a confident "0 words" on one machine and 66 on another. Never
        silently degrade a measurement — say which backend is in use.
        """
        toks, times = [], []
        try:
            toks = list(rec.tokens(stream))
            times = list(rec.timestamps(stream))
        except Exception as e:
            if "tokens" not in warned:
                warned.append("tokens")
                print(f"    note: rec.tokens()/timestamps() unavailable "
                      f"({type(e).__name__}); falling back to get_result() text. "
                      f"stabilisation_latency will be NaN.", flush=True)
        if toks:
            if len(times) != len(toks):
                times = times + [0.0] * (len(toks) - len(times))
            words = tokens_to_words(toks, times)
            if words:
                return words
        # fallback: plain text, no acoustic times
        if "text" not in warned:
            warned.append("text")
        txt = rec.get_result(stream) or ""
        return [(w, float("nan")) for w in txt.split()]

    prev_len = 0
    reset_at = None
    for i in range(0, len(audio), step):
        stream.accept_waveform(sr, np.ascontiguousarray(audio[i:i + step]))
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        t_audio = min((i + step) / sr, len(audio) / sr)
        w = snapshot(t_audio)
        # An endpoint reset truncates the hypothesis. Mixing pre- and
        # post-reset hypotheses would fabricate huge "revisions", so record
        # where it happened and analyse only up to that point.
        if len(w) < prev_len - 1 and reset_at is None:
            reset_at = t_audio
        prev_len = len(w)
        snaps.append((t_audio, w))
    if reset_at is not None:
        snaps = [(t, w) for t, w in snaps if t <= reset_at]

    # flush: tail padding so the decoder emits whatever it is still holding
    stream.accept_waveform(sr, np.zeros(int(0.5 * sr), np.float32))
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    t_end = len(audio) / sr
    final = snapshot(t_end)
    snaps.append((t_end, final))

    # ---- per-word stability against the final hypothesis ----
    rows: List[Dict[str, object]] = []
    n = len(final)
    for i in range(n):
        word, w_start = final[i]
        # acoustic end of word i ~= start of word i+1 (last word: audio end).
        # NaN when the timestamp backend was unavailable -- stabilisation
        # latency is then undefined and must not be silently invented.
        w_end = final[i + 1][1] if i + 1 < n else t_end

        emit_t: Optional[float] = None
        stable_t: Optional[float] = None
        for t, hyp in snaps:
            present = len(hyp) > i
            if present and emit_t is None:
                emit_t = t
            matches = present and hyp[i][0] == word
            if matches:
                if stable_t is None:
                    stable_t = t
            else:
                stable_t = None      # changed again -> restart the clock
        if emit_t is None or stable_t is None:
            continue
        rows.append({
            "word_index": i,
            "word": word,
            "acoustic_start_s": round(w_start, 3),
            "acoustic_end_s": round(w_end, 3),
            "emit_time_s": round(emit_t, 3),
            "stable_time_s": round(stable_t, 3),
            "revision_span_ms": round((stable_t - emit_t) * 1000, 1),
            "stabilisation_latency_ms": (
                round((stable_t - w_end) * 1000, 1) if w_end == w_end
                else float("nan")),
            "was_revised": bool(stable_t > emit_t),
        })
    return {"n_words": n,
            "duration_s": round(t_end, 2),
            "final_text": " ".join(w for w, _ in final),
            "n_snapshots": len(snaps),
            "words": rows}


def tradeoff_curve(spans_ms: Sequence[float],
                   timeouts=(0, 50, 100, 200, 300, 400, 500, 700, 1000, 1500, 2000)):
    """For each commit timeout T: fraction of words released while unstable."""
    s = np.asarray(spans_ms, float)
    out = []
    for T in timeouts:
        premature = float((s > T).mean()) if len(s) else float("nan")
        out.append({"commit_timeout_ms": T,
                    "frac_released_unstable": round(premature, 4),
                    "pct_released_unstable": round(100 * premature, 2)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=DEFAULT_MODELS)
    ap.add_argument("--wavs", default=None,
                    help="glob; default = the sherpa model's own test_wavs")
    ap.add_argument("--step-ms", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fp32", action="store_true",
                    help="skip int8 and use the fp32 encoder directly")
    ap.add_argument("--tag", default="zipformer",
                    help="Default names the MODEL, not the host: this measures\n                         decoder revision behaviour, which is deterministic and\n                         hardware-independent. Only re-tag if you change model.")
    ap.add_argument("--out-prefix", default=None)
    a = ap.parse_args()

    md = os.path.abspath(a.models_dir)
    pattern = a.wavs or os.path.join(md, ASR_DIR, "test_wavs", "*.wav")
    paths = sorted(p for p in glob.glob(pattern, recursive=True)
                   if not os.path.basename(p).startswith("8k"))
    if not paths:
        sys.exit(f"no wavs matched {pattern}")

    out_prefix = a.out_prefix or os.path.join(
        os.path.dirname(__file__), "..", "results", "raw", f"commit_delay_{a.tag}")

    quantised = not a.fp32
    rec = build_recognizer(md, a.threads, quantised=quantised)

    # Sanity-decode one clip before measuring anything. The int8 encoder has
    # been observed to return an empty hypothesis on some platform builds of
    # sherpa-onnx while working on others, with the same weights and the same
    # audio -- so probe, and fall back to fp32 rather than silently reporting
    # "0 words".
    _probe = read_wav(paths[0])
    _s = rec.create_stream()
    _s.accept_waveform(16_000, _probe)
    _s.input_finished()
    while rec.is_ready(_s):
        rec.decode_stream(_s)
    if not (rec.get_result(_s) or "").strip() and quantised:
        print("  int8 encoder produced an empty hypothesis on this platform; "
              "retrying with the fp32 encoder (354 MB).")
        quantised = False
        rec = build_recognizer(md, a.threads, quantised=False)

    print(f"{len(paths)} utterances, {a.step_ms:g} ms feed step, "
          f"{a.threads} thread(s), encoder={'int8' if quantised else 'fp32'}\n")

    all_words: List[Dict[str, object]] = []
    per_utt = []
    for p in paths:
        audio = read_wav(p)
        r = trace_utterance(rec, audio, a.step_ms)
        for w in r["words"]:
            w["utt"] = os.path.basename(p)
        # Guard the tokenisation failure mode above: one "word" for a
        # multi-second utterance means the boundary marker was not recognised
        # and every downstream statistic would be silently wrong.
        if r["n_words"] <= 1 and r["duration_s"] > 2.0:
            print(f"\n!! {os.path.basename(p)}: {r['n_words']} word(s) in "
                  f"{r['duration_s']}s — the recogniser produced no usable "
                  f"hypothesis. Diagnostics:")
            print(diagnose(md, audio))
            sys.exit("aborting: cannot measure commit delay without hypotheses")
        all_words.extend(r["words"])
        per_utt.append({"file": os.path.basename(p), "duration_s": r["duration_s"],
                        "n_words": r["n_words"], "text": r["final_text"][:90]})
        print(f"  {os.path.basename(p):<10} {r['duration_s']:>5.1f}s  "
              f"{r['n_words']:>3} words  \"{r['final_text'][:60]}...\"")

    if not all_words:
        sys.exit("no words traced -- check the model paths")

    spans = [float(w["revision_span_ms"]) for w in all_words]
    stab = [float(w["stabilisation_latency_ms"]) for w in all_words
            if float(w["stabilisation_latency_ms"]) == float(w["stabilisation_latency_ms"])]
    have_times = len(stab) > 0
    revised = [w for w in all_words if w["was_revised"]]

    d_span = _dist(spans)
    d_stab = _dist(stab) if have_times else {k: float("nan") for k in
                                             ("n","mean","p50","p90","p95","p99","max","stdev")}
    curve = tradeoff_curve(spans)

    # smallest tested timeout that leaves <1% and <5% of words unstable
    def timeout_for(frac):
        for c in curve:
            if c["frac_released_unstable"] <= frac:
                return c["commit_timeout_ms"]
        return None

    summary = {
        "tag": a.tag,
        "n_utterances": len(paths),
        "n_words": len(all_words),
        "n_revised": len(revised),
        "frac_revised": round(len(revised) / len(all_words), 4),
        "feed_step_ms": a.step_ms,
        "model_decode_chunk_ms": 320.0,
        "revision_span_ms": {k: round(v, 1) for k, v in d_span.items()},
        "stabilisation_latency_ms": ({k: round(v, 1) for k, v in d_stab.items()}
                                     if have_times else None),
        "timestamps_available": have_times,
        "tradeoff_curve": curve,
        "timeout_for_99pct_stable_ms": timeout_for(0.01),
        "timeout_for_95pct_stable_ms": timeout_for(0.05),
        "accentbridge_current_commit_timeout_ms": 700.0,
        "per_utterance": per_utt,
        "caveats": [
            "Quantised to the model's 320 ms decode chunk -- a real property, "
            "not measurement error.",
            "Stability is measured against this utterance's final hypothesis; "
            "a word wrong from the start and never corrected counts as stable. "
            "This measures instability, not accuracy.",
            "Clean read speech. Accented speech should revise more; re-run "
            "with --wavs over L2-ARCTIC to check.",
        ],
    }

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    write_csv(out_prefix + ".csv", all_words)
    with open(out_prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*62}\nCOMMIT DELAY -- how long until a word stops changing\n{'='*62}")
    print(f"{len(all_words)} words, {len(revised)} revised after first emission "
          f"({100*len(revised)/len(all_words):.1f}%)\n")
    print(f"{'':24}{'p50':>8}{'p90':>8}{'p95':>8}{'p99':>8}{'max':>8}")
    shown = [("revision span (ms)", d_span)]
    if have_times:
        shown.append(("stabilisation lat (ms)", d_stab))
    else:
        print("  (stabilisation latency unavailable: no token timestamps "
              "from this sherpa-onnx build)")
    for name, d in shown:
        print(f"{name:<24}{d['p50']:>8.0f}{d['p90']:>8.0f}{d['p95']:>8.0f}"
              f"{d['p99']:>8.0f}{d['max']:>8.0f}")
    print(f"\n{'commit timeout':>16}   {'% words released while unstable':>32}")
    for c in curve:
        bar = "#" * int(round(c["pct_released_unstable"] / 2))
        print(f"{c['commit_timeout_ms']:>13} ms   {c['pct_released_unstable']:>6.2f}%  {bar}")
    print(f"\n<1% unstable needs {summary['timeout_for_99pct_stable_ms']} ms; "
          f"<5% needs {summary['timeout_for_95pct_stable_ms']} ms. "
          f"accentbridge currently uses 700 ms.")
    print(f"\nwrote {out_prefix}.csv and {out_prefix}_summary.json")


if __name__ == "__main__":
    main()
