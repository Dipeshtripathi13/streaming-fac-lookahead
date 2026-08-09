"""Measure t_buffer: the I/O and jitter-buffer term of the latency budget.

t_buffer is the last estimated number in the cascade budget (it has been
carried as a ~30 ms placeholder). It is the delay contributed by the audio
subsystem rather than by the model:

    t_buffer = capture_beyond_block + playback + jitter_safety

and it is *system*-dependent, which is why it cannot be inferred from the
model config the way t_algorithmic can, or from a forward pass the way
t_compute can. It has to be measured on the machine that will run the demo.

WHAT THIS DOES NOT DOUBLE-COUNT
-------------------------------
The single easiest way to get t_buffer wrong is to count block accumulation
twice. The model already refuses to emit until it has a full chunk, and that
wait is t_algorithmic's `chunk_ms` term. If the audio device's blocksize
equals the model's chunk, then the device's "input latency" *is* that same
wait, and adding it to t_buffer inflates the headline number by a whole chunk.

So we report `capture_beyond_block`: device input latency minus the block
accumulation time, floored at zero. Pass --blocksize-ms and --chunk-ms
separately if they differ; if they are equal (the usual streaming design) the
subtraction is exact.

THREE MEASUREMENTS, INCREASING IN HONESTY
-----------------------------------------
1. `probe`     -- what the device *claims* (PortAudio reported latency).
                  Cheap, and usually optimistic: it omits ADC/DAC group delay
                  and any buffering the driver does not disclose.
2. `jitter`    -- how late callbacks actually arrive under load. This sets the
                  jitter buffer you need to avoid dropouts, which is a real
                  latency cost and is invisible to (1).
3. `loopback`  -- the ground truth: emit a chirp, record it, cross-correlate.
                  Captures everything, including what the driver hides. Works
                  acoustically (built-in speaker -> built-in mic) so it needs
                  no cable; pass --distance-m to subtract air propagation.

Report (3) when you can and (1)+(2) when you cannot, and say which. A number
from (1) alone should not go in a paper as measured.

Usage
-----
    python bench/bench_tbuffer.py --self-test          # no audio hardware needed
    python bench/bench_tbuffer.py --mode probe
    python bench/bench_tbuffer.py --mode jitter --seconds 30
    python bench/bench_tbuffer.py --mode loopback --reps 20 --distance-m 0.3
    python bench/bench_tbuffer.py --mode all --out results/raw/tbuffer_m4.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from sfac.latency import _dist  # reuse the paper's percentile estimator
except Exception:  # pragma: no cover - keeps the bench runnable standalone
    def _dist(xs: List[float]) -> Dict[str, float]:
        s = sorted(xs)
        n = len(s)

        def q(p: float) -> float:
            if n == 1:
                return s[0]
            k = min(n - 1, max(0, int(round(p * (n - 1)))))
            return s[k]

        return {
            "n": float(n), "mean": statistics.fmean(s), "p50": q(0.50),
            "p90": q(0.90), "p95": q(0.95), "p99": q(0.99), "max": s[-1],
            "stdev": statistics.pstdev(s) if n > 1 else 0.0,
        }


SPEED_OF_SOUND_M_S = 343.0  # 20 C, dry air


# ==========================================================================
# Estimators -- pure functions, so they can be tested without a sound card
# ==========================================================================

def chirp(n: int, sr: int, f0: float = 500.0, f1: float = 8000.0) -> np.ndarray:
    """Linear chirp. Better than a click for delay estimation: the energy is
    spread over time so it survives a low peak-amplitude limit, while the
    autocorrelation stays sharp."""
    t = np.arange(n, dtype=np.float64) / sr
    dur = max(t[-1], 1e-9)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))
    w = np.hanning(n)
    return (np.sin(phase) * w).astype(np.float32)


def estimate_delay_samples(rec: np.ndarray, ref: np.ndarray) -> Tuple[int, float]:
    """Cross-correlation delay of `ref` inside `rec`, by FFT.

    Returns (lag_samples, normalised_peak). The peak is the correlation
    coefficient at the best lag; treat anything below ~0.2 as a failed
    measurement (too quiet, wrong device, clipping) rather than a real delay.
    Silently accepting a low-confidence peak is how you get a plausible-looking
    latency number that is pure noise.
    """
    rec = np.asarray(rec, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if rec.size < ref.size:
        rec = np.pad(rec, (0, ref.size - rec.size))
    rec = rec - rec.mean()
    ref = ref - ref.mean()

    n = 1 << int(math.ceil(math.log2(rec.size + ref.size)))
    R = np.fft.rfft(rec, n)
    F = np.fft.rfft(ref, n)
    cc = np.fft.irfft(R * np.conj(F), n)[: rec.size]

    lag = int(np.argmax(cc))
    denom = math.sqrt(float((rec ** 2).sum()) * float((ref ** 2).sum()))
    peak = float(cc[lag] / denom) if denom > 0 else 0.0
    return lag, peak


def required_jitter_buffer_ms(
    arrivals_ms: List[float], period_ms: float, target_underrun: float = 1e-3
) -> Dict[str, float]:
    """How much buffer do late callbacks force us to hold?

    `arrivals_ms` are monotonic callback timestamps. A callback that arrives
    later than its nominal deadline starves the output unless we already hold
    enough audio to cover the gap. The buffer we need is a high quantile of
    the *lateness* distribution, not its mean -- the mean is ~0 by
    construction and tells you nothing about dropouts.

    Lateness is measured against a drift-corrected ideal clock. Using the
    first timestamp plus i*period instead would let a small constant clock
    offset accumulate into a fake linear trend that dwarfs the real jitter.
    """
    if len(arrivals_ms) < 8:
        return {"n": float(len(arrivals_ms)), "insufficient_samples": 1.0}

    a = np.asarray(arrivals_ms, dtype=np.float64)
    a = a - a[0]
    i = np.arange(a.size, dtype=np.float64)

    # least-squares fit removes drift between the audio clock and perf_counter
    slope, intercept = np.polyfit(i, a, 1)
    ideal = intercept + slope * i
    lateness = a - ideal

    late_only = np.maximum(lateness, 0.0)
    q = 100.0 * (1.0 - target_underrun)
    buf = float(np.percentile(late_only, min(q, 99.99)))

    gaps = np.diff(a)
    return {
        "n": float(a.size),
        "measured_period_ms": float(slope),
        "nominal_period_ms": float(period_ms),
        "clock_drift_pct": float(100.0 * (slope - period_ms) / period_ms)
        if period_ms
        else 0.0,
        "lateness_p50_ms": float(np.percentile(lateness, 50)),
        "lateness_p95_ms": float(np.percentile(lateness, 95)),
        "lateness_p99_ms": float(np.percentile(lateness, 99)),
        "lateness_max_ms": float(lateness.max()),
        "gap_max_ms": float(gaps.max()) if gaps.size else 0.0,
        "late_callback_rate": float((lateness > 0.5 * period_ms).mean()),
        "target_underrun_prob": target_underrun,
        "required_buffer_ms": buf,
    }


def capture_beyond_block_ms(device_input_latency_ms: float, blocksize_ms: float) -> float:
    """Device input latency minus block accumulation, floored at zero.

    See the module docstring: the block wait is already t_algorithmic's chunk
    term. Only the excess belongs in t_buffer.
    """
    return max(0.0, device_input_latency_ms - blocksize_ms)


# ==========================================================================
# Audio-dependent measurements
# ==========================================================================

def _import_sd():
    try:
        import sounddevice as sd  # type: ignore
        return sd
    except Exception as e:
        print(
            "sounddevice is required for audio modes.\n"
            "  pip install sounddevice\n"
            "  (macOS also needs PortAudio: brew install portaudio)\n"
            f"import error: {e}",
            file=sys.stderr,
        )
        return None


def mode_probe(blocksize_ms: float, sr: int) -> Dict[str, object]:
    sd = _import_sd()
    if sd is None:
        return {"error": "sounddevice unavailable"}

    block = int(round(sr * blocksize_ms / 1000.0))
    out: Dict[str, object] = {"blocksize_frames": block, "samplerate": sr}
    try:
        out["default_device"] = sd.query_devices(kind="input")["name"]
        out["default_output"] = sd.query_devices(kind="output")["name"]
    except Exception as e:
        out["device_query_error"] = str(e)

    # PortAudio only fills in latency once a stream exists.
    for latency_hint in ("low", "high"):
        try:
            with sd.Stream(
                samplerate=sr, blocksize=block, channels=1,
                dtype="float32", latency=latency_hint,
            ) as s:
                in_ms = float(s.latency[0]) * 1000.0
                out_ms = float(s.latency[1]) * 1000.0
                out[f"{latency_hint}_input_ms"] = round(in_ms, 3)
                out[f"{latency_hint}_output_ms"] = round(out_ms, 3)
                out[f"{latency_hint}_reported_roundtrip_ms"] = round(in_ms + out_ms, 3)
                out[f"{latency_hint}_capture_beyond_block_ms"] = round(
                    capture_beyond_block_ms(in_ms, blocksize_ms), 3
                )
        except Exception as e:
            out[f"{latency_hint}_error"] = str(e)
    # Does the reported latency scale with the requested block? If it does, the
    # figure is dominated by however many blocks PortAudio chose to queue, not
    # by the hardware -- which makes it useless as a device characterisation.
    # Observed on an M4: 412 ms reported for a 40 ms block, 10x the block, with
    # the low/high hint ignored. That needs explaining before it is cited.
    scaling = {}
    for bms in (10.0, 20.0, 40.0, 80.0):
        try:
            with sd.Stream(samplerate=sr, blocksize=int(round(sr * bms / 1000.0)),
                           channels=1, dtype="float32", latency="low") as s2:
                scaling[f"{bms:g}ms_block"] = {
                    "input_ms": round(float(s2.latency[0]) * 1000.0, 2),
                    "output_ms": round(float(s2.latency[1]) * 1000.0, 2),
                    "input_over_block": round(
                        float(s2.latency[0]) * 1000.0 / bms, 2),
                }
        except Exception as e:
            scaling[f"{bms:g}ms_block"] = {"error": str(e)}
    out["blocksize_scaling"] = scaling
    ratios = [v["input_over_block"] for v in scaling.values()
              if isinstance(v, dict) and "input_over_block" in v]
    if ratios and max(ratios) > 3.0:
        out["reported_latency_verdict"] = (
            f"reported input latency is {min(ratios):.1f}-{max(ratios):.1f}x the "
            "block size -- dominated by driver queueing, NOT a hardware figure. "
            "Do not cite it as t_buffer; the loopback is the only valid source."
        )
    out["caveat"] = (
        "PortAudio-reported latency omits ADC/DAC group delay and undisclosed "
        "driver buffering. Treat as a lower bound; prefer loopback."
    )
    return out


def mode_jitter(
    blocksize_ms: float, sr: int, seconds: float, target_underrun: float
) -> Dict[str, object]:
    sd = _import_sd()
    if sd is None:
        return {"error": "sounddevice unavailable"}

    block = int(round(sr * blocksize_ms / 1000.0))
    arrivals: List[float] = []
    statuses: List[str] = []

    def cb(indata, outdata, frames, time_info, status):  # noqa: ANN001
        arrivals.append(time.perf_counter_ns() / 1e6)
        if status:
            statuses.append(str(status))
        outdata[:] = 0.0

    try:
        with sd.Stream(
            samplerate=sr, blocksize=block, channels=1,
            dtype="float32", latency="low", callback=cb,
        ):
            time.sleep(seconds)
    except Exception as e:
        return {"error": f"stream failed: {e}"}

    warm = 5  # first few callbacks pay allocation and page-fault costs
    arr = arrivals[warm:] if len(arrivals) > warm else arrivals
    res = required_jitter_buffer_ms(arr, blocksize_ms, target_underrun)
    res["dropped_warmup_callbacks"] = float(warm)
    res["xrun_statuses"] = statuses[:20]
    res["xrun_count"] = float(len(statuses))
    if statuses:
        res["note"] = (
            "PortAudio reported over/underflows. The required_buffer_ms below "
            "is optimistic: the stream already failed at this blocksize."
        )
    return res


def amplitude_invariance(sd, sr: int, chirp_ms: float, tail_ms: float,
                         lo_amp: float = 0.25, hi_amp: float = 0.95,
                         reps: int = 4) -> Dict[str, object]:
    """Does the captured level respond to output amplitude at all?

    This is the decisive diagnostic and it is cheap. If the speaker signal is
    reaching the microphone -- however attenuated -- then nearly quadrupling the
    output amplitude must raise the captured peak roughly proportionally. If the
    captured level is *invariant*, the output contributes nothing: either it is
    not being emitted (routing/mute) or it is being actively cancelled (macOS
    echo cancellation, which scales with its reference and so defeats louder
    signals by design).

    Observed on an M4: 0.5 -> 0.95 amplitude moved the captured peak from
    0.00505 to 0.00468, i.e. slightly DOWN. That is noise, not signal.
    """
    n_ref = int(round(sr * chirp_ms / 1000.0))
    pad = int(round(sr * tail_ms / 1000.0))
    out: Dict[str, object] = {"lo_amp": lo_amp, "hi_amp": hi_amp, "reps": reps}
    levels = {}
    for tag, amp in (("lo", lo_amp), ("hi", hi_amp)):
        peaks = []
        ref = chirp(n_ref, sr) * amp
        play = np.concatenate([ref, np.zeros(pad, dtype=np.float32)])
        for _ in range(reps):
            try:
                rec = sd.playrec(play, samplerate=sr, channels=1, blocking=True)
            except Exception as e:
                return {"error": f"playrec failed: {e}"}
            peaks.append(float(np.max(np.abs(np.asarray(rec[:, 0])))))
            time.sleep(0.05)
        levels[tag] = float(np.median(peaks))
    out["captured_peak_lo"] = round(levels["lo"], 6)
    out["captured_peak_hi"] = round(levels["hi"], 6)
    amp_ratio = hi_amp / max(lo_amp, 1e-9)
    cap_ratio = levels["hi"] / max(levels["lo"], 1e-12)
    out["amplitude_ratio"] = round(amp_ratio, 2)
    out["captured_ratio"] = round(cap_ratio, 3)
    # If output were reaching the mic, captured_ratio should track amp_ratio.
    # Allow a wide margin; we only need to catch total invariance.
    out["output_reaches_mic"] = cap_ratio > 1.5
    if not out["output_reaches_mic"]:
        out["diagnosis"] = (
            f"Captured level is INVARIANT to output amplitude "
            f"({amp_ratio:.1f}x louder gave {cap_ratio:.2f}x captured). The "
            "speaker signal is not reaching the microphone at all. Either the "
            "output is not being emitted (check volume/routing -- can you HEAR "
            "the chirps?) or it is being actively cancelled by the OS. macOS "
            "echo cancellation scales with its reference, so a louder chirp "
            "cannot defeat it. An acoustic loopback is not usable here; use a "
            "wired loopback with --distance-m 0, or measure on a platform "
            "without AEC in the path."
        )
    return out


def mode_loopback(
    sr: int, reps: int, chirp_ms: float, tail_ms: float,
    distance_m: float, amplitude: float, min_peak: float,
) -> Dict[str, object]:
    sd = _import_sd()
    if sd is None:
        return {"error": "sounddevice unavailable"}

    n_ref = int(round(sr * chirp_ms / 1000.0))
    ref = chirp(n_ref, sr) * amplitude
    pad = int(round(sr * tail_ms / 1000.0))
    play = np.concatenate([ref, np.zeros(pad, dtype=np.float32)])

    lat_ms: List[float] = []
    peaks: List[float] = []
    rms_in: List[float] = []
    peak_in: List[float] = []
    rejected = 0
    for _ in range(reps):
        try:
            rec = sd.playrec(play, samplerate=sr, channels=1, blocking=True)
        except Exception as e:
            return {"error": f"playrec failed: {e}"}
        ch = np.asarray(rec[:, 0], dtype=np.float64)
        # Capture-level diagnostics. Without these, a failed loopback is
        # indistinguishable between "the microphone returned silence" (macOS
        # denies mic access by returning zeros, not an error) and "the chirp
        # was captured but decorrelated" (echo cancellation, wrong device).
        # Those need opposite fixes, so guessing is not good enough.
        rms_in.append(float(np.sqrt(np.mean(ch ** 2))))
        peak_in.append(float(np.max(np.abs(ch))) if ch.size else 0.0)
        lag, peak = estimate_delay_samples(ch, ref)
        peaks.append(peak)
        if peak < min_peak:
            rejected += 1
            continue
        lat_ms.append(1000.0 * lag / sr)
        time.sleep(0.05)

    air_ms = 1000.0 * distance_m / SPEED_OF_SOUND_M_S
    med_rms = float(np.median(rms_in)) if rms_in else 0.0
    med_peak_in = float(np.median(peak_in)) if peak_in else 0.0
    SILENT_RMS = 1e-4      # below this the input is silence, not quiet audio
    out: Dict[str, object] = {
        "reps": reps,
        "accepted": len(lat_ms),
        "rejected_low_correlation": rejected,
        "min_peak_threshold": min_peak,
        "peak_p50": round(float(np.median(peaks)), 4) if peaks else 0.0,
        "air_propagation_ms": round(air_ms, 3),
        "distance_m": distance_m,
        "input_rms_p50": round(med_rms, 6),
        "input_peak_p50": round(med_peak_in, 6),
        "input_silent": med_rms < SILENT_RMS,
        "input_clipped": med_peak_in > 0.99,
    }
    if not lat_ms:
        if med_rms < SILENT_RMS:
            out["diagnosis"] = (
                f"INPUT IS SILENT (rms {med_rms:.2e}). The microphone returned "
                "zeros, so this is a capture problem, not an acoustics problem. "
                "Almost always macOS microphone permission: System Settings > "
                "Privacy & Security > Microphone, enable the app that launched "
                "this (Terminal / iTerm). macOS returns silence rather than an "
                "error when access is denied. Raising --amplitude will NOT help."
            )
        elif med_peak_in > 0.99:
            out["diagnosis"] = (
                f"INPUT IS CLIPPING (peak {med_peak_in:.3f}). Lower --amplitude "
                "or reduce input gain; a clipped chirp decorrelates."
            )
        else:
            out["diagnosis"] = (
                f"Input has signal (rms {med_rms:.2e}, peak {med_peak_in:.3f}) "
                "but it does not correlate with the chirp. Likely the built-in "
                "mic's voice isolation / echo cancellation suppressing it, or "
                "output routed to a different device. Try: System Settings > "
                "Sound > Input, disable Voice Isolation; or use a wired "
                "loopback and pass --distance-m 0."
            )
        out["error"] = "every rep failed the correlation threshold"
        # The loopback failed. Establish whether the output reaches the mic at
        # all -- that single fact separates "cancelled/muted" from "present but
        # decorrelated", and it took three manual runs to notice by hand.
        out["amplitude_invariance"] = amplitude_invariance(
            sd, sr, chirp_ms, tail_ms)
        return out

    raw = _dist(lat_ms)
    out["roundtrip_raw"] = {k: round(v, 3) for k, v in raw.items()}
    out["roundtrip_minus_air"] = {
        k: round(v - air_ms, 3) if k in ("mean", "p50", "p90", "p95", "p99", "max") else round(v, 3)
        for k, v in raw.items()
    }
    out["note"] = (
        "Round-trip covers output + input. It includes one block of capture "
        "accumulation, which t_algorithmic already counts -- subtract "
        "blocksize_ms before adding to the budget."
    )
    return out


# ==========================================================================
# Self-test: validates the estimators against known answers
# ==========================================================================

def self_test() -> int:
    print("bench_tbuffer self-test (no audio hardware required)")
    fails = 0
    sr = 48000

    # --- 1. cross-correlation recovers a known delay ---
    ref = chirp(int(0.02 * sr), sr)
    for true_lag in (0, 137, 1024, 5000):
        rec = np.zeros(true_lag + ref.size + 2000, dtype=np.float32)
        rec[true_lag: true_lag + ref.size] = ref
        lag, peak = estimate_delay_samples(rec, ref)
        ok = lag == true_lag and peak > 0.9
        print(f"  delay {true_lag:>5} -> {lag:>5}  peak={peak:.3f}  {'ok' if ok else 'FAIL'}")
        fails += 0 if ok else 1

    # --- 2. survives noise and attenuation, as an acoustic path would ---
    rng = np.random.default_rng(0)
    true_lag = 911
    rec = np.zeros(true_lag + ref.size + 4000, dtype=np.float32)
    rec[true_lag: true_lag + ref.size] = 0.05 * ref
    rec += 0.01 * rng.standard_normal(rec.size).astype(np.float32)
    lag, peak = estimate_delay_samples(rec, ref)
    ok = abs(lag - true_lag) <= 2
    print(f"  noisy/attenuated {true_lag} -> {lag}  peak={peak:.3f}  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 3. rejects a correlation-free recording instead of inventing a lag ---
    noise = (0.01 * rng.standard_normal(20000)).astype(np.float32)
    _, peak = estimate_delay_samples(noise, ref)
    ok = peak < 0.2
    print(f"  pure noise peak={peak:.4f} (want <0.2)  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 4. jitter buffer: clean periodic stream needs ~nothing ---
    period = 40.0
    clean = [i * period for i in range(500)]
    r = required_jitter_buffer_ms(clean, period)
    ok = r["required_buffer_ms"] < 1.0 and abs(r["measured_period_ms"] - period) < 1e-6
    print(f"  clean stream buffer={r['required_buffer_ms']:.3f} ms (want ~0)  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 5. jitter buffer: a known 12 ms spike must show up in the tail ---
    spiky = []
    for i in range(500):
        t = i * period
        if i % 50 == 7:
            t += 12.0
        spiky.append(t)
    r = required_jitter_buffer_ms(spiky, period, target_underrun=1e-2)
    ok = 5.0 < r["required_buffer_ms"] <= 13.0 and r["lateness_max_ms"] > 10.0
    print(f"  spiky stream buffer={r['required_buffer_ms']:.2f} ms, max={r['lateness_max_ms']:.2f} "
          f"(want spike visible)  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 6. drift must not be mistaken for jitter ---
    drifty = [i * period * 1.001 for i in range(500)]
    r = required_jitter_buffer_ms(drifty, period)
    ok = r["required_buffer_ms"] < 1.0 and abs(r["clock_drift_pct"] - 0.1) < 0.01
    print(f"  drifting clock buffer={r['required_buffer_ms']:.3f} ms, "
          f"drift={r['clock_drift_pct']:.3f}%  {'ok' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # --- 7. no double counting of block accumulation ---
    cases = [(40.0, 40.0, 0.0), (52.0, 40.0, 12.0), (10.0, 40.0, 0.0)]
    for dev, blk, want in cases:
        got = capture_beyond_block_ms(dev, blk)
        ok = abs(got - want) < 1e-9
        print(f"  capture_beyond_block(dev={dev}, block={blk}) = {got} (want {want})  "
              f"{'ok' if ok else 'FAIL'}")
        fails += 0 if ok else 1

    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["probe", "jitter", "loopback", "all"])
    ap.add_argument("--self-test", action="store_true",
                    help="validate the estimators against known answers, "
                         "then exit. Needs no sound card.")
    ap.add_argument("--samplerate", type=int, default=48000)
    ap.add_argument("--blocksize-ms", type=float, default=40.0,
                    help="audio device block. Default matches the model chunk.")
    ap.add_argument("--chunk-ms", type=float, default=40.0,
                    help="model chunk, for the double-counting subtraction.")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--chirp-ms", type=float, default=20.0)
    ap.add_argument("--tail-ms", type=float, default=400.0)
    ap.add_argument("--distance-m", type=float, default=0.0,
                    help="speaker-to-mic distance for acoustic loopback; "
                         "subtracts air propagation. 0 = wired loopback.")
    ap.add_argument("--amplitude", type=float, default=0.5)
    ap.add_argument("--min-peak", type=float, default=0.2)
    ap.add_argument("--target-underrun", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.self_test:
        sys.exit(self_test())

    res: Dict[str, object] = {
        "schema": "tbuffer/1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "config": {
            "samplerate": a.samplerate,
            "blocksize_ms": a.blocksize_ms,
            "chunk_ms": a.chunk_ms,
        },
    }

    if a.mode in ("probe", "all"):
        print("== probe: device-reported latency ==")
        res["probe"] = mode_probe(a.blocksize_ms, a.samplerate)
        print(json.dumps(res["probe"], indent=2))

    if a.mode in ("jitter", "all"):
        print(f"\n== jitter: {a.seconds:.0f}s duplex stream ==")
        res["jitter"] = mode_jitter(
            a.blocksize_ms, a.samplerate, a.seconds, a.target_underrun
        )
        print(json.dumps(res["jitter"], indent=2))

    if a.mode in ("loopback", "all"):
        kind = "wired" if a.distance_m == 0 else f"acoustic @{a.distance_m} m"
        print(f"\n== loopback ({kind}): {a.reps} chirps ==")
        if a.distance_m == 0:
            print("  distance-m=0: assuming a wired/virtual loopback. For the "
                  "built-in speaker->mic path pass e.g. --distance-m 0.3")
        res["loopback"] = mode_loopback(
            a.samplerate, a.reps, a.chirp_ms, a.tail_ms,
            a.distance_m, a.amplitude, a.min_peak,
        )
        print(json.dumps(res["loopback"], indent=2))

    # ---- the number that goes in the budget ----
    verdict: Dict[str, object] = {}
    lb = res.get("loopback") or {}
    pr = res.get("probe") or {}
    jt = res.get("jitter") or {}
    jitter_ms = float(jt.get("required_buffer_ms", 0.0) or 0.0)

    if isinstance(lb, dict) and "roundtrip_minus_air" in lb:
        rt = float(lb["roundtrip_minus_air"]["p50"])
        io_ms = max(0.0, rt - a.blocksize_ms)
        verdict["source"] = "loopback (measured)"
        verdict["io_ms"] = round(io_ms, 2)
    elif isinstance(pr, dict) and "low_reported_roundtrip_ms" in pr:
        io_ms = float(pr.get("low_capture_beyond_block_ms", 0.0)) + float(
            pr.get("low_output_ms", 0.0)
        )
        verdict["source"] = "device-reported (LOWER BOUND -- not measured)"
        verdict["io_ms"] = round(io_ms, 2)
    else:
        io_ms = float("nan")
        verdict["source"] = "unavailable"

    if not math.isnan(io_ms):
        verdict["jitter_buffer_ms"] = round(jitter_ms, 2)
        verdict["t_buffer_ms"] = round(io_ms + jitter_ms, 2)
        verdict["excludes"] = (
            f"block accumulation ({a.blocksize_ms} ms), already counted in "
            "t_algorithmic as chunk_ms"
        )
    res["verdict"] = verdict

    print("\n== t_buffer ==")
    print(json.dumps(verdict, indent=2))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"\nwrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
