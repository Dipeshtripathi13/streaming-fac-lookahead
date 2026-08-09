# t_buffer on the M4: jitter measured, I/O still unmeasured

**Status: partially resolved.** The jitter term is measured and is effectively
zero. The I/O term is **not** measured, and the number the tool falls back to
must not be cited.

Evidence: `results/raw/tbuffer_m4.json`, two runs on 9 Aug 2026, macOS 26.5
(25F71), Apple M4, on AC power. Tool: `bench/bench_tbuffer.py` (12 estimator
self-tests passing, validated with no sound card).

## 1. Measured: the jitter buffer is ~0.1 ms

Two independent 30 s duplex runs at 40 ms blocks:

| | run 1 | run 2 |
|---|---:|---:|
| callbacks | 748 | 747 |
| lateness p99 | 0.065 ms | 0.060 ms |
| lateness max | 0.166 ms | 0.084 ms |
| required buffer (1e-3 underrun) | 0.143 ms | **0.080 ms** |
| xruns | 0 | 0 |
| late-callback rate | 0.0 | 0.0 |
| clock drift | 2.2e-5 % | -1.2e-4 % |

**Take 0.1 ms, or simply "negligible".** CoreAudio on Apple Silicon delivers
40 ms callbacks with sub-0.1 ms tail jitter and zero dropouts, so the jitter
safety margin contributes nothing meaningful to the latency budget. This
replaces guesswork with a measurement and it is the part of the ~30 ms
placeholder that can now be retired.

## 2. NOT measured: the I/O term. Do not cite 640 ms.

The acoustic loopback failed on both runs — 20/20 reps rejected, correlation
peak 0.042–0.046 against a threshold of 0.20 (pure noise scores 0.029 in the
self-test). With no loopback, the tool falls back to device-reported latency and
labels it `LOWER BOUND -- not measured`. That fallback figure is
**t_buffer = 639.7 ms**, and it is not a hardware property:

| requested block | reported input latency | ÷ block | excess over block |
|---|---:|---:|---:|
| 10 ms | 211.69 ms | 21.2× | 201.69 |
| 20 ms | 221.69 ms | 11.1× | **201.69** |
| 40 ms | 412.35 ms | 10.3× | 372.35 |
| 80 ms | 793.69 ms | 9.9× | 713.69 |

At 10 and 20 ms the excess over the block is **identically 201.69 ms** — a fixed
constant, then roughly doubling and quadrupling. That is PortAudio/CoreAudio
queueing, not ADC/DAC delay. The `low` and `high` latency hints also return
identical values, i.e. the hint is ignored. Real macOS audio applications
achieve sub-20 ms round-trip, so 680 ms is wrong by more than an order of
magnitude.

The tool now emits this verdict automatically when the ratio exceeds 3×.

## 3. Why the loopback failed, and it is not what we first guessed

First hypothesis was macOS microphone permission, because macOS returns
*silence* rather than an error when access is denied. The capture-level
diagnostics added after run 1 refute it:

```
input_rms_p50   0.001213      input_silent   false
input_peak_p50  0.005046      input_clipped  false
```

The microphone is capturing — that level is ambient room noise, about −46 dBFS —
but the chirp specifically is absent. Signal present, target absent.

**Leading explanation: macOS echo cancellation.** When one process holds both
the microphone and the speaker, macOS applies AEC whose explicit purpose is to
remove the speaker's output from the microphone input. An acoustic loopback
through built-in speaker → built-in mic is therefore structurally hostile on
this platform: the OS is designed to defeat exactly this measurement.

*Methodological note worth keeping:* without the RMS/peak diagnostics, "silent
input" and "chirp captured but decorrelated" are indistinguishable, and they
need opposite fixes — permissions versus disabling audio processing. Raising
`--amplitude` would have been the wrong response to the wrong diagnosis, and we
would have concluded "mic permission" and been wrong.

## 4. What to do next

In order of cost:

1. **Disable mic processing.** Launch the benchmark, then *while it runs* set
   Control Center → Mic Mode → **Wide Spectrum** (the control only appears while
   an app holds the microphone). Raise speaker volume and use
   `--amplitude 0.95`.
2. **Wired loopback.** A 3.5 mm headphone-to-mic cable, or any USB audio
   interface, with `--distance-m 0`. The signal never goes acoustic so AEC
   cannot touch it. This is the definitive measurement.
3. **Failing both**, report the budget with the jitter term measured (~0.1 ms)
   and the I/O term as an explicit stated limitation. Do **not** substitute the
   device-reported number.

## 5. What the paper should say now

- t_buffer's **jitter component is measured at ~0.1 ms on Apple Silicon** and is
  negligible; two independent runs, zero xruns.
- t_buffer's **I/O component remains unmeasured on this hardware.** The §7
  limitation stays, but it is now sharper: it is not "we did not measure it", it
  is "an acoustic loopback cannot measure it on macOS because the OS cancels the
  test signal, and the driver-reported figure is queueing rather than hardware".
- Keep the `capture_beyond_block` framing: whatever the I/O term turns out to be,
  the block-accumulation part of it is already counted in `t_algorithmic` as
  `chunk_ms` and must not be added twice.
