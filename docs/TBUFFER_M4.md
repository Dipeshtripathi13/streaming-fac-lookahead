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

The acoustic loopback failed on both runs, 20/20 reps rejected, correlation
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

At 10 and 20 ms the excess over the block is **identically 201.69 ms**, a fixed
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

The microphone is capturing, that level is ambient room noise, about −46 dBFS,
but the chirp specifically is absent. Signal present, target absent.

**Then a third run settled it: the captured level is invariant to output
amplitude.**

| output amplitude | captured peak | captured rms | correlation peak |
|---:|---:|---:|---:|
| 0.50 | 0.00505 | 0.001213 | 0.042 |
| 0.95 | 0.00468 | 0.001128 | 0.029 |
| 0.95 (repeat) | 0.00446 | 0.000956 | 0.033 |

Nearly doubling the output changed the captured signal by **nothing**, it
drifted slightly *down*, consistent with room-noise variation. If the chirp were
reaching the microphone at all, even heavily attenuated, doubling amplitude would
roughly double the captured peak. It does not. **The speaker contributes zero
measurable energy to the microphone input.**

That leaves exactly two causes, distinguished by one free observation, whether
the chirps are audible:

- **Audible** → output works, so the OS is cancelling it. macOS applies echo
  cancellation whenever one process holds both mic and speaker, and adaptive
  cancellation scales with its reference, which is precisely why a louder chirp
  cannot defeat it. An acoustic loopback is then *structurally* unmeasurable on
  this platform: the OS is designed to defeat this measurement.
- **Inaudible** → the output is not being emitted (routing or volume), and
  amplitude is equally irrelevant.

Either way the acoustic route is a dead end on this machine, and amplitude is
not the lever.

*Tooling consequence:* `amplitude_invariance()` now runs automatically whenever
the loopback fails, comparing captured level at 0.25 vs 0.95 amplitude. It took
three manual runs to notice the invariance by hand; the tool now reports it in
one.

*Methodological note worth keeping:* without the RMS/peak diagnostics, "silent
input" and "chirp captured but decorrelated" are indistinguishable, and they
need opposite fixes, permissions versus disabling audio processing. Raising
`--amplitude` would have been the wrong response to the wrong diagnosis, and we
would have concluded "mic permission" and been wrong.

## 4. What to do next

`--amplitude` has been tried and ruled out (see the invariance table above).
Remaining options, in order of value:

1. **Wired loopback.** A 3.5 mm headphone-to-mic cable, or any USB audio
   interface, with `--distance-m 0`. The signal never goes acoustic, so AEC
   cannot touch it. This is the definitive measurement and the only one that
   fully settles the I/O term on this machine.
2. **Measure on a platform without AEC in the path.** The project already
   targets a Raspberry Pi and ARM64 Linux (§8), neither of which applies
   system-wide echo cancellation. Arguably *more* relevant than the Mac anyway:
   the paper's claim is about commodity and embedded deployment, and an
   AEC-free Linux measurement would characterise the class of device the
   deployment argument is actually about.
3. **Report the limitation as it stands.** Jitter measured (~0.1 ms), I/O term
   explicitly unmeasured with the reason given. Do **not** substitute the
   device-reported number.

Option 2 is probably the better use of effort than hunting for a cable: it
serves the paper's actual claim rather than just closing a hole on hardware the
paper does not centre.

## 4b. The virtual-device route was tried too, and it also fails

**Attempted 7 Sep 2026, rejected.** Option 1 above suggests a wired loopback so
the signal never goes acoustic. A virtual audio device is the same idea without
the cable: BlackHole 2ch is installed on this machine, exposes 2 in / 2 out, and
a duplex stream on it returns the signal digitally, where echo cancellation
cannot reach. `bench_tbuffer.py` gained a `--device` option for this.

The path works mechanically -- a 1 kHz tone written to BlackHole comes back at
peak 0.057 -- but it does not yield a usable number, for two reasons, and the
second is the disqualifying one.

**The detections are not trustworthy.** At the tool's default `--min-peak 0.2`
every rep is rejected: the returned signal peaks around 0.014. Lowering the
threshold to 0.01 makes 33 of 40 reps "pass", and the resulting distribution is
not a latency distribution at all:

| statistic | value |
|---|---:|
| p50 | 206.0 ms |
| p90 | 491.1 ms |
| p99 | 573.5 ms |
| stdev | **117.4 ms** |

p90 is 2.4x p50. With detection peaks sitting barely above the noise floor, the
cross-correlation is locking onto spurious lags. The honest description of what
happened is that a threshold was lowered until the tool produced output, which
is the same failure this repository already documents three times.

**And it would measure the wrong thing even if it were clean.** BlackHole is a
software device with its own buffering and no converter. A number from it
characterises BlackHole, not the ADC/DAC path that `t_buffer`'s I/O term is
supposed to cover. It could at best corroborate the queueing interpretation in
section 2 -- and the p50 of 206 ms is indeed close to the driver-reported
201.69 ms excess -- but corroboration from an untrustworthy estimator is worth
nothing.

**Conclusion unchanged: the I/O term remains unmeasured.** A wired loopback
through real converters, or a platform without AEC, is still what it takes.

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
