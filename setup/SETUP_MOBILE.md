# Setup — phone (`mobile-android`) as the on-device condition

Hardware class **`mobile-android`**. This is the condition that makes
contribution #3 non-trivial, and it replaces `embedded-pi` as the primary
on-device measurement. `SETUP_RASPBERRY_PI.md` stays valid and is kept as the
reproducible fallback (see "Why keep the Pi around" below).

## Why a phone is the better condition, not merely the cheaper one

The Pi doc justifies itself as "a device you can put in a headset". A phone is
that device, and it is the one this application actually ships on: nobody
deploys real-time accent conversion on a Raspberry Pi.

It also lets the paper land a punch it currently only gestures at. §2 calls out
StreamVC for claiming operation "even on a mobile platform" while naming no
device, core count or thread count -- verified against its abstract on
1 Sep 2026, and there is no latency figure in it. **A named phone, with reported
SoC, core layout and thread count, is the falsifiable version of the claim they
made unfalsifiably.** A Pi cannot do that; it is not a mobile platform.

And §6's actual finding -- that embedded latency cannot be projected from
specifications, because a peak-FLOPS derivation puts a Pi 5 at 37--56x the M4
while an achieved-throughput derivation puts it far closer -- is tested just as
well by a phone. The finding survives the substitution.

## The trap this condition nearly walked into

On Termux, `platform.system()` returns `Linux` and `platform.machine()` returns
`aarch64` -- **byte-identical to what a Neoverse-N1 cloud instance reports.**
Without explicit detection a phone is silently filed as `cpu-arm64` and pooled
with a datacentre VM, and the paper would then average a handset together with a
server under one label.

`hardware_probe.py` now detects Android (`ANDROID_ROOT`/`ANDROID_DATA`, a Termux
`PREFIX`, or `/system/build.prop`), records `getprop` SoC and model, and returns
`mobile-android` *before* the generic arm64 fallback. Two of the seven
`--self-test` cases exist solely to pin that ordering, because they cannot be
tested on the machine that writes them.

## Getting a Python on the phone

**Do not use the Play Store Termux.** It was abandoned around 2020 when Google
policy blocked apps that download executable code; it is frozen at v0.101 and
`pkg install` fails because the repositories moved. It installs fine and then
breaks at the second step, which is the worst way for a dependency to fail.

If the F-Droid install errors with a vague "App not installed", the usual cause
is a **signature conflict**: an older Play Store Termux is still present under
the same package name but a different signing key. Uninstall it first. Also
check Settings -> Apps -> Special access -> Install unknown apps.

Simplest route, skipping the F-Droid client entirely: take the APK from
`github.com/termux/termux-app/releases` (`...arm64-v8a.apk`). Same app, same
key.

Fallbacks if Termux cannot be made to work, both on the Play Store:

| runtime | how it differs | caveat |
|---|---|---|
| **UserLAnd** | real Debian/Ubuntu userspace; `apt install python3 python3-numpy` | runs under proot, so `/system` may not be mounted inside the guest |
| **Pydroid 3** | bundles Python and numpy, runs natively | no git; copy the repo across manually |

`android_device()` therefore checks **six independent signals** -- `ANDROID_ROOT`
/`ANDROID_DATA`, a `com.termux` PREFIX, `/system/build.prop`, `/system/bin`,
"android" in `/proc/version`, and Android filesystem markers -- so that any of
the three runtimes is recognised. Two `--self-test` cases assert that the env
signal alone and the Termux signal alone are each sufficient, because the
failure is silent: an undetected phone is filed as `cpu-arm64` and averaged in
with a datacentre VM.

## Android (primary path)

1. Install **Termux**, by whichever of the routes above works.
2. ```bash
   pkg update && pkg install python clang cmake libopenblas
   pip install numpy onnxruntime
   ```
   `bench_encoder_scaling.py` is numpy-only and needs no port.
   `bench_cascade_onnx.py` needs `onnxruntime`; sherpa-onnx publishes Android
   builds if the full cascade is wanted.
3. ```bash
   python3 bench/hardware_probe.py --self-test          # must pass first
   python3 bench/hardware_probe.py --out results/raw/hw_phone.json
   ```
   Check `hw_class` reads `mobile-android` and `core_layout.is_big_little`
   is populated **before** spending time on a sweep.

## The thermal protocol, which is weaker here and must be reported as such

The Pi has `vcgencmd get_throttled`: a single authoritative bitmask, non-zero
invalidates the run. **Android has no equivalent.** The substitute
`hardware_probe.py` records is evidence, not proof:

* `cpu_cur_over_max` -- summed current clock over summed per-core maximum. A
  sustained value well below 1.0 is this platform's version of `throttled != 0`.
* `thermal_zones_c_max` -- hottest thermal zone.
* `max_freq_mhz_per_cpu` -- per-core maxima, which is how the big.LITTLE split
  becomes visible and interpretable.

Protocol, and it is stricter than the Pi's precisely because the signal is
softer:

* airplane mode, screen at minimum, no other apps, **on charge**;
* 60 s idle soak, probe, run, probe again;
* discard any run where `cpu_cur_over_max` fell more than 15% between the two
  probes, and say in the paper how many runs were discarded;
* at least 3 repetitions, report spread rather than a single number;
* record the exact device, SoC, OS build. "A phone" is not a measurement.

## What is genuinely lost against the Pi

1. **Reproducibility.** "Raspberry Pi 5, 8 GB" is an $80 spec any reviewer can
   buy and replicate. A specific handset on a specific OS build is not. This is
   the real cost and it should be stated in Limitations, not hidden.
2. **Throttle proof** becomes throttle *evidence*, as above.
3. **Core affinity.** F4 (`num_threads = cpu_count` costs 3.2x on big.LITTLE) is
   the finding most worth re-testing on a phone -- phones are the most
   big.LITTLE devices in existence -- and it is also where control is weakest.
   `taskset` may work under Termux; if it does not, sweep thread count without
   pinning and say so.

## iOS

Not recommended, and not because of effort snobbery. There is no shell, no
Python, no thread control and no clock introspection, so the existing benchmarks
cannot run at all -- a Swift harness over ONNX Runtime or Core ML would have to
be written from scratch. Worse for the argument: Apple phone cores are the same
family as the M4 reference platform, so the hardware-class contrast is weaker
than a Cortex-A76 board would have given. If iOS is the only device available,
buy the Pi instead; $80 is cheaper than that port.

## Why keep the Pi around

Do the phone first: it costs nothing and it is the better deployment argument.
Add the Pi if a reviewer asks for a spec they can buy, or if the thermal
evidence above proves too soft to defend. The two are complementary --
`embedded-pi` is reproducible, `mobile-android` is relevant -- and reporting
both would make contribution #3 stronger than either alone.
