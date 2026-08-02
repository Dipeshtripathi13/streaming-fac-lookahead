# Setup — Raspberry Pi 5 / ARM64 embedded condition

Hardware class **`embedded-pi`**. This is the condition that makes contribution
#3 non-trivial: nobody has published streaming accent-conversion latency on a
device you can put in a headset.

## First, a correction to the plan

**Online Raspberry Pi simulators cannot produce a usable number here.**
The browser-based ones (`wokwi`, `pi-simulator` sites) emulate GPIO and a shell;
they do not emulate the Cortex-A76 microarchitecture, its NEON throughput, its
cache hierarchy, or its thermal behaviour. A latency measured there is a
measurement of someone else's server. If it appears in the paper, it is fabricated
data.

There are exactly three honest options, in descending order of preference:

| Option | Cost | Fidelity | Use it for |
|---|---|---|---|
| **1. Buy a Pi 5 (8 GB)** | ~$80–120 | exact | the paper |
| **2. Rent an ARM64 cloud instance** | ~$0.04–0.10/hr | architecture-correct, wrong performance class | development, CI, scaling trends |
| **3. QEMU `aarch64` emulation on your Mac** | free | correct ISA, ~10–50× slow, useless timings | debugging ARM-only bugs only |

Option 1 is $80 against a project whose compute budget is $130. Buy the Pi.
Options 2 and 3 are for getting the code working before it arrives.

---

## Option 1 — real Raspberry Pi 5

### Hardware
- Raspberry Pi 5, **8 GB** (4 GB will OOM on Whisper and on fp32 encoders)
- Active cooler — **not optional**. Without it the SoC throttles within minutes
  and the second half of your sweep is slower than the first for thermal
  reasons, which will look exactly like a lookahead effect if you sweep in order.
- 27 W USB-C PSU. An underpowered supply triggers under-voltage throttling that
  `vcgencmd get_throttled` reports and `hardware_probe.py` captures.
- NVMe HAT or a fast A2 microSD.

### OS

Raspberry Pi OS **64-bit** (Bookworm). Verify:

```bash
uname -m          # must print aarch64, not armv7l
```

A 32-bit userland silently halves your ONNX Runtime throughput and has no
`sherpa-onnx` wheels.

### Install

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev \
     libsndfile1 ffmpeg cmake build-essential
cd ~/accent_con/research
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements/embedded.txt
```

`requirements/embedded.txt` deliberately omits torch, speechbrain, whisper and
matplotlib. The Pi runs *inference and timing only*; scoring and plotting
happen on the Mac from the wavs and CSVs the Pi produces. Trying to install
torch on a Pi is a multi-hour detour that buys nothing.

### Verify

```bash
python3 tests/test_causal.py
python3 tests/test_pipeline_invariants.py
python3 bench/hardware_probe.py --out results/raw/hw_pi5.json
```

`hardware_probe.py` reads `/proc/device-tree/model` and `vcgencmd get_throttled`,
so the Pi's identity and throttle state end up in the results automatically.

### Thermal protocol — read this before running the sweep

`vcgencmd get_throttled` returns a bitmask. Anything non-zero invalidates the
run:

| bit | meaning |
|---|---|
| 0 | under-voltage now |
| 1 | ARM frequency capped now |
| 2 | **currently throttled** |
| 16 | under-voltage occurred since boot |
| 18 | **throttling occurred since boot** |

Procedure:

```bash
vcgencmd get_throttled              # must be throttled=0x0
sudo cpufreq-set -g performance 2>/dev/null || true
# 60 s idle soak so you start from a known thermal state
sleep 60 && vcgencmd measure_temp

python3 bench/hardware_probe.py --out results/raw/hw_pi5_pre.json
python3 bench/bench_encoder_scaling.py --preset small tiny --reps 30 \
        --out-prefix results/raw/encoder_scaling_pi5
python3 bench/bench_cascade_onnx.py --threads 1 2 4 --skip-fp32 \
        --out-prefix results/raw/cascade_pi5

vcgencmd get_throttled              # must STILL be 0x0
python3 bench/hardware_probe.py --out results/raw/hw_pi5_post.json
```

Report pre- and post-run temperature in the paper's appendix. If
`get_throttled` is non-zero afterwards, discard the run, improve cooling,
repeat. Reporting a throttled sweep is the embedded-benchmarking equivalent of
not setting a seed.

### Memory

8 GB is enough for int8 models and not enough for fp32 `large-v3`. Pass
`--skip-fp32`. Watch it:

```bash
watch -n1 free -m
```

If you see swap activity, your p95 is measuring the SD card.

### Expected order of magnitude

A Pi 5 (Cortex-A76 @ 2.4 GHz, 4 cores) delivers roughly **30–60 GFLOP/s** fp32
via NEON, against ~400 GFLOP/s for a 4-core Neoverse-N1 server core and well
over 1 TFLOP/s for an M4 P-core cluster. Budget for the Pi being **5–15×**
slower than the Mac on the same workload. That is precisely why it is in the
matrix: it is where RTF crosses 1 and the feasibility question becomes real
instead of theoretical.

---

## Option 2 — ARM64 cloud instance (for development before the Pi arrives)

Architecture-correct, so ARM-specific bugs surface. Performance class is wrong
(server cores, not mobile cores), so **label these results `cpu-arm64`, never
`embedded-pi`.** `bench/hardware_probe.py` already makes this distinction
automatically: it only returns `embedded-pi` when `/proc/device-tree/model`
names a Pi.

| Provider | Instance | vCPU | ~cost |
|---|---|---|---|
| AWS | `t4g.xlarge` (Graviton2) | 4 | ~$0.13/hr |
| Oracle Cloud | Ampere A1 free tier | up to 4 | free |
| Hetzner | CAX21 (Ampere) | 4 | ~€6/mo |
| Scaleway | COPARM1-2C-8G | 2 | ~€0.03/hr |

Oracle's always-free Ampere A1 allocation is the obvious choice for CI.

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev libsndfile1 ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/embedded.txt
python3 bench/hardware_probe.py     # will report hw_class = cpu-arm64
```

---

## Option 3 — QEMU on the Mac (last resort, debugging only)

```bash
brew install qemu
# Raspberry Pi OS Lite arm64 image
qemu-system-aarch64 -M virt -cpu cortex-a76 -smp 4 -m 4096 \
  -kernel kernel8.img -drive file=raspios.img,format=raw,if=virtio \
  -append "root=/dev/vda2 rw console=ttyAMA0" -nographic
```

Correct instruction set, wrong everything else. Use it to find "this ONNX op
has no ARM kernel" class bugs. **Never** put a QEMU timing in a results table.

---

## What the Pi condition actually contributes to the paper

Not "we ran it on a Pi and it was slow". The specific claims it enables:

1. **Where RTF crosses 1.** On the Mac every configuration is feasible, so the
   feasibility frontier is invisible. The Pi is the only condition where chunk
   size, quantisation, and model size have consequences you can plot.
2. **Whether the CPU/GPU gap is parameter count or operator support.** H4 says
   it is the vocoder and non-causal ops. The Pi has the weakest operator
   coverage, so if H4 is right the gap widens *disproportionately* there —
   a directional prediction that a Mac alone cannot test.
3. **A number practitioners can act on.** "Streaming FAC needs L ms of
   lookahead" is a modelling result. "…and on a $80 board that costs you an
   extra N ms of compute, so the total budget is X" is a deployment result.
