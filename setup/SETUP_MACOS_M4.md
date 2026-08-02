# Setup — Apple Silicon (M4), the primary CPU condition

This is hardware class **`cpu-apple-silicon`** in the experimental matrix, and
it is the machine you already own, so it is the reference platform. Everything
else in this repo is validated against it.

## 0. What this machine is and is not good for

| Task | M4 | Note |
|---|---|---|
| Running the sweep's CPU conditions | yes | this is the point |
| Latency benchmarking | yes, with caveats below | |
| Whisper large-v3 scoring | yes, slowly | use `faster-whisper` int8, not `openai-whisper` |
| Training the accent translator | **no** | use Colab/RunPod; MPS lacks kernels and silently falls back to CPU |
| Representing "a laptop CPU" generally | **no** | M4 has ~2–4× the per-core throughput of a mid-range x86 laptop. Report it as its own class. |

## 1. Toolchain

```bash
xcode-select --install                      # if not already
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11 ffmpeg cmake sox
```

Use Python 3.11, not 3.13 — several audio wheels (`sherpa-onnx`, `praat-parselmouth`,
some `librosa` deps) still lag on 3.13, and you do not want to debug that in November.

## 2. Environment

```bash
cd ~/Desktop/accent_con/research
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements/cpu.txt
```

## 3. Verify

```bash
python3 tests/test_causal.py
python3 tests/test_pipeline_invariants.py
python3 eval/metrics.py
python3 eval/phoneme_analysis.py
python3 bench/hardware_probe.py --out results/raw/hw_m4.json
```

All five must pass before you trust a single number out of this repo.

## 4. Benchmarking correctly on macOS

Apple Silicon has three ways to make your latency numbers lies. Guard all three.

**(a) P-core / E-core scheduling — and it is worse than a scheduling nuisance.**
Measured on this M4 (4 P-cores + 6 E-cores), asking for more threads makes
everything monotonically slower:

| ONNX Runtime threads | ASR encoder step p50 | Kokoro RTF |
|---:|---:|---:|
| 1 | **11.8 ms** | **0.78** |
| 2 | 12.6 | 0.85 |
| 4 | 17.9 | 0.92 |
| 8 | 37.7 (**3.2× slower**) | **1.10 — past real time** |

Once the pool exceeds the P-core count, work lands on E-cores and the op waits
on its slowest thread. **Set `num_threads=1` for streaming speech on Apple
Silicon.** `accentbridge.py` currently uses 2; 1 is better on this machine.

For the QoS clamp: **the accepted clamp names differ by macOS release.** On
26.5, `taskpolicy -c user-interactive` is rejected outright and the command
fails — which silently killed a whole benchmark stage on the first run here.
Probe before relying on it:

```bash
for c in userinteractive user-interactive utility; do
  taskpolicy -c "$c" true 2>/dev/null && echo "use: $c" && break
done
```

`run_m4.command` does this automatically and falls back to default priority.
Report core counts from `hardware_probe.py` (`mac_perf_cores` / `mac_eff_cores`).

**(b) Thermal throttling.** Sustained sweeps on a fanless MacBook Air will clock
down after a few minutes and the second half of your sweep will be slower than
the first for reasons that have nothing to do with lookahead. Randomise
condition order (the bench scripts do not do this for you yet — add
`--shuffle` before the November full sweep) and log:

```bash
sudo pmset -g thermlog &        # run alongside the sweep
```

**(c) Accelerate vs OpenBLAS.** NumPy on Apple Silicon may link Accelerate or
OpenBLAS depending on how it was installed, and they differ by ~2× on sgemm.
`hardware_probe.py` records which one; put it in the paper's reproducibility
appendix. Pin threads explicitly:

```bash
export VECLIB_MAXIMUM_THREADS=4
export OMP_NUM_THREADS=4
```

## 5. Real-time audio (for the live demo and the S1 cascade)

The existing `accentbridge` prototype routes to a virtual device:

```bash
brew install blackhole-2ch     # then reboot
cd ../accentbridge && python3 accentbridge.py --tts kokoro
```

Grant microphone permission in System Settings → Privacy & Security → Microphone,
otherwise the input stream is silent and `accentbridge.py` will warn but keep running.

`BLOCKSIZE` in `audio_util.py` sets `t_buffer`. CoreAudio adds roughly one
buffer of input plus one of output, so a 512-sample block at 16 kHz is ~32 ms
each way. **Measure it rather than assuming** — `bench/bench_cascade_onnx.py`
reports compute only, and the I/O term has to be added separately or the
end-to-end number is understated.

## 6. Running the CPU conditions of the sweep

Just double-click **`run_m4.command`** in Finder. It probes the QoS clamp,
creates the venv, runs every stage independently (so one failure does not abort
the rest), and logs to `results/raw/run_m4_<timestamp>.log`.

`run_selftest.command` runs only the causality proof (~1 min) for fast iteration.

Equivalent by hand:

```bash
source .venv/bin/activate
export VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4

python3 bench/hardware_probe.py --out results/raw/hw_m4.json
python3 bench/bench_encoder_scaling.py --preset all --reps 30 \
    --out-prefix results/raw/encoder_scaling_m4
python3 bench/bench_cascade_onnx.py --threads 1 2 4 8 \
    --out-prefix results/raw/cascade_m4
python3 bench/bench_content_degradation.py --selftest --tag m4   # causality proof
python3 bench/bench_content_degradation.py --n-utts 48 --tag m4  # needs HF login
python3 bench/make_figures.py
```

## 7. Known traps

- `openai-whisper` on MPS is slower than on CPU for `large-v3` and occasionally
  produces different text. Use `faster-whisper` with `compute_type="int8"` and
  `device="cpu"`. Scoring must be deterministic across the whole sweep or your
  WER differences are noise.
- `speechbrain` pulls a large torch; if you only need ECAPA, install it in a
  separate venv and score offline from saved wavs.
- **Do not benchmark on battery.** `pmset -g ps` must show `AC Power`; the
  first run here was on battery and the CPU was downclocked throughout.
  `run_m4.command` prints the power source in stage 0 — read it.
- `KoelLabs/L2Arctic` is a **gated** HF dataset (CC-BY-NC-4.0). Accept the terms
  on the dataset page, then `huggingface-cli login` inside the venv, once.
