# Setup — Windows / x86 laptop CPU condition

Hardware class **`cpu-x86`**. This is the condition that represents "a normal
laptop", which is the deployment target most readers care about and the one
no streaming accent-conversion paper reports.

You need a machine with an Intel i5/i7 or AMD Ryzen 5/7 class CPU. Borrowed is
fine — the benchmark takes under an hour and writes a single results directory.

## Option A — native Windows (recommended for the paper)

Native is what a real user runs, and ONNX Runtime's Windows CPU provider is not
identical to Linux's. If you benchmark under WSL and call it "x86 laptop CPU",
a careful reviewer is entitled to object.

### 1. Python

Install **Python 3.11** from python.org (not the Microsoft Store build — it
sandboxes paths and breaks `sherpa-onnx` model loading). Tick "Add to PATH".

```powershell
python --version        # must print 3.11.x
```

### 2. Environment

```powershell
cd $HOME\Desktop\accent_con\research
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel
pip install -r requirements\cpu.txt
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 3. ffmpeg

```powershell
winget install Gyan.FFmpeg
```

Reopen the terminal, then `ffmpeg -version`.

### 4. Verify

```powershell
python tests\test_causal.py
python tests\test_pipeline_invariants.py
python eval\metrics.py
python eval\phoneme_analysis.py
python bench\hardware_probe.py --out results\raw\hw_win.json
```

### 5. Benchmark correctly

Three Windows-specific things will corrupt your numbers:

**(a) Power plan.** The default "Balanced" plan parks cores and scales
frequency. Set High performance for the duration of the run:

```powershell
powercfg /setactive SCHEME_MIN      # High performance
powercfg /getactivescheme           # confirm
# restore afterwards:
# powercfg /setactive SCHEME_BALANCED
```

**(b) Defender real-time scanning.** It intercepts the first read of each
`.onnx` file and adds hundreds of milliseconds to model load, and occasionally
to first inference. Load time is reported separately by the bench scripts, so
this mostly shows up as a fat warm-up tail — which is why `drop_warmup=3`
exists. Exclude the models folder if you can:

```powershell
Add-MpPreference -ExclusionPath "$HOME\Desktop\accent_con\accentbridge\models"
```

**(c) Timer resolution.** `time.perf_counter_ns()` is backed by QPC and is fine.
Do **not** substitute `time.time()`, which on Windows can have ~15 ms
granularity — larger than several of the quantities being measured.

### 6. Run the sweep

```powershell
$env:OMP_NUM_THREADS=4; $env:MKL_NUM_THREADS=4
python bench\hardware_probe.py --out results\raw\hw_win.json
python bench\bench_encoder_scaling.py --preset all --reps 30 --out-prefix results\raw\encoder_scaling_win
python bench\bench_cascade_onnx.py --threads 1 2 4 --out-prefix results\raw\cascade_win
```

Copy `results\raw\*_win*` back to the main repo and commit.

### 7. Live audio on Windows

`sounddevice` uses WASAPI/MME. For the loopback demo, install
[VB-CABLE](https://vb-audio.com/Cable/) as the Windows equivalent of BlackHole,
then:

```powershell
python ..\accentbridge\list_devices.py
```

and pass the CABLE Input index. WASAPI shared mode adds ~10 ms buffer;
exclusive mode gets to ~3 ms but blocks other apps. Report which you used.

---

## Option B — WSL2 (convenient, but label it honestly)

```bash
wsl --install -d Ubuntu-22.04
```

then follow `SETUP_RASPBERRY_PI.md`'s Linux instructions (the x86 path is the
same). WSL2 runs under a Hyper-V VM; CPU-bound numbers are within a few percent
of native but I/O and thread scheduling are not. **Do not mix WSL and native
results in the same table.** If you use WSL, add a column saying so.

---

## Option C — no Windows machine available

The `cpu-x86` condition can be filled by a cheap cloud instance, and this is a
legitimate substitute *if* you say what it was:

| Instance | vCPU | Rough cost | Represents |
|---|---|---|---|
| AWS `c7i.xlarge` | 4 (Sapphire Rapids) | ~$0.18/hr | modern x86 laptop, optimistic |
| AWS `t3.xlarge` | 4 (Skylake) | ~$0.17/hr | ~2019 laptop, realistic |
| Hetzner CX32 | 4 (shared) | ~€7/mo | budget, noisy neighbours — avoid for p95 |

A shared-vCPU instance will give you a p95 that reflects other tenants, not
your model. Use dedicated instances for anything you intend to publish.
