# Setup — GPU condition (training + the `gpu` hardware class)

Two separate needs, often conflated:

1. **Training** the causal accent translator. ~380 GPU-hours across the project.
2. **The `gpu` row of the results table** — one inference sweep, ~40 GPU-hours,
   which exists only so the CPU numbers have something to be compared against.

Colab is fine for (1) during development and for (2) entirely. It is a poor
choice for the long training runs in October–November because sessions get
reclaimed. Use RunPod/Vast for those.

---

## A. Google Colab

### Free tier
T4, 16 GB, ~12 h wall clock, disconnects when idle. Enough to:
- reproduce an offline FAC baseline,
- train the accent probe classifier,
- run the whole `gpu` inference sweep (it is 40 hours of *cheap* inference; split it),
- debug the training loop before paying for anything.

### Colab Pro (~$10/mo) / Pro+ (~$50/mo)
L4 or A100 40 GB, background execution on Pro+. Pro+ background execution is
the feature that matters — it is what lets a 12-hour run survive you closing
the laptop.

### Notebook

`notebooks/colab_streaming_fac.ipynb` in this repo is set up to:
- mount Drive, clone/sync the repo,
- install a pinned CUDA torch,
- assert GPU type and log it into the results JSON,
- run `bench/hardware_probe.py` so the GPU row carries the same metadata as every other row,
- checkpoint to Drive every N steps so a reclaimed session costs minutes, not hours.

Open it via **File → Upload notebook**, or from GitHub once the repo is pushed.

### The three Colab traps

**(a) You do not get the GPU you asked for.** Always assert:

```python
import torch, subprocess
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total",
                      "--format=csv"], capture_output=True, text=True).stdout)
assert torch.cuda.is_available()
```

and record `torch.cuda.get_device_name(0)` **into the results file**. A GPU
latency row that does not say which GPU is not a result.

**(b) Sessions vanish.** Checkpoint every 1000 steps to Drive, not to `/content`.

**(c) Timing on GPU needs synchronisation.** CUDA kernels are asynchronous;
`time.perf_counter()` around a forward pass measures launch time, not execution.
The GPU latency row must use CUDA events:

```python
s, e = torch.cuda.Event(True), torch.cuda.Event(True)
torch.cuda.synchronize(); s.record()
model(x)
e.record(); torch.cuda.synchronize()
ms = s.elapsed_time(e)
```

This is the single most common way published GPU latency numbers are wrong,
and calling it out explicitly is worth a sentence in the paper's methods.

---

## B. Rented GPUs — the actual training budget

Prices move; re-check before committing. As of the proposal's costing:

| GPU | VRAM | RunPod Community | Vast.ai | Good for |
|---|---|---|---|---|
| RTX 4090 | 24 GB | ~$0.34/hr | ~$0.30–0.40 | **default choice** |
| RTX 3090 | 24 GB | ~$0.22/hr | ~$0.20 | cheaper, ~40% slower |
| A100 40 GB | 40 GB | ~$0.52–0.60/hr | similar | only if you OOM at 24 GB |
| A100 80 GB | 80 GB | ~$1.07 on-demand / ~$0.60 spot | | you should not need this |
| H100 | 80 GB | ~$2+/hr | | you definitely do not need this |

The proposal's ~378 GPU-hours at $0.34 ≈ **$130**. That estimate already carries
a 40% retry margin, which is realistic for a first-time training pipeline.

**Do not train on Colab Pro+ by the month if you need >100 hours** — at
$50/month with usage limits it is more expensive per useful hour than RunPod
Community, and it can be interrupted.

### Spot vs on-demand
Spot is roughly half price and can be reclaimed with ~30 s notice. That is fine
*if and only if* you checkpoint every ~1000 steps. Write the checkpointing
before you start renting; retrofitting it after losing a 9-hour run is a
predictable and avoidable waste.

### Storage
Keep datasets on a network volume (~$0.05/GB/month) rather than re-downloading
L2-ARCTIC + VCTK into each pod. VCTK alone is ~11 GB and downloading it four
times costs more in GPU-idle time than the volume costs in a month.

---

## C. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements/gpu.txt
```

`requirements/gpu.txt` pins the CUDA wheel index. Verify before doing anything else:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python bench/hardware_probe.py --out results/raw/hw_gpu.json
python tests/test_causal.py
python tests/test_pipeline_invariants.py     # the torch test stops being skipped here
```

---

## D. What the GPU condition must report

The point of the GPU row is **not** to show that GPUs are fast. It is to
isolate H4: *the CPU/GPU gap is dominated by the vocoder and by non-causal
operations that resist quantisation, not by parameter count.*

To test that, the GPU sweep must report per-stage times (encoder / conversion /
vocoder) separately, on identical inputs to the CPU sweep, using the same
`StageTimer`. Then:

```
gap_stage = t_compute_cpu[stage] / t_compute_gpu[stage]
```

If H4 holds, `gap_vocoder >> gap_encoder` despite the vocoder having fewer
parameters. If instead the gaps are roughly proportional to parameter count,
H4 is refuted — which is also a publishable finding, and a more actionable one
(it would say: just shrink the model).

Record for every GPU row: GPU name, driver, CUDA version, torch version, batch
size (must be 1 — this is a latency measurement, not throughput), and whether
CUDA events or wall clock were used.

---

## E. Cost control

- Set a spending cap on day one.
- `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 60` logged to a file.
  If utilisation sits under ~70%, you are paying for a GPU to wait on the
  dataloader — fix that before renting more hours.
- Kill idle pods. The most common way this budget blows past $130 is a pod left
  running over a weekend, which alone is ~$16 of nothing.
