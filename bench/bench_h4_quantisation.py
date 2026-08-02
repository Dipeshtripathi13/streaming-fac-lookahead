"""RQ4 / H4: what actually dominates the CPU/GPU gap?

H4 (pre-registered): the gap is dominated by the **vocoder and by non-causal
operations that resist quantisation**, not by parameter count. If true, the
optimisation effort should go into operator choice rather than shrinking the
model — a directly actionable redirection. If false — if the gap tracks
parameter count — then "just use a smaller model" is the right advice, which is
also worth knowing and is the opposite recommendation.

Every benchmark in this study so far passed `--skip-fp32`, so the quantisation
half of H4 has never been tested. This does that.

Two axes, measured per stage on the same inputs:

  1. **Quantisation sensitivity.** int8 vs fp32 for each ONNX stage. An op that
     quantises well shows a large int8 speedup; one that resists shows ~1x.
     H4 predicts the *vocoder* resists more than the *encoder*.

  2. **Parameter count vs speedup.** If the int8 speedup correlates with
     parameter count, the gap is a size story. If it correlates with op mix
     instead, it is an operator story. Reported as both.

Deliberately CPU-only and ONNX-based: that is where the quantisation question
actually bites. The GPU side of H4 needs per-stage CUDA-event timing of the
same graph and belongs in the Colab notebook, because a GPU quantisation
comparison on a T4 (which has fast int8 tensor cores) answers a different
question from a CPU one.

Usage
-----
    python3 bench/bench_h4_quantisation.py
    python3 bench/bench_h4_quantisation.py --threads 1 --reps 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sfac.latency import StageTimer, write_csv, _dist  # noqa: E402

DEFAULT_MODELS = os.path.join(os.path.dirname(__file__), "..", "..",
                              "accentbridge", "models")
ASR_DIR = "sherpa-onnx-streaming-zipformer-en-2023-06-21"


def onnx_stage_info(path: str) -> Dict[str, object]:
    """Parameter count and operator histogram for one ONNX graph.

    The operator histogram is the point: H4 is a claim about op mix, so we need
    to report which ops each stage is made of, not just how big it is.
    """
    import onnx
    m = onnx.load(path, load_external_data=False)
    ops: Dict[str, int] = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    n_params = 0
    for init in m.graph.initializer:
        sz = 1
        for d in init.dims:
            sz *= d
        n_params += sz
    return {"n_params": int(n_params), "n_nodes": len(m.graph.node),
            "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1]))}


def time_session(path: str, feeds: Dict[str, np.ndarray], threads: int,
                 reps: int) -> Dict[str, float]:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    for _ in range(3):                       # warm: kernel selection + alloc
        sess.run(None, feeds)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        sess.run(None, feeds)
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    return _dist(ts)


def make_feeds(path: str) -> Optional[Dict[str, np.ndarray]]:
    """Random inputs matching the graph's declared shapes (batch/dyn -> 1)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {}
    rng = np.random.default_rng(0)
    for i in sess.get_inputs():
        shape = [1 if (not isinstance(d, int) or d <= 0) else d for d in i.shape]
        t = i.type
        if "float" in t:
            feeds[i.name] = rng.standard_normal(shape).astype(np.float32) * 0.1
        elif "int64" in t:
            feeds[i.name] = np.zeros(shape, dtype=np.int64)
        elif "int32" in t:
            feeds[i.name] = np.zeros(shape, dtype=np.int32)
        else:
            return None
    return feeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=DEFAULT_MODELS)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--tag", default="local")
    a = ap.parse_args()

    md = os.path.abspath(a.models_dir)
    d = os.path.join(md, ASR_DIR)
    # (stage, fp32 path, int8 path) -- only stages that ship both
    pairs = [
        ("asr_encoder", f"{d}/encoder-epoch-99-avg-1.onnx",
         f"{d}/encoder-epoch-99-avg-1.int8.onnx"),
        ("asr_joiner", f"{d}/joiner-epoch-99-avg-1.onnx",
         f"{d}/joiner-epoch-99-avg-1.int8.onnx"),
        ("asr_decoder", f"{d}/decoder-epoch-99-avg-1.onnx",
         f"{d}/decoder-epoch-99-avg-1.int8.onnx"),
    ]

    print(f"H4: quantisation sensitivity per stage  (threads={a.threads}, "
          f"reps={a.reps})\n")
    rows: List[Dict[str, object]] = []
    for name, fp32, int8 in pairs:
        if not (os.path.exists(fp32) and os.path.exists(int8)):
            print(f"  {name}: missing fp32 or int8 graph, skipped")
            continue
        try:
            info32 = onnx_stage_info(fp32)
        except Exception as e:
            info32 = {"n_params": None, "ops": {}, "err": str(e)}
        feeds = make_feeds(int8)
        if feeds is None:
            print(f"  {name}: unsupported input dtype, skipped")
            continue
        try:
            t32 = time_session(fp32, feeds, a.threads, a.reps)
            t8 = time_session(int8, feeds, a.threads, a.reps)
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__}: {e}")
            continue
        speedup = t32["p50"] / t8["p50"] if t8["p50"] else float("nan")
        size32 = os.path.getsize(fp32) / 1e6
        size8 = os.path.getsize(int8) / 1e6
        top_ops = list(info32.get("ops", {}).items())[:6]
        rows.append({
            "stage": name, "threads": a.threads,
            "n_params_M": round((info32.get("n_params") or 0) / 1e6, 2),
            "fp32_MB": round(size32, 1), "int8_MB": round(size8, 1),
            "size_ratio": round(size32 / size8, 2) if size8 else None,
            "fp32_p50_ms": round(t32["p50"], 4),
            "int8_p50_ms": round(t8["p50"], 4),
            "int8_speedup": round(speedup, 3),
            "top_ops": ";".join(f"{k}x{v}" for k, v in top_ops),
        })
        print(f"  {name:<12} params {rows[-1]['n_params_M']:>6.2f}M   "
              f"size {size32:>6.1f}->{size8:<6.1f}MB ({rows[-1]['size_ratio']}x)   "
              f"time {t32['p50']:>8.3f}->{t8['p50']:<8.3f}ms   "
              f"speedup {speedup:>5.2f}x")
        print(f"               ops: {', '.join(f'{k}x{v}' for k,v in top_ops)}")

    if not rows:
        sys.exit("no stages measured")

    print("\n" + "=" * 70)
    print("H4 READOUT")
    print("=" * 70)
    sp = {r["stage"]: r["int8_speedup"] for r in rows}
    pm = {r["stage"]: r["n_params_M"] for r in rows}
    print(f"  int8 speedup by stage : "
          f"{', '.join(f'{k} {v:.2f}x' for k, v in sp.items())}")
    print(f"  parameter count (M)   : "
          f"{', '.join(f'{k} {v:.2f}' for k, v in pm.items())}")

    if len(rows) >= 3:
        x = np.array([r["n_params_M"] for r in rows], float)
        y = np.array([r["int8_speedup"] for r in rows], float)
        if x.std() > 0 and y.std() > 0:
            rho = float(np.corrcoef(np.log10(x + 1e-9), y)[0, 1])
            print(f"\n  corr(log params, int8 speedup) = {rho:+.3f}")
            print("  -> " + ("speedup tracks SIZE: the gap is a parameter-count "
                             "story, and 'use a smaller model' is the right advice "
                             "(H4 REFUTED for these stages)."
                             if rho > 0.7 else
                             "speedup does NOT track size: it is an operator-mix "
                             "story, so quantisability depends on which ops a "
                             "stage is built from (H4 direction SUPPORTED)."))

    print("\n  Scope: ASR stages only. The vocoder half of H4 needs an ONNX")
    print("  HiFi-GAN/Kokoro pair shipping both precisions, which this model set")
    print("  does not include -- Kokoro ships int8 only. Until that is measured,")
    print("  the vocoder clause of H4 is untested, and the paper must say so.")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "raw",
                       f"h4_quantisation_{a.tag}.csv")
    write_csv(out, rows)
    with open(out.replace(".csv", ".json"), "w") as f:
        json.dump({"rows": rows, "threads": a.threads, "reps": a.reps}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
