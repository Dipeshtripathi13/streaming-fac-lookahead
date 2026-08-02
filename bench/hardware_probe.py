"""Record exactly what machine a benchmark ran on.

Every latency number in this project is meaningless without this. The paper
reports four hardware classes; the reproducibility claim requires that a
reader can tell whether their machine is comparable. This writes a JSON
sidecar next to every result file.

Also measures a small set of *calibration* microbenchmarks (BLAS sgemm,
memory bandwidth, single-core scalar throughput) so that results from
machines we did not test can be placed on the same axis.

Usage:
    python3 bench/hardware_probe.py
    python3 bench/hardware_probe.py --out results/raw/hw_<name>.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, Optional

import numpy as np


def _sh(cmd: str) -> Optional[str]:
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        s = out.stdout.strip()
        return s or None
    except Exception:
        return None


def cpu_model() -> str:
    s = platform.system()
    if s == "Darwin":
        return _sh("sysctl -n machdep.cpu.brand_string") or "apple-silicon-unknown"
    if s == "Linux":
        for line in (_sh("cat /proc/cpuinfo") or "").splitlines():
            if line.lower().startswith(("model name", "hardware", "cpu part")):
                return line.split(":", 1)[-1].strip()
        return _sh("lscpu | grep -i 'model name'") or platform.processor() or "linux-unknown"
    if s == "Windows":
        return _sh("wmic cpu get name /value") or platform.processor() or "windows-unknown"
    return platform.processor() or "unknown"


def raspberry_pi_model() -> Optional[str]:
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(p, "rb") as f:
                return f.read().decode("utf-8", "ignore").strip("\x00").strip()
        except Exception:
            pass
    return None


def mem_total_gb() -> Optional[float]:
    s = platform.system()
    if s == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return round(int(line.split()[1]) / 1024 / 1024, 2)
        except Exception:
            return None
    if s == "Darwin":
        v = _sh("sysctl -n hw.memsize")
        return round(int(v) / 1024**3, 2) if v and v.isdigit() else None
    return None


def thermal_state() -> Dict[str, object]:
    """Thermal throttling silently invalidates sustained latency benchmarks.

    On a Pi this is the single most common source of an unreproducible
    result: the first 60 s look great, then the SoC clocks down. Capture the
    state before and after every sweep.
    """
    out: Dict[str, object] = {}
    thr = _sh("vcgencmd get_throttled")           # Raspberry Pi
    if thr:
        out["vcgencmd_throttled"] = thr
    t = _sh("vcgencmd measure_temp")
    if t:
        out["soc_temp"] = t
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            out["thermal_zone0_c"] = int(f.read().strip()) / 1000.0
    except Exception:
        pass
    if platform.system() == "Darwin":
        out["note"] = "macOS: no user-space throttle counter; run `pmset -g thermlog` alongside"
    return out


# --------------------------------------------------------------------------
# Calibration microbenchmarks
# --------------------------------------------------------------------------

def bench_sgemm(n: int = 512, reps: int = 20) -> Dict[str, float]:
    """BLAS matmul throughput -- the dominant cost in any transformer encoder."""
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    a @ b  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        a @ b
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    ts.sort()
    med = ts[len(ts) // 2]
    gflops = (2.0 * n**3) / (med / 1000.0) / 1e9
    return {"n": n, "median_ms": round(med, 4), "gflops": round(gflops, 2)}


def bench_memcpy(mb: int = 64, reps: int = 10) -> Dict[str, float]:
    """Streaming memory bandwidth -- bounds vocoder/upsampling throughput."""
    a = np.zeros(mb * 1024 * 1024 // 4, dtype=np.float32)
    b = np.empty_like(a)
    np.copyto(b, a)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        np.copyto(b, a)
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    ts.sort()
    med = ts[len(ts) // 2]
    return {"mb": mb, "median_ms": round(med, 3), "gb_per_s": round(mb / 1024 / (med / 1000), 2)}


def bench_single_core() -> Dict[str, float]:
    """Scalar loop -- proxy for the non-vectorisable glue (framing, VAD logic)."""
    t0 = time.perf_counter_ns()
    x = 0
    for i in range(2_000_000):
        x += i % 7
    return {"loop_2m_ms": round((time.perf_counter_ns() - t0) / 1e6, 2)}


def blas_info() -> Dict[str, object]:
    info: Dict[str, object] = {}
    try:
        cfg = np.show_config(mode="dicts")  # numpy >= 2
        info["numpy_build"] = {
            k: v for k, v in cfg.get("Build Dependencies", {}).items() if k == "blas"
        }
    except Exception:
        info["numpy_build"] = "unavailable"
    info["threading_env"] = {
        k: os.environ.get(k)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    }
    try:
        import onnxruntime as ort
        info["onnxruntime_version"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        info["onnxruntime_version"] = None
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["torch_cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        info["torch_mps"] = bool(getattr(torch.backends, "mps", None)
                                 and torch.backends.mps.is_available())
    except Exception:
        info["torch_version"] = None
    return info


def probe(run_calibration: bool = True) -> Dict[str, object]:
    d: Dict[str, object] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),          # arm64 / aarch64 / x86_64
        "python": sys.version.split()[0],
        "cpu_model": cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "mem_total_gb": mem_total_gb(),
        "pi_model": raspberry_pi_model(),
        "thermal": thermal_state(),
        "libs": blas_info(),
    }
    if platform.system() == "Darwin":
        d["mac_perf_cores"] = _sh("sysctl -n hw.perflevel0.logicalcpu")
        d["mac_eff_cores"] = _sh("sysctl -n hw.perflevel1.logicalcpu")
        d["mac_model"] = _sh("sysctl -n hw.model")
    if run_calibration:
        d["calibration"] = {
            "sgemm_512": bench_sgemm(512),
            "sgemm_1024": bench_sgemm(1024, reps=8),
            "memcpy_64mb": bench_memcpy(),
            "single_core": bench_single_core(),
        }
    d["hw_class"] = classify(d)
    return d


def classify(d: Dict[str, object]) -> str:
    """Map onto the four hardware classes in the experimental matrix."""
    sysname, mach = d["system"], str(d["machine"]).lower()
    libs = d.get("libs", {}) or {}
    if libs.get("torch_cuda"):
        return "gpu"
    if d.get("pi_model"):
        return "embedded-pi"
    if sysname == "Darwin" and mach in ("arm64", "aarch64"):
        return "cpu-apple-silicon"
    if mach in ("aarch64", "arm64"):
        return "cpu-arm64"      # Pi-class proxy / ARM server
    if mach in ("x86_64", "amd64"):
        return "cpu-x86"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-calibration", action="store_true")
    a = ap.parse_args()
    d = probe(run_calibration=not a.no_calibration)
    txt = json.dumps(d, indent=2, default=str)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(txt)
        print(f"\nwrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
