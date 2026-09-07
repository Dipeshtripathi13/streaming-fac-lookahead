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


def android_device() -> Optional[Dict[str, str]]:
    """Identify an Android host, including under Termux.

    This matters more than it looks. On Termux `platform.system()` is "Linux"
    and `platform.machine()` is "aarch64", which is EXACTLY what a Neoverse
    cloud instance reports -- so without this a phone is silently filed as
    `cpu-arm64` and pooled with a datacentre VM. The two are not the same
    hardware class and must never share a label.
    """
    hints = []
    if os.environ.get("ANDROID_ROOT") or os.environ.get("ANDROID_DATA"):
        hints.append("env")
    if "com.termux" in (os.environ.get("PREFIX", "") or ""):
        hints.append("termux")
    if os.path.exists("/system/build.prop"):
        hints.append("build.prop")
    if not hints:
        return None
    out = {"detected_by": ",".join(hints)}
    for key, prop in (("model", "ro.product.model"),
                      ("brand", "ro.product.brand"),
                      ("soc", "ro.soc.model"),
                      ("soc_manufacturer", "ro.soc.manufacturer"),
                      ("android_release", "ro.build.version.release")):
        v = _sh(f"getprop {prop}")
        if v:
            out[key] = v
    return out


def cpu_core_layout() -> Dict[str, object]:
    """Per-core maximum frequency, which is how a big.LITTLE split shows up.

    F4 (num_threads = cpu_count is the wrong default) is a big.LITTLE finding,
    and a phone is the most big.LITTLE device most people own. Recording the
    cluster layout is what lets a thread-count result be interpreted rather
    than merely reported.
    """
    out: Dict[str, object] = {}
    freqs = []
    try:
        import glob as _glob
        for path in sorted(_glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq")):
            try:
                freqs.append(int(open(path).read().strip()) // 1000)
            except Exception:
                pass
    except Exception:
        pass
    if freqs:
        out["max_freq_mhz_per_cpu"] = freqs
        clusters = sorted(set(freqs))
        out["distinct_max_freqs_mhz"] = clusters
        out["is_big_little"] = len(clusters) > 1
        out["n_cores"] = len(freqs)
    return out


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
    # Android has no vcgencmd equivalent, so the throttle evidence has to be
    # reconstructed: current clock against each core's own maximum. A sustained
    # ratio well below 1.0 is the phone's version of `throttled != 0x0`.
    try:
        import glob as _glob
        cur, mx = [], []
        for c in sorted(_glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq")):
            try:
                cur.append(int(open(c + "/scaling_cur_freq").read().strip()))
                mx.append(int(open(c + "/cpuinfo_max_freq").read().strip()))
            except Exception:
                pass
        if cur and mx and sum(mx):
            out["cpu_cur_over_max"] = round(sum(cur) / sum(mx), 3)
            out["scaling_cur_freq_mhz"] = [v // 1000 for v in cur]
    except Exception:
        pass
    zones = []
    try:
        import glob as _glob
        for z in sorted(_glob.glob("/sys/class/thermal/thermal_zone*/temp")):
            try:
                zones.append(int(open(z).read().strip()) / 1000.0)
            except Exception:
                pass
    except Exception:
        pass
    if zones:
        out["thermal_zones_c_max"] = max(zones)
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
    d["android"] = android_device()
    d["core_layout"] = cpu_core_layout()
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
    if d.get("android"):
        return "mobile-android"   # never fall through to cpu-arm64: see android_device()
    if sysname == "Darwin" and mach in ("arm64", "aarch64"):
        return "cpu-apple-silicon"
    if mach in ("aarch64", "arm64"):
        return "cpu-arm64"      # Pi-class proxy / ARM server
    if mach in ("x86_64", "amd64"):
        return "cpu-x86"
    return "unknown"



def _self_test() -> int:
    """Exercise classify() on synthetic hosts. The phone cases matter most:
    they cannot be tested on the machine that writes them."""
    print("hardware_probe self-test")
    cases = [
        ("Android phone under Termux",
         {"system": "Linux", "machine": "aarch64", "libs": {},
          "android": {"model": "Pixel 7"}}, "mobile-android"),
        ("Android phone must NOT be filed as cpu-arm64",
         {"system": "Linux", "machine": "aarch64", "libs": {},
          "android": {"model": "SM-S911B"}}, "mobile-android"),
        ("ARM64 cloud VM (no android markers)",
         {"system": "Linux", "machine": "aarch64", "libs": {}}, "cpu-arm64"),
        ("Raspberry Pi still wins over arm64",
         {"system": "Linux", "machine": "aarch64", "libs": {},
          "pi_model": "Raspberry Pi 5 Model B"}, "embedded-pi"),
        ("GPU beats everything",
         {"system": "Linux", "machine": "x86_64", "libs": {"torch_cuda": True},
          "android": {"model": "x"}}, "gpu"),
        ("Apple Silicon Mac",
         {"system": "Darwin", "machine": "arm64", "libs": {}}, "cpu-apple-silicon"),
        ("x86 laptop",
         {"system": "Linux", "machine": "x86_64", "libs": {}}, "cpu-x86"),
    ]
    ok = True
    for name, d, want in cases:
        got = classify(d)
        good = got == want
        ok &= good
        print(f"  {name:<48s} {'ok' if good else 'FAIL'} ({got})")
    layout = cpu_core_layout()
    print(f"  cpu_core_layout() returns a dict on this host       "
          f"{'ok' if isinstance(layout, dict) else 'FAIL'}")
    th = thermal_state()
    print(f"  thermal_state() returns a dict on this host         "
          f"{'ok' if isinstance(th, dict) else 'FAIL'}")
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-calibration", action="store_true")
    a = ap.parse_args()
    if getattr(a, "self_test", False):
        raise SystemExit(_self_test())
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
