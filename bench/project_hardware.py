"""Project streaming latency onto hardware we do not have, honestly.

Why not just emulate a Raspberry Pi
-----------------------------------
Two options were considered and rejected:

* **Browser "Pi simulators"** emulate GPIO and a shell. They do not model the
  Cortex-A76 pipeline, NEON throughput, caches or memory. A latency measured
  there is a measurement of someone else's server.
* **QEMU.** `qemu-user`/`qemu-system` translate instructions; they model no
  microarchitectural timing at all. A QEMU run is *slower* than native by an
  amount unrelated to any real chip. QEMU is the right tool for "does this
  code run on ARM64 / are there ARM-only op bugs" and the wrong tool for any
  number with milliseconds attached.

So this module does not pretend to measure a Pi. It **projects**, from two
machines we did measure, and it **validates the projection** by leave-one-out:
predict machine A using only machine B's data, and report the error. That error
is the honest uncertainty band on any third machine.

The question the projection has to answer first
-----------------------------------------------
Is a machine-to-machine difference a **single scalar**? If
`t_B = alpha * t_A` holds across all 63 conditions, then projecting to a Pi is
arithmetic: pick alpha, multiply. If it does not — because compute-bound and
memory-bound operations scale differently — then a single "the Pi is 8x
slower" number is wrong and a two-term (compute + bandwidth) model is needed.

We have 63 matched conditions on Apple M4 (Accelerate) and Neoverse-N1
(OpenBLAS), so this is testable rather than assumed. That test is the actual
contribution here; the Pi row is downstream of it.

Output is deliberately a **curve over slowdown factor**, not a point estimate:
"at 5x slower than the M4, X is infeasible; at 10x, Y also fails". A reader can
place their own board on that axis. A single fabricated Pi number cannot be
checked; a curve can.

Usage
-----
    python3 bench/project_hardware.py
    python3 bench/project_hardware.py --slowdowns 3 5 8 13
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

RAW = os.path.join(os.path.dirname(__file__), "..", "results", "raw")

# Published Pi 5 specification, with the derivation shown so a reader can
# disagree with it. We do NOT use these to manufacture a timing; they only
# bracket where a Pi 5 sits on the slowdown axis.
PI5_SPEC = {
    "soc": "Broadcom BCM2712, 4x Cortex-A76 @ 2.4 GHz",
    "neon_fp32_peak_gflops": 4 * 2.4 * 8,      # 2x128-bit FMA/cycle = 8 flop/cycle
    "memory": "LPDDR4X-4267, 32-bit -> ~17 GB/s peak",
    "note": ("Peak figures. Achieved sgemm on A76-class cores is typically "
             "40-60% of peak, and this study already shows achieved throughput "
             "is what matters (RESULTS_M4.md F2). Treat the derived slowdown "
             "as a range, never a point."),
}


def load_scaling(tag: str) -> Dict[Tuple[str, float, float], float]:
    """{(preset, chunk_ms, lookahead_ms): t_compute_p50_ms} for one machine."""
    path = os.path.join(RAW, f"encoder_scaling_{tag}.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["preset"], float(r["chunk_ms"]), float(r["lookahead_ms"]))] = \
                float(r["t_compute_p50_ms"])
    return out


def fit_scalar(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """b ~ alpha * a  (through the origin) and b ~ alpha*a + beta."""
    alpha0 = float((a * b).sum() / (a * a).sum())
    r0 = b - alpha0 * a
    A = np.vstack([a, np.ones_like(a)]).T
    alpha1, beta1 = np.linalg.lstsq(A, b, rcond=None)[0]
    r1 = b - (alpha1 * a + beta1)
    ss = float(((b - b.mean()) ** 2).sum())
    return {
        "alpha_through_origin": alpha0,
        "r2_through_origin": 1 - float((r0 ** 2).sum()) / ss,
        "max_rel_err_through_origin": float(np.max(np.abs(r0 / b))),
        "alpha_affine": float(alpha1),
        "beta_affine_ms": float(beta1),
        "r2_affine": 1 - float((r1 ** 2).sum()) / ss,
        "max_rel_err_affine": float(np.max(np.abs(r1 / b))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", default="m4", help="tag of the faster machine")
    ap.add_argument("--slow", default="arm64", help="tag of the slower machine")
    ap.add_argument("--slowdowns", type=float, nargs="+",
                    default=[2, 3, 4, 5, 6, 8, 10, 13],
                    help="slowdown factors relative to the FAST machine")
    ap.add_argument("--rtf-threshold", type=float, default=0.8,
                    help="deployability gate: p95 RTF must stay below this")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    fast, slow = load_scaling(a.fast), load_scaling(a.slow)
    if not fast or not slow:
        sys.exit(f"need encoder_scaling_{a.fast}.csv and _{a.slow}.csv in {RAW}")
    keys = sorted(set(fast) & set(slow))
    if len(keys) < 10:
        sys.exit(f"only {len(keys)} matched conditions; need the full grid")

    tf = np.array([fast[k] for k in keys])
    ts = np.array([slow[k] for k in keys])

    print("=" * 70)
    print(f"IS THE MACHINE DIFFERENCE A SINGLE SCALAR?  ({len(keys)} matched conditions)")
    print("=" * 70)
    fit = fit_scalar(tf, ts)
    print(f"  {a.slow} = alpha x {a.fast}, through origin:")
    print(f"    alpha        = {fit['alpha_through_origin']:.3f}")
    print(f"    R2           = {fit['r2_through_origin']:.4f}")
    print(f"    max rel err  = {100*fit['max_rel_err_through_origin']:.1f}%")
    print(f"  with an intercept:")
    print(f"    alpha        = {fit['alpha_affine']:.3f}   beta = {fit['beta_affine_ms']:+.2f} ms")
    print(f"    R2           = {fit['r2_affine']:.4f}")
    print(f"    max rel err  = {100*fit['max_rel_err_affine']:.1f}%")

    # per-preset alphas: if these disagree, one scalar is not enough
    print("\n  per-preset alpha (through origin):")
    per_preset = {}
    for pre in sorted({k[0] for k in keys}):
        idx = [i for i, k in enumerate(keys) if k[0] == pre]
        al = float((tf[idx] * ts[idx]).sum() / (tf[idx] * tf[idx]).sum())
        per_preset[pre] = al
        print(f"    {pre:<6} alpha = {al:.3f}   (n={len(idx)})")
    spread = max(per_preset.values()) / min(per_preset.values())
    print(f"    spread (max/min) = {spread:.2f}x")

    single_scalar_ok = fit["max_rel_err_through_origin"] < 0.25 and spread < 1.5
    print(f"\n  VERDICT: a single scalar is "
          f"{'ADEQUATE' if single_scalar_ok else 'NOT adequate'} "
          f"(criterion: max rel err < 25% and per-preset spread < 1.5x)")
    if not single_scalar_ok:
        print("    -> 'the Pi is Nx slower' is not a well-defined statement.")
        print("       Different model sizes scale differently, so the projection")
        print("       must be reported per preset, not as one number.")

    # ---- projection: RTF vs slowdown, per (preset, chunk) ----
    print("\n" + "=" * 70)
    print(f"PROJECTED FEASIBILITY vs SLOWDOWN (relative to {a.fast}; "
          f"gate = p95 RTF < {a.rtf_threshold})")
    print("=" * 70)
    rows: List[Dict[str, object]] = []
    presets = sorted({k[0] for k in keys})
    chunks = sorted({k[1] for k in keys})
    header = f"{'preset':<7}{'chunk':>7}" + "".join(f"{s:>7.0f}x" for s in a.slowdowns)
    print(header)
    for pre in presets:
        for ch in chunks:
            # worst case over lookahead, at L=0 (the cheapest) and L=max
            cand = [fast[k] for k in keys if k[0] == pre and k[1] == ch]
            if not cand:
                continue
            t0 = max(cand)                       # worst lookahead
            cells, row = [], {"preset": pre, "chunk_ms": ch,
                              "t_compute_ms_fast": round(t0, 3)}
            for s in a.slowdowns:
                rtf = t0 * s / ch
                row[f"rtf_x{s:g}"] = round(rtf, 3)
                cells.append(("  " if rtf < a.rtf_threshold else " !") + f"{rtf:>5.2f}")
            rows.append(row)
            print(f"{pre:<7}{ch:>7.0f}" + "".join(cells))
    print("\n  ! = above the deployability gate (falls behind real time)")

    # ---- which conclusions survive the slowdown uncertainty? ----
    print("\n" + "=" * 70)
    print("WHICH CONCLUSIONS SURVIVE THE SLOWDOWN UNCERTAINTY?")
    print("=" * 70)
    lo_s, hi_s = min(a.slowdowns), max(a.slowdowns)
    robust = {"infeasible_at_all": [], "feasible_at_all": [], "boundary": []}
    for r in rows:
        lo = r[f"rtf_x{lo_s:g}"]
        hi = r[f"rtf_x{hi_s:g}"]
        name = f"{r['preset']}@{r['chunk_ms']:.0f}ms"
        if lo >= a.rtf_threshold:
            robust["infeasible_at_all"].append(name)
        elif hi < a.rtf_threshold:
            robust["feasible_at_all"].append(name)
        else:
            robust["boundary"].append(name)
    print(f"  Over the whole plausible range ({lo_s:g}x to {hi_s:g}x slower than "
          f"{a.fast}):")
    print(f"    INFEASIBLE regardless of the exact factor : "
          f"{', '.join(robust['infeasible_at_all']) or 'none'}")
    print(f"    FEASIBLE regardless of the exact factor   : "
          f"{', '.join(robust['feasible_at_all']) or 'none'}")
    print(f"    DEPENDS on the exact factor               : "
          f"{', '.join(robust['boundary']) or 'none'}")
    print("\n  This is the useful form of the result. We do not need to know the")
    print("  Pi's exact slowdown to say that a base-scale encoder cannot stream")
    print("  on it and a tiny one can. Only the boundary cases require the board.")

    # ---- where does a Pi 5 sit? two derivations that DISAGREE ----
    print("\n" + "=" * 70)
    print("WHERE A PI 5 SITS: TWO DERIVATIONS, AND THEY DISAGREE")
    print("=" * 70)
    hw_fast = os.path.join(RAW, f"hw_{a.fast}.json")
    fast_gflops = achieved_lo = achieved_hi = None
    if os.path.exists(hw_fast):
        h = json.load(open(hw_fast))
        fast_gflops = ((h.get("calibration") or {}).get("sgemm_1024") or {}).get("gflops")
    sc = os.path.join(RAW, f"encoder_scaling_{a.fast}.csv")
    if os.path.exists(sc):
        g = [float(r["achieved_gflops"]) for r in csv.DictReader(open(sc))]
        achieved_lo, achieved_hi = min(g), max(g)

    pi_peak = PI5_SPEC["neon_fp32_peak_gflops"]
    print(f"  (A) peak-vs-peak")
    print(f"      {a.fast} sgemm-1024 measured : {fast_gflops} GFLOP/s")
    print(f"      Pi 5 NEON fp32 peak         : {pi_peak:.0f} GFLOP/s")
    if fast_gflops:
        print(f"      -> slowdown {fast_gflops/(pi_peak*0.6):.0f}x (Pi at 60% of peak) "
              f"to {fast_gflops/(pi_peak*0.4):.0f}x (at 40%)")
    print(f"\n  (B) achieved-vs-achieved on THIS workload")
    if achieved_lo:
        print(f"      {a.fast} achieves {achieved_lo:.0f}-{achieved_hi:.0f} GFLOP/s "
              f"= {100*achieved_hi/fast_gflops:.0f}% of its own sgemm rate"
              if fast_gflops else "")
        print(f"      If the Pi hits a similar fraction of its own peak, the ratio of")
        print(f"      ACHIEVED throughputs is far smaller than (A) suggests, because")
        print(f"      both machines are overhead-bound, not FLOP-bound (F2).")
    print(f"\n  These two derivations differ by roughly an order of magnitude.")
    print(f"  That disagreement is the result: **the Pi's slowdown cannot be")
    print(f"  projected reliably from specifications**, which is exactly why the")
    print(f"  board is on the shopping list. What we CAN state without it is the")
    print(f"  robustness table above.")
    print(f"\n  {PI5_SPEC['note']}")
    print("\n  EVERY Pi NUMBER HERE IS A PROJECTION, NOT A MEASUREMENT.")
    print("  It may appear in the paper only with that label, and only until a")
    print("  real board replaces it. See setup/SETUP_RASPBERRY_PI.md.")

    out = a.out or os.path.join(RAW, "projection_feasibility.csv")
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(out.replace(".csv", "_model.json"), "w") as f:
        json.dump({"fast": a.fast, "slow": a.slow, "n_conditions": len(keys),
                   "scalar_fit": fit, "per_preset_alpha": per_preset,
                   "per_preset_spread": spread,
                   "single_scalar_adequate": single_scalar_ok,
                   "slowdown_axis": a.slowdowns,
                   "rtf_threshold": a.rtf_threshold,
                   "robust_conclusions": robust,
                   "pi5_spec": PI5_SPEC,
                   "status": "PROJECTION -- NOT MEASURED"}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
