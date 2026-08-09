"""Figure: the dense lookahead curve, and why it has no knee to report.

Panel A  the curve on a log2 axis, with the log-linear fit and the two
         candidate breakpoints from the axis sweep drawn as a *band* rather
         than a line -- the point being that they disagree.
Panel B  marginal return per 20 ms frame against the measured noise floor,
         which is where the saturation point comes from and is the quantity a
         deployment engineer needs.

Every number is read from results/analysis_dense_knee.json. Nothing is
hardcoded, so the figure cannot drift from the analysis the way
make_3seed_figure.py did.
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "analysis_dense_knee.json")
OUT = os.path.join(HERE, "..", "results", "figures", "fig_dense_no_knee")

CONV = "#1b4f72"
WARN = "#a93226"
GREY = "#7f8c8d"


def main() -> None:
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} not found -- run eval/analyse_dense_knee.py first")
    d = json.load(open(SRC))
    Ls = [c["L_ms"] for c in d["curve"]]
    ys = [c["mean_per"] for c in d["curve"]]
    sp = [c["seed_spread"] for c in d["curve"]]
    mr = d["marginal_returns"]
    floor = mr["noise_2sigma"]
    sat = mr["saturation_ms"]
    sweep = d["axis_sweep"]

    # L=0 cannot sit on a log axis; draw it at half a frame and say so.
    xs = [math.log2((10.0 if L == 0 else L)) for L in Ls]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 3.9))

    # ---------------- Panel A ----------------
    a.plot(xs, ys, "o-", color=CONV, lw=1.8, ms=4.5, label="conversion (g2p), 2 seeds")
    a.fill_between(xs, [y - s / 2 for y, s in zip(ys, sp)],
                   [y + s / 2 for y, s in zip(ys, sp)],
                   color=CONV, alpha=0.18, lw=0)

    # log-linear fit on L >= 20
    fx = [x for x, L in zip(xs, Ls) if L >= 20]
    fy = [y for y, L in zip(ys, Ls) if L >= 20]
    n = len(fx)
    mx, my = sum(fx) / n, sum(fy) / n
    sxx = sum((x - mx) ** 2 for x in fx)
    slope = sum((x - mx) * (y - my) for x, y in zip(fx, fy)) / sxx
    icept = my - slope * mx
    ss = sum((y - (icept + slope * x)) ** 2 for x, y in zip(fx, fy))
    sst = sum((y - my) ** 2 for y in fy)
    a.plot(fx, [icept + slope * x for x in fx], "--", color=GREY, lw=1.2,
           label=f"log-linear fit: {slope:+.4f}/doubling, $R^2$={1 - ss / sst:.3f}")

    # the disagreeing breakpoints, as a band
    locs = [s["knee_ms"] for s in sweep if s["knee_ms"] is not None]
    if locs:
        lo, hi = min(locs), max(locs)
        a.axvspan(math.log2(lo), math.log2(hi), color=WARN, alpha=0.10, lw=0)
        a.text(math.log2(math.sqrt(lo * hi)), min(ys) + 0.030,
               "breakpoint estimates disagree\n"
               f"{int(lo)}-{int(hi)} ms, none identifiable",
               color=WARN, fontsize=7.0, ha="center", va="bottom")

    if sat is not None:
        a.axvline(math.log2(sat), color="#117864", lw=1.4, ls=":")
        a.text(math.log2(sat) + 0.10, max(ys) * 0.80,
               f"saturation\n{sat:.0f} ms", color="#117864",
               fontsize=7.4, va="top")

    a.set_title("A  smooth curvature, no locatable knee", fontsize=10, loc="left")
    a.set_xlabel("lookahead $L$ (ms, $\\log_2$ axis; $L{=}0$ drawn at 10 ms)")
    a.set_ylabel("test PER (speaker-disjoint)")
    ticks = [10, 20, 40, 80, 160, 320, 640]
    a.set_xticks([math.log2(t) for t in ticks])
    a.set_xticklabels(["0" if t == 10 else str(t) for t in ticks])
    a.grid(alpha=0.25)
    a.legend(fontsize=7.2, loc="upper right")

    # ---------------- Panel B ----------------
    mids, gains, cols = [], [], []
    for s in mr["steps"]:
        mids.append(math.log2(math.sqrt(max(10.0, s["from_ms"]) * s["to_ms"])))
        gains.append(s["gain_per_frame"])
        cols.append(CONV if s["resolvable"] else WARN)
    b.bar(mids, gains, width=0.26, color=cols)
    b.axhline(floor, color="k", lw=1.2, ls="--")
    b.text(mids[0], floor * 1.25, f"2$\\sigma$ noise floor = {floor:.4f}",
           fontsize=7.4, va="bottom")
    if sat is not None:
        b.axvline(math.log2(sat), color="#117864", lw=1.4, ls=":")
    b.set_yscale("log")
    b.set_title("B  marginal return per 20 ms frame vs measured noise",
                fontsize=10, loc="left")
    b.set_xlabel("lookahead step (ms, $\\log_2$ axis)")
    b.set_ylabel("PER gain per added frame")
    b.set_xticks([math.log2(t) for t in ticks])
    b.set_xticklabels(["0" if t == 10 else str(t) for t in ticks])
    b.grid(alpha=0.25, axis="y")
    b.text(0.97, 0.95,
           f"red = below floor:\nnot measurably useful\n"
           f"from {sat:.0f} ms on",
           transform=b.transAxes, fontsize=7.2, color=WARN,
           va="top", ha="right")

    prov = d["provenance"]
    fig.suptitle(f"{prov['n_rows']} runs: {prov['n_lookaheads']} lookaheads "
                 f"$\\times$ {len(prov['seeds'])} seeds, {prov['target']} arm, "
                 f"padding-fixed, 1200 steps, T4", fontsize=8.6, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=190, bbox_inches="tight")
    print(f"wrote {OUT}.{{png,pdf}}")
    print(f"  slope {slope:+.4f}/doubling  R2 {1 - ss / sst:.4f}  "
          f"saturation {sat} ms  breakpoints {locs}")


if __name__ == "__main__":
    main()
