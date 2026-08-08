"""Figure for §5.2: why the paper reports the margin and not PER.

One panel would not make the point. The claim is comparative -- two quantities
measured on the *same 42 runs*, one of which is too seed-noisy to support the
conclusions people normally draw from curves like this, and one of which is not.
So the figure is two panels sharing an x-axis with error bars on the same
visual footing, and the reader is meant to notice that the left-hand ribbons
overlap almost everywhere while the right-hand ones do not.

Panel A: own-target PER, mean +- SD over seeds, both arms.
Panel B: preference margin (cross - own), same seeds, same axes treatment.

Two annotations earn their space:
  * the 160->320 ms step in panel A, which is -0.0009 PER with 1/3 seeds
    improving -- the single-seed "plateau" that turned out to be nothing;
  * the seed-SD ratio, printed rather than described, because "10x more
    stable" is the whole argument for switching instruments.

x is log2-spaced with L=0 placed at 10 ms, matching eval/phoneme_analysis.py
and the BIC fit, so the panels are readable against the knee analysis rather
than on a separate scale.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from analyse_3seed import CROSS, L, OWN, SEEDS, margin, mean, sd  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

X = [math.log2(max(x, 10)) for x in L]      # 0 ms drawn at 10 ms
ARMS = {"native": ("conversion (g2p)", "#1b4f72", "o", "-"),
        "produced": ("transcription (ipa)", "#b03a2e", "s", "--")}


def series(arm, kind):
    if kind == "per":
        return [[OWN[arm][s][i] for s in SEEDS] for i in range(len(L))]
    return [[margin(arm, s)[i] for s in SEEDS] for i in range(len(L))]


def panel(ax, kind, title, ylabel):
    for arm, (lab, col, mk, ls) in ARMS.items():
        vals = series(arm, kind)
        m = [mean(v) for v in vals]
        e = [sd(v) for v in vals]
        ax.fill_between(X, [a - b for a, b in zip(m, e)],
                        [a + b for a, b in zip(m, e)], color=col, alpha=0.18,
                        linewidth=0)
        ax.errorbar(X, m, yerr=e, color=col, marker=mk, ls=ls, ms=5, lw=1.6,
                    capsize=3, label=lab, zorder=3)
        # individual seeds, so the reader sees the actual spread not just +-SD
        for j, s in enumerate(SEEDS):
            ax.plot(X, [vals[i][j] for i in range(len(L))], color=col,
                    alpha=0.28, lw=0.7, zorder=2)
    ax.set_xticks(X)
    ax.set_xticklabels([str(v) for v in L])
    ax.set_xlabel("lookahead $L$ (ms, log$_2$ axis; 0 drawn at 10)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, frameon=False)


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.7))

    panel(a, "per", "A  own-target PER: adjacent doublings are not resolved",
          "test PER (speaker-disjoint)")
    # the step that mattered
    i0, i1 = L.index(160), L.index(320)
    y = mean([OWN["native"][s][i1] for s in SEEDS])
    a.annotate("160$\\to$320 ms:\n$-$0.0009 PER,\n1/3 seeds improve",
               xy=(X[i1], y), xytext=(X[i0] - 0.55, y + 0.085),
               fontsize=7.4, color="#1b4f72",
               arrowprops=dict(arrowstyle="->", color="#1b4f72", lw=0.8))

    panel(b, "margin", "B  preference margin: monotone in every seed",
          "PER(other target) $-$ PER(own target)")

    per_sd = mean([sd([OWN["native"][s][i] for s in SEEDS])
                   for i in range(len(L))])
    mar_sd = mean([sd(v) for v in series("native", "margin")])
    b.text(0.03, 0.955,
           f"mean seed SD  {mar_sd:.4f}  vs  {per_sd:.4f} in panel A\n"
           f"({per_sd / mar_sd:.0f}$\\times$ more stable)",
           transform=b.transAxes, fontsize=7.6, va="top",
           bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999", lw=0.6))

    fig.suptitle("42 runs: 7 lookaheads $\\times$ 2 targets $\\times$ 3 seeds, "
                 "1200 steps, frozen causal WavLM-base, Tesla T4",
                 fontsize=8.6, y=1.005, color="#444")
    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"results/figures/fig_3seed_margin_vs_per.{ext}",
                    dpi=200, bbox_inches="tight")
    print("wrote results/figures/fig_3seed_margin_vs_per.{pdf,png}")
    print(f"  panel A mean seed SD {per_sd:.4f}   panel B {mar_sd:.4f}   "
          f"ratio {per_sd / mar_sd:.1f}x")


if __name__ == "__main__":
    main()
