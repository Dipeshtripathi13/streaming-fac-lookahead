"""Figures from whatever result CSVs exist in results/raw/.

Skips anything missing rather than failing, so it is safe to run mid-sweep.
Writes SVG (vector, Interspeech-safe) and PNG.

Figures produced:
  fig1_compute_vs_lookahead   -- RQ4: t_compute is ~flat in L
  fig2_algo_vs_compute        -- the decomposition, stacked
  fig3_rtf_feasibility        -- which (hw, chunk) pairs are deployable at all
  fig4_quality_vs_lookahead   -- RQ1, once quality numbers exist (H1 knee)
  fig5_commit_delay           -- the cascade's commit-timeout trade-off
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

RAW = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
FIG = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


def load(pattern: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in sorted(glob.glob(os.path.join(RAW, pattern))):
        with open(p) as f:
            for r in csv.DictReader(f):
                r["_src"] = os.path.basename(p)
                rows.append(r)
    return rows


def f(x, d=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("pip install matplotlib")

    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                         "axes.grid": True, "grid.alpha": 0.3})

    rows = load("encoder_scaling*.csv")
    if not rows:
        print("no encoder_scaling*.csv yet -- run bench/bench_encoder_scaling.py")
    else:
        # ---- fig 1: compute vs lookahead ----
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
        presets = sorted({r["preset"] for r in rows})
        for ax, pre in zip(axes, presets):
            for chunk in sorted({f(r["chunk_ms"]) for r in rows if r["preset"] == pre}):
                d = sorted((f(r["lookahead_ms"]), f(r["t_compute_p50_ms"]))
                           for r in rows if r["preset"] == pre
                           and f(r["chunk_ms"]) == chunk)
                ax.plot([x for x, _ in d], [y for _, y in d], "o-",
                        label=f"chunk {chunk:g} ms", ms=3, lw=1.2)
            # Anchor y at zero. With an auto-scaled axis a 6% rise fills the
            # panel and the figure argues the opposite of the finding. The
            # claim is "flat"; the axis has to let the reader see flat.
            ax.set_ylim(bottom=0)
            ax.set_title(f"encoder: {pre}")
            ax.set_xlabel("lookahead L (ms)")
            ax.set_xscale("symlog", linthresh=20)
            ax.set_xticks([0, 20, 40, 80, 160, 320, 640])
            ax.set_xticklabels(["0", "20", "40", "80", "160", "320", "640"],
                               rotation=45, fontsize=7)
        axes[0].set_ylabel("t_compute p50 (ms / chunk)")
        axes[-1].legend(fontsize=7)
        fig.suptitle("Per-chunk compute is nearly flat in lookahead; "
                     "chunk size is what moves it", fontsize=10)
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(os.path.join(FIG, f"fig1_compute_vs_lookahead.{ext}"))
        plt.close(fig)
        print("wrote fig1_compute_vs_lookahead")

        # ---- fig 2: the decomposition ----
        sel = [r for r in rows if r["preset"] == ("small" if "small" in presets else presets[0])
               and f(r["chunk_ms"]) == 40]
        sel.sort(key=lambda r: f(r["lookahead_ms"]))
        if sel:
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            xs = list(range(len(sel)))
            algo = [f(r["t_algorithmic_ms"]) for r in sel]
            comp = [f(r["t_compute_p95_ms"]) for r in sel]
            ax.bar(xs, algo, label="t_algorithmic (chunk + lookahead)")
            ax.bar(xs, comp, bottom=algo, label="t_compute p95")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{f(r['lookahead_ms']):g}" for r in sel])
            ax.set_xlabel("lookahead L (ms)")
            ax.set_ylabel("latency (ms)")
            ax.set_title("Latency decomposition: the inherent term dominates")
            ax.legend(fontsize=7)
            fig.tight_layout()
            for ext in ("svg", "png"):
                fig.savefig(os.path.join(FIG, f"fig2_algo_vs_compute.{ext}"))
            plt.close(fig)
            print("wrote fig2_algo_vs_compute")

        # ---- fig 3: RTF feasibility ----
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        for pre in presets:
            d = defaultdict(list)
            for r in rows:
                if r["preset"] == pre:
                    d[f(r["chunk_ms"])].append(f(r["rtf_p95"]))
            xs = sorted(d)
            ax.plot(xs, [max(d[c]) for c in xs], "s-", label=pre, ms=4)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.axhline(0.8, color="r", ls=":", lw=1)
        ax.text(0.02, 0.82, "deployability threshold (0.8)", transform=ax.transAxes,
                fontsize=6, color="r")
        ax.set_xlabel("chunk size (ms)")
        ax.set_ylabel("worst-case RTF (p95)")
        ax.set_yscale("log")
        ax.set_title("Small chunks are the feasibility problem, not lookahead")
        ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(os.path.join(FIG, f"fig3_rtf_feasibility.{ext}"))
        plt.close(fig)
        print("wrote fig3_rtf_feasibility")

    # ---- fig 5: the commit-delay trade-off ----
    import json as _json
    cd = os.path.join(RAW, "commit_delay_zipformer_summary.json")
    if os.path.exists(cd):
        d = _json.load(open(cd))
        curve = d["tradeoff_curve"]
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        xs = [c["commit_timeout_ms"] for c in curve]
        ys = [c["pct_released_unstable"] for c in curve]
        ax.plot(xs, ys, "o-", ms=4, color="tab:red")
        ax.axvline(320, ls=":", lw=1, color="k")
        ax.text(325, max(ys) * 0.9, " model decode chunk\n 320 ms", fontsize=6)
        ax.axvline(700, ls="--", lw=1, color="tab:blue")
        ax.text(705, max(ys) * 0.55, " accentbridge\n default", fontsize=6,
                color="tab:blue")
        ax.set_xlabel("commit timeout (ms)")
        ax.set_ylabel("% words released while still unstable")
        ax.set_title("Cascade commit delay: the cliff sits at the decode chunk")
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(os.path.join(FIG, f"fig5_commit_delay.{ext}"))
        plt.close(fig)
        print("wrote fig5_commit_delay")
    else:
        print("no commit_delay summary yet -- run bench/bench_commit_delay.py")

    # ---- fig 4: quality vs lookahead (only once quality exists) ----
    q = load("quality*.csv") + load("conditions*.csv")
    if q:
        from phoneme_analysis import find_knee
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        for mode in sorted({r.get("mode", "AC") for r in q}):
            d = sorted((f(r["lookahead_ms"]), f(r.get("accent_p_nonnative")))
                       for r in q if r.get("mode", "AC") == mode)
            d = [(x, y) for x, y in d if y == y]
            if len(d) < 3:
                continue
            xs, ys = zip(*d)
            ax.plot(xs, ys, "o-", label=mode, ms=4)
            k = find_knee(xs, ys)
            if k["knee_ms"] == k["knee_ms"]:
                ax.axvline(k["knee_ms"], ls=":", lw=1)
                ax.text(k["knee_ms"], max(ys), f" knee {k['knee_ms']:.0f} ms",
                        fontsize=7, va="top")
        ax.set_xscale("symlog", linthresh=20)
        ax.set_xlabel("lookahead L (ms)")
        ax.set_ylabel("accent probe p(non-native)")
        ax.set_title("RQ1/H1: quality vs lookahead, AC vs VC-only")
        ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(os.path.join(FIG, f"fig4_quality_vs_lookahead.{ext}"))
        plt.close(fig)
        print("wrote fig4_quality_vs_lookahead")
    else:
        print("no quality CSVs yet -- fig4 waits for the November sweep")

    print(f"\nfigures in {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()
