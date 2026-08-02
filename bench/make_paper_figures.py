"""Paper figures. Single-column Interspeech width (3.4in), vector output.

Distinct from make_figures.py, which is exploratory. These are the four the
draft actually cites, sized and styled for submission.
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

RAW = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
FIG = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")

def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 7, "axes.grid": True, "grid.alpha": .3,
                         "figure.dpi": 300, "savefig.bbox": "tight",
                         "axes.labelsize": 7, "legend.fontsize": 6})
    W = 3.4

    # --- F1: causality ablation ---
    fig, ax = plt.subplots(figsize=(W, 1.9))
    labels = ["mask\nonly", "+pos-\nconv", "+group\nnorm", "both\npatches"]
    vals = [1.1438e-2, 5.9956e-3, 2.5983e-2, 6.14e-6]
    cols = ["tab:red"]*3 + ["tab:green"]
    ax.bar(range(4), vals, color=cols)
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.axhline(1e-4, ls="--", c="k", lw=.8)
    ax.text(3.35, 1.4e-4, "causal", fontsize=5.5, ha="right")
    ax.set_ylabel("relative L2 deviation")
    ax.set_title("Truncation proof: only both patches are causal", fontsize=7)
    for e in ("pdf","png"): fig.savefig(f"{FIG}/fig1_causality.{e}")
    plt.close(fig); print("fig1_causality")

    # --- F2: compute flat in lookahead, chunk moves it ---
    rows=[r for r in csv.DictReader(open(f"{RAW}/encoder_scaling_m4.csv"))]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    for ch,mk in zip([20.,40.,80.],["o","s","^"]):
        d=sorted((float(r["lookahead_ms"]),float(r["t_compute_p50_ms"]))
                 for r in rows if r["preset"]=="small" and float(r["chunk_ms"])==ch)
        ax.plot([x for x,_ in d],[y for _,y in d],mk+"-",ms=2.5,lw=1,label=f"chunk {ch:.0f} ms")
    ax.set_ylim(bottom=0); ax.set_xscale("symlog", linthresh=20)
    ax.set_xticks([0,20,40,80,160,320,640]); ax.set_xticklabels(["0","20","40","80","160","320","640"],rotation=45)
    ax.set_xlabel("lookahead $L$ (ms)"); ax.set_ylabel("$t_{compute}$ p50 (ms/chunk)")
    ax.set_title("Compute is flat in lookahead (Apple M4)", fontsize=7); ax.legend()
    for e in ("pdf","png"): fig.savefig(f"{FIG}/fig2_compute_flat.{e}")
    plt.close(fig); print("fig2_compute_flat")

    # --- F3: the exchange rate (dense GPU run, deduplicated) ---
    L=[0,20,40,60,80,100,120,140,160,180,200,240,280,320,480,640]
    y=[0.49716,0.42209,0.37166,0.33439,0.30919,0.28621,0.26562,0.24664,
       0.23045,0.21639,0.20509,0.18769,0.17298,0.16078,0.12768,0.10737]
    fig,(a1,a2)=plt.subplots(2,1,figsize=(W,3.0),height_ratios=[2,1.4])
    a1.semilogx(L[1:], y[1:], "o-", ms=2.5, lw=1, base=2)
    a1.scatter([40],[y[2]],s=20,facecolors="none",edgecolors="tab:red",lw=1,zorder=5)
    a1.annotate("PHONOS\n40 ms", xy=(40, y[2]), xytext=(70, 0.44), fontsize=5.5,
                color="tab:red", arrowprops=dict(arrowstyle="->", lw=.6, color="tab:red"))
    a1.set_ylabel("drift from bidirectional"); a1.set_xlabel("")
    a1.set_title("Log-linear in lookahead ($R^2$=0.994), no cliff", fontsize=7)
    a1.tick_params(labelbottom=False)
    prof=[(20,40,.0504),(40,80,.0625),(80,160,.0787),(100,200,.0811),
          (160,320,.0697),(320,640,.0534)]
    a2.plot([p[0] for p in prof],[p[2] for p in prof],"s-",ms=3,lw=1,c="tab:orange")
    a2.axvline(100,ls=":",c="k",lw=.8); a2.set_xscale("log", base=2)
    a2.set_xlabel("lookahead $L$ (ms), start of doubling")
    a2.set_ylabel("gain per\ndoubling")
    a2.annotate("best marginal\nreturn 100-200 ms", xy=(100,.0811), xytext=(160,.055),
                fontsize=5.5, arrowprops=dict(arrowstyle="->", lw=.6))
    for e in ("pdf","png"): fig.savefig(f"{FIG}/fig3_exchange_rate.{e}")
    plt.close(fig); print("fig3_exchange_rate")

    # --- F4: projected feasibility heat table ---
    pr=[r for r in csv.DictReader(open(f"{RAW}/projection_feasibility.csv"))]
    sd=[c for c in pr[0] if c.startswith("rtf_x")]
    xs=[float(c.replace("rtf_x","")) for c in sd]
    M=np.array([[float(r[c]) for c in sd] for r in pr])
    labs=[f"{r['preset']}@{float(r['chunk_ms']):.0f}" for r in pr]
    fig,ax=plt.subplots(figsize=(W,2.2))
    im=ax.imshow(np.log10(M),aspect="auto",cmap="RdYlGn_r",vmin=-1.4,vmax=1.4)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:.0f}x" for x in xs],fontsize=6)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j,i,f"{M[i,j]:.1f}",ha="center",va="center",fontsize=4.5,
                    color="white" if abs(np.log10(M[i,j]))>.8 else "black")
    ax.set_xlabel("slowdown relative to Apple M4"); ax.grid(False)
    ax.set_title("PROJECTED p95 RTF (green<0.8 deployable)", fontsize=7)
    for e in ("pdf","png"): fig.savefig(f"{FIG}/fig4_projected_feasibility.{e}")
    plt.close(fig); print("fig4_projected_feasibility")
    print(f"\nwrote to {os.path.abspath(FIG)}")

if __name__ == "__main__":
    main()
