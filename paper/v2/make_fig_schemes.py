"""Figure: PER against lookahead under the three masking schemes (WavLM base+).

    ../../.venv/bin/python make_fig_schemes.py   ->  fig_schemes.pdf / .png
"""
import collections
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "raw")


def from_jsonl(*paths, target="native"):
    d = collections.defaultdict(list)
    for p in paths:
        for line in open(os.path.join(RAW, p)):
            r = json.loads(line)
            if r["target"] == target:
                d[float(r["lookahead_ms"])].append(r["test_per"])
    return d


def from_csv(path):
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(RAW, path))):
        d[float(r["lookahead_ms"])].append(float(r["test_per"]))
    return d


SERIES = [  # label, data, colour (validated categorical slots 1-3), marker
    ("per-layer (leaks)", from_jsonl("translator_dense.jsonl", "translator_dense_plus_sat.jsonl"),
     "#2a78d6", "o"),
    ("encoder-first (exact)", from_csv("e2e/translator_encfirst.csv"), "#eb6834", "s"),
    ("end-to-end (exact)", from_jsonl("e2e/translator_e2e_dense.jsonl"), "#1baf7a", "^"),
]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.45, 2.35))
x_of = lambda L: np.log2(np.asarray(L) + 20.0)  # noqa: E731  (L = 0 must sit on the axis)
for label, d, col, mk in SERIES:
    Ls = sorted(d)
    m = np.array([np.mean(d[L]) for L in Ls])
    ax.plot(x_of(Ls), m, color=col, lw=1.6, marker=mk, ms=3.6, mec="white", mew=0.6,
            label=label, zorder=3)
    ax.annotate(label.split(" (")[0], (x_of(Ls[-1]), m[-1]), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=7, color="#333333")
ticks = [0, 20, 40, 80, 160, 320, 640]
ax.set_xticks(x_of(ticks), [str(t) for t in ticks])
ax.set_xlim(x_of(0) - 0.2, x_of(640) + 2.3)
ax.set_xlabel("nominal lookahead $L$ (ms)")
ax.set_ylabel("test PER")
ax.grid(axis="y", color="#e4e4e0", lw=0.5, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout(pad=0.3)
out = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out, "fig_schemes.pdf"))
fig.savefig(os.path.join(out, "fig_schemes.png"), dpi=300)
