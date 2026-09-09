"""Does the lookahead curve replicate on a second SSL encoder?

A reviewer asked whether the headline exchange rate is a property of speech or
specific to the WavLM configuration tested. The replication holds everything
fixed -- corpus, split,
head, optimiser, step budget, seed, chunk, lookback, grid -- and changes only
`--encoder-name`, from `microsoft/wavlm-base-plus` to `facebook/wav2vec2-base`,
both causalised by the same patches and both verified by the same truncation
proof.

Absolute PER is NOT expected to match. WavLM base+ is pretrained on ~94k hours
with denoising and speaker-conditioned objectives; wav2vec 2.0 base sees 960 h
of clean read LibriSpeech. Layer 9 of the two is not the same representation
and the weaker one starts much further from the floor.

So the comparison is of SHAPE, at two levels:

  slope_per_doubling  the paper's headline, in raw PER units. Directly
                      comparable only if the two encoders have similar
                      dynamic range, which they do not, so it is reported
                      but not leaned on.
  relative slope      the same fit after dividing each curve by its own L=0
                      PER. This asks whether each encoder recovers the same
                      FRACTION of its available headroom per doubling, which
                      is the claim that should transfer across backbones.

Reporting only the raw slope would let a difference in starting PER masquerade
as a difference in lookahead sensitivity.

    python3 eval/analyse_second_encoder.py --self-test
    python3 eval/analyse_second_encoder.py
"""
from __future__ import annotations

import argparse, json, math, os, sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

GRID = [0, 20, 40, 80, 160, 320, 640]


def fit_log2(Ls: Sequence[float], pers: Sequence[float]) -> Dict[str, float]:
    """Least squares of PER on log2(L), excluding L=0 (log2 undefined).

    Same treatment as the paper's headline fit: L >= 20 only, so the reported
    slope is not driven by the axis convention chosen for the zero point.
    """
    pts = [(math.log2(L), p) for L, p in zip(Ls, pers) if L > 0]
    if len(pts) < 3:
        return {"slope_per_doubling": float("nan"), "r2": float("nan"), "n": len(pts)}
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    tot = sum((y - my) ** 2 for y in ys)
    return {"slope_per_doubling": b, "intercept": a,
            "r2": 1 - ss / tot if tot > 0 else float("nan"), "n": n}


def summarise(Ls: Sequence[float], pers: Sequence[float]) -> Dict[str, object]:
    raw = fit_log2(Ls, pers)
    base = pers[0]                       # PER at L=0: this encoder's headroom
    rel = fit_log2(Ls, [p / base for p in pers])
    return {
        "per_at_L0": base,
        "per_at_Lmax": pers[-1],
        "relative_gain_0_to_max": (base - pers[-1]) / base,
        "raw": raw,
        "normalised": rel,
    }


def _self_test() -> None:
    print("analyse_second_encoder self-test")
    ok = True
    def chk(n, c, x=""):
        nonlocal ok; ok &= bool(c); print(f"  {n:<52s} {'ok' if c else 'FAIL'} {x}")

    # exact log-linear curve: slope must come back exactly
    Ls = [0, 20, 40, 80, 160, 320, 640]
    ys = [0.9] + [0.5 - 0.04 * math.log2(L) for L in Ls[1:]]
    f = fit_log2(Ls, ys)
    chk("recovers a known slope", abs(f["slope_per_doubling"] + 0.04) < 1e-9,
        f'{f["slope_per_doubling"]:.6f}')
    chk("R2 = 1 on an exact line", abs(f["r2"] - 1) < 1e-9)
    chk("L=0 excluded from the fit", f["n"] == 6, f'n={f["n"]}')

    # scaling a curve scales the raw slope but not the normalised one
    s = summarise(Ls, ys)
    s2 = summarise(Ls, [y * 2 for y in ys])
    chk("raw slope scales with the curve",
        abs(s2["raw"]["slope_per_doubling"] - 2 * s["raw"]["slope_per_doubling"]) < 1e-9)
    chk("normalised slope is scale-invariant",
        abs(s2["normalised"]["slope_per_doubling"]
            - s["normalised"]["slope_per_doubling"]) < 1e-9,
        "this is the point of the normalisation")
    chk("relative gain is scale-invariant",
        abs(s2["relative_gain_0_to_max"] - s["relative_gain_0_to_max"]) < 1e-9)
    chk("too few points -> nan, not a fabricated slope",
        math.isnan(fit_log2([0, 20], [0.9, 0.5])["slope_per_doubling"]))
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    if not ok: sys.exit(1)


def load_jsonl(path: str, target="native") -> Dict[float, List[float]]:
    out: Dict[float, List[float]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("target") == target:
                out[float(r["lookahead_ms"])].append(float(r["test_per"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavlm", default="results/raw/translator_dense.jsonl")
    ap.add_argument("--w2v2", default="results/raw/translator_w2v2.jsonl")
    ap.add_argument("--out", default="results/analysis_second_encoder.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test: _self_test(); return

    w = load_jsonl(a.wavlm)
    v = load_jsonl(a.w2v2)
    grid = [L for L in GRID if L in w and L in v]
    missing = [L for L in GRID if L not in v]
    if missing:
        print(f"NOTE: wav2vec2 missing {missing}; comparing on {grid}")
    if len(grid) < 4:
        sys.exit(f"only {len(grid)} shared lookaheads; not enough to compare shape")

    wl = [sum(w[L]) / len(w[L]) for L in grid]
    vl = [sum(v[L]) / len(v[L]) for L in grid]

    print(f"{'L (ms)':>7}  {'WavLM':>8}  {'wav2vec2':>9}")
    for L, x, y in zip(grid, wl, vl):
        print(f"{L:>7.0f}  {x:>8.4f}  {y:>9.4f}")

    W, V = summarise(grid, wl), summarise(grid, vl)
    print(f"\n{'':22s}{'WavLM':>10}{'wav2vec2':>11}")
    print(f"  {'PER at L=0':<20}{W['per_at_L0']:>10.4f}{V['per_at_L0']:>11.4f}")
    print(f"  {'PER at L=max':<20}{W['per_at_Lmax']:>10.4f}{V['per_at_Lmax']:>11.4f}")
    print(f"  {'relative gain':<20}{W['relative_gain_0_to_max']:>10.3f}"
          f"{V['relative_gain_0_to_max']:>11.3f}")
    print(f"  {'raw slope/doubling':<20}{W['raw']['slope_per_doubling']:>10.4f}"
          f"{V['raw']['slope_per_doubling']:>11.4f}")
    print(f"  {'  R2':<20}{W['raw']['r2']:>10.3f}{V['raw']['r2']:>11.3f}")
    print(f"  {'normalised slope':<20}{W['normalised']['slope_per_doubling']:>10.4f}"
          f"{V['normalised']['slope_per_doubling']:>11.4f}")
    print(f"  {'  R2':<20}{W['normalised']['r2']:>10.3f}{V['normalised']['r2']:>11.3f}")

    rn = V["normalised"]["slope_per_doubling"] / W["normalised"]["slope_per_doubling"]
    rr = V["raw"]["slope_per_doubling"] / W["raw"]["slope_per_doubling"]
    print(f"\n  ratio wav2vec2/WavLM  raw {rr:.2f}   normalised {rn:.2f}")

    json.dump({"grid_ms": grid, "wavlm": {"per": wl, **W},
               "wav2vec2": {"per": vl, **V},
               "ratio_raw": rr, "ratio_normalised": rn,
               "note": "absolute PER is not expected to match; the claim under "
                       "test is the shape of the curve, best read from the "
                       "headroom-normalised slope"},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
