"""Fold a finished translator sweep into the paper, with the H3 test.

Run this once `translator_sweep_summary.json` lands (download it from Colab into
results/raw/). It reads the curves, applies the same knee machinery used for the
encoder result -- so the task-level and representation-level answers are judged
by identical criteria -- and prints a paper-ready block for section 5.2.

    python3 bench/ingest_sweep.py
    python3 bench/ingest_sweep.py --summary /path/to/translator_sweep_summary.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from phoneme_analysis import find_knee  # noqa: E402

RAW = os.path.join(os.path.dirname(__file__), "..", "results", "raw")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(RAW, "translator_sweep_summary.json"))
    a = ap.parse_args()
    if not os.path.exists(a.summary):
        sys.exit(f"not found: {a.summary}\n"
                 f"Download it from Colab ({{RES}}/translator_sweep_summary.json) "
                 f"into results/raw/ first.")
    s = json.load(open(a.summary))

    print("=" * 68)
    print("TRAINED TRANSLATOR SWEEP  (section 5.2)")
    print("=" * 68)
    c = s.get("corpus", {})
    print(f"corpus: {c.get('n_items')} utts, train/val/test "
          f"{c.get('n_train')}/{c.get('n_val')}/{c.get('n_test')}, "
          f"vocab {c.get('vocab_size')}")
    print(f"speaker-disjoint: val={c.get('val_speakers')} test={c.get('test_speakers')}")
    print(f"mean PER(g2p, ipa) = {c.get('mean_per_g2p_vs_ipa')}  "
          f"(> 0.02 means the two arms are genuinely different tasks)\n")

    curves = {}
    for arm in ("native", "produced"):
        cur = s.get(f"curve_{arm}")
        if not cur:
            continue
        Ls, per = cur["lookaheads_ms"], cur["test_per"]
        curves[arm] = (Ls, per)
        k = find_knee(Ls, per)
        label = {"native": "ACCENT CONVERSION (target g2p)",
                 "produced": "TRANSCRIPTION control (target ipa)"}[arm]
        print(f"--- {label} ---")
        print(f"{'L (ms)':>8}{'test PER':>11}")
        for L, p in zip(Ls, per):
            print(f"{L:>8.0f}{p:>11.4f}")
        print(f"  R2 log-linear {k['r2_loglinear']}   dBIC {k['delta_bic_piecewise_minus_loglinear']}"
              f"   has_knee {k['has_knee']}   underpowered {k['underpowered_for_bic']}")
        print(f"  slope/doubling {k['slope_per_doubling']}")
        if k["gain_per_doubling_profile"]:
            print("  gain per doubling: " + ", ".join(
                f"{p['from_ms']:.0f}->{p['to_ms']:.0f}:{p['delta']:+.4f}"
                for p in k["gain_per_doubling_profile"]))
        print()

    if len(curves) == 2:
        (Ln, pn), (Lp, pp) = curves["native"], curves["produced"]
        gn = (pn[0] - pn[-1]) / pn[0] if pn[0] else float("nan")
        gp = (pp[0] - pp[-1]) / pp[0] if pp[0] else float("nan")
        print("=" * 68)
        print("H3: does CONVERSION need more lookahead than TRANSCRIPTION?")
        print("=" * 68)
        print(f"  relative gain 0 -> max lookahead")
        print(f"    conversion (g2p)   : {gn:.4f}")
        print(f"    transcription(ipa) : {gp:.4f}")
        supported = gn > gp
        print(f"  -> H3 {'SUPPORTED' if supported else 'NOT supported'} "
              f"(difference {gn-gp:+.4f})")
        print("\n  Read with care: the two arms share architecture, capacity, seed,")
        print("  data order and step count, so the difference is attributable to the")
        print("  target. But PER floors differ between targets, so compare the")
        print("  RELATIVE gain, not the absolute PER.")
        # per-L difference, the shape comparison
        print(f"\n  per-L relative-to-own-L0 improvement:")
        print(f"{'L':>6}{'conv':>9}{'trans':>9}{'diff':>9}")
        for i, L in enumerate(Ln):
            rn = (pn[0] - pn[i]) / pn[0] if pn[0] else float("nan")
            rp = (pp[0] - pp[i]) / pp[0] if pp[0] else float("nan")
            print(f"{L:>6.0f}{rn:>9.4f}{rp:>9.4f}{rn-rp:>+9.4f}")

    print("\nPaste-ready for the draft: see paper/interspeech2027_draft.md 5.2")


if __name__ == "__main__":
    main()
