"""Measure the future context the whole model (encoder + probe) really gets.

The truncation proof in bench/ runs at L = 0, where nothing can compound. It
shows the encoder is causal; it does not show that a lookahead LABEL equals the
future context behind a result. Applying the same lookahead mask in every layer
makes right context grow with depth, so "L = 40 ms" over 9 encoder layers and 4
probe blocks is several hundred ms of future, not 40.

The test: replace all audio after a cut point and find the earliest output
frame that moves. Under lookahead_scope="end_to_end" that frame must be exactly
the one the mask predicts, for every budget. Under "per_layer" it is far
earlier.

    python eval/effective_lookahead.py                 # both scopes, table
    python eval/effective_lookahead.py --encoder-name facebook/wav2vec2-base \
        --encoder-layer 2
"""
import argparse
import json
import os
import sys
from typing import Dict, Sequence

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "bench"))

from sfac.translator import TranslatorConfig, PhoneVocab, build_translator  # noqa: E402

FRAME = 320           # samples per 20 ms frame at 16 kHz
TOL = 1e-4            # same threshold as the truncation proof; float32 noise is ~1e-5


def _first_moved(cfg: TranslatorConfig, encoder, wav_a, wav_b):
    """Earliest output frame that differs between two inputs, and the largest
    relative change among the frames before it."""
    import torch
    from bench_content_degradation import MaskInjector

    vocab = PhoneVocab([f"p{i}" for i in range(36)])
    probe, _ = build_translator(cfg, vocab)
    probe.eval()

    def run(w):
        with torch.no_grad(), MaskInjector(encoder, cfg.encoder_geometry):
            h = encoder(torch.from_numpy(w)[None],
                        output_hidden_states=True).hidden_states[cfg.encoder_layer]
            return probe(h)[0]

    a, b = run(wav_a), run(wav_b)
    d = ((a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-9)).numpy()
    moved = np.nonzero(d > TOL)[0]
    first = int(moved[0]) if len(moved) else len(d)
    return first, float(d[:first].max()) if first else 0.0


def measure(encoder_name: str, encoder_layer: int, lookaheads: Sequence[float],
            scope: str, chunk_ms: float = 40.0, lookback_ms: float = 2000.0,
            seconds: float = 8.0, cut_s: float = 6.0):
    """Per budget: labelled lookahead, the future context the mask predicts for
    a truly end-to-end lookahead, and the future context measured."""
    from bench_content_degradation import load_encoder

    rng = np.random.default_rng(0)
    wav = (rng.standard_normal(int(16_000 * seconds)) * 0.1).astype(np.float32)
    encoder, _ = load_encoder(encoder_name, "cpu", causal_pos_conv=True,
                              causal_group_norm=True)
    rows = []
    for L in lookaheads:
        cfg = TranslatorConfig(lookahead_ms=L, chunk_ms=chunk_ms,
                               lookback_ms=lookback_ms, encoder_name=encoder_name,
                               encoder_layer=encoder_layer, lookahead_scope=scope)
        g = cfg.geometry
        C, NL = g.chunk_frames, g.lookahead_frames
        # A frame's future context depends on where it sits in its chunk, so
        # one cut point sees only one alignment. Shift the cut across a whole
        # chunk and keep the furthest reach.
        exp_ms = got_ms = 0.0
        exact, leak = True, 0.0
        for shift in range(C):
            cut = int(16_000 * cut_s) + shift * FRAME
            wav2 = wav.copy()
            wav2[cut:] = rng.standard_normal(len(wav) - cut).astype(np.float32)
            # The front end's 25 ms window means the frame before the cut
            # already overlaps the changed samples.
            changed = cut // FRAME - 1
            # earliest frame whose window (end of its chunk + NL) reaches it
            expected = next(i for i in range(changed + 1)
                            if (i // C + 1) * C - 1 + NL >= changed)
            first, lk = _first_moved(cfg, encoder, wav, wav2)
            exp_ms = max(exp_ms, (changed - expected) * g.frame_ms)
            got_ms = max(got_ms, (changed - first) * g.frame_ms)
            exact &= first == expected
            leak = max(leak, lk)
        rows.append({
            "lookahead_ms": L,
            "expected_future_ms": exp_ms,
            "measured_future_ms": got_ms,
            "exact": exact,
            "max_rel_change_before": leak,
        })
    return rows


def reach_selftest(encoder_name: str, encoder_layer: int,
                   lookaheads: Sequence[float], chunk_ms: float = 40.0,
                   lookback_ms: float = 2000.0) -> Dict[str, object]:
    """Gate for a sweep: every budget's measured future context must equal the
    one its label promises."""
    rows = measure(encoder_name, encoder_layer, lookaheads, "end_to_end",
                   chunk_ms=chunk_ms, lookback_ms=lookback_ms)
    return {"scope": "end_to_end", "encoder": encoder_name,
            "layer": encoder_layer, "tolerance": TOL, "rows": rows,
            "ok": all(r["exact"] for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-name", default="microsoft/wavlm-base-plus")
    ap.add_argument("--encoder-layer", type=int, default=9)
    ap.add_argument("--lookaheads", type=float, nargs="+",
                    default=[0, 20, 40, 60, 80, 160, 320, 640])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    report = {"encoder": a.encoder_name, "layer": a.encoder_layer, "tolerance": TOL}
    for scope in ("end_to_end", "per_layer"):
        rows = measure(a.encoder_name, a.encoder_layer, a.lookaheads, scope)
        report[scope] = rows
        print(f"\n{scope}  ({a.encoder_name}, layer {a.encoder_layer}, chunk 40 ms)")
        print(f"{'L (ms)':>7} {'label implies':>14} {'measured':>9}  exact")
        for r in rows:
            print(f"{r['lookahead_ms']:>7g} {r['expected_future_ms']:>14g} "
                  f"{r['measured_future_ms']:>9g}  {r['exact']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
