"""Train the causal accent translator and sweep lookahead. Real training.

One command trains all 14 runs (7 lookaheads x 2 targets) and writes the RQ1
and RQ3 curves. Runs on Colab T4/L4, a rented 4090, or (slowly) an M4 CPU.

    python3 train/train_translator.py --steps 8000 --out results/raw/translator_sweep
    python3 train/train_translator.py --smoke          # 60 steps, proves it runs
    python3 train/train_translator.py --lookaheads 0 80 640 --targets native

What is being trained
---------------------
A causal conversion stack + CTC head over a frozen, lookahead-masked WavLM.
Target is the CANONICAL phone sequence (`g2p`) for the accent-conversion arm,
and the PRODUCED phone sequence (`ipa`) for the transcription control. See
`src/sfac/translator.py` for why this is the right task for this dataset.

Read before trusting any output
-------------------------------
* The encoder is frozen by default. That makes the sweep a question about how
  much context the *converter* needs given a fixed representation. Unfreezing
  lets the encoder re-learn and confounds the comparison unless every run gets
  identical steps -- allowed, but pass --unfreeze deliberately and say so.
* WavLM is NOT causal out of the box, in TWO places, and masking attention
  fixes neither:
    - `pos_conv_embed`: depthwise conv, kernel 128, symmetric padding
      -> 1.28 s of future in every frame;
    - the feature-encoder GroupNorm (`feat_extract_norm="group"` on all base
      checkpoints): normalises each channel over the WHOLE utterance
      -> an unbounded dependency on all future frames.
  We patch both. Measured on wavlm-base-plus, relative L2 under a truncation
  test: mask only 1.14e-2, +pos-conv 6.00e-3, both 6.14e-6 (causal).
  `--verify-causality` (on by default) re-runs that proof and ABORTS training
  if it fails -- otherwise every L label in the sweep would be a lie.
* Speakers are split disjointly across train/val/test. A random utterance
  split leaks speaker identity and inflates every score.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from sfac.translator import (  # noqa: E402
    TranslatorConfig, Target, PhoneVocab, build_translator, greedy_ctc_decode,
    tokenize_phones, per, sweep_configs, assert_only_L_varies,
)
from sfac.causal import StreamGeometry  # noqa: E402
from phoneme_analysis import find_knee   # noqa: E402


# ==========================================================================
# Data
# ==========================================================================

def build_splits(ds, val_speakers: int = 3, test_speakers: int = 3, seed: int = 0):
    """Speaker-disjoint splits, stratified so every L1 appears in every split.

    L2-ARCTIC has 24 speakers over 6 L1s (4 each). A random utterance split
    would put the same speaker on both sides and turn this into a
    speaker-memorisation task. Holding out whole speakers, balanced by L1, is
    the only split that measures what we claim to measure.
    """
    by_l1: Dict[str, List[str]] = defaultdict(list)
    spk_l1: Dict[str, str] = {}
    for i in range(len(ds)):
        r = ds[i]
        spk, l1 = r.get("speaker_code", "?"), r.get("speaker_native_language", "?")
        if spk not in spk_l1:
            spk_l1[spk] = l1
            by_l1[l1].append(spk)
    rng = random.Random(seed)
    val, test = set(), set()
    n_l1 = max(1, len(by_l1))
    for l1, spks in sorted(by_l1.items()):
        s = sorted(spks)
        rng.shuffle(s)
        nv = max(1, val_speakers // n_l1) if len(s) > 2 else 0
        nt = max(1, test_speakers // n_l1) if len(s) > 3 else 0
        val.update(s[:nv])
        test.update(s[nv:nv + nt])
    return val, test, spk_l1


def load_corpus(n_max: Optional[int] = None, seed: int = 0, cache: Optional[str] = None):
    from bench_content_degradation import decode_audio, open_l2arctic

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))):
        raise RuntimeError(
            "KoelLabs/L2Arctic is gated (CC-BY-NC-4.0). Accept the terms at "
            "https://huggingface.co/datasets/KoelLabs/L2Arctic then run "
            "`hf auth login`, or export HF_TOKEN=hf_...")

    ds, _mode = open_l2arctic("scripted")
    val_spk, test_spk, spk_l1 = build_splits(ds, seed=seed)

    items: List[Dict[str, object]] = []
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    for i in idx:
        r = ds[int(i)]
        g2p = tokenize_phones(r.get("g2p", "") or "")
        ipa = tokenize_phones(r.get("ipa", "") or "")
        if len(g2p) < 3 or len(ipa) < 3:
            continue
        wav, sr = decode_audio(r["audio"])
        if sr != 16_000:
            from scipy.signal import resample_poly
            g = math.gcd(int(sr), 16_000)
            wav = resample_poly(wav, 16_000 // g, int(sr) // g).astype(np.float32)
        dur = len(wav) / 16_000
        if not (0.8 <= dur <= 10.0):
            continue
        spk = r.get("speaker_code", "?")
        items.append({
            "wav": wav.astype(np.float32), "g2p": g2p, "ipa": ipa,
            "speaker": spk, "l1": spk_l1.get(spk, "?"),
            "split": "val" if spk in val_spk else ("test" if spk in test_spk else "train"),
        })
        if n_max and len(items) >= n_max:
            break

    vocab = PhoneVocab.build([it["g2p"] for it in items] + [it["ipa"] for it in items])
    stats = {
        "n_items": len(items),
        "n_train": sum(1 for i in items if i["split"] == "train"),
        "n_val": sum(1 for i in items if i["split"] == "val"),
        "n_test": sum(1 for i in items if i["split"] == "test"),
        "val_speakers": sorted(val_spk), "test_speakers": sorted(test_spk),
        "vocab_size": len(vocab),
        "per_l1": dict(sorted(
            {l1: sum(1 for i in items if i["l1"] == l1)
             for l1 in {i["l1"] for i in items}}.items())),
        "mean_dur_s": round(float(np.mean([len(i["wav"]) / 16_000 for i in items])), 2),
        # How different are the two targets? If g2p == ipa everywhere, the
        # accent-conversion arm and the transcription control are the same task
        # and RQ3 is unanswerable. Check, do not assume.
        "mean_per_g2p_vs_ipa": round(float(np.mean(
            [per(i["g2p"], i["ipa"]) for i in items[:500]])), 4),
    }
    return items, vocab, stats


# ==========================================================================
# Feature extraction (frozen, masked encoder)
# ==========================================================================

class Frontend:
    """Lookahead-masked WavLM with a causal positional conv."""

    def __init__(self, cfg: TranslatorConfig, device: str):
        from bench_content_degradation import load_encoder, MaskInjector
        self.MaskInjector = MaskInjector
        self.model, self.encinfo = load_encoder(cfg.encoder_name, device,
                                               causal_pos_conv=True,
                                               causal_group_norm=True)
        self.cfg, self.device = cfg, device

    def __call__(self, wavs: List[np.ndarray]):
        """-> (feats (B,T,D), key_padding_mask (B,T) True=pad)"""
        import torch
        lens = [len(w) for w in wavs]
        mx = max(lens)
        x = torch.zeros(len(wavs), mx, dtype=torch.float32)
        for i, w in enumerate(wavs):
            x[i, :len(w)] = torch.from_numpy(w)
        x = x.to(self.device)
        # Padding mask: without it the encoder attends into the zero-pad tail,
        # which contaminates the last frames of every short utterance in a
        # batch. Our lookahead mask is added to the position bias and leaves
        # the model's own key_padding_mask path intact, so both apply.
        am = torch.zeros(len(wavs), mx, dtype=torch.long)
        for i, n in enumerate(lens):
            am[i, :n] = 1
        am = am.to(self.device)
        geom = self.cfg.geometry
        with torch.no_grad(), self.MaskInjector(self.model, geom):
            out = self.model(x, attention_mask=am, output_hidden_states=True)
        h = out.hidden_states[self.cfg.encoder_layer]
        T = h.shape[1]
        # WavLM frame count for a given sample count
        fl = self.model._get_feat_extract_output_lengths(
            torch.tensor(lens, device=self.device)).cpu().numpy()
        kpm = torch.ones(len(wavs), T, dtype=torch.bool, device=self.device)
        for i, n in enumerate(fl):
            kpm[i, :int(n)] = False
        return h, kpm, [int(n) for n in fl]


# ==========================================================================
# Frozen-encoder feature cache
# ==========================================================================

def cache_features(fe: "Frontend", items, device: str, batch: int = 8,
                   log_every: int = 400):
    """Run the frozen encoder ONCE per utterance and keep the result.

    The encoder is frozen and the lookahead mask is fixed within a condition,
    so its output for a given utterance never changes during training. The
    first sweep recomputed it every time an utterance came round -- with 1800
    training utterances and 1200 steps at batch 8, that is ~5.3 redundant
    forward passes per utterance, and it dominated the 0.50 s/step budget.

    Caching turns a ~10.6 min condition into ~2.5 min, which is what makes
    3 seeds x 14 conditions affordable. It changes nothing scientifically:
    identical inputs, identical mask, identical frozen weights.

    Stored on CPU in fp16 (~1 GB for 3600 utterances of ~3.7 s), moved to the
    device per batch. fp16 is safe here because these are inputs to a trained
    head, not accumulations -- but we cast back to fp32 on use so the head
    itself trains in full precision.
    """
    import torch
    out = []
    t0 = time.time()
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        h, kpm, fl = fe([c["wav"] for c in chunk])
        h = h.half().cpu()
        kpm = kpm.cpu()
        for j, c in enumerate(chunk):
            out.append({"feat": h[j, :fl[j]].clone(), "n": fl[j], "item": c})
        if log_every and (i // batch) % (log_every // batch) == 0 and i:
            print(f"      cached {i}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    return out


def collate_cached(entries, device):
    """Pad a list of cached (T_i, D) features into (B, T, D) + key-padding mask."""
    import torch
    n = [e["n"] for e in entries]
    mx = max(n)
    D = entries[0]["feat"].shape[-1]
    x = torch.zeros(len(entries), mx, D, dtype=torch.float32)
    kpm = torch.ones(len(entries), mx, dtype=torch.bool)
    for i, e in enumerate(entries):
        x[i, :e["n"]] = e["feat"].float()
        kpm[i, :e["n"]] = False
    return x.to(device), kpm.to(device), n


# ==========================================================================
# Train one condition
# ==========================================================================

def train_one(cfg: TranslatorConfig, items, vocab: PhoneVocab, device: str,
              steps: int, ckpt_dir: Optional[str], log_every: int = 200,
              eval_every: int = 1000, unfreeze: bool = False,
              cached=None):
    """Train one (lookahead, target, seed). Returns (result, feature_cache).

    `cached` lets the caller reuse the frozen-encoder features across seeds of
    the SAME condition -- exact, because the encoder does not depend on the
    seed.
    """
    import torch
    import torch.nn as nn

    key = "g2p" if cfg.target is Target.NATIVE else "ipa"
    train = [i for i in items if i["split"] == "train"]
    val = [i for i in items if i["split"] == "val"]
    test = [i for i in items if i["split"] == "test"]

    if cached is None:
        fe = Frontend(cfg, device)
        t_cache = time.time()
        print(f"    caching frozen-encoder features "
              f"({len(train)}+{len(val[:200])}+{len(test[:400])} utts)...", flush=True)
        cached = (cache_features(fe, train, device),
                  cache_features(fe, val[:200], device, log_every=0),
                  cache_features(fe, test[:400], device, log_every=0))
        del fe
        import gc as _gc; _gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"    cached in {time.time()-t_cache:.0f}s "
              f"(reused for every seed of this condition)", flush=True)
    c_train, c_val, c_test = cached

    model, info = build_translator(cfg, vocab)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=steps, pct_start=min(0.3, cfg.warmup_steps / steps))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    # identical data order across every condition -- part of the invariant
    rng = random.Random(cfg.seed)
    order = list(range(len(c_train)))
    rng.shuffle(order)

    def batches(pool, bs, shuffled_order=None, loop=True):
        idxs = shuffled_order if shuffled_order is not None else list(range(len(pool)))
        p = 0
        while True:
            if p + bs > len(idxs):
                if not loop:
                    if p < len(idxs):
                        yield [pool[i] for i in idxs[p:]]
                    return
                p = 0
            yield [pool[i] for i in idxs[p:p + bs]]
            p += bs

    gen = batches(c_train, cfg.batch_size, order)

    def evaluate(pool, max_n=400):
        model.eval()
        refs, hyps, items_seen = [], [], []
        with torch.no_grad():
            for b in batches(pool[:max_n], cfg.batch_size, loop=False):
                feats, kpm, fl = collate_cached(b, device)
                logits = model(feats, kpm)
                dec = greedy_ctc_decode(logits, vocab)
                for e, d in zip(b, dec):
                    refs.append(e["item"][key])
                    hyps.append(d)
                    items_seen.append(e["item"])
        model.train()
        pers = [per(r, h) for r, h in zip(refs, hyps)]
        # the cross-score: how well does this model predict the OTHER target?
        other = "ipa" if key == "g2p" else "g2p"
        cross = [per(x[other], h) for x, h in zip(items_seen, hyps)]
        return {"per": float(np.mean(pers)),
                "per_cross": float(np.mean(cross)),
                "n": len(pers)}

    history = []
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        b = next(gen)
        feats, kpm, fl = collate_cached(b, device)
        logits = model(feats, kpm)                    # (B,T,V)
        logp = logits.log_softmax(-1).transpose(0, 1)  # (T,B,V)
        tgt = [torch.tensor(vocab.encode(e["item"][key]), dtype=torch.long) for e in b]
        tgt_len = torch.tensor([len(t) for t in tgt], dtype=torch.long)
        in_len = torch.tensor(fl, dtype=torch.long)
        # CTC requires input_length >= target_length; drop the rare violation
        keep = (in_len >= tgt_len)
        if keep.sum() == 0:
            continue
        loss = ctc(logp[:, keep], torch.cat([t for t, k in zip(tgt, keep) if k]).to(device),
                   in_len[keep], tgt_len[keep])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if step % log_every == 0:
            print(f"    {cfg.tag()} step {step}/{steps} loss {loss.item():.4f} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % eval_every == 0 or step == steps:
            e = evaluate(c_val, max_n=200)
            history.append({"step": step, "loss": float(loss.item()), **e})
            print(f"    {cfg.tag()} step {step}  val PER {e['per']:.4f}  "
                  f"cross-PER {e['per_cross']:.4f}", flush=True)

    final = evaluate(c_test, max_n=400)
    result = {
        "tag": cfg.tag(),
        "lookahead_ms": cfg.lookahead_ms,
        "target": cfg.target.value,
        "chunk_ms": cfg.chunk_ms,
        "t_algorithmic_ms": cfg.geometry.algorithmic_ms,
        "t_algorithmic_honest_ms": cfg.geometry.algorithmic_ms_honest,
        "test_per": final["per"],
        "test_per_cross": final["per_cross"],
        "n_test": final["n"],
        "params": info["params"],
        "steps": steps,
        "wall_s": round(time.time() - t0, 1),
        "history": history,
    }
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg.fingerprint(),
                    "lookahead_ms": cfg.lookahead_ms, "vocab": vocab.to_dict(),
                    "result": {k: v for k, v in result.items() if k != "history"}},
                   os.path.join(ckpt_dir, f"{cfg.tag()}.pt"))
    del model
    import gc, torch as _t
    gc.collect()
    if device == "cuda":
        _t.cuda.empty_cache()
    return result, cached


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookaheads", type=float, nargs="+",
                    default=[0, 20, 40, 80, 160, 320, 640])
    ap.add_argument("--targets", nargs="+", default=["native", "produced"],
                    choices=["native", "produced"])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337],
                    help="one training run per seed per condition. Multiple "
                         "seeds are what turn a per-condition point into an "
                         "interval; the single-seed sweep could not resolve "
                         "differences below ~0.01 PER.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk-ms", type=float, default=40.0)
    ap.add_argument("--lookback-ms", type=float, default=2000.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--unfreeze", action="store_true")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--out", default="results/raw/translator_sweep")
    ap.add_argument("--smoke", action="store_true",
                    help="60 steps, 3 lookaheads, 120 utts -- proves the pipeline runs")
    ap.add_argument("--verify-causality", action="store_true", default=True)
    a = ap.parse_args()

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu   : {torch.cuda.get_device_name(0)}")

    if a.smoke:
        a.steps, a.lookaheads, a.n_max = 60, [0, 80, 640], 120
        a.targets = ["native"]

    # ---- causality proof before anything expensive ----
    if a.verify_causality:
        from bench_content_degradation import causality_selftest
        st = causality_selftest(device="cpu")
        print(json.dumps(st, indent=2))
        if not st["with_fix"]["causal"]:
            sys.exit("ABORT: the causal pos_conv patch did not produce a causal "
                     "encoder. Every lookahead label would be wrong. Fix first.")

    # ---- data ----
    t0 = time.time()
    items, vocab, stats = load_corpus(n_max=a.n_max)
    print(json.dumps(stats, indent=2))
    print(f"corpus loaded in {time.time()-t0:.0f}s")
    if stats["mean_per_g2p_vs_ipa"] < 0.02:
        print("\nWARNING: g2p and ipa are nearly identical (mean PER "
              f"{stats['mean_per_g2p_vs_ipa']:.4f}). The conversion arm and the "
              "transcription control would be the same task and RQ3 would be "
              "unanswerable. Inspect the columns before trusting the sweep.")

    # ---- sweep ----
    cfgs = sweep_configs(
        lookaheads=tuple(a.lookaheads),
        targets=tuple(Target(t) for t in a.targets),
        chunk_ms=a.chunk_ms, lookback_ms=a.lookback_ms,
        batch_size=a.batch_size, train_steps=a.steps,
        freeze_encoder=not a.unfreeze)
    for t in a.targets:
        assert_only_L_varies([c for c in cfgs if c.target.value == t])
    print(f"\n{len(cfgs)} conditions x {a.steps} steps\n")

    results = []
    total = len(cfgs) * len(a.seeds)
    k = 0
    for i, cfg in enumerate(cfgs, 1):
        # Cache the frozen-encoder features ONCE per condition and reuse them
        # for every seed. The encoder does not depend on the seed, so this is
        # exact -- and it is what makes 3 seeds cost ~1.2x a single seed rather
        # than 3x.
        shared = None
        for sd in a.seeds:
            k += 1
            c = TranslatorConfig(**{**cfg.fingerprint(), "lookahead_ms":
                                    cfg.lookahead_ms, "target": cfg.target,
                                    "seed": sd})
            print(f"[{k}/{total}] {c.tag()} seed={sd}  {c.geometry.describe()}")
            r, shared = train_one(c, items, vocab, device, a.steps,
                                  a.ckpt_dir, unfreeze=a.unfreeze,
                                  cached=shared)
            r["seed"] = sd
            results.append(r)

    # ---- curves ----
    summary: Dict[str, object] = {"corpus": stats, "device": device,
                                  "steps": a.steps, "chunk_ms": a.chunk_ms,
                                  "results": results}
    for t in a.targets:
        arm_all = [r for r in results if r["target"] == t]
        Ls = sorted({r["lookahead_ms"] for r in arm_all})
        # aggregate seeds: mean, sd and range per condition
        agg = []
        for L in Ls:
            v = [r["test_per"] for r in arm_all if r["lookahead_ms"] == L]
            c = [r["test_per_cross"] for r in arm_all if r["lookahead_ms"] == L]
            agg.append({"lookahead_ms": L, "n_seeds": len(v),
                        "per_mean": float(np.mean(v)),
                        "per_sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                        "per_min": float(np.min(v)), "per_max": float(np.max(v)),
                        "per_all": v,
                        "cross_mean": float(np.mean(c)),
                        "preference_mean": float(np.mean(c) - np.mean(v))})
        summary[f"per_condition_{t}"] = agg
        pers = [x["per_mean"] for x in agg]
        if len(Ls) >= 3:
            k = find_knee(Ls, pers)
            sds = [x["per_sd"] for x in agg]
            summary[f"curve_{t}"] = {
                "lookaheads_ms": Ls, "test_per": pers,
                "test_per_sd": sds,
                "n_seeds": agg[0]["n_seeds"],
                "pooled_sd": float(np.sqrt(np.mean(np.square(sds)))),
                "knee": k,
                "relative_gain_0_to_max":
                    (pers[0] - pers[-1]) / pers[0] if pers[0] else None}

    if "native" in a.targets and "produced" in a.targets:
        cn = summary.get("curve_native", {})
        cp = summary.get("curve_produced", {})
        gn, gp = cn.get("relative_gain_0_to_max"), cp.get("relative_gain_0_to_max")
        summary["H3"] = {
            "gain_conversion": gn, "gain_transcription": gp,
            "supported": bool(gn is not None and gp is not None and gn > gp),
            "reading": ("H3 predicts conversion benefits MORE from lookahead than "
                        "transcription does. Supported iff gain_conversion > "
                        "gain_transcription."),
        }

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    import csv
    with open(a.out + ".csv", "w", newline="") as f:
        cols = [c for c in results[0] if c != "history"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r[c] for c in cols})

    print("\n" + "=" * 76)
    print(f"{'target':<12}{'L (ms)':>8}{'seeds':>7}{'PER mean':>10}{'sd':>8}"
          f"{'range':>16}{'prefers g2p':>13}")
    for t in a.targets:
        for x in summary.get(f"per_condition_{t}", []):
            print(f"{t:<12}{x['lookahead_ms']:>8.0f}{x['n_seeds']:>7}"
                  f"{x['per_mean']:>10.4f}{x['per_sd']:>8.4f}"
                  f"{x['per_min']:>8.4f}-{x['per_max']:<7.4f}"
                  f"{x['preference_mean']:>+13.4f}")
    for t in a.targets:
        c = summary.get(f"curve_{t}")
        if c and c.get("pooled_sd") is not None:
            print(f"  {t}: pooled seed sd = {c['pooled_sd']:.4f} PER "
                  f"-> differences below ~{2*c['pooled_sd']:.4f} are not resolved")
    for t in a.targets:
        c = summary.get(f"curve_{t}")
        if c:
            print(f"\n{t}: knee at {c['knee']['knee_ms']:.0f} ms, "
                  f"relative gain 0->max {c['relative_gain_0_to_max']:.3f}")
    if "H3" in summary:
        print(f"\nH3 {'SUPPORTED' if summary['H3']['supported'] else 'NOT supported'}: "
              f"conversion gain {summary['H3']['gain_conversion']:.3f} vs "
              f"transcription gain {summary['H3']['gain_transcription']:.3f}")
    print(f"\nwrote {a.out}.csv and {a.out}_summary.json")


if __name__ == "__main__":
    main()
