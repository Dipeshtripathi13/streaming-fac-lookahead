"""Build a listening test from predicted phone sequences, and score it.

The gap this closes
-------------------
Every result so far is phone error rate or representation drift. Neither is
accentedness. A model can lower PER by fixing phones no listener notices, or
raise it while sounding more native. Until someone listens, the paper cannot
say anything perceptual — and §7 of the draft says so.

Design decisions, and why
-------------------------
**Pairwise, not MOS.** With the handful of raters a small project can afford,
absolute MOS is dominated by per-rater scale differences. Forced-choice
"which of these two sounds more like a native speaker of American English?" is
far more reliable at the same cost, and Bradley–Terry recovers a latent score
per condition with confidence intervals.

**Two ground-truth anchors in every set.**
  * *ceiling* — synthesised from `g2p`, the canonical phones: perfect conversion.
  * *floor* — synthesised from `ipa`, what the speaker actually produced:
    perfect transcription of the accented production.
Without them a rater's scale is uncalibrated and the numbers mean nothing. The
ceiling-vs-floor pair doubles as the **attention check**: a rater who cannot
pick the ceiling is not listening, and their data is dropped before analysis.

**One fixed synthetic voice for every stimulus.** This deliberately removes
speaker identity and prosody, so the test isolates whether the *phone output*
carries the accent correction. It is a narrower claim than "the system works",
and it also neutralises the usual confound: because every clip comes from the
same vocoder, raters cannot be responding to differences in artefact level.
Say this in the paper; do not let the claim drift.

**Pair sampling is balanced and blind.** Every condition appears equally often,
A/B order is randomised, and filenames are hashed so a rater inspecting the
page cannot infer the condition.

Usage
-----
    python3 eval/listening_test.py build --hyp-dir results/raw --n-utts 12
    python3 eval/listening_test.py score --votes votes.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import itertools
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "synth"))


# ==========================================================================
# Build
# ==========================================================================

def build(hyp_dir: str, out_dir: str, n_utts: int, seed: int,
          target: str, pairs_per_rater: int) -> None:
    from phones_to_audio import PiperPhonemeSynth, write_wav

    files = sorted(glob.glob(os.path.join(hyp_dir, f"hyps_L*_{target}_s*.json")))
    if not files:
        sys.exit(f"no hyps_L*_{target}_s*.json in {hyp_dir}\n"
                 f"Re-run the sweep with --save-hyps.")
    by_L: Dict[float, dict] = {}
    for f in files:
        d = json.load(open(f))
        by_L.setdefault(d["lookahead_ms"], d)      # first seed per condition
    Ls = sorted(by_L)
    print(f"{len(Ls)} lookahead conditions: {[int(L) for L in Ls]}")

    # utterances present in every condition, so each item is fully crossed
    key = lambda h: (h["speaker"], " ".join(h["g2p"]))        # noqa: E731
    common = set(map(key, by_L[Ls[0]]["hyps"]))
    for L in Ls[1:]:
        common &= set(map(key, by_L[L]["hyps"]))
    common = sorted(common)
    rng = random.Random(seed)
    rng.shuffle(common)
    chosen = common[:n_utts]
    print(f"{len(common)} utterances common to all conditions; using {len(chosen)}")
    if len(chosen) < n_utts:
        print(f"  WARNING: only {len(chosen)} available")

    synth = PiperPhonemeSynth()
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    stimuli: List[Dict[str, object]] = []

    for ui, k in enumerate(chosen):
        idx = {key(h): h for h in by_L[Ls[0]]["hyps"]}[k]
        conds: List[Tuple[str, List[str]]] = [
            ("ceiling", idx["g2p"]), ("floor", idx["ipa"])]
        for L in Ls:
            h = {key(x): x for x in by_L[L]["hyps"]}[k]
            conds.append((f"L{int(L)}", h["pred"]))
        for cond, phones in conds:
            if not phones:
                continue
            try:
                wav, meta = synth.synth(phones)
            except Exception as e:
                print(f"  utt{ui} {cond}: synth failed ({e})")
                continue
            # hashed filename: a curious rater must not be able to read the
            # condition off the URL
            fid = hashlib.sha1(f"{seed}:{ui}:{cond}".encode()).hexdigest()[:16]
            write_wav(os.path.join(audio_dir, f"{fid}.wav"), wav, synth.sr)
            stimuli.append({"utt": ui, "cond": cond, "file": f"{fid}.wav",
                            "speaker": idx["speaker"], "l1": idx["l1"],
                            "n_phones": len(phones),
                            "duration_s": meta["duration_s"]})

    # balanced pair sampling within each utterance
    conds = sorted({s["cond"] for s in stimuli})
    pairs: List[Dict[str, object]] = []
    for ui in sorted({s["utt"] for s in stimuli}):
        got = {s["cond"]: s for s in stimuli if s["utt"] == ui}
        for a, b in itertools.combinations([c for c in conds if c in got], 2):
            x, y = (a, b) if rng.random() < .5 else (b, a)
            pairs.append({"utt": ui, "A": got[x]["file"], "B": got[y]["file"],
                          "cond_A": x, "cond_B": y,
                          "is_attention_check": {x, y} == {"ceiling", "floor"}})
    rng.shuffle(pairs)

    manifest = {"target": target, "lookaheads_ms": Ls, "conditions": conds,
                "n_utts": len(chosen), "n_stimuli": len(stimuli),
                "n_pairs": len(pairs), "pairs_per_rater": pairs_per_rater,
                "seed": seed, "voice": "piper en_US-ryan-medium (fixed)",
                "stimuli": stimuli, "pairs": pairs}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    _write_page(out_dir, pairs_per_rater)

    n_check = sum(1 for p in pairs if p["is_attention_check"])
    print(f"\n{len(stimuli)} clips, {len(pairs)} pairs "
          f"({n_check} attention checks), {len(conds)} conditions")
    if n_check < 3:
        print(f"  WARNING: only {n_check} attention-check pairs "
              f"(one per utterance). Use --n-utts >= 6 so every rater can be "
              f"screened on at least 3.")
    print(f"  every rater is served ALL available checks "
          f"(up to max(3, {pairs_per_rater}//6)) before random pairs")
    print(f"  each rater sees {pairs_per_rater} pairs -> "
          f"{max(1, round(len(pairs)/pairs_per_rater))} raters for one full pass")
    print(f"  Recommended: 3 passes (~{3*max(1, round(len(pairs)/pairs_per_rater))} "
          f"raters) so each pair has 3 independent votes.")
    print(f"\nwrote {out_dir}/index.html  — open locally, or host the folder")
    print(f"Collected votes go to {out_dir}/votes.csv, then:")
    print(f"  python3 eval/listening_test.py score --votes {out_dir}/votes.csv")
    if synth.unmapped:
        print(f"\n  NOTE unmapped phone symbols: {dict(list(synth.unmapped.items())[:10])}")


def _write_page(out_dir: str, per_rater: int) -> None:
    html = """<!doctype html><meta charset=utf-8>
<title>Accentedness listening test</title>
<style>
 body{font:15px/1.5 system-ui;max-width:640px;margin:40px auto;padding:0 16px}
 button{font:inherit;padding:10px 18px;margin:6px 8px 6px 0;cursor:pointer}
 .pick{background:#0b5;color:#fff;border:0;border-radius:6px}
 .play{border:1px solid #999;border-radius:6px;background:#fff}
 #bar{height:6px;background:#eee;border-radius:3px;margin:18px 0}
 #fill{height:6px;background:#0b5;border-radius:3px;width:0}
 .q{font-weight:600;margin:22px 0 6px}
 small{color:#666}
</style>
<h2>Which one sounds more like a native speaker of American English?</h2>
<p><small>Both clips say the same sentence in the same synthetic voice. Judge only
the <b>pronunciation of the sounds</b> — not audio quality, speed or which you
prefer. Use headphones. You can replay each clip.</small></p>
<div id=bar><div id=fill></div></div>
<div id=app></div>
<script>
const PER_RATER = __PER__;
let M=null, order=[], i=0, votes=[], rid=Math.random().toString(36).slice(2,10);
fetch('manifest.json').then(r=>r.json()).then(m=>{
  M=m;
  // Every rater MUST see the attention checks. Sampling uniformly does not
  // guarantee that: with a handful of ceiling-vs-floor pairs among a hundred,
  // most raters would draw none and could never be screened. So take all the
  // checks, then fill the rest at random, then shuffle.
  const chk=[], oth=[];
  m.pairs.forEach((p,k)=>(p.is_attention_check?chk:oth).push(k));
  const sh=a=>{for(let k=a.length-1;k>0;k--){const j=Math.random()*(k+1)|0;[a[k],a[j]]=[a[j],a[k]];}return a;};
  order=sh(chk).slice(0,Math.min(chk.length,Math.max(3,PER_RATER/6|0)))
        .concat(sh(oth).slice(0,Math.max(0,PER_RATER-Math.min(chk.length,Math.max(3,PER_RATER/6|0)))));
  order=sh(order); render();
});
function play(f){const a=new Audio('audio/'+f);a.play();}
function pick(w){
  const p=M.pairs[order[i]];
  votes.push({rater:rid,pair:order[i],utt:p.utt,cond_A:p.cond_A,cond_B:p.cond_B,
              winner:w==='A'?p.cond_A:p.cond_B,
              is_attention_check:p.is_attention_check,t:Date.now()});
  i++; render();
}
function render(){
  const app=document.getElementById('app');
  document.getElementById('fill').style.width=(100*i/order.length)+'%';
  if(i>=order.length){
    const hdr='rater,pair,utt,cond_A,cond_B,winner,is_attention_check,t';
    const csv=hdr+'\\n'+votes.map(v=>[v.rater,v.pair,v.utt,v.cond_A,v.cond_B,
       v.winner,v.is_attention_check,v.t].join(',')).join('\\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
    app.innerHTML='<p><b>Done — thank you.</b></p><p><a download="votes_'+rid+
      '.csv" href="'+url+'">Download your responses</a> and send the file back.</p>';
    return;
  }
  const p=M.pairs[order[i]];
  app.innerHTML='<div class=q>Pair '+(i+1)+' of '+order.length+'</div>'+
   '<button class=play onclick="play(\\''+p.A+'\\')">▶ Play A</button>'+
   '<button class=play onclick="play(\\''+p.B+'\\')">▶ Play B</button><br>'+
   '<button class=pick onclick="pick(\\'A\\')">A sounds more native</button>'+
   '<button class=pick onclick="pick(\\'B\\')">B sounds more native</button>';
}
</script>"""
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html.replace("__PER__", str(per_rater)))


# ==========================================================================
# Score
# ==========================================================================

def bradley_terry(wins: Dict[Tuple[str, str], int], conds: Sequence[str],
                  iters: int = 3000) -> Dict[str, float]:
    """Latent ability per condition; anchored so the mean log-score is 0."""
    idx = {c: i for i, c in enumerate(conds)}
    p = np.ones(len(conds))
    W = np.zeros((len(conds), len(conds)))
    for (a, b), n in wins.items():
        W[idx[a], idx[b]] += n
    for _ in range(iters):
        for i in range(len(conds)):
            num = W[i].sum()
            den = sum((W[i, j] + W[j, i]) / (p[i] + p[j])
                      for j in range(len(conds)) if j != i)
            if den > 0 and num > 0:
                p[i] = num / den
        p /= np.exp(np.mean(np.log(p)))
    return {c: float(np.log(p[idx[c]])) for c in conds}


def score(votes_csv: str, manifest: Optional[str], boot: int) -> None:
    rows = list(csv.DictReader(open(votes_csv)))
    if not rows:
        sys.exit("no votes")
    print(f"{len(rows)} votes from {len({r['rater'] for r in rows})} raters")

    # attention checks first: drop raters who fail
    chk = defaultdict(lambda: [0, 0])
    for r in rows:
        if str(r["is_attention_check"]).lower() == "true":
            chk[r["rater"]][1] += 1
            if r["winner"] == "ceiling":
                chk[r["rater"]][0] += 1
    bad = {k for k, (ok, n) in chk.items() if n >= 2 and ok / n < 0.6}
    unscreened = {r["rater"] for r in rows} - set(chk)
    if unscreened:
        print(f"\n  WARNING: {len(unscreened)} rater(s) saw no attention check "
              f"and cannot be screened: {sorted(unscreened)}")
    if chk:
        print(f"\nattention checks (ceiling should beat floor):")
        for k, (ok, n) in sorted(chk.items()):
            print(f"  {k}: {ok}/{n}" + ("   DROPPED" if k in bad else ""))
    if bad:
        print(f"  dropping {len(bad)} rater(s); a listener who cannot separate "
              f"the ground-truth anchors is not doing the task")
    rows = [r for r in rows if r["rater"] not in bad]
    if not rows:
        sys.exit("no votes survive the attention check")

    conds = sorted({r["cond_A"] for r in rows} | {r["cond_B"] for r in rows})
    def tally(rs):
        w = defaultdict(int)
        for r in rs:
            lose = r["cond_B"] if r["winner"] == r["cond_A"] else r["cond_A"]
            w[(r["winner"], lose)] += 1
        return w

    fit = bradley_terry(tally(rows), conds)
    rng = np.random.default_rng(0)
    bs = defaultdict(list)
    for _ in range(boot):
        samp = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        try:
            f = bradley_terry(tally(samp), conds, iters=400)
            for c in conds:
                bs[c].append(f[c])
        except Exception:
            pass

    print(f"\n{'condition':<12}{'BT score':>10}{'95% CI':>22}{'win rate':>10}")
    for c in sorted(conds, key=lambda c: -fit[c]):
        n_w = sum(1 for r in rows if r["winner"] == c)
        n_t = sum(1 for r in rows if c in (r["cond_A"], r["cond_B"]))
        lo, hi = (np.percentile(bs[c], [2.5, 97.5]) if bs[c] else (np.nan, np.nan))
        print(f"{c:<12}{fit[c]:>10.3f}   [{lo:>6.3f}, {hi:>6.3f}]"
              f"{(n_w/n_t if n_t else float('nan')):>10.3f}")

    print("\n  ceiling and floor are ground truth, not system outputs: they")
    print("  bound the scale. A condition scoring near ceiling means its phone")
    print("  output is as native-sounding as the canonical transcription.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--hyp-dir", default="results/raw")
    b.add_argument("--out-dir", default="results/listening_test")
    b.add_argument("--n-utts", type=int, default=12)
    b.add_argument("--target", default="native")
    b.add_argument("--pairs-per-rater", type=int, default=30)
    b.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--votes", required=True)
    s.add_argument("--manifest", default=None)
    s.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.hyp_dir, a.out_dir, a.n_utts, a.seed, a.target, a.pairs_per_rater)
    else:
        score(a.votes, a.manifest, a.boot)


if __name__ == "__main__":
    main()
