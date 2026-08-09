# Decision log

Short, dated entries for choices that would otherwise get silently reversed or
quietly forgotten. Each records what was decided, what it costs, and what would
reopen it.

---

## 2026-08-09 — No paid human listening test for now

**Decided:** do not run the paid rater study (~$450, ~30 raters × ~30 stimuli ×
3 rounds).

**What this does not block.** The infrastructure is finished and the inputs
exist. `eval/listening_test.py build` synthesises stimuli directly from the
hypothesis dumps via `synth/phones_to_audio.py` (Piper, one fixed voice), with
ceiling/floor attention checks, condition-blind filenames and a Bradley–Terry
scorer already written. There are **74 hyps files on disk** (42 from the
3-seed sweep, 32 from the dense sweep). Generating the stimuli costs nothing.
Only the raters cost money.

So this is purely a budget decision, not a capability gap. That is worth stating
because it is easy to misremember later as "the listening test wasn't ready".

**What it costs the paper.** `eval/listening_test.py`'s own docstring puts the
objection better than a summary would: *a model can lower PER by fixing phones
no listener notices, or raise it while sounding more native.* Concretely:

- The **saturation point (~240 ms)** is a statement about *phone error rate*,
  not about audibility. It must not be worded as "listeners stop noticing
  improvement beyond 240 ms".
- The **"40% of achievable reduction forgone at 40 ms"** figure is
  PER-relative, not quality-relative.
- **H3 survives unaffected** — it lives on the preference margin, which is an
  objective quantity requiring no listeners.
- For an *accent conversion* paper, accentedness is definitionally perceptual.
  Absence of any perceptual evaluation is the most likely reviewer objection at
  Interspeech, and it is a fair one.

**Consequent wording discipline.** The paper claims a characterisation of the
latency–PER trade-off plus a public harness. It must not claim anything about
perceived quality, naturalness or accentedness. Every headline number is a
phone-sequence measurement and should read as one.

**What would reopen this, cheapest first:**

1. **Build the stimuli anyway (free).** `listening_test.py build` on the
   existing hyps, then listen informally, or run 3–5 colleagues through the
   HTML page. Not publishable as a rater study, but it would catch the
   embarrassing case where the PER ordering and the audible ordering disagree —
   which is exactly the failure the docstring warns about.
2. **NISQA-MOS on the synthesised stimuli (free, automatic).** Gives a
   naturalness axis with no raters. Already referenced in §6 as the automatic
   proxy, with `metrics.degradation_control` as the guard against "less
   accented because more degraded".
3. **Fund the rater study** if a reviewer requires it, or if (1) shows the PER
   ordering does not match what is audible.

**Status of the artefacts:** keep `results/raw/hyps/` in the repository. It is
the input to all three options above and regenerating it costs a full sweep.
