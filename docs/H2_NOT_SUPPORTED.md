# RQ2 / H2: the manner-class grouping is not the structure in the data

**Status: H2 not supported.** Every phone class benefits substantially from
lookahead, and the pre-registered split into "breaks first" and "survives short
lookahead" explains far less variance than the spread *within* those groups.
This is a clean negative result on a pre-registered hypothesis, and it should be
reported as one.

Evidence: `results/analysis_h2_sequences.json`, from `eval/analyse_h2_sequences.py`
(21 self-tests). Input: the **21 native-arm** hypothesis dumps of the 3-seed
sweep (7 lookaheads × 3 seeds; the directory holds 42 files, the produced arm is
excluded), 1 200 utterance-instances per lookahead after pooling seeds.
**42 255 reference phones per lookahead, 295 785 in the native arm, and zero
falling outside the seven pre-registered classes** — a coverage assertion across
all 42 files (591 570 tokens, both arms) also returns zero unclassified.

## What was measured

For every condition, `pred` was Levenshtein-aligned to `g2p` and each
substitution or deletion charged to the class of the **reference** phone, so the
denominator is "opportunities to get this class right". Insertions have no
reference phone and are counted globally rather than attributed to a neighbour.
Class-wise error rate was then regressed on log₂ lookahead.

## The result

Relative error-rate reduction from L=0 to L=640 ms, sorted:

| rank | class | group | relative gain | reference phones |
|---|---|---|---:|---:|
| 1 | nasal | **survives** | 0.716 | 4 020 |
| 2 | approximant | breaks-first | 0.700 | 4 575 |
| 3 | fricative | **survives** | 0.687 | 7 659 |
| 4 | vowel_mono | breaks-first | 0.673 | 13 008 |
| 5 | vowel_diph | breaks-first | 0.654 | 4 824 |
| 6 | stop | survives | 0.615 | 7 071 |
| 7 | affricate | survives | 0.525 | 1 098 |

Group means differ in the predicted **direction** — breaks-first 0.676 versus
survives 0.636 — but that is the weakest possible reading, and three harder
checks all fail:

- **Effect size 0.66.** The between-group difference (0.040) is smaller than the
  pooled SD across classes (0.060).
- **The survives group's internal spread is 0.190 — 4.7× the between-group
  difference.** Whatever separates nasals (0.716) from affricates (0.525) is a
  much stronger effect than anything the H2 grouping captures.
- **5 of 12 pairwise orderings are violated.** `nasal` beats all three
  breaks-first classes; `fricative` beats both vowel classes.

By L1, only **3 of 6** support even the directional reading, and the gain ratio
straddles 1.0 (0.92 Chinese to 1.23 Vietnamese), i.e. the sign of the effect
depends on which L1 you look at.

## Reading it honestly

The finding is not "lookahead does not help vowels". Every class improves by
52–72% relative, all resolvable against the 0.0044 noise floor. The finding is
that **manner class is the wrong axis**: it does not predict which sounds need
right context.

Two candidate explanations, neither tested here:

1. **Sequence-level PER may not expose the phenomenon H2 is about.** H2's
   rationale is formant trajectories over 80–250 ms — a property of the acoustic
   realisation. A discrete phone-label error rate can be blind to it: a vowel can
   be labelled correctly while its formant trajectory is wrong. The
   mel-cepstral-distortion form in `phoneme_analysis.py` is the test that would
   settle this, and it needs synthesis plus alignment.
2. **The grouping may be genuinely wrong.** The observed ordering — nasals and
   fricatives gaining most, affricates least — is not obviously about
   coarticulatory locality. Affricates gaining least is consistent with them
   being short and locally determined; nasals gaining most is not.

Either way the pre-registered hypothesis, as stated and as tested at the
sequence level, is **not supported**.

## Methodological note: the bug that nearly produced a fake answer

The first run of this analysis reported a plausible-looking H2 verdict that was
meaningless. `phoneme_analysis.CLASS_OF` is keyed on **ARPAbet** (`IY`, `TH`,
`NG`); the sweep emits **IPA** (`i`, `θ`, `ŋ`). Applying one to the other sent
**8 094 of 14 085 reference tokens (57%) to class "other"** — including *every
vowel* and /r/ — leaving `breaks_first` with `l` and `w` alone. The script still
printed per-class slopes, an H2 verdict and a by-L1 table.

Two features of this transcription then forced context-sensitive mapping, both
verified against the data before being encoded:

- **Diphthongs are decomposed**: `a`+`ɪ`, `a`+`ʊ`, `e`+`ɪ`, `o`+`ʊ`, `ɔ`+`ɪ`.
  `a` is followed by a glide 100% of the time, but `ɪ` follows a diphthong onset
  only 38.6% of the time and `ʊ` 78.5% — so a context-free map would relabel
  thousands of genuine monophthongs as diphthongs.
- **Affricates are decomposed** as `t`+`ʃ` and `d`+`ʒ`, so without sequence
  detection the pre-registered `affricate` class is empty and H2's survives group
  silently loses a member.

Guards added: a coverage assertion that no declared phone maps to `other`, ten
contextual class tests, and a reachability test that all seven pre-registered
classes can be produced. The boolean criterion was also replaced — direction
alone called a 6% group difference "supported", so it now additionally requires
effect size ≥ 1.0 and a minority of ordering violations.

*The general lesson for the paper's reproducibility section: an analysis that
silently buckets unmatched symbols into "other" will produce confident output on
the wrong alphabet. Assert coverage, do not infer it from the absence of an
error.*

## What to change in the paper

1. Report **H2 as not supported** at the sequence level, with the effect size,
   the within-group spread and the 5/12 ordering violations. Do not soften it to
   "partially supported" on the strength of the group-mean direction.
2. Keep the pre-registered prediction visible. The value of a pre-registered
   hypothesis is precisely that it can come out false.
3. State that all classes improve by 52–72% relative — the negative result is
   about the *grouping*, not about lookahead being useless.
4. Record that the audio-domain form of H2 remains untested and is the
   experiment that would distinguish "wrong grouping" from "PER is blind to the
   phenomenon".
5. Add the ARPAbet/IPA coverage failure to the methods-lessons list alongside the
   knee-estimator and padding findings.
