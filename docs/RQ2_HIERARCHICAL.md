# RQ2's uncertainty belongs at the speaker level, and the by-L1 split is confounded

**Status: strengthens the H2 verdict in
[`H2_NOT_SUPPORTED.md`](H2_NOT_SUPPORTED.md); does not change it.** Written
into the paper as §RQ2.

Evidence: `eval/rq2_hierarchical.py` (6 self-tests) →
`results/analysis_rq2_hierarchical.json`.

## The problem

The manner-class analysis pooled **42,255 reference-phone tokens** per
condition. Those tokens are not independent: they are nested in **six
speakers**, and the claim at issue is about accented speech, for which the
independent replicate is a talker, not a phone. Token-level intervals are
roughly an order of magnitude too narrow.

## The result

Bootstrapping the class rates over speakers (2000 draws):

| class | rel. gain | speaker-clustered 95% CI |
|---|---:|---|
| nasal | 0.716 | [0.668, 0.779] |
| approximant | 0.700 | [0.628, 0.774] |
| fricative | 0.687 | [0.608, 0.774] |
| vowel (mono) | 0.673 | [0.595, 0.767] |
| vowel (diph) | 0.654 | [0.577, 0.758] |
| stop | 0.615 | [0.550, 0.689] |
| affricate | 0.525 | [0.397, 0.633] |

Every individual class gain stays **large and clearly non-zero** — lookahead
unambiguously helps all seven. What fails is the *grouping*:

> **between-group difference (breaks-first − survives): +0.040,
> 95% CI [−0.010, +0.101] — includes zero.**

The estimator is not simply too blunt for six speakers: the *same* procedure
applied to RQ3 gives [−0.0211, −0.0069], which **excludes** zero. So the null
here is a property of the manner grouping, not of the method.

**Leave-one-speaker-out**, the non-asymptotic companion (six clusters do not
supply the asymptotics a cluster bootstrap assumes): dropping each talker in
turn gives **+0.0217 to +0.0592** against a pooled +0.0401. No single speaker
drives the estimate, and the entire range sits inside an interval containing
zero.

**H2's verdict is unchanged. Its support is now weaker, not stronger.**

## The by-L1 table cannot carry weight

In the speaker-disjoint test split, **each L1 is represented by exactly one
speaker**:

| Arabic | Chinese | Hindi | Korean | Spanish | Vietnamese |
|---|---|---|---|---|---|
| ABA | LXC | SVBI | HJK | EBVS | PNV |

L1 and talker are therefore **perfectly confounded**. The by-L1 breakdown
cannot separate a language effect from one person's idiosyncrasy, and every
row is equally readable as a single speaker's variation. The paper reports it
as a consistency check on the pooled result — the reversals matter because the
pooled direction survives *no* disaggregation — and explicitly not as evidence
about native language.

One clarification while checking this: the paper's "the predicted direction
appears in only 3 of 6 L1s" is correct but comes from a **two-part criterion**
(the gain ratio *and* the mean slope must both move as predicted). By gain
ratio alone it is 4 of 6 — Spanish, at 1.02, clears the gain test but not the
slope test. The sentence previously read as if it followed from the ratios
printed beside it.
