# TASLP submission (in progress)

Target: IEEE/ACM Transactions on Audio, Speech and Language Processing.
No deadline. Free to publish on the traditional route: IEEE page charges are
explicitly voluntary and "not a prerequisite for publication"; only the
mandatory *overlength* charge applies, above 10 pages for a full paper. Staying
within 10 pages therefore costs nothing, and requires no travel or conference
registration -- which ICASSP would have, at USD 900-1240 plus mandatory
in-person presentation.

Grown from `../icassp/icassp2027.tex`, the 5-page conference compression.

## Status against the second review

DONE (this pass):
  * consistent phone-probe framing through Method, contributions and RQ
    headings, not only in Limitations
  * RQ3 interaction hardened: speaker-clustered bootstrap and phone-matched
    control (eval/rq3_interaction.py)
  * preregistration made verifiable by citing the commit that fixed
    H2_PREDICTION seven days before the results it is tested on
  * removed the informal-listening anecdote, which contradicted the ethics
    statement's "no human subjects"
  * frame *rate* 20 ms -> frame *hop* h = 20 ms (dimensional error)
  * compute claim now reports both platforms (1.9% M4, 7.6% ARM64) rather than
    the M4 range alone, and "two architectures" -> "two hardware/software
    platforms" (an M4 is itself ARM64)
  * saturation stated as a provisional 200-280 ms region everywhere, including
    the abstract
  * the phrase-level substitutions from the reviewer's table

STILL REQUIRED, needs GPU:
  1. **Convergence check.** Everything is 1200 steps. If lookahead conditions
     converge at different rates the exchange rate is partly an optimization
     curve. Train 0/40/160/240/640 ms to convergence, or show ranking and
     slopes stable across checkpoints. The reviewer rates this the largest
     remaining threat and it is not on the earlier schedule.
  2. **Extra saturation conditions** at 220/260/300/360 ms so "per 20 ms
     equivalent" can become "per observed 20 ms step".
  3. **More seeds** in 160-320 ms, and a noise-floor estimator with a CI --
     including whether a sqrt(2) correction applies to sigma from paired
     differences.
  4. **Second encoder** (e.g. causalized wav2vec 2.0 base) over 5-7 points, to
     show the shape replicates. Until then every quantitative claim is scoped
     to the WavLM probe.

STILL REQUIRED, no GPU:
  5. Restore t_buffer, all three measurement failures, the by-L1 breakdown and
     the acoustic-weighting method in full -- there is room now.
  6. Hierarchical uncertainty (seed -> speaker -> utterance) for RQ2 and the
     by-L1 analysis, replacing the global 0.0044 floor for class-conditioned
     rates, where token counts differ by an order of magnitude.
  7. Phonetics references for the 80-250 ms formant-trajectory claim.
  8. Formal dependency argument alongside the truncation test, if "proof" is to
     be used at all.
