# ICASSP 2027 submission

Deadline **16 Sep 2026**, Toronto, notification 13 Jan 2027.

`icassp2027.tex` is the ICASSP-formatted version. `../main.tex` is the original
and is deliberately kept: at 4 pages + 2 reference pages it fits **SSW14**
(4 + up to 2 for references) with no trimming at all, and is the base for an
Interspeech submission. The two are copies rather than includes, because the
venues need different front matter and different trims -- keep them in sync by
hand.

## Official kit, vendored here

`spconf.sty` and `IEEEbib.bst` downloaded 7 Sep 2026 from
`https://cmsworkshops.com/ICASSP2027/papers/PaperFormat/`. `Template.tex` is
kept for reference. Do not substitute IEEEtran: ICASSP requires these.

## Hard constraints, verified against the paper kit

* **5 pages total, no exceptions.** Pages 1-4 carry technical content, figures
  and references; page 5 may contain ONLY references, funding acknowledgements
  and the Compliance with Ethical Standards statement. This build is 5 pages
  with the bibliography starting on page 5, confirmed from the TeX log and
  from PDF metadata.
* **Reviewing is not blind.** The author block stays in.
* Two columns, 86 mm wide, 6 mm gutter; Times or Computer Modern, 9 pt minimum.

## What differs from ../main.tex

* `spconf` front matter (`\name`, `\address`) instead of `\author`; `\ninept`.
* `\paragraph` is not defined by spconf and was replaced with
  `\noindent\textbf{...}`.
* `Limitations` and `Remaining work` merged into one section. Two separate
  sections is unusual at this length, and the merge is what brought the paper
  from 6 pages to 5.
* A **Compliance with Ethical Standards** statement, which ICASSP requires and
  the other venues do not.
* End matter compressed and reordered: Conclusion, then data availability,
  acknowledgements, ethics, references.

## Before submitting

* Re-read the CFP for any late change to the page limit.
* The paper claims no perceptual evaluation anywhere; keep it that way unless
  a listening study is actually run.
