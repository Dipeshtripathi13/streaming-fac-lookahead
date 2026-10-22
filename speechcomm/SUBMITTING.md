# Submitting to Speech Communication

Journal: Speech Communication (Elsevier, for EURASIP and ISCA).
Portal: https://submit.elsevier.com/SPECOM
Article type: **Original full-length research article**.
Review: single anonymised, so the author block stays in the manuscript.

## Cost: choose the subscription route

The journal charges authors **nothing** on the subscription route. When the
portal asks about publishing options, choose subscription, **not** open
access; open access costs about USD 3,280.

## Files to upload

| portal item | file |
|---|---|
| Manuscript (LaTeX source) | `manuscript.tex` |
| LaTeX supporting files | `refs.bib`, `manuscript.bbl` |
| Figure 1 | `fig_schemes.pdf` (vector) |
| Highlights | `highlights.docx` (5 bullets, each under 85 characters) |
| Declaration of interest | `declaration_of_competing_interest.docx`; the portal may instead ask you to fill in Elsevier's declarations tool, which produces the same statement |
| Cover letter | `cover_letter.docx` |

`manuscript.pdf` is a local build for checking. The portal builds its own PDF
from the source, so check that one before approving.

Rebuild after any edit: `tectonic manuscript.tex`.

## Form answers

- **Title:** Per-layer lookahead masks leak future context: re-measuring the
  future-context requirement of streaming accent conversion
- **Abstract:** copy from the manuscript (250 words, the limit).
- **Keywords (6):** accent conversion; streaming speech processing;
  lookahead; latency; self-supervised models; reproducibility
- **Funding:** none.
- **Data availability:** "Data and code are available at
  https://github.com/Dipeshtripathi13/streaming-fac-lookahead. The L2-ARCTIC
  speech corpus is third-party data under CC-BY-NC-4.0 and is not
  redistributed."
- **Generative AI:** declared in the manuscript, in its own section before the
  acknowledgements. Elsevier requires this whenever AI tools were used beyond
  grammar and spelling checks.
- **SSRN preprint option:** optional, and has no effect on the decision.
- **Suggested reviewers, if asked:** do not suggest anyone from the Texas A&M
  PSI lab (authors of PHONOS, TVTSyn and DarkStream); the paper discusses
  their systems.

## Before you press submit

1. Read the portal-generated PDF in full.
2. arXiv: the existing preprint carries the superseded claims. Replace it with
   this version (same arXiv ID) so the public record matches the submission.
3. If the form asks whether the work has a preprint, answer yes and give the
   arXiv identifier.
