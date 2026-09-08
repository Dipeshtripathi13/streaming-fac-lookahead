# arXiv submission

The package is built by `make_arxiv_package.sh`, which produces
`arxiv_taslp.tar.gz` from `taslp/`. Everything below is what the arXiv
web form asks for.

## What goes in the tarball

    taslp.tex               the paper
    taslp.bbl               REQUIRED -- arXiv does not reliably run BibTeX
    refs.bib                included as well, harmless, covers the case where it does
    fig_dense_no_knee.png   the only figure

`IEEEtran.cls` is *not* included: arXiv ships it. Do not add it.

The `refs.bib` inside `taslp/` is a symlink to `paper/refs.bib` (one source of
truth for the three paper variants). The package script copies with `cp -L` so
the tarball contains a real file -- a symlink would arrive empty.

## Form fields

**Title**

    Future-Context Requirements for Streaming Accent Conversion:
    A Controlled Study Using Causal Phone Translation

**Authors**

    Dipesh Tripathi (University of South Dakota)

**Primary category**: `eess.AS` (Audio and Speech Processing)
**Cross-list**: `cs.SD` (Sound), `cs.CL` (Computation and Language)

**Comments field** (free text, worth filling in):

    Submitted to IEEE/ACM TASLP. Code and per-condition outputs:
    https://github.com/Dipeshtripathi13/streaming-fac-lookahead

**Licence**: arXiv's default (`arXiv.org perpetual, non-exclusive licence`) is
the right choice unless you have a reason otherwise. It does not conflict with
a later TASLP submission.

**Abstract**: paste from `arxiv_abstract.txt` (plain text, LaTeX stripped).

## Order of operations, given TASLP

Posting to arXiv first is fine and does not count as prior publication for
IEEE. IEEE's policy permits an author-prepared preprint on arXiv; on
acceptance you add the DOI and the IEEE copyright line to the arXiv version.
So: post the preprint, then submit the same source to TASLP.

## What I could not do for you

Submitting requires your arXiv login, and a first-time submission to `eess.AS`
may need endorsement -- if your USD address is recognised it is usually
automatic. Upload `arxiv_taslp.tar.gz` at https://arxiv.org/submit and the
fields above are everything the form asks for.
