# arXiv submission

Everything needed to submit is in this folder. **The only file you upload is
`arxiv_taslp.tar.gz`.** The rest is here so you can check what you are
uploading and copy the metadata across.

| file | what it is |
|---|---|
| `arxiv_taslp.tar.gz` | **the upload.** LaTeX source, 216 KB |
| `taslp.pdf` | what that source compiles to, for your reference. Do not upload this |
| `abstract.txt` | the abstract as plain text, LaTeX stripped, ready to paste |
| `metadata.txt` | title, authors, categories, comments, licence |
| `source/` | the tarball unpacked, if you want to look inside |

Regenerate the tarball at any time with `../paper/make_arxiv_package.sh`.

---

## Before you start

**You need an arXiv account, and you may need endorsement.** arXiv requires
endorsement before a user's first submission to a category. Registering with
your `@coyotes.usd.edu` address gets you expedited consideration, and that is
the path to take. If the system still asks for endorsement for `eess.AS`, it
will tell you during submission and give you a code to send to someone who can
endorse. A faculty member in your department who has posted to `eess.AS` or
`cs.CL` in the last few years is the person to ask. This can add a few days,
so register before the day you want to post.

**Upload source, not PDF.** arXiv explicitly prefers LaTeX source and does not
accept PDF generated from LaTeX. The tarball is already in the right shape.

---

## The steps

Go to <https://arxiv.org/submit> and work through the nine stages.

**1. Start a new submission.** Choose the article type; this is a regular
submission, not a cross-list or a replacement.

**2. Upload `arxiv_taslp.tar.gz`.** Nothing else. Do not add the PDF.

**3. Let it detect the compiler and top-level file.** It should find
`taslp.tex` as the main file. If it asks about the compiler, the paper builds
under pdfLaTeX and XeLaTeX; accept the default.

**4. Check the file list.** You should see exactly four files:
`taslp.tex`, `taslp.bbl`, `refs.bib`, `fig_dense_no_knee.png`. If anything
starting with `._` appears, stop and rebuild the tarball; the build script
strips those and fails if any leak, so it should not happen.

**5. Compile and preview.** This is the step that matters most. arXiv builds
the paper on its own machines, so read the preview rather than assuming it
matches `taslp.pdf`. Check:

  - 10 pages
  - the bibliography is present and numbered `[1]` to `[22]`, with no `[?]`
  - Figure 1 on page 4 is labelled *saturation onset ~340 ms* and
    *fixed-seed repeatability floor 2 sigma = 0.0029*. The numbers 0.0044 and
    240 ms do appear on that page, but in the body text, in the sentence
    explaining that the superseded floor is what produced the earlier 240 ms
    figure. That sentence is meant to be there; the figure labels are what to
    check
  - Table IV on page 5 sits inside its column and does not overlap the text
    beside it

  If the bibliography is missing, the `.bbl` did not get picked up. It is in
  the tarball and its name matches the `.tex` file, which is arXiv's
  requirement, so this should not happen.

**6. Enter the metadata.** Copy from `metadata.txt`. Paste the abstract from
`abstract.txt` rather than from the PDF; the PDF text has ligature and
line-break artefacts, and arXiv's abstract box does not render LaTeX.

**7. Licence.** Take the default arXiv perpetual non-exclusive licence. It
does not conflict with submitting to TASLP.

**8. Submit.** You get an identifier immediately, but the paper is announced
on the next cycle. Submissions before 14:00 US Eastern on a weekday are
announced that evening; after that, the next business day. It goes live at
`arxiv.org/abs/<id>`.

**9. Moderation.** A moderator may reclassify the primary category. That is
routine and needs nothing from you.

---

## After it is live

- Add the arXiv ID to the repository README so the code and paper point at
  each other.
- Submit to TASLP. Posting the preprint first does not count as prior
  publication for IEEE.
- On acceptance, post a v2 with the journal reference and DOI filled in, and
  add the IEEE copyright line.

## If you change the paper

Edit under `../paper/taslp/`, then run `../paper/make_arxiv_package.sh`, which
rebuilds the tarball and refuses to ship it if a citation is unresolved, a
table overflows its column, or macOS metadata leaks in. Copy the new tarball
here and submit it as a replacement, not a new submission, so the versions
stay under one identifier.
