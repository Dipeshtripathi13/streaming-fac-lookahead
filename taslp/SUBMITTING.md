# Submitting to TASLP, end to end

Written 8 Sep 2026. Everything below that I could verify against IEEE or SPS
documentation is stated plainly; everything I could not is marked, because
guessing at a submission procedure wastes your time rather than mine.

---

## Step 0: confirm which journal, and where it submits

**Do this first. It is the only step that can waste months.**

Searching for TASLP's submission page turns up two similar names:

- **IEEE/ACM Transactions on Audio, Speech, and Language Processing**, the
  joint IEEE/ACM journal this paper targets.
- **IEEE Transactions on Audio, Speech and Language Processing**, referred to
  by SPS as *TASLPRO*, listed with its own scope and impact factor.

I could not establish from public pages whether TASLPRO is a new IEEE-only
journal, a rename, or something running alongside the joint one, nor whether
the IEEE/ACM title is still accepting submissions. Those two possibilities
lead to different portals.

Resolve it in one of these ways, in order of speed:

1. Open a recent paper in the journal you mean to target on IEEE Xplore and
   follow its "Submit Manuscript" link. That link is definitionally correct
   for that journal.
2. Go to the SPS publications page for the journal and use its Information for
   Authors link.
3. Email the SPS publications office. Slower, but definitive, and worth it if
   the first two disagree.

IEEE has been migrating journals from **ScholarOne Manuscripts** to the
**IEEE Author Portal**. Either is possible here. The steps below apply to both,
since the fields are substantially the same; only the layout differs.

## Step 1: do NOT anonymise

SPS journals use **single-anonymised** review: reviewers see who you are, you
do not see who they are. So submit `manuscript.pdf`, the version carrying your
name and affiliation.

`make_anonymous.sh` and `anonymous/` exist because this was unresolved
earlier. They are not part of this submission. Do not run the script and do
not upload it.

Still worth confirming: **are source files required at submission?** Some IEEE
journals take a PDF for review and ask for source only on acceptance.
`source/` and `taslp_source.tar.gz` are ready either way.

## Step 2: account

Register or sign in. Use `dipesh.tripathi@coyotes.usd.edu`, the address on the
manuscript. Have your ORCID ready; IEEE asks for it and will offer to create
one if you have none.

## Step 3: start the submission

Manuscript type: **Regular Paper**. Not a Letter, Correspondence or Overview.
Confirm the category names on the journal's own page, since they vary.

## Step 4: metadata

From `metadata.txt`:

- **Title.** Copy exactly; do not retype and risk a typo.
- **Abstract.** The manuscript's own abstract, 177 words, which sits inside
  the SPS requirement of 150 to 250 words with no references, footnotes,
  displayed equations or abbreviations. This is *not* the arXiv abstract in
  `../arxiv/abstract.txt`, which was written to a different limit.
- **Index terms.** accent conversion, streaming speech, latency, lookahead,
  self-supervised models.
- **Author.** You alone, corresponding and submitting.

## Step 5: upload

`manuscript.pdf`, nine pages, with your name on it. Upload
`taslp_source.tar.gz` only if the portal asks for source. Do not upload
`metadata.txt`, `COVER_LETTER_NOTES.md`, `make_anonymous.sh` or anything in
`anonymous/`; those are working files.

## Step 6: the questions that cost money

These are the ones to read rather than click through.

- **Open access.** Decline. SPS journals are hybrid; traditional publication
  costs the author nothing.
- **Voluntary page charges**, $110 per page for the first ten. They are
  voluntary. Decline.
- **Overlength charges**, $220 per page beyond ten *published* pages, and
  mandatory. The manuscript is nine pages, so you owe nothing today. Guard
  that margin through revision.

## Step 7: the policy questions

- **Prior publication.** No.
- **Under consideration elsewhere.** No.
- **Preprint.** Yes, declare it. A preprint has been submitted to arXiv and is
  awaiting announcement. IEEE permits author-prepared preprints and this does
  not count as prior publication. Once arXiv assigns the public identifier,
  supply it; the submission number 8054723 is internal and resolves to
  nothing.
- **Funding.** None. The paper says so.
- **Conflicts.** None to declare, but see reviewers below.

## Step 8: reviewers

If the form asks for suggestions, you may prefer not to name Quamer, Tseng,
Nasrallah or Gutierrez-Osuna, who wrote PHONOS, TVTSyn and DarkStream, the
three systems this paper measures itself against.

Do not mark them as conflicts of interest. IEEE means something specific by
that: shared institution, recent collaboration, or a supervisory relationship.
Having authored work you compare against is not one, and declaring it as such
would be inaccurate.

## Step 9: cover letter

Paste `cover_letter.md`. It contains the letter and nothing else, so it needs
no editing before use. `COVER_LETTER_NOTES.md` holds the two things worth
checking first (the salutation, and swapping in the arXiv identifier if it has
been announced); that file is for you and is not submitted.

## Step 9b: the LLM policy, which applies to you

Read this properly rather than clicking through it.

SPS policy: authors take full responsibility and ownership for the manuscript.
Improving language and clarity, and accelerating code development, are
acceptable uses. What is not acceptable is generating most or significant
components of a manuscript with an LLM, or using LLM-generated text or code
without thorough verification of correctness. Submitting the paper is itself
your confirmation that you have read the policy and done that verification.

This project used an LLM heavily: for drafting manuscript text, writing the
analysis code, running experiments and interpreting results. That is well
beyond "language polishing", so the confirmation you give at submission is not
a formality. Before you click Submit you should be able to:

- state what every number in the paper means and where it came from;
- defend every methodological choice, including the ones that changed under
  review, such as clustering on speakers rather than tokens, treating the
  repeatability floor as a floor rather than a significance threshold, and
  matching encoder layers by function rather than index;
- explain the code well enough to answer a reviewer asking how a result was
  computed. The self-tests help, but they are not a substitute for
  understanding.

If any of that is not yet true, the fix is to read the manuscript and the
analysis scripts end to end before submitting, not to submit and hope. The
policy does not require you to disclose LLM use in the manuscript, but it does
require the verification to have actually happened.

## Step 10: approve the generated PDF

The system builds its own PDF from what you uploaded. **Read it before
approving**, the same way you would read arXiv's. Check that Figure 1 renders,
that no table overlaps the column beside it, and that the reference list is
complete and numbered 1 to 22.

## After submission

- You will get a manuscript number. Keep it.
- Review typically takes months. Nothing is required of you meanwhile.
- On acceptance: sign the copyright form, then post the accepted version to
  arXiv as a **replacement** with the journal reference and DOI added, so it
  stays under the same identifier, and add the DOI to the repository README.
- If revisions are requested, watch the page count. Nine pages leaves one page
  of headroom before overlength charges start.
