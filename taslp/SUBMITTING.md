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

## Step 1: also confirm these two

- **Is review double-blind?** If yes, submit
  `anonymous/manuscript_anon.pdf`, not `manuscript.pdf`. Removing your name
  is not enough on its own: the repository URL contains your username and
  appears twice in the identified version. `make_anonymous.sh` handles both
  and refuses to emit a PDF still containing an identifying string.
- **Are source files required at submission?** Some IEEE journals take a PDF
  for review and ask for source only on acceptance. `source/` and
  `taslp_source.tar.gz` are ready either way.

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
- **Abstract.** The manuscript's own abstract, 318 words. Note this is
  *not* the shortened arXiv one in `../arxiv/abstract.txt`, which was cut to
  fit arXiv's 1920-character field. If the portal has its own limit, tell me
  and I will cut one to fit.
- **Index terms.** accent conversion, streaming speech, latency, lookahead,
  self-supervised models.
- **Author.** You alone, corresponding and submitting.

## Step 5: upload

Identified or anonymised per Step 1. `manuscript.pdf` is nine pages.

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

If the form asks for suggestions, that is fine. If it asks for exclusions, or
if you are choosing whom to suggest, **exclude Texas A&M PSI**: Waris Quamer,
Mu-Ruei Tseng, Ghady Nasrallah and Ricardo Gutierrez-Osuna authored PHONOS,
TVTSyn and DarkStream, the three systems this paper measures itself against.
That is a conflict of interest, not a recommendation.

## Step 9: cover letter

Paste `cover_letter.md`. Two things to adjust before you do:

- It opens "Dear Editor-in-Chief". If the portal names a handling editor, use
  their name.
- If arXiv has announced by then, replace the awaiting-announcement sentence
  with the real identifier. The replacement sentence is at the top of the file.

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
