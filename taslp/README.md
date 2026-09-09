# TASLP submission

Files for submitting to IEEE/ACM Transactions on Audio, Speech, and Language
Processing. Not to be confused with `../paper/taslp/`, which is where the
manuscript is *written*; this folder is what gets *sent*.

| file | what it is |
|---|---|
| `manuscript.pdf` | the paper, 10 pages |
| `source/` | LaTeX source: `taslp.tex`, `taslp.bbl`, `refs.bib`, the figure |
| `cover_letter.md` | draft cover letter. Read before sending |
| `metadata.txt` | title, author, index terms, and what to declare |
| `make_anonymous.sh` | builds a blinded version, only if it is required |
| `anonymous/` | output of that script |

Regenerate the manuscript with `../paper/make_arxiv_package.sh`, which also
runs the citation and table-width checks, then copy the PDF here.

---

## Read this before you start

**Four things I could not verify, and you must.** I tried to confirm these
from the IEEE Signal Processing Society and IEEE Xplore pages and could not
reach a current, authoritative statement for any of them. Do not submit on my
guess. Open the journal's **Information for Authors** page and check:

1. **Is review double-blind?** This is the one that can cost you a desk
   rejection. If it is, submit `anonymous/manuscript_anon.pdf`, not
   `manuscript.pdf`. Note that removing your name is not sufficient on its
   own: the repository URL is `github.com/Dipeshtripathi13/...`, which names
   you, and it appears twice in the identified version. The script handles
   both and refuses to produce a PDF that still contains any identifying
   string.
2. **Page limit and overlength charges.** The manuscript is 10 pages. You told
   me you did not want to pay page charges, so confirm where the free limit
   sits before submitting rather than after acceptance. If 10 pages is over
   it, tell me and I will cut; the measurement-audit section is the most
   compressible without losing a result.
3. **Which submission system.** IEEE has been migrating journals from
   ScholarOne Manuscripts to the IEEE Author Portal. The journal page will say
   which one TASLP uses and link to it.
4. **Whether source is required at submission.** Some IEEE journals take a
   PDF for review and ask for source only on acceptance. `source/` is ready
   either way.

**What I did verify:** the manuscript itself. 10 pages, 22 references, no
unresolved citations, every table inside its column, and the anonymous build
compiles and contains none of "Tripathi", "Dipeshtripathi13", "coyotes.usd.edu"
or "South Dakota".

---

## Order of operations with arXiv

Post the arXiv preprint first if you want a timestamp and a citable link;
IEEE permits an author-prepared preprint and it does not count as prior
publication. Then declare it on the TASLP form, which asks. If you would
rather not have the preprint visible during review, submit to TASLP first and
post the preprint afterwards. Either order is allowed; the only mistake is
not declaring it.

`cover_letter.md` has a bracketed `arXiv:[ID]` to fill in, or a paragraph to
cut if you go to TASLP first.

The preprint is going up under **cs.SD** as primary, with eess.AS requested as
a cross-list, because eess.AS requires an endorsement the account does not
have. This has no bearing on the TASLP submission; arXiv classification and
journal scope are unrelated.

---

## The submission itself

1. Create or sign in to your account on whichever system the journal names.
2. Start a new submission, category **regular paper**.
3. Upload the manuscript. Identified or anonymised, per point 1 above.
4. Paste the metadata from `metadata.txt`. Have your ORCID to hand.
5. Paste the cover letter.
6. Declare the arXiv preprint if it is live.
7. Suggested reviewers, if asked: **exclude anyone from Texas A&M PSI.**
   Quamer, Tseng, Nasrallah and Gutierrez-Osuna wrote PHONOS, TVTSyn and
   DarkStream, all three of which this paper measures itself against. That is
   a conflict of interest, not a suggestion.
8. Approve the system-generated PDF before completing. Check the figure and
   tables survived, as with any conversion step.

## On acceptance

- Sign the IEEE copyright form.
- Update the arXiv version: add the journal reference, the DOI and the IEEE
  copyright line, and submit it as a **replacement** so it stays under the
  same arXiv identifier.
- Add the DOI to the repository README.
