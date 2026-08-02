# Literature — verified, with corrections to proposal v1

Verified 2 August 2026 against arXiv abstracts and listing pages. Anything
marked **[unverified]** was not confirmed from a primary source in this pass
and must be checked before it appears in a submission.

---

## Corrections to proposal v1 — read these first

Four errors in v1, one of them strategically significant.

### 1. PHONOS uses **≤40 ms lookahead**, not "inherited from TVTSyn"

> "…supervise a causal accent translator that maps non-native content tokens to
> native equivalents **with at most 40 ms look-ahead**, trained using joint
> cross-entropy and CTC losses."
> — PHONOS abstract, arXiv:2603.27001

This is the single most consequential correction. Proposal v1's table left the
PHONOS lookahead blank and H1 predicted that "the knee is above the ~80–140 ms
used by current streaming systems, meaning existing budgets are
under-provisioned for AC."

PHONOS is an accent-conversion system operating at **40 ms** — half of TVTSyn's
80 ms and well under DarkStream's 140 ms — and reporting an 81% reduction in
non-native accent confidence with listening tests that agree. H1 as written is
therefore in direct tension with a published result, and needs restating rather
than deleting. See PROPOSAL_v2 §3.

### 2. PHONOS was posted **27 March 2026**, not July 2026

Submitted to Interspeech 2026. Four months older than v1 assumed, which means
the follow-up work is further along than v1's risk table implies.

### 3. PHONOS is from **the same lab** as TVTSyn and DarkStream

Authors: Waris Quamer, Mu-Ruei Tseng, Ghady Nasrallah, Ricardo Gutierrez-Osuna
(Texas A&M PSI Lab). v1's related-work table implicitly treats PHONOS as an
outside result. It is not — it is the third paper in one continuous programme
(DarkStream → TVTSyn → PHONOS). That concentrates the competitive risk into a
single group and makes the "email them in August" action item much more
important than v1 rated it. It also means the 40 ms budget was chosen by people
who had already built two streaming systems, so "they never asked the question"
is a weaker claim than v1 assumes.

### 4. The survey is titled differently than v1 records

arXiv:2604.27281 = *"Accent Conversion: A Problem-Driven Survey of
Sociolinguistic and Technical Constraints"* (30 April 2026). v1 lists it as
*"Accent Conversion: A Problem-Driven Survey"*. Its framing — sociolinguistic
constraints, the accent-modification vs speaker-identity-preservation
trade-off — is directly relevant to §12's contribution claims and should be
cited in the introduction, not just the related work.

### Also new since v1 was drafted (not in the v1 table at all)

- **Stream-Voice-Anon** — arXiv:2601.13948, real-time speaker anonymisation via
  neural audio codec + LM. **[unverified: venue]**
- **StreamVoiceAnon+** — arXiv:2603.06079, emotion-preserving streaming
  anonymisation. **[unverified: venue]**
- **End-to-end streaming model for low-latency speech anonymization** —
  arXiv:2406.09277.

The streaming-anonymisation space is more crowded than v1's table shows. This
does not threaten the AC-specific angle, but it does mean a reviewer will know
the area well.

---

## Verified related work

| Work | ID | Date | Task | Streaming | Lookahead | Latency | HW | Status |
|---|---|---|---|---|---|---|---|---|
| **PHONOS** | arXiv:2603.27001 | 27 Mar 2026 | FAC + anonymisation | yes | **≤40 ms** | **<241 ms** | **single GPU** | verified |
| **TVTSyn** | arXiv:2602.09389 | Feb 2026 | streaming VC + anon | yes | **[unverified]** ~80 ms | **<80 ms** | **GPU** | abstract verified; ICLR 2026 acceptance **[unverified]** |
| **DarkStream** | arXiv:2509.04667 | Sep 2025 | real-time anonymisation | yes | short lookahead buffer, **140 ms [unverified]** | low | not stated in abstract | verified |
| **StreamVC** | arXiv:2401.03078 | Jan 2024 | streaming VC | yes | causal | low | **"even on a mobile platform"**, device unnamed | verified |
| Stream-Voice-Anon | arXiv:2601.13948 | Jan 2026 | streaming anon | yes | — | — | — | new to v2 |
| StreamVoiceAnon+ | arXiv:2603.06079 | Mar 2026 | streaming anon + emotion | yes | — | — | — | new to v2 |
| E2E streaming anon | arXiv:2406.09277 | Jun 2024 | anonymisation | yes | — | — | — | new to v2 |
| AC survey | arXiv:2604.27281 | 30 Apr 2026 | survey | — | — | — | — | verified; **read first** |
| LLVC | Sadov et al. 2023 | 2023 | VC | yes | minimal | 20 ms | — | **[unverified]** |
| Zhao et al., reference-free FAC | TASLP 29 | 2021 | FAC | no | — | offline | — | **[unverified]** |
| Accentron | CSL 72 | 2022 | zero-shot FAC | no | — | offline | — | **[unverified]** |

**Every [unverified] row must be checked against the paper, not the abstract,
before submission.** Three of them (TVTSyn's exact token count, DarkStream's
140 ms, LLVC's 20 ms) are load-bearing numbers in the argument.

---

## What survives of the gap analysis

| v1 claimed gap | Verdict after checking |
|---|---|
| **No CPU numbers** | **Holds, and is the strongest gap.** PHONOS reports GPU. TVTSyn reports GPU. StreamVC says "mobile platform" without naming a device, a core count, or a thread count — unfalsifiable as a deployment claim. DarkStream's abstract names no hardware at all. Nobody has published a CPU or embedded latency–quality curve for streaming AC. |
| **Justified lookahead budgets** | **Weakened but alive.** PHONOS states 40 ms; it does not report an ablation showing 40 ms was *necessary* or *sufficient*, nor what happens at 0/20/80/160 ms. The gap is now "no one has swept it", not "no one has chosen one". That is a narrower and more honest claim, and it is still a real gap. |
| **Phoneme-level analysis** | **Holds.** Every system reports aggregate accentedness / MOS / WER. No published decomposition of degradation by phoneme class under a latency constraint. |
| **Algorithmic vs computational latency** | **Holds, and is under-appreciated.** "≤241 ms on single GPU" fuses a 40 ms modelling decision with an unstated GPU inference cost and an unstated buffer. A reader cannot tell from that number whether a faster chip helps. This is cheap to fix and worth a table. |
| **AC vs VC comparison** | **Holds.** No study runs one pipeline in both configurations at matched capacity to isolate what is accent-specific. |

Net: **five gaps in, four and a half out.** The CPU gap and the
algorithmic/computational decomposition are the two that no amount of
follow-up from TAMU is likely to close incidentally, because their research
programme is aimed at anonymisation quality on GPU.

---

## Datasets — verified

**L2-ARCTIC / CMU ARCTIC parallelism: confirmed by documentation.**
L2-ARCTIC speakers each read the **1132 phonetically balanced CMU ARCTIC
prompts**. 24 non-native speakers, gender-balanced, **6 L1s** (Hindi, Korean,
Mandarin, Spanish, Arabic, **Vietnamese** — v1 omits Vietnamese), **26,867
utterances** total, ~1 h each. Phoneme-level mispronunciation annotations ship
for a per-speaker subset.

Two things documentation cannot tell you, which is why
`data/verify_prompt_overlap.py` exists and is a gate, not a formality:

1. Whether every speaker actually read every prompt (utterance IDs drift; some
   speakers skipped items).
2. Whether shared IDs have *textually identical* prompts. Some L2-ARCTIC
   transcripts were corrected to match what the speaker said rather than the
   card. Those are precisely the utterances where "use the prompt as WER ground
   truth" silently breaks, and they will be a small, systematically-biased
   subset — the hardest sentences.

The script also counts annotated utterances per speaker, because the H2
phoneme-class analysis is powered by that subset and by nothing else. If it
comes back at ~150/speaker, four L1s gives roughly 600 annotated utterances per
accent pair: enough for class-level slopes, not enough for per-phone slopes.
Better to discover that in September than in December.

**CMU ARCTIC voice choice matters.** `bdl`, `slt`, `clb`, `rms` are General
American. `jmk` is Canadian and `awb` is Scottish — using either as a "GA
target" would invalidate the accent probe. `data/download.sh` fetches only the
four GA voices by default.

---

## Venue facts — verified

- **Interspeech 2027**: São Paulo, Brazil, **29 Aug – 2 Sep 2027**. Main
  conference paper deadline **not yet announced** as of 2 Aug 2026. Historical
  range 25 Feb – 21 Mar; planning for **1 March 2027** remains reasonable.
  Re-check `interspeech2027.org` monthly — this is action item #6 and it is
  the one that quietly slips.
- **SSW14** (14th ISCA Speech Synthesis Workshop, Interspeech 2027 satellite):
  submission deadline **Sunday 20 April 2027, 23:59 AoE** — verified. That is a
  seven-week buffer after the Interspeech deadline, which makes it a genuinely
  usable fallback rather than a theoretical one.
- **ICASSP 2027**: deadline ~16 Sep 2026 (six weeks out). Correctly ruled out.

---

## Reading order

1. **arXiv:2604.27281** — the survey. Read it first; it will save you from
   re-deriving the field's framing.
2. **arXiv:2603.27001** — PHONOS. Defines what you can and cannot claim as
   novel. Read the method section for the golden-target generation recipe: it
   is the part worth reusing, because it makes the training target independent
   of L, which is what makes RQ1 answerable at all (see
   `notebooks/colab_streaming_fac.ipynb` §6b).
3. **arXiv:2602.09389** — TVTSyn. The likely architectural base.
4. **arXiv:2509.04667** — DarkStream. Confirm the 140 ms figure from the paper.
5. Zhao et al. TASLP 2021 — the offline reference-free FAC fallback if no code
   is released.

## Sources

- [PHONOS (arXiv:2603.27001)](https://arxiv.org/abs/2603.27001)
- [TVTSyn (arXiv:2602.09389)](https://arxiv.org/abs/2602.09389)
- [DarkStream (arXiv:2509.04667)](https://arxiv.org/abs/2509.04667)
- [StreamVC (arXiv:2401.03078)](https://arxiv.org/abs/2401.03078)
- [Accent Conversion survey (arXiv:2604.27281)](https://arxiv.org/abs/2604.27281)
- [Stream-Voice-Anon (arXiv:2601.13948)](https://arxiv.org/html/2601.13948)
- [StreamVoiceAnon+ (arXiv:2603.06079)](https://arxiv.org/pdf/2603.06079)
- [L2-ARCTIC documentation, TAMU PSI Lab](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/)
- [Interspeech 2027](https://interspeech2027.org/)
- [SSW14 @ Interspeech 2027 — LINGUIST List 37.2247](https://linguistlist.org/issues/37/2247/)
