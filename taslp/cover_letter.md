# Cover letter

Draft. Read it before sending; it makes claims on your behalf.

As written it says the preprint is awaiting announcement, which is true today.
Once arXiv announces it you will get a public identifier of the form
arXiv:2609.NNNNN. That is the number to quote, not the submission number
(8054723), which is an internal tracking ID an editor cannot look up. When the
public ID exists, replace the sentence with:

  A preprint of this work is available at arXiv:2609.NNNNN.

---

Dear Editor-in-Chief,

I am submitting "Future-Context Requirements for Streaming Accent Conversion:
A Controlled Study Using Causal Phone Translation" for consideration in
IEEE/ACM Transactions on Audio, Speech, and Language Processing.

Streaming foreign accent conversion systems each choose a lookahead budget and
report a single end-to-end latency. To my knowledge no prior streaming accent-conversion work
densely sweeps that budget while separately reporting algorithmic and
computational latency, so the field has operating points without a characterisation of what
they cost. This paper measures that relationship directly, using a causal
phone translator as a controlled probe rather than an end-to-end conversion
system, and reports what the probe can and cannot establish.

Five results follow. Phone error rate falls smoothly with lookahead at -0.045
PER per doubling with no locatable knee, and saturation onset near 340 ms
measured against a repeatability floor from deliberate fixed-seed repeats. A
pre-specified hypothesis that vowels and approximants would benefit more from
future context than stops, fricatives, affricates and nasals is not supported,
and remains unsupported when errors are reweighted by a spectral-distortion
proxy for acoustic cost. Canonical-phone prediction benefits more from future
context than realized-phone transcription at matched capacity. Repeating the
sweep on a second causalised encoder shows that neither the exchange rate nor
the detailed curve shape transfers, so a future-context budget has to be
characterised for the deployed representation rather than inherited. Per-chunk
compute depends only weakly on lookahead.

Two aspects may interest reviewers particularly. First, the manuscript
documents the measurement failures encountered along the way: a padding bug
whose correction grew with lookahead and so distorted the central curve rather
than offsetting it, and a breakpoint estimator that returned a confident knee
with a tight bootstrap interval on a curve that has none. Second, the negative
result on the pre-specified phoneme-class hypothesis is reported as a negative
result, with the prediction fixed in a public commit before the outputs it is
tested on were generated.

The manuscript is nine pages, within the ten-page limit for a Regular Paper.
The work is single-author and used no external funding and no institutional
compute; all GPU results come from free-tier Google Colab. The harness,
per-condition outputs and analysis scripts are public at
https://github.com/Dipeshtripathi13/streaming-fac-lookahead.

A preprint of this work has been submitted to arXiv and is awaiting
announcement; I will supply the identifier once it is assigned. That version
is one page longer, with the same results: the manuscript submitted here moves
two supporting tables to the public repository and shortens the abstract. The
work has not been published elsewhere and is not under consideration by
another journal.

Thank you for your consideration.

Dipesh Tripathi
University of South Dakota, Vermillion, SD, USA
dipesh.tripathi@coyotes.usd.edu
