Dear Editor-in-Chief,

I am submitting "Future-Context Requirements for Streaming Accent Conversion:
A Controlled Study Using Causal Phone Translation" for consideration as a
Regular Paper in the IEEE/ACM Transactions on Audio, Speech, and Language
Processing.

Streaming foreign accent conversion systems choose a lookahead budget, yet
prior work does not densely characterise how that choice trades future context
for latency and accuracy. This manuscript studies that relationship using a
causal phone translator as a controlled probe rather than claiming an
end-to-end accent-conversion evaluation.

The study evaluates 21 lookahead budgets and finds a strong but
encoder-dependent future-context response. For the WavLM-based probe, phone
error rate decreases approximately 0.045 per doubling of lookahead, with no
identifiable knee and a repeatability-aware saturation onset near 340 ms. A
pre-specified manner-class hypothesis is not supported, while canonical-phone
prediction benefits more strongly from future context than realized-phone
transcription. A matched wav2vec 2.0 experiment shows that neither the
exchange rate nor the detailed response curve transfers across encoders. The
manuscript also separates algorithmic from computational latency and documents
measurement failures that materially affected earlier analyses.

The manuscript is nine double-column pages and is submitted as a Regular
Paper. Code, configurations, analysis scripts, causality tests and
per-condition outputs are publicly available at the repository identified in
the manuscript.

A preprint has been submitted to arXiv and is awaiting announcement. The work
has not been published elsewhere and is not under consideration by another
journal.

Thank you for your consideration.

Dipesh Tripathi
University of South Dakota
dipesh.tripathi@coyotes.usd.edu
