# Future-Context Requirements for Streaming Accent Conversion

How much future context does streaming foreign accent conversion actually
need? Streaming systems each choose a lookahead budget and report a single
end-to-end latency, but the budget itself is rarely characterised. This
repository holds the harness, data and analysis behind a controlled study of
that trade-off, using a causal phone translator as the probe.

**Paper:** *Future-Context Requirements for Streaming Accent Conversion: A
Controlled Study Using Causal Phone Translation.*
[`paper/taslp/taslp.pdf`](paper/taslp/taslp.pdf) — submitted to IEEE/ACM
Transactions on Audio, Speech, and Language Processing; preprint on arXiv.

---

## What the study found

Twenty-one lookahead budgets from 0 to 640 ms, every other factor held fixed.

| | |
|---|---|
| **Exchange rate** | Phone error rate falls ≈0.045 per doubling of lookahead (R² = 0.98) |
| **No locatable knee** | A piecewise fit is preferred under every axis treatment, yet the breakpoint moves between 40 and 180 ms and the narrowest bootstrap interval spans 2.6 octaves |
| **Saturation onset ≈340 ms** | Measured against a repeatability floor from deliberate fixed-seed repeats, with a 300–480 ms sensitivity range |
| **Cost of a 40 ms budget** | A further 45.3% relative PER reduction remains available by 340 ms |
| **A pre-specified hypothesis fails** | Vowels and approximants do *not* benefit more than stops, fricatives, affricates and nasals: +0.040, 95% CI [−0.010, +0.101] under speaker-level resampling |
| **Encoder dependence** | Repeating the sweep on causalised wav2vec 2.0 gives +3.9% endpoint gain against WavLM's +63.3%, from near-identical zero-lookahead PER. Neither the exchange rate nor the curve shape transfers |
| **Compute is nearly flat** | Per-chunk compute moves at most 7.6% across a 17× change in algorithmic latency |

Two methodological results travel with these. Masking attention does **not**
make a base-size self-supervised encoder causal: the positional convolution
and the feature-encoder GroupNorm each leak future context, and both need
patching before any lookahead label is valid. And a forced one-breakpoint
estimator will return a confident knee, with a tight bootstrap interval, on a
curve that has none.

---

## Repository layout

```
src/sfac/          lookahead masks, frame arithmetic, latency decomposition
bench/             causality audit, encoder scaling, compute and cascade benchmarks
train/             the causal phone translator and its lookahead sweep
eval/              analysis for each research question, each with self-tests
data/              corpus download and prompt-overlap verification
results/           raw per-condition outputs and the analyses computed from them
paper/             the manuscript, its figure, and the arXiv package builder
arxiv/  taslp/     submission packages
docs/              design decisions, literature notes, and the findings log
```

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/cpu.txt

# the causality audit gates everything else; if it fails, no lookahead
# result in this repository means anything
python3 bench/bench_content_degradation.py --selftest

# analysis self-tests, no GPU or corpus needed
python3 eval/analyse_dense_knee.py     --self-test
python3 eval/rq2_hierarchical.py       --self-test
python3 eval/rq3_interaction.py        --self-test
python3 eval/analyse_second_encoder.py --self-test
```

The sweeps themselves need a GPU and the L2-ARCTIC corpus; see
[`setup/`](setup/) for the per-platform protocol. All GPU results in the paper
come from free-tier Google Colab on a Tesla T4.

## Data and licence

Speech data is [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/), used
via the `KoelLabs/L2Arctic` release under CC-BY-NC-4.0. The corpus is not
redistributed here. Given that licence we release configurations and
per-condition outputs rather than model weights.

Code in this repository is under the licence in [`LICENSE`](LICENSE).

## Reading further

| | |
|---|---|
| [`docs/FINDINGS_LOG.md`](docs/FINDINGS_LOG.md) | Every measurement in the order it was made, superseded entries included and annotated |
| [`docs/SECOND_ENCODER.md`](docs/SECOND_ENCODER.md) | The encoder-transfer result and the three explanations ruled out before it |
| [`docs/DENSE_SWEEP_RQ1_ANSWERED.md`](docs/DENSE_SWEEP_RQ1_ANSWERED.md) | Why there is no knee to report, and what replaced it |
| [`docs/RQ2_HIERARCHICAL.md`](docs/RQ2_HIERARCHICAL.md) | Speaker-level uncertainty, and why the by-L1 split cannot carry weight |
| [`docs/PADDING_FIX_RESOLVED.md`](docs/PADDING_FIX_RESOLVED.md) | A bug whose correction grew with lookahead, and so bent the curve being measured |
| [`docs/LITERATURE.md`](docs/LITERATURE.md) | Verified citations, with corrections to earlier drafts |

## Scope

This is a measurement study, not a system paper. It characterises an axis that
streaming accent-conversion systems parameterise but do not report, and it
uses a phone-translation probe rather than an end-to-end converter. Phone
error rate is not perceptual quality, and the paper is explicit about what the
probe cannot establish.

## Citation

```bibtex
@article{tripathi2026lookahead,
  author  = {Tripathi, Dipesh},
  title   = {Future-Context Requirements for Streaming Accent Conversion:
             A Controlled Study Using Causal Phone Translation},
  year    = {2026},
  note    = {Submitted to IEEE/ACM Transactions on Audio, Speech, and
             Language Processing}
}
```
