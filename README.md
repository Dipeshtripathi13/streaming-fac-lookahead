# Per-Layer Lookahead Masks Leak Future Context

How much future context does streaming foreign accent conversion need? This
repository set out to measure that with a causal phone-translation probe over
a frozen self-supervised encoder, and found on the way that the usual way of
imposing a lookahead budget does not impose it. A lookahead mask applied in
every layer compounds with depth, so the model sees far more of the future
than its label says. This repository holds the harness, the tests that detect
the problem, the corrected measurement and all raw results.

**Paper:** *Per-Layer Lookahead Masks Leak Future Context: Re-measuring the
Future-Context Requirement of Streaming Accent Conversion.*
[`paper/v2/paper.pdf`](paper/v2/paper.pdf)

> **The first arXiv version and the TASLP submission
> ([`paper/taslp/`](paper/taslp/)) are superseded.** Their headline results
> were measured with compounding per-layer masks and do not hold under exact
> budgets. See [`docs/E2E_RERUN.md`](docs/E2E_RERUN.md).

---

## What the study found

**1. Per-layer masks leak.** The model is a frozen WavLM base+ encoder tapped
at layer 9 plus a four-block probe. Measured effective future context:

| nominal lookahead | per-layer mask | exact schemes |
|---:|---:|---:|
| 20 ms | at least 200 ms | 40 ms |
| 40 ms | at least 220 ms | 60 ms |
| 160 ms | at least 780 ms | 180 ms |
| 640 ms | at least 2.8 s | 660 ms |

(Nominal = rest of the 40 ms chunk + lookahead. Per-layer values are lower
bounds.)

**2. With exact budgets the benefit of lookahead is small.** Phone error rate,
canonical-phone probe, WavLM base+, mean of two seeds:

| scheme | 0 ms | 80 ms | 640 ms | reduction 0 to 640 ms |
|---|---:|---:|---:|---:|
| per-layer (leaks) | 0.426 | 0.275 | 0.156 | 63.3% |
| encoder-first (exact, lookahead in encoder layer 0) | 0.424 | 0.388 | 0.371 | 12.5% |
| end-to-end (exact, lookahead after the encoder) | 0.452 | 0.434 | 0.435 | 3.7% |

Under end-to-end lookahead nearly all of the gain arrives within 80 ms, and
longer training (6000 steps) raises it only to 7% at 240 ms.

**3. Where the context enters matters.** Letting encoder frames see the rest
of their own 40 ms chunk is worth 6% on its own, more than 640 ms of lookahead
added after the encoder.

**4. Two earlier findings do not survive exact budgets.** Canonical-phone
prediction no longer benefits measurably more than realized-phone
transcription, and WavLM and wav2vec 2.0 no longer differ; both effects came
from different amounts of leaked context.

**5. Also measured.** Masking attention does not make a base-size encoder
causal (the positional convolution and the feature-encoder GroupNorm both
leak, and both are patched). Per-chunk compute is nearly flat in lookahead.

---

## Repository layout

```
src/sfac/          lookahead masks (all three schemes), frame arithmetic, latency
bench/             causality audit, encoder scaling, compute benchmarks
train/             the causal phone translator and its lookahead sweep
eval/              analyses, including the effective-context test
notebooks/         Colab notebook that re-runs every training experiment
results/raw/       per-condition outputs; results/raw/e2e/ holds the exact-budget runs
paper/v2/          the current manuscript and its figure script
paper/taslp/, arxiv/, taslp/   superseded first version and its packages
docs/              design decisions, findings log, and the re-run analysis
```

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/cpu.txt

# encoder causality at L = 0 (truncation test)
python3 bench/bench_content_degradation.py --selftest

# effective future context under all three schemes; the exact schemes must
# match the nominal budget on every row
python3 eval/effective_lookahead.py

python3 tests/test_causal.py && python3 tests/test_pipeline_invariants.py
```

Training needs a GPU and the gated L2-ARCTIC release. Run
[`notebooks/colab_rerun_end_to_end.ipynb`](notebooks/colab_rerun_end_to_end.ipynb)
on a Colab T4, or call the trainer directly:

```bash
python3 train/train_translator.py --lookahead-scope end_to_end \
    --lookaheads 0 20 40 80 160 320 640 --targets native --seeds 1337 7 \
    --steps 1200 --device cuda --out results/raw/my_sweep
```

`--lookahead-scope` is `end_to_end` (default), `encoder_first`, or
`per_layer`. The last reproduces the first version and leaks future context;
use it only for comparison. Before training, the exact scopes run the
effective-context test for every budget and abort if any label is wrong.

## Data and licence

Speech data is [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/), used
via the `KoelLabs/L2Arctic` release under CC-BY-NC-4.0. The corpus is not
redistributed here. Given that licence we release configurations and
per-condition outputs rather than model weights.

Code in this repository is under the licence in [`LICENSE`](LICENSE).

## Reading further

| | |
|---|---|
| [`docs/E2E_RERUN.md`](docs/E2E_RERUN.md) | The exact-budget re-run, encoder-first experiment and what each overturns |
| [`docs/FINDINGS_LOG.md`](docs/FINDINGS_LOG.md) | Every measurement in the order it was made, superseded entries included |
| [`docs/PADDING_FIX_RESOLVED.md`](docs/PADDING_FIX_RESOLVED.md) | A bug whose correction grew with lookahead |
| [`docs/LITERATURE.md`](docs/LITERATURE.md) | Verified citations |

## Scope

This is a measurement study with a phone-translation probe, not a conversion
system. Phone error rate is not perceived accentedness, the encoder is frozen,
and the two exact masking schemes are two of many possible designs. The
general lesson does not depend on those limits: a lookahead result should
come with a measured effective context.

## Citation

```bibtex
@misc{tripathi2026lookaheadleak,
  author = {Tripathi, Dipesh},
  title  = {Per-Layer Lookahead Masks Leak Future Context: Re-measuring the
            Future-Context Requirement of Streaming Accent Conversion},
  year   = {2026},
  note   = {Manuscript}
}
```
