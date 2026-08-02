#!/bin/bash
# Re-run the two fast experiments after a code change. Double-click in Finder.
#   - content-degradation pilot on real L2-ARCTIC  (~40 s once cached)
#   - commit-delay measurement                     (~60 s)
# Full pipeline is run_m4.command; causality proof alone is run_selftest.command.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
mkdir -p results/raw results/figures
LOG="results/raw/quick_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
source .venv/bin/activate 2>/dev/null || { echo "run run_m4.command first"; exit 1; }
export TOKENIZERS_PARALLELISM=false

echo "=== content-degradation pilot (real L2-ARCTIC) ==="
python bench/bench_content_degradation.py --n-utts 48 --chunk-ms 40 --device cpu --tag m4

echo
echo "=== commit delay ==="
python bench/bench_commit_delay.py --threads 1 --tag zipformer

echo
echo "=== figures ==="
python bench/make_figures.py

echo
echo "log: $LOG"
echo "This window can be closed."
