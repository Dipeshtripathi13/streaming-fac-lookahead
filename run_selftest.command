#!/bin/bash
# Fast iteration: causality proof only (~1 min). Double-click in Finder.
# The full pipeline is run_m4.command.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
mkdir -p results/raw
LOG="results/raw/selftest_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
source .venv/bin/activate 2>/dev/null || { echo "run run_m4.command first to create .venv"; exit 1; }
export TOKENIZERS_PARALLELISM=false
echo "HF token present: $([ -n "${HF_TOKEN:-}" ] || [ -f "$HOME/.cache/huggingface/token" ] && echo yes || echo NO)"
python bench/bench_content_degradation.py --selftest --tag m4
echo
echo "log: $LOG"
echo "This window can be closed."
