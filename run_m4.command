#!/bin/bash
# ============================================================================
#  Streaming FAC — full M4 run.  Double-click this file in Finder.
#
#  Stages, each independently resumable. If one fails the rest still run and
#  the log says exactly what broke.
#
#    0  environment
#    1  unit tests + invariant guards
#    2  hardware probe (the cpu-apple-silicon row)
#    3  encoder scaling benchmark
#    4  cascade benchmark (S1)
#    5  causality self-test  <- proves masked WavLM is / isn't actually causal
#    6  content-degradation pilot on real L2-ARCTIC  <- the RQ1 evidence
#    7  figures
#
#  Everything is logged to results/raw/run_m4_<timestamp>.log
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p results/raw results/figures
LOG="results/raw/run_m4_${TS}.log"
TAG="m4"

exec > >(tee -a "$LOG") 2>&1

banner() { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }
ok()     { echo "  [OK]   $*"; }
fail()   { echo "  [FAIL] $*"; FAILED="${FAILED:-} $1"; }
FAILED=""

banner "0. ENVIRONMENT"
echo "repo   : $ROOT"
echo "log    : $LOG"
sw_vers 2>/dev/null | tr '\n' ' '; echo
sysctl -n machdep.cpu.brand_string 2>/dev/null
echo "P-cores: $(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null)  E-cores: $(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null)"
pmset -g ps 2>/dev/null | head -1
echo
echo "NOTE: benchmarks should run on AC power. Battery = downclocked = wrong p95."

PY=""
for c in python3.11 python3.12 python3; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
[ -z "$PY" ] && { echo "no python3 found. Install: brew install python@3.11"; exit 1; }
echo "python : $PY ($($PY -V 2>&1))"

if [ ! -d .venv ]; then
  echo "creating .venv ..."
  "$PY" -m venv .venv || { echo "venv creation failed"; exit 1; }
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q -U pip wheel 2>&1 | tail -1

echo "installing dependencies (first run downloads ~2 GB, later runs are instant) ..."
python -m pip install -q \
  numpy scipy soundfile onnxruntime sherpa-onnx matplotlib pandas \
  torch transformers datasets huggingface_hub 2>&1 | tail -3
ok "dependencies"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
# MPS has gaps in coverage and silently falls back; keep the CPU row honest.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Keep the benchmark on P-cores. The accepted QoS clamp names differ across
# macOS releases (older builds take "user-interactive"; 26.x rejects it), so
# probe rather than assume -- a silent taskpolicy failure would otherwise abort
# the whole stage, which is exactly what happened on the first run.
QOS=""
for clamp in userinteractive user-interactive utility; do
  if taskpolicy -c "$clamp" true >/dev/null 2>&1; then QOS="taskpolicy -c $clamp"; break; fi
done
if [ -z "$QOS" ]; then
  echo "QoS clamp: unavailable on this macOS build; running at default priority."
else
  echo "QoS clamp: $QOS"
fi

banner "1. UNIT TESTS AND INVARIANT GUARDS"
for t in tests/test_causal.py tests/test_pipeline_invariants.py eval/metrics.py eval/phoneme_analysis.py; do
  if python "$t" >/dev/null 2>&1; then ok "$t"; else fail "$t"; python "$t" 2>&1 | tail -15; fi
done

banner "2. HARDWARE PROBE"
python bench/hardware_probe.py --out "results/raw/hw_${TAG}.json" | tail -25 && ok "hardware probe" || fail "hardware probe"

banner "3. ENCODER SCALING SWEEP  (compute vs lookahead x chunk x model size)"
if $QOS python bench/bench_encoder_scaling.py \
     --preset all --reps 30 --out-prefix "results/raw/encoder_scaling_${TAG}"; then
  ok "encoder scaling"
else
  fail "encoder scaling"
fi

banner "4. CASCADE BENCHMARK (S1)"
if [ -d ../accentbridge/models ]; then
  $QOS python bench/bench_cascade_onnx.py \
    --threads 1 2 4 8 --skip-fp32 --seconds 30 \
    --out-prefix "results/raw/cascade_${TAG}" && ok "cascade" || fail "cascade"
else
  echo "  skipped: ../accentbridge/models not found"
fi

banner "5. CAUSALITY SELF-TEST"
echo "Does masking attention alone make WavLM causal? (spoiler: no -- pos_conv)"
python bench/bench_content_degradation.py --selftest --tag "$TAG" && ok "causality self-test" || fail "causality self-test"

banner "6. CONTENT-DEGRADATION PILOT ON REAL L2-ARCTIC  (the RQ1 evidence)"
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
  echo "  !! No Hugging Face token found."
  echo "  !! KoelLabs/L2Arctic is gated. Accept terms at"
  echo "  !!   https://huggingface.co/datasets/KoelLabs/L2Arctic"
  echo "  !! then run:  source .venv/bin/activate && huggingface-cli login"
  echo "  !! Falling back to synthetic audio so the pipeline is still exercised."
  python bench/bench_content_degradation.py --n-utts 12 --synthetic \
    --tag "${TAG}_synthetic" && ok "pilot (synthetic)" || fail "pilot (synthetic)"
else
  python bench/bench_content_degradation.py --n-utts 48 --chunk-ms 40 \
    --device cpu --tag "$TAG" && ok "content-degradation pilot" || fail "content-degradation pilot"
fi

banner "6b. COMMIT-DELAY MEASUREMENT (cascade's dominant term)"
echo "How long until a streaming ASR stops revising a word? Needs no dataset."
if [ -d ../accentbridge/models ]; then
  python bench/bench_commit_delay.py --threads 1 --tag zipformer \
    && ok "commit delay" || fail "commit delay"
else
  echo "  skipped: ../accentbridge/models not found"
fi

banner "7. TRANSLATOR SMOKE TEST"
echo "Proves the training pipeline runs end-to-end (60 steps, 3 lookaheads)."
echo "The full 14-condition sweep belongs on a GPU -- see notebooks/."
if python -c "import sys;sys.path.insert(0,'src');import runpy;runpy.run_module('sfac.translator',run_name='__main__')"; then
  ok "translator self-test"
else
  fail "translator self-test"
fi
if [ -n "${HF_TOKEN:-}" ] || [ -f "$HOME/.cache/huggingface/token" ]; then
  python train/train_translator.py --smoke --device cpu \
    --out "results/raw/translator_smoke_${TAG}" && ok "translator smoke" || fail "translator smoke"
  # Opt-in reduced CPU sweep. Hours on an M4 -- only if you ask for it.
  if [ "${RUN_CPU_SWEEP:-0}" = "1" ]; then
    python train/train_translator.py --lookaheads 0 40 80 160 640 \
      --targets native produced --steps 1500 --device cpu \
      --out "results/raw/translator_sweep_${TAG}" && ok "cpu sweep" || fail "cpu sweep"
  else
    echo "  (set RUN_CPU_SWEEP=1 to also run a reduced 10-condition CPU sweep)"
  fi
else
  echo "  skipped: no Hugging Face token (see stage 6)"
fi

banner "8. FIGURES"
python bench/make_figures.py && ok "figures" || fail "figures"

banner "DONE"
if [ -n "$FAILED" ]; then
  echo "STAGES THAT FAILED:$FAILED"
  echo "Full log: $LOG"
else
  echo "All stages completed."
fi
echo
echo "Results:"
ls -la results/raw/ | tail -25
echo
echo "This window can be closed."
