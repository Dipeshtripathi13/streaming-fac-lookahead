#!/bin/bash
# ============================================================================
#  Streaming FAC — t_buffer measurement.  Double-click this file in Finder.
#
#  Replaces the ~30 ms placeholder in the latency budget with a measured
#  number, and closes the stated limitation in the proposal's section 7.
#
#  Stages:
#    0  environment
#    1  estimator self-test          <- runs with no sound card; must pass
#    2  device-reported latency      <- cheap, optimistic
#    3  callback jitter              <- sets the jitter buffer
#    4  acoustic loopback            <- ground truth, needs speaker + mic
#
#  Takes about two minutes. Stage 4 will make a short chirp sound several
#  times -- that is expected, and it needs the speaker unmuted.
#
#  IMPORTANT: run on AC power. On battery, macOS downclocks and the p95
#  callback lateness is not the one your users will see.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p results/raw
LOG="results/raw/tbuffer_${TS}.log"
OUT="results/raw/tbuffer_m4.json"
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
pmset -g ps 2>/dev/null | head -1
echo
echo "Audio devices are listed in stage 2. If the defaults are wrong, set them"
echo "in System Settings > Sound before re-running."

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

# numpy is the only hard dependency of the estimators; sounddevice is needed
# for the three audio stages and needs PortAudio underneath it.
python -m pip install -q numpy 2>&1 | tail -1
if ! python -c "import sounddevice" >/dev/null 2>&1; then
  echo "installing sounddevice ..."
  if ! command -v brew >/dev/null 2>&1; then
    echo "  !! Homebrew not found; PortAudio may be missing."
    echo "  !! See https://brew.sh, then: brew install portaudio"
  else
    brew list portaudio >/dev/null 2>&1 || brew install portaudio 2>&1 | tail -3
  fi
  python -m pip install -q sounddevice 2>&1 | tail -2
fi
if python -c "import sounddevice" >/dev/null 2>&1; then
  ok "sounddevice"
  AUDIO=1
else
  echo "  !! sounddevice unavailable -- stages 2-4 will be skipped."
  echo "  !! The self-test in stage 1 still runs and still means something:"
  echo "  !! it validates the estimators, just not this machine's hardware."
  AUDIO=0
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

banner "1. ESTIMATOR SELF-TEST  (no sound card needed)"
echo "Validates cross-correlation delay recovery, low-confidence rejection,"
echo "drift-vs-jitter separation, and the no-double-counting subtraction."
if python bench/bench_tbuffer.py --self-test; then
  ok "self-test"
else
  fail "self-test"
  echo
  echo "The estimators are wrong on this machine. Stop here -- a measured"
  echo "number from a broken estimator is worse than the placeholder."
  exit 1
fi

if [ "$AUDIO" = "0" ]; then
  banner "DONE (estimators only)"
  echo "Install sounddevice and re-run to get the measured number."
  exit 0
fi

banner "2. DEVICE-REPORTED LATENCY"
python bench/bench_tbuffer.py --mode probe --blocksize-ms 40 --chunk-ms 40 \
  && ok "probe" || fail "probe"

banner "3. CALLBACK JITTER  (30 s duplex stream)"
echo "Measures how late callbacks actually arrive; the tail sets the buffer."
python bench/bench_tbuffer.py --mode jitter --seconds 30 --blocksize-ms 40 \
  && ok "jitter" || fail "jitter"

banner "4. ACOUSTIC LOOPBACK  (ground truth)"
echo "You will hear ~20 short chirps. Unmute the speaker."
echo "Default assumes ~30 cm from speaker to mic; edit --distance-m if not."
python bench/bench_tbuffer.py --mode all --reps 20 --distance-m 0.3 \
  --blocksize-ms 40 --chunk-ms 40 --out "$OUT" \
  && ok "loopback" || fail "loopback"

banner "DONE"
if [ -n "$FAILED" ]; then
  echo "STAGES THAT FAILED:$FAILED"
else
  echo "All stages completed."
fi
echo
echo "Measured t_buffer is the 'verdict' block in:"
echo "  $OUT"
echo
echo "If 'source' says 'device-reported (LOWER BOUND)', the loopback did not"
echo "succeed and the number should not be cited as measured -- check that the"
echo "speaker was unmuted and rerun stage 4."
echo
echo "Log: $LOG"
echo "This window can be closed."
