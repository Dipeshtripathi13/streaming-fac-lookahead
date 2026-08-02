#!/usr/bin/env bash
# Dataset acquisition for the streaming-FAC lookahead study.
#
# L2-ARCTIC: the phoneme-annotated subset is now available on Hugging Face and
# needs no signed form -- only a click-through and a login:
#
#   https://huggingface.co/datasets/KoelLabs/L2Arctic     (gated, CC-BY-NC-4.0)
#   huggingface-cli login
#
# That release is 3599 scripted utterances with BOTH `g2p` (canonical phones)
# and `ipa` (produced phones) per item -- which is what makes the trainable
# RQ1/RQ3 sweep in train/train_translator.py possible. It is the ANNOTATED
# SUBSET, not the full 26,867-utterance corpus.
#
# For the full corpus (needed for the synthesis study, and for parallel
# CMU ARCTIC pairing) still request it from TAMU -- days-to-weeks turnaround, so
# start it now even though the HF subset unblocks the near-term work:
#
#   https://psi.engr.tamu.edu/l2-arctic-corpus/
#
# NOTE the licence: CC-BY-NC-4.0 is NON-COMMERCIAL. This settles the
# model-weights release question in the proposal's deliverables.
#
# Everything else below downloads without an account.

set -euo pipefail
DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "$0")" && pwd)/corpora}"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"
echo "data root: $DATA_ROOT"

have() { command -v "$1" >/dev/null 2>&1; }
have curl || { echo "need curl"; exit 1; }

# ---------------------------------------------------------------- CMU ARCTIC
# Native reference renditions of the same 1132 prompts L2-ARCTIC uses.
# bdl (US male), slt (US female), clb (US female), rms (US male) are the
# General American voices; jmk (Canadian) and awb (Scottish) are NOT General
# American and must not be used as GA targets -- an easy and fatal mistake.
CMU_VOICES="${CMU_VOICES:-bdl slt clb rms}"
mkdir -p cmu_arctic && cd cmu_arctic
for v in $CMU_VOICES; do
  f="cmu_us_${v}_arctic-0.95-release.tar.bz2"
  [ -d "cmu_us_${v}_arctic" ] && { echo "  have $v"; continue; }
  echo "  fetching CMU ARCTIC $v ..."
  curl -fL --retry 3 -O "http://festvox.org/cmu_arctic/cmu_arctic/packed/$f"
  tar xjf "$f" && rm -f "$f"
done
# canonical prompt list (needed by verify_prompt_overlap.py)
[ -f cmuarctic.data ] || curl -fL --retry 3 -O "http://festvox.org/cmu_arctic/cmuarctic.data"
cd ..

# --------------------------------------------------------------------- VCTK
# 109 speakers / 11 accents. Used to TRAIN the accent probe and to supply
# target speakers. CC BY 4.0.
if [ ! -d VCTK-Corpus-0.92 ]; then
  echo "  fetching VCTK (~11 GB, this takes a while) ..."
  curl -fL --retry 3 -o VCTK.zip \
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
  mkdir -p VCTK-Corpus-0.92 && (cd VCTK-Corpus-0.92 && unzip -q ../VCTK.zip) && rm -f VCTK.zip
fi

# ------------------------------------------------------------- Common Voice
cat <<'EOF'

Common Voice (accent-labelled, CC0) needs a click-through:
  https://commonvoice.mozilla.org/en/datasets
Download the English "Delta Segment" (small, recent) rather than the full
release -- the probe needs accent diversity, not hours. Place under:
  $DATA_ROOT/common_voice/

EdAcc (Edinburgh International Accents of English, spontaneous speech,
used only as a read-speech-overfitting guard):
  https://groups.inf.ed.ac.uk/edacc/

L2-ARCTIC, annotated subset (no form -- click-through + login):
  pip install datasets huggingface_hub && huggingface-cli login
  https://huggingface.co/datasets/KoelLabs/L2Arctic

L2-ARCTIC, FULL corpus (still a request form; needed for the synthesis study):
  https://psi.engr.tamu.edu/l2-arctic-corpus/
Unpack to:
  $DATA_ROOT/l2arctic_release_v5/

Then verify the parallel-prompt assumption before writing training code:
  python3 data/verify_prompt_overlap.py \
      --l2-arctic  $DATA_ROOT/l2arctic_release_v5 \
      --cmu-arctic $DATA_ROOT/cmu_arctic
EOF
