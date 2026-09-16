#!/usr/bin/env bash
# Copy the Russian videos to ext4 (fast seeking for segment cuts) and run the batch analysis.
# Usage: run_batch_ru.sh [SRC_WIN_DIR_IN_WSL] [--explain]
set -uo pipefail
SRC="${1:-/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/video_ru}"
EXTRA="${2:-}"
LOCAL="$HOME/bs/video_ru"
OUT="$HOME/bs/ru_runs"
DST_WIN="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/results/video_ru"
PY="$HOME/bs/venv/bin/python"
S="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts"
mkdir -p "$LOCAL" "$OUT" "$DST_WIN"
cd "$HOME"
export PYTHONWARNINGS=ignore
start=$(date +%s)
rsync -a --include='*.mp4' --include='*.MP4' --include='*.mov' --exclude='*' "$SRC/" "$LOCAL/"
echo "copied $(ls "$LOCAL" | wc -l) files in $(( $(date +%s) - start ))s"
FILT="SyntaxWarning|invalid escape|libEGL|gl_context|inference_feedback|landmark_projection|XNNPACK|absl|FutureWarning|warnings.warn|newly initialized|TRAIN this|forced_decoder_ids|attention mask|Token indices|^\s*$"
# a sticky CUDA error kills the process (exit 3); restart a clean one, finished videos are cached
for attempt in 1 2 3 4 5; do
  "$PY" "$S/batch_ru.py" "$LOCAL" "$OUT" --lang ru $EXTRA 2>&1 | grep -vE "$FILT"
  rc=${PIPESTATUS[0]}
  echo "batch_ru.py attempt $attempt exited with $rc"
  [ "$rc" -eq 0 ] && break
  sleep 5
done
# copy result.json + summary + timelines (no video segments) into the project
rsync -a --exclude='*.mp4' --exclude='*.wav' "$OUT/" "$DST_WIN/"
echo "batch done in $(( $(date +%s) - start ))s; results in $DST_WIN"
