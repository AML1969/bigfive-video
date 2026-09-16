#!/usr/bin/env bash
# Wait for run_batch_ru.sh to finish, then: (1) replay the segments around the first CUDA assert of the
# Seraphim video with CUDA_LAUNCH_BLOCKING=1 (debug log), (2) build PDF reports for all videos, (3) copy results.
set -uo pipefail
S="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts"
PY="$HOME/bs/venv/bin/python"
OUT="$HOME/bs/ru_runs"
LOCAL="$HOME/bs/video_ru"
DST_WIN="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/results/video_ru"
LOG="$HOME/bs/logs"
FILT="SyntaxWarning|invalid escape|libEGL|gl_context|inference_feedback|landmark_projection|XNNPACK|absl|FutureWarning|warnings.warn|newly initialized|TRAIN this|forced_decoder_ids|attention mask|Token indices|^\s*$"
cd "$HOME"
export PYTHONWARNINGS=ignore
while pgrep -f 'run_batch_r[u].sh' >/dev/null; do sleep 30; done
echo "batch finished at $(date +%H:%M:%S)"
bash "$S/batch_status.sh" "$OUT"

echo "=== debug replay Seraphim segments 18-21 (CUDA_LAUNCH_BLOCKING=1) -> $LOG/debug_seraphim.log"
CUDA_LAUNCH_BLOCKING=1 "$PY" "$S/debug_segments.py" "$OUT/Seraphim-short-interview" 18 19 20 21 --lang ru 2>&1 \
  | grep -vE "$FILT" > "$LOG/debug_seraphim.log"
grep -aE "^=====|^  (mm|oceanai):|Error|assert" "$LOG/debug_seraphim.log" | cut -c1-200

echo "=== PDF reports"
"$PY" "$S/make_batch_pdfs.py" "$OUT" "$LOCAL" --lang ru 2>&1 | grep -vE "$FILT" | cut -c1-200

rsync -a --exclude='*.mp4' --exclude='*.wav' "$OUT/" "$DST_WIN/"
echo "after_batch done at $(date +%H:%M:%S); results in $DST_WIN"
