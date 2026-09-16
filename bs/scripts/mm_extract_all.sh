#!/usr/bin/env bash
# Extract all MM-PSYCHE-style features for FIV2 (train, dev, test) into ~/data/fiv2/features.
# Face extraction (video decoding + MediaPipe + CLIP) runs in N parallel shards per split.
# Usage: mm_extract_all.sh [N_SHARDS] [SPLITS...]
set -uo pipefail
N="${1:-6}"
shift 1 2>/dev/null || true
SPLITS=("$@"); [ ${#SPLITS[@]} -eq 0 ] && SPLITS=(test dev train)
PY="$HOME/bs/venv/bin/python"
LOG="$HOME/bs/logs/mm_extract"; mkdir -p "$LOG"
cd "$HOME"
export PYTHONWARNINGS=ignore
start=$(date +%s)
for split in "${SPLITS[@]}"; do
  echo "===== $split: audio + text + behavior ====="
  "$PY" -m bs_bigfive.mm.extract --split "$split" --modalities audio,text,behavior --threads 8 \
      2>&1 | grep -vE "SyntaxWarning|invalid escape|libEGL|gl_context|inference_feedback|XNNPACK|absl" | tee "$LOG/${split}_atb.log" | grep -E "INFO bs|WARNING|Error" | tail -20
  echo "===== $split: face in $N shards ====="
  for i in $(seq 0 $((N-1))); do
    "$PY" -m bs_bigfive.mm.extract --split "$split" --modalities face --shard "$i/$N" --threads 3 \
        > "$LOG/${split}_face_$i.log" 2>&1 &
  done
  wait
  grep -hE "stats:|shard .* saved|WARNING" "$LOG"/${split}_face_*.log | tail -"$((N+2))"
  "$PY" -m bs_bigfive.mm.extract --split "$split" --modalities face --merge 2>&1 | grep -E "INFO bs|Error" | tail -3
  echo "----- $split done at $(( $(date +%s) - start ))s -----"
done
ls -la "$HOME/data/fiv2/features/"*/
echo "extract done in $(( $(date +%s) - start ))s"
