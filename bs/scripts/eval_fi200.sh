#!/usr/bin/env bash
# Evaluate a backend on a FIV2 eval folder (<stem>.mp4 + <stem>.txt Rev transcripts + labels.csv).
# Usage: eval_fi200.sh [oceanai|sslmepr] [EVAL_DIR] [EXTRA_ARGS...]   (e.g. --asr to use Whisper instead of .txt)
# Output: ~/bs/eval/results/<backend>_<eval-dir-name>[_asr].json / .pred.csv / .log
set -uo pipefail
BACKEND="${1:-oceanai}"
EVAL_DIR="${2:-$HOME/bs/eval/fi_test200}"
shift 2 2>/dev/null || shift $#
OUT_DIR="$HOME/bs/eval/results"
mkdir -p "$OUT_DIR"
cd "$HOME"
NAME="${BACKEND}_$(basename "$EVAL_DIR")"
case " $* " in *" --asr "*) NAME="${NAME}_asr";; esac
LOG="$OUT_DIR/$NAME.log"
start=$(date +%s)
"$HOME/bs/venv/bin/bs" eval-fiv2 --backend "$BACKEND" --dir "$EVAL_DIR" --out "$OUT_DIR/$NAME.json" "$@" \
  2>&1 | grep -vE "SyntaxWarning|invalid escape|libEGL|DEBUG urllib3|gl_context|inference_feedback|landmark_projection|absl::InitializeLog|XNNPACK" | tee "$LOG" | tail -40
echo "exit=${PIPESTATUS[0]} elapsed=$(( $(date +%s) - start ))s"
