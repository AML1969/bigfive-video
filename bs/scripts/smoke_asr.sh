#!/usr/bin/env bash
# Smoke test of the real pipeline: video only, transcript from Whisper (no .txt), OCEAN-AI backend.
set -uo pipefail
EVAL_DIR="${1:-$HOME/bs/eval/fi_test200}"
OUT="${2:-$HOME/bs/eval/smoke_asr.json}"
cd "$HOME"
V=$(ls "$EVAL_DIR"/*.mp4 | head -1)
TMP=$(mktemp -d /tmp/bs_asr_XXXX)
cp "$V" "$TMP/clip.mp4"                       # no .txt next to it -> forces ASR
echo "clip: $V (copied to $TMP/clip.mp4)"
start=$(date +%s)
"$HOME/bs/venv/bin/bs" infer "$TMP/clip.mp4" --out "$OUT" 2>&1 \
  | grep -vE "SyntaxWarning|invalid escape|libEGL|DEBUG urllib3|gl_context|inference_feedback|landmark_projection|absl::InitializeLog|XNNPACK|^\s*$" | tail -30
echo "exit=${PIPESTATUS[0]} elapsed=$(( $(date +%s) - start ))s"
python3 - "$OUT" "${V%.mp4}.txt" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("ASR transcript :", d["transcript"][:300])
print("Rev transcript :", open(sys.argv[2], encoding="utf-8").read()[:300])
print({k: v["score"] for k, v in d["traits"].items()}, d["timings_sec"])
PY
rm -rf "$TMP"
