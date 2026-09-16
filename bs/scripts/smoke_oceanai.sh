#!/usr/bin/env bash
# Smoke test: score one FIV2 clip with the OCEAN-AI backend using its Rev transcript (no ASR).
# Usage: smoke_oceanai.sh [EVAL_DIR] [OUT_JSON] [BACKEND]
set -uo pipefail
EVAL_DIR="${1:-$HOME/bs/eval/fi_test200}"
OUT="${2:-$HOME/bs/eval/smoke_oceanai.json}"
BACKEND="${3:-oceanai}"
BS="$HOME/bs/venv/bin/bs"
cd "$HOME"
V=$(ls "$EVAL_DIR"/*.mp4 | head -1)
T="${V%.mp4}.txt"
echo "clip: $V"
echo "transcript: $(head -c 120 "$T")"
start=$(date +%s)
"$BS" infer "$V" --backend "$BACKEND" --transcript "$T" --out "$OUT" -v 2>&1 | grep -vE "SyntaxWarning|invalid escape|libEGL|^\s*$" | tail -40
echo "exit=${PIPESTATUS[0]} elapsed=$(( $(date +%s) - start ))s"
[ -f "$OUT" ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['traits'], indent=1)); print('timings', d['timings_sec'])" "$OUT"
