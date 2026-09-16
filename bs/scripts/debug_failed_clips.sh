#!/usr/bin/env bash
# Re-run OCEAN-AI verbosely on the clips it dropped during the full-test evaluation to see the library's reason.
set -uo pipefail
SRC="$HOME/bs/eval/fi_test_all"
DST="$HOME/bs/eval/failed_clips"
rm -rf "$DST"; mkdir -p "$DST"
for n in JmAQlC-FEV8.000 L-rmZZP_wj8.005 _plk5k7PBEg.004; do
  cp "$SRC/$n.mp4" "$SRC/$n.txt" "$DST/" 2>/dev/null
  echo "== $n: $(ffprobe -v error -show_entries stream=codec_type,duration -of csv=p=0 "$DST/$n.mp4" | tr '\n' ' ')"
done
cd "$HOME"
"$HOME/bs/venv/bin/python" /mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts/debug_oceanai.py "$DST" 3 2>&1 \
  | grep -vE "SyntaxWarning|invalid escape|libEGL|DEBUG urllib3|gl_context|inference_feedback|landmark_projection|absl::InitializeLog|XNNPACK|^\s*$" | tail -30
