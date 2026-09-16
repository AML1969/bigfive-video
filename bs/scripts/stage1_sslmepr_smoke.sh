#!/usr/bin/env bash
# Make sure the SSL-MEPR encoders are cached (CLIP, BGE, wav2vec2), then smoke-test the sslmepr backend on one clip.
set -uo pipefail
S="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts"
PY="$HOME/bs/venv/bin/python"
cd "$HOME"
echo "===== prefetch encoders ====="
"$PY" "$S/prefetch_hf.py" openai/clip-vit-base-patch32 BAAI/bge-small-en-v1.5 audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim 2>&1 | grep -vE "it/s|s/it"
echo "===== smoke test (sslmepr) ====="
bash "$S/smoke_oceanai.sh" "$HOME/bs/eval/fi_test200" "$HOME/bs/eval/smoke_sslmepr.json" sslmepr 2>&1 \
  | grep -vE "DEBUG urllib3|gl_context|inference_feedback|landmark_projection|absl::InitializeLog|XNNPACK"
