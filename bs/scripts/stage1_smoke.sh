#!/usr/bin/env bash
# Stage 1 smoke run inside WSL: make sure BERT multilingual is cached, then score one FIV2 clip with OCEAN-AI.
set -uo pipefail
S="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts"
PY="$HOME/bs/venv/bin/python"
cd "$HOME"
echo "===== prefetch BERT ====="
"$PY" "$S/prefetch_hf.py" google-bert/bert-base-multilingual-cased 2>&1 | grep -vE "it/s|s/it"
echo "===== smoke test (oceanai) ====="
bash "$S/smoke_oceanai.sh" "$HOME/bs/eval/fi_test200" "$HOME/bs/eval/smoke_oceanai.json" oceanai
