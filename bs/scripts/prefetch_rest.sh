#!/usr/bin/env bash
# Prefetch the remaining Hugging Face models used by the two backends (run after the smoke test).
set -uo pipefail
S="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts"
PY="$HOME/bs/venv/bin/python"
cd "$HOME"
"$PY" "$S/prefetch_hf.py" \
  openai/clip-vit-base-patch32 \
  BAAI/bge-small-en-v1.5 \
  audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim \
  openai/whisper-large-v3-turbo 2>&1 | grep -vE "it/s|s/it"
du -sh "$HOME"/.cache/huggingface/hub/models--* | sort -h
