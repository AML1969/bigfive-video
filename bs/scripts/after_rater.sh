#!/usr/bin/env bash
# Wait for the LLM-rater experiment (Ollama, ~20 GB VRAM) to finish, then run the population diagnostic on the
# Russian batch (OCEAN-AI FIV2 weights + own model without audio) and copy the result into the project.
set -uo pipefail
S="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts"
PY="$HOME/bs/venv/bin/python"
DST_WIN="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/results/video_ru"
cd "$HOME"
export PYTHONWARNINGS=ignore
sleep 120
while pgrep -f 'run_rater_experimen[t]|after_dia[g]' >/dev/null; do sleep 60; done
echo "rater finished at $(date +%H:%M:%S); running diagnostic"
"$PY" "$S/diag_ru_population.py" "$HOME/bs/ru_runs" 2>&1 | sed "s/\x1b\[[0-9;]*[A-Za-z]//g" \
  | grep -aE "^[A-Za-z0-9_-]+: |^    |^\||^#|Error|Traceback|File \"/mnt" | cut -c1-200
cp "$HOME/bs/ru_runs"/diag_population.* "$DST_WIN/" 2>/dev/null
echo "diagnostic done at $(date +%H:%M:%S)"
