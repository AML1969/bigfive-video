#!/usr/bin/env bash
# Runs after the rater experiment and the population diagnostic: frame-count ablation of the LLM rater on
# test200 (32 frames at 448 px, 8 frames at 640 px; the 16-frame run is the baseline), evaluation of each with
# the same dev200 calibration, a comparison table, and a copy of all rater results into the project.
# Usage: after_all.sh [MODEL]
set -uo pipefail
MODEL="${1:-qwen3-vl:30b}"
TAG="$(echo "$MODEL" | tr ':/' '__')"
S="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts"
PY="$HOME/bs/venv/bin/python"
OUT="$HOME/bs/eval/results"
T="$HOME/bs/eval/fi_test200"
D="$HOME/bs/eval/fi_dev200"
DST_WIN="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/results/rater"
mkdir -p "$DST_WIN"
cd "$HOME"
export PYTHONWARNINGS=ignore
sleep 180
while pgrep -f 'after_dia[g]|after_rate[r]|run_rater_experimen[t]|diag_ru_populatio[n]' >/dev/null; do sleep 60; done
echo "previous chains finished at $(date +%H:%M:%S)"
if [ ! -f "$OUT/rater_${TAG}_dev200.csv" ]; then echo "no dev200 rater csv -> main experiment did not finish; stop"; exit 1; fi
start=$(date +%s)
for cfg in "32 448" "8 640"; do
  set -- $cfg; F=$1; SIDE=$2
  echo "===== ablation: $F frames @ ${SIDE}px ====="
  "$PY" "$S/llm_rater.py" "$T" "$OUT/rater_${TAG}_f${F}_s${SIDE}_test200.csv" --model "$MODEL" --frames "$F" --max-side "$SIDE" 2>&1 | grep -vE "SyntaxWarning|invalid escape"
  "$PY" "$S/rater_eval.py" "$T/labels.csv" "$OUT/rater_${TAG}_f${F}_s${SIDE}_test200.csv" "$D/labels.csv" "$OUT/rater_${TAG}_dev200.csv" \
    "$OUT/oceanai_fi_test_all.pred.csv" "$HOME/bs/mm_runs_seeds/test_pred_mean.csv" "$OUT/rater_${TAG}_f${F}_s${SIDE}_eval.json" 2>&1 | grep -vE "SyntaxWarning|invalid escape" | tail -14
done
"$PY" "$S/rater_ablation_summary.py" "$OUT/rater_${TAG}_ablation.md" "16f@640=$OUT/rater_${TAG}_eval.json" \
  "32f@448=$OUT/rater_${TAG}_f32_s448_eval.json" "8f@640=$OUT/rater_${TAG}_f8_s640_eval.json"
cp "$OUT"/rater_* "$DST_WIN/" 2>/dev/null
echo "ablation done in $(( $(date +%s) - start ))s; results in $DST_WIN"
