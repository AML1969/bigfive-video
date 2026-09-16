#!/usr/bin/env bash
# LLM-as-rater experiment: build a 200-clip dev set, rate test200 and dev200 with an Ollama VLM, evaluate.
# Usage: run_rater_experiment.sh [MODEL] [FRAMES]
set -uo pipefail
MODEL="${1:-qwen3-vl:30b}"
FRAMES="${2:-16}"
TAG="$(echo "$MODEL" | tr ':/' '__')"
S="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts"
CSV="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/MM-PSYCHE/data/fiv2"
PY="$HOME/bs/venv/bin/python"
OUT="$HOME/bs/eval/results"
cd "$HOME"
export PYTHONWARNINGS=ignore
start=$(date +%s)
echo "===== dev200 eval set ====="
python3 "$S/make_eval_set.py" --src "$HOME/data/fiv2/video/dev" --csv "$CSV/dev_full_with_description.csv" --dst "$HOME/bs/eval/fi_dev200" --n 200
echo "===== rating test200 with $MODEL ====="
"$PY" "$S/llm_rater.py" "$HOME/bs/eval/fi_test200" "$OUT/rater_${TAG}_test200.csv" --model "$MODEL" --frames "$FRAMES" 2>&1 | grep -vE "SyntaxWarning|invalid escape"
echo "===== rating dev200 with $MODEL ====="
"$PY" "$S/llm_rater.py" "$HOME/bs/eval/fi_dev200" "$OUT/rater_${TAG}_dev200.csv" --model "$MODEL" --frames "$FRAMES" 2>&1 | grep -vE "SyntaxWarning|invalid escape"
echo "===== evaluation ====="
"$PY" "$S/rater_eval.py" "$HOME/bs/eval/fi_test200/labels.csv" "$OUT/rater_${TAG}_test200.csv" \
  "$HOME/bs/eval/fi_dev200/labels.csv" "$OUT/rater_${TAG}_dev200.csv" \
  "$OUT/oceanai_fi_test_all.pred.csv" "$HOME/bs/mm_runs_seeds/test_pred_mean.csv" "$OUT/rater_${TAG}_eval.json" 2>&1 | grep -vE "SyntaxWarning|invalid escape"
echo "rater experiment done in $(( $(date +%s) - start ))s"
