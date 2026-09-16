#!/usr/bin/env bash
# Second ablation arm after after_all.sh: Ollama's Qwen3-VL spends ~1 050 tokens per frame regardless of the frame
# resolution, so 32 frames (33.5k tokens) do not fit the 32k context -> use 28 frames at 640 px instead.
# Then rebuild the comparison table (16 / 8 / 28 frames) and copy the results. Usage: after_all2.sh [MODEL]
set -uo pipefail
MODEL="${1:-qwen3-vl:30b}"
TAG="$(echo "$MODEL" | tr ':/' '__')"
S="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts"
PY="$HOME/bs/venv/bin/python"
OUT="$HOME/bs/eval/results"
T="$HOME/bs/eval/fi_test200"
D="$HOME/bs/eval/fi_dev200"
DST_WIN="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/results/rater"
cd "$HOME"
export PYTHONWARNINGS=ignore
sleep 60
while pgrep -f 'after_al[l].sh|llm_rate[r]' >/dev/null; do sleep 60; done
echo "after_all finished at $(date +%H:%M:%S)"
start=$(date +%s)
F=28; SIDE=640
echo "===== ablation: $F frames @ ${SIDE}px ====="
"$PY" "$S/llm_rater.py" "$T" "$OUT/rater_${TAG}_f${F}_s${SIDE}_test200.csv" --model "$MODEL" --frames "$F" --max-side "$SIDE" 2>&1 | grep -vE "SyntaxWarning|invalid escape"
"$PY" "$S/rater_eval.py" "$T/labels.csv" "$OUT/rater_${TAG}_f${F}_s${SIDE}_test200.csv" "$D/labels.csv" "$OUT/rater_${TAG}_dev200.csv" \
  "$OUT/oceanai_fi_test_all.pred.csv" "$HOME/bs/mm_runs_seeds/test_pred_mean.csv" "$OUT/rater_${TAG}_f${F}_s${SIDE}_eval.json" 2>&1 | grep -vE "SyntaxWarning|invalid escape" | tail -14
"$PY" "$S/rater_ablation_summary.py" "$OUT/rater_${TAG}_ablation.md" "16f@640=$OUT/rater_${TAG}_eval.json" \
  "8f@640=$OUT/rater_${TAG}_f8_s640_eval.json" "28f@640=$OUT/rater_${TAG}_f28_s640_eval.json"
cp "$OUT"/rater_* "$DST_WIN/" 2>/dev/null
curl -s --max-time 30 -X POST http://172.26.80.1:11434/api/generate -d "{\"model\":\"$MODEL\",\"keep_alive\":0}" >/dev/null
echo "ablation 2 done in $(( $(date +%s) - start ))s; results in $DST_WIN; Ollama model unloaded"
