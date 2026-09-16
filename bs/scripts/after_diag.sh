#!/usr/bin/env bash
# After the population diagnostic (GPU): recompute the Russian batch under variant A (main score = OCEAN-AI MuPTA,
# pool percentiles; stored behaviour descriptions are reused), rebuild PDFs, copy results, then resume the
# LLM-as-rater experiment (Ollama VLM). Usage: after_diag.sh [MODEL] [FRAMES]
set -uo pipefail
S="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts"
PY="$HOME/bs/venv/bin/python"
OUT="$HOME/bs/ru_runs"
LOCAL="$HOME/bs/video_ru"
DST_WIN="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/results/video_ru"
FILT="SyntaxWarning|invalid escape|libEGL|gl_context|inference_feedback|landmark_projection|XNNPACK|absl|FutureWarning|warnings.warn|newly initialized|TRAIN this|forced_decoder_ids|attention mask|Token indices|^\s*$"
cd "$HOME"
export PYTHONWARNINGS=ignore
while pgrep -f 'diag_ru_populatio[n]' >/dev/null; do sleep 30; done
echo "diagnostic finished at $(date +%H:%M:%S)"

echo "=== batch under variant A (force, descriptions reused)"
for attempt in 1 2 3; do
  "$PY" "$S/batch_ru.py" "$LOCAL" "$OUT" --lang ru $( [ "$attempt" = 1 ] && echo --force ) 2>&1 | grep -vE "$FILT" | sed "s/\x1b\[2K//g" | cut -c1-180
  rc=${PIPESTATUS[0]}
  echo "batch_ru.py attempt $attempt exited with $rc"
  [ "$rc" -eq 0 ] && break
  sleep 5
done
echo "=== PDF reports (explanations recomputed on the new representative segments)"
"$PY" "$S/make_batch_pdfs.py" "$OUT" "$LOCAL" --lang ru --force 2>&1 | grep -vE "$FILT" | cut -c1-200
rsync -a --exclude='*.mp4' --exclude='*.wav' "$OUT/" "$DST_WIN/"
cp "$OUT"/diag_population.* "$DST_WIN/" 2>/dev/null
echo "batch A done at $(date +%H:%M:%S)"

echo "=== rater experiment ${1:-qwen3-vl:30b}"
bash "$S/run_rater_experiment.sh" "${1:-qwen3-vl:30b}" "${2:-16}" 2>&1 | grep -vE "^\s*$" | cut -c1-200
