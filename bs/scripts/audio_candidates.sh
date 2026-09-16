#!/usr/bin/env bash
# Overnight comparison of language-independent speech encoders against CLAP in the own model:
# for every candidate: extract features for train/dev/test, train 5 seeds with the candidate replacing CLAP and
# 5 seeds with the candidate added next to CLAP, then one comparison table. emotion2vec+ runs in its own venv
# (funasr) and is last, so a failure there cannot block the others.
# Usage: audio_candidates.sh [CANDIDATES...]   default: audio_whisper audio_xlsr audio_egemaps audio_w2v_emo audio_e2v
set -uo pipefail
CANDS=("$@"); [ ${#CANDS[@]} -eq 0 ] && CANDS=(audio_whisper audio_xlsr audio_egemaps audio_w2v_emo audio_e2v)
S="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts"
PY="$HOME/bs/venv/bin/python"
ROOT="$HOME/data/fiv2"
RUN="$HOME/bs/mm_runs_audio"
BASE="$HOME/bs/mm_runs_seeds"
DST_WIN="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/results/mm_runs_audio"
LOG="$HOME/bs/logs/audio_candidates"; mkdir -p "$LOG" "$RUN" "$DST_WIN"
FILT="SyntaxWarning|invalid escape|libEGL|gl_context|XNNPACK|absl|FutureWarning|warnings.warn|newly initialized|TRAIN this|^\s*$"
cd "$HOME"
export PYTHONWARNINGS=ignore
start=$(date +%s)

extract_std() {   # candidate: extract three splits with the main venv
  local c="$1"
  for split in train dev test; do
    if [ -f "$ROOT/features/$split/$c.pt" ]; then echo "  $split/$c exists"; continue; fi
    "$PY" -m bs_bigfive.mm.extract --split "$split" --modalities "$c" --threads 8 2>&1 | grep -vE "$FILT" \
      | tee "$LOG/${split}_$c.log" | grep -E "stats:|saved|Error|Traceback" | tail -3 | cut -c1-200
  done
}

extract_e2v() {
  local V="$HOME/bs/venv_e2v"
  if [ ! -x "$V/bin/python" ]; then
    echo "  creating venv for funasr"
    python3 -m venv "$V" && "$V/bin/pip" install -q --upgrade pip \
      && "$V/bin/pip" install -q torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130 \
      && "$V/bin/pip" install -q funasr modelscope soundfile soxr "numpy<2" 2>&1 | tail -3
  fi
  for split in train dev test; do
    if [ -f "$ROOT/features/$split/audio_e2v.pt" ]; then echo "  $split/audio_e2v exists"; continue; fi
    "$V/bin/python" "$S/extract_e2v.py" "$ROOT" "$split" 2>&1 | grep -vE "$FILT" | tee "$LOG/${split}_audio_e2v.log" \
      | grep -E "loaded|saved|failed|Error|Traceback" | tail -4 | cut -c1-200
  done
}

train_both() {   # candidate -> replace and add configurations, 5 seeds each
  local c="$1"
  for split in train dev test; do [ -f "$ROOT/features/$split/$c.pt" ] || { echo "  no features for $split/$c, skip training"; return; }; done
  echo "  --- train: $c replaces CLAP"
  bash "$S/mm_train_seeds.sh" "face,$c,text,behavior" big5+interview "$RUN/${c}_replace" 2>&1 | grep -vE "$FILT" | tail -8 | cut -c1-160
  echo "  --- train: $c next to CLAP"
  bash "$S/mm_train_seeds.sh" "face,audio,$c,text,behavior" big5+interview "$RUN/${c}_add" 2>&1 | grep -vE "$FILT" | tail -8 | cut -c1-160
}

for c in "${CANDS[@]}"; do
  echo "===== $c ($(date +%H:%M:%S)) ====="
  if [ "$c" = "audio_e2v" ]; then extract_e2v; else extract_std "$c"; fi
  if [ "$c" = "audio_egemaps" ]; then "$PY" "$S/standardize_features.py" "$ROOT" audio_egemaps 2>&1 | grep -vE "$FILT" | tail -3; fi
  train_both "$c"
  "$PY" "$S/audio_candidates_summary.py" "$RUN" "$BASE" "$RUN/audio_candidates.md" >/dev/null 2>&1
  rsync -a --include='*/' --include='seed_ensemble.json' --include='result.json' --include='*.md' --exclude='*' "$RUN/" "$DST_WIN/"
done
echo "===== summary ====="
"$PY" "$S/audio_candidates_summary.py" "$RUN" "$BASE" "$RUN/audio_candidates.md"
rsync -a --include='*/' --include='seed_ensemble.json' --include='result.json' --include='*.md' --exclude='*' "$RUN/" "$DST_WIN/"
echo "audio candidates done in $(( ($(date +%s) - start) / 60 )) min; results in $DST_WIN"
