#!/usr/bin/env bash
# Smoke test of the candidate speech encoders on 12 clips in a temporary feature root (symlinked media),
# followed by a 2-epoch training on those clips. Usage: audio_smoke.sh [CANDIDATES...]
set -uo pipefail
CANDS=("$@"); [ ${#CANDS[@]} -eq 0 ] && CANDS=(audio_whisper audio_xlsr audio_egemaps audio_w2v_emo)
PY="$HOME/bs/venv/bin/python"
ROOT="$HOME/data/fiv2"
T="/tmp/bs_audio_smoke"
rm -rf "$T"; mkdir -p "$T/features"
ln -s "$ROOT/audio" "$T/audio"; ln -s "$ROOT/video" "$T/video"
for split in train dev test; do
  mkdir -p "$T/features/$split"
  for m in face audio text behavior; do ln -s "$ROOT/features/$split/$m.pt" "$T/features/$split/$m.pt"; done
done
cd "$HOME"
export PYTHONWARNINGS=ignore
for c in "${CANDS[@]}"; do
  echo "===== $c"
  for split in train dev test; do
    "$PY" -m bs_bigfive.mm.extract --root "$T" --split "$split" --modalities "$c" --limit 12 --threads 4 2>&1 \
      | grep -E "stats:|saved|Error|Traceback|failed" | tail -2 | cut -c1-200
  done
  if [ "$c" = "audio_egemaps" ]; then "$PY" /mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts/standardize_features.py "$T" audio_egemaps | tail -1; fi
  "$PY" -m bs_bigfive.mm.train --root "$T" --modalities "face,$c,text,behavior" --targets big5+interview --epochs 2 --patience 2 \
      --out "$T/run_$c" 2>&1 | grep -E "feature table|DONE|Error|Traceback" | tail -3 | cut -c1-200
done
"$PY" -c "import torch; d=torch.load('$T/features/test/audio_whisper.pt'); print('whisper feats', tuple(d['x'].shape), 'nan', bool(torch.isnan(d['x']).any()))" 2>/dev/null
