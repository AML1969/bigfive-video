#!/usr/bin/env bash
# Description-shift check for the own model on the 200-clip eval set:
#   A) behaviour descriptions from the paper (Qwen3-VL-4B on the whole clip, from the MM-PSYCHE csv)
#   B) behaviour descriptions generated locally by Ollama (<stem>.behavior.txt already in the eval folder)
# Both use the Rev transcripts (.txt), i.e. no ASR, so only the description source differs.
# Usage: mm_eval_desc_shift.sh [CKPT] [EVAL_DIR]
set -uo pipefail
CKPT="${1:-$HOME/bs/mm_runs/all4_interview/best.pt}"
SRC="${2:-$HOME/bs/eval/fi_test200}"
PAPER="${SRC}_paperdesc"
OUT="$HOME/bs/eval/results"
PY="$HOME/bs/venv/bin/python"
BS="$HOME/bs/venv/bin/bs"
CSV="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/MM-PSYCHE/data/fiv2/test_full_with_description.csv"
mkdir -p "$PAPER" "$OUT"
cd "$HOME"
export PYTHONWARNINGS=ignore
# folder A: links to the clips + transcripts, paper descriptions as .behavior.txt
"$PY" - "$SRC" "$PAPER" "$CSV" <<'PY'
import csv, os, sys
from pathlib import Path
src, dst, csvp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
desc = {r["video_name"]: r["text_llm"] for r in csv.DictReader(open(csvp, encoding="utf-8"))}
n = 0
for mp4 in sorted(src.glob("*.mp4")):
    for ext in (".mp4", ".txt"):
        link = dst / (mp4.stem + ext)
        if not link.exists():
            os.symlink((src / (mp4.stem + ext)).resolve(), link)
    (dst / (mp4.stem + ".behavior.txt")).write_text(desc[mp4.stem], encoding="utf-8")
    n += 1
if not (dst / "labels.csv").exists():
    os.symlink((src / "labels.csv").resolve(), dst / "labels.csv")
print("paper-description folder ready:", dst, n, "clips")
PY
FILT="SyntaxWarning|invalid escape|libEGL|gl_context|inference_feedback|XNNPACK|absl"
echo "===== A) paper descriptions ====="
"$BS" eval-fiv2 --backend mm --mm-ckpt "$CKPT" --dir "$PAPER" --out "$OUT/mm_fi_test200_paperdesc.json" 2>&1 | grep -vE "$FILT" | grep -E "mACC|mCCC|^\{|\"n\"|\[ok\]" | head -8
echo "===== B) Ollama descriptions ====="
"$BS" eval-fiv2 --backend mm --mm-ckpt "$CKPT" --dir "$SRC" --out "$OUT/mm_fi_test200_ollamadesc.json" 2>&1 | grep -vE "$FILT" | grep -E "mACC|mCCC|^\{|\"n\"|\[ok\]" | head -8
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
for name in ("mm_fi_test200_paperdesc", "mm_fi_test200_ollamadesc"):
    p = Path(sys.argv[1]) / f"{name}.json"
    if p.exists():
        r = json.load(open(p))
        print(f"{name:28s} n={r['n']} mACC={r['mACC']:.4f} mCCC={r['mCCC']:.4f} sec/clip={r['seconds_per_clip']}")
PY
