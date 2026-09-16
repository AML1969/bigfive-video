#!/usr/bin/env bash
# Train the personality fusion model on cached features for several modality / target sets (MM-PSYCHE recipe).
# Usage: mm_train_all.sh [RUN_ROOT]   -> RUN_ROOT/<config>/{best.pt,result.json,test_pred.csv}
set -uo pipefail
RUN="${1:-$HOME/bs/mm_runs}"
PY="$HOME/bs/venv/bin/python"
mkdir -p "$RUN"
cd "$HOME"
export PYTHONWARNINGS=ignore
start=$(date +%s)
# name | modalities | targets
declare -a CONFIGS=(
  "all4|face,audio,text,behavior|big5"
  "all4_interview|face,audio,text,behavior|big5+interview"
  "face|face|big5"
  "face_audio|face,audio|big5"
  "face_audio_text|face,audio,text|big5"
  "face_behavior|face,behavior|big5"
  "audio_text_behavior|audio,text,behavior|big5"
)
for cfg in "${CONFIGS[@]}"; do
  IFS="|" read -r name mods targets <<< "$cfg"
  echo "===== $name ($mods; $targets) ====="
  "$PY" -m bs_bigfive.mm.train --modalities "$mods" --targets "$targets" --out "$RUN/$name" 2>&1 \
    | grep -vE "SyntaxWarning|invalid escape" | tee "$RUN/$name.log" | grep -E "DONE|early stopping|feature table" | tail -5
done
echo "===== summary ====="
"$PY" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
rows = []
for p in sorted(Path(sys.argv[1]).glob("*/result.json")):
    r = json.load(open(p))
    iv = r["test"].get("interview", {})
    rows.append((p.parent.name, ",".join(r["modalities"]), r["best_epoch"], r["dev"]["mACC"], r["dev"]["mCCC"],
                 r["test"]["mACC"], r["test"]["mCCC"], r["test"]["CCC_flat"], iv.get("acc", float("nan")), iv.get("ccc", float("nan")), r["test"]["n"]))
print(f"{'run':20s} {'modalities':27s} {'ep':>3s} {'dev mACC':>8s} {'dev mCCC':>8s} {'tst mACC':>8s} {'tst mCCC':>8s} {'CCCflat':>7s} {'iv ACC':>6s} {'iv CCC':>6s} {'n':>5s}")
for r in rows:
    print(f"{r[0]:20s} {r[1]:27s} {r[2]:3d} {r[3]:8.4f} {r[4]:8.4f} {r[5]:8.4f} {r[6]:8.4f} {r[7]:7.4f} {r[8]:6.3f} {r[9]:6.3f} {r[10]:5d}")
PY
echo "train-all done in $(( $(date +%s) - start ))s"
