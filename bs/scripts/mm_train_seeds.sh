#!/usr/bin/env bash
# Train one configuration with several seeds and average the test predictions (seed ensemble).
# Usage: mm_train_seeds.sh [MODALITIES] [TARGETS] [RUN_ROOT] [SEEDS...]
set -uo pipefail
MODS="${1:-face,audio,text,behavior}"
TARGETS="${2:-big5+interview}"
RUN="${3:-$HOME/bs/mm_runs_seeds}"
shift 3 2>/dev/null || shift $#
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 1 2 3 4)
PY="$HOME/bs/venv/bin/python"
mkdir -p "$RUN"
cd "$HOME"
export PYTHONWARNINGS=ignore
start=$(date +%s)
for s in "${SEEDS[@]}"; do
  d="$RUN/seed$s"
  if [ -f "$d/result.json" ]; then echo "seed $s exists"; continue; fi
  "$PY" -m bs_bigfive.mm.train --modalities "$MODS" --targets "$TARGETS" --seed "$s" --out "$d" 2>&1 \
    | grep -vE "SyntaxWarning|invalid escape" | tee "$RUN/seed$s.log" | grep -E "DONE" | cut -c1-200
done
"$PY" - "$RUN" <<'PY'
import glob, json, sys
import numpy as np, pandas as pd
from pathlib import Path
from bs_bigfive.mm.train import metrics
from bs_bigfive.mm.data import load_split_table, TARGET_SETS
run = Path(sys.argv[1])
preds = {p.parent.name: pd.read_csv(p).set_index("Path") for p in sorted(run.glob("seed*/test_pred.csv"))}
names = sorted(set.intersection(*(set(v.index) for v in preds.values())))
cols = list(next(iter(preds.values())).columns)
targets = "big5+interview" if "Interview" in cols else "big5"
lab = load_split_table("test", with_interview=(targets == "big5+interview")).set_index("video_name")
t = lab.loc[[n[:-4] for n in names], TARGET_SETS[targets]].to_numpy(dtype=np.float32)
rows = []
for k, v in preds.items():
    m = metrics(t, v.loc[names, cols].to_numpy(dtype=np.float32))
    rows.append((k, m["mACC"], m["mCCC"], m["CCC_flat"], m.get("interview", {}).get("ccc", float("nan"))))
avg = sum(v.loc[names, cols].to_numpy(dtype=np.float32) for v in preds.values()) / len(preds)
m = metrics(t, avg)
rows.append((f"mean of {len(preds)} seeds", m["mACC"], m["mCCC"], m["CCC_flat"], m.get("interview", {}).get("ccc", float("nan"))))
print(f"{'run':22s} {'mACC':>7s} {'mCCC':>7s} {'CCCflat':>8s} {'iv CCC':>7s}")
for r in rows:
    print(f"{r[0]:22s} {r[1]:7.4f} {r[2]:7.4f} {r[3]:8.4f} {r[4]:7.4f}")
df = pd.DataFrame(avg, columns=cols); df.insert(0, "Path", names); df.to_csv(run / "test_pred_mean.csv", index=False)
json.dump({"seeds": list(preds), "mean": m, "per_seed": {r[0]: {"mACC": r[1], "mCCC": r[2], "CCC_flat": r[3]} for r in rows[:-1]}},
          open(run / "seed_ensemble.json", "w"), indent=2)
print("wrote", run / "test_pred_mean.csv")
PY
echo "seeds done in $(( $(date +%s) - start ))s"
