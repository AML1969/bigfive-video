"""Post-hoc check: does averaging OCEAN-AI with the SSL-MEPR scene branch help?

Reads two .pred.csv files produced by `bs eval-fiv2` (oceanai and sslmepr backends) and a labels.csv.
Usage: ensemble_check.py OCEANAI_PRED_CSV SSLMEPR_PRED_CSV LABELS_CSV [OUT_JSON]
"""
import json
import sys
from pathlib import Path

import pandas as pd

from bs_bigfive.evaluate import evaluate
from bs_bigfive.norms import OCEANAI_COLUMNS

oa_csv, sm_csv, labels_csv = sys.argv[1:4]
out_json = sys.argv[4] if len(sys.argv) > 4 else None
labels = pd.read_csv(labels_csv)
oa = pd.read_csv(oa_csv)
sm = pd.read_csv(sm_csv)
oa["video_name"] = oa["Path"].str.replace(r"\.[^.]+$", "", regex=True)
sm["video_name"] = sm["Path"].str.replace(r"\.[^.]+$", "", regex=True)
m = oa.merge(sm, on="video_name", suffixes=("_oa", "_sm"))
print("clips in both:", len(m))
has_audio = f"audio:{OCEANAI_COLUMNS[0]}" in m.columns


def make(cols_fn):
    df = pd.DataFrame({"Path": m["Path_oa"]})
    for c in OCEANAI_COLUMNS:
        df[c] = cols_fn(c)
    return df


variants = {
    "oceanai": lambda c: m[f"{c}_oa"],
    "sslmepr_scene": lambda c: m[f"scene:{c}"],
    "mean(oceanai, scene)": lambda c: (m[f"{c}_oa"] + m[f"scene:{c}"]) / 2,
    "0.6*oceanai + 0.4*scene": lambda c: 0.6 * m[f"{c}_oa"] + 0.4 * m[f"scene:{c}"],
}
if has_audio:
    variants["sslmepr_late(scene,audio,text)"] = lambda c: m[f"{c}_sm"]
    variants["mean(oceanai, scene, audio)"] = lambda c: (m[f"{c}_oa"] + m[f"scene:{c}"] + m[f"audio:{c}"]) / 3

rows, full = [], {}
for name, fn in variants.items():
    r = evaluate(make(fn), labels)
    full[name] = r
    rows.append({"variant": name, "n": r["n"], "mACC": r["mACC"], "mCCC": r["mCCC"],
                 **{f"ccc_{k[:5]}": v["ccc"] for k, v in r["per_trait"].items()}})
print(pd.DataFrame(rows).to_string(index=False))
if out_json:
    Path(out_json).write_text(json.dumps(full, indent=2), encoding="utf-8")
    print("wrote", out_json)
