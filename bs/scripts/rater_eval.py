"""Evaluate the LLM rater: raw, linearly calibrated on a dev subset, and in ensembles with OCEAN-AI and the own model.

Usage: rater_eval.py TEST_LABELS TEST_RATER_CSV DEV_LABELS DEV_RATER_CSV OCEANAI_PRED OWN_PRED OUT_JSON
All prediction csvs: Path + Openness..Non-Neuroticism columns (OCEAN-AI / own preds may cover more clips).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bs_bigfive.evaluate import evaluate
from bs_bigfive.norms import FIV2_COLUMNS, OCEANAI_COLUMNS

test_lab, test_rater, dev_lab, dev_rater, oa_csv, own_csv, out_json = sys.argv[1:8]


def load_pred(path):
    df = pd.read_csv(path)
    df["video_name"] = df["Path"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    return df.set_index("video_name")[OCEANAI_COLUMNS]


tl, dl = pd.read_csv(test_lab), pd.read_csv(dev_lab)
tr, dr = load_pred(test_rater), load_pred(dev_rater)
oa, own = load_pred(oa_csv), load_pred(own_csv)

# --- calibration: per-trait linear fit rater -> label on dev
dev = dr.join(dl.set_index("video_name")[FIV2_COLUMNS], how="inner")
calib = {}
for c, t in zip(OCEANAI_COLUMNS, FIV2_COLUMNS):
    x, y = dev[c].to_numpy(dtype=float), dev[t].to_numpy(dtype=float)
    if np.std(x) < 1e-6:
        a, b = 0.0, float(y.mean())
    else:
        a, b = np.polyfit(x, y, 1)
    calib[c] = (float(a), float(b))
print("dev clips for calibration:", len(dev))
print("calibration (slope, intercept):", {c: (round(a, 3), round(b, 3)) for c, (a, b) in calib.items()})
tr_cal = tr.copy()
for c, (a, b) in calib.items():
    tr_cal[c] = (a * tr[c] + b).clip(0, 1)

common = sorted(set(tr.index) & set(oa.index) & set(own.index))
print("test clips in all systems:", len(common))
systems = {"rater_raw": tr.loc[common], "rater_calibrated": tr_cal.loc[common],
           "oceanai": oa.loc[common], "own5": own.loc[common]}
rows, full = [], {}


def ev(name, frame):
    f = frame.copy()
    f.insert(0, "Path", [n + ".mp4" for n in f.index])
    r = evaluate(f.reset_index(drop=True), tl)
    full[name] = r
    rows.append({"variant": name, "n": r["n"], "mACC": r["mACC"], "mCCC": r["mCCC"],
                 **{f"ccc_{k[:5]}": v["ccc"] for k, v in r["per_trait"].items()}})


for n, s in systems.items():
    ev(n, s)
ev("mean(oceanai,own5)", (systems["oceanai"] + systems["own5"]) / 2)
for rn in ("rater_raw", "rater_calibrated"):
    ev(f"mean(oceanai,own5,{rn})", (systems["oceanai"] + systems["own5"] + systems[rn]) / 3)
    ev(f"0.4*oceanai+0.4*own5+0.2*{rn}", 0.4 * systems["oceanai"] + 0.4 * systems["own5"] + 0.2 * systems[rn])
# correlation of the rater with the other systems (how independent are its errors?)
err = {n: (systems[n].to_numpy() - tl.set_index("video_name").loc[common, FIV2_COLUMNS].to_numpy()).ravel()
       for n in ("oceanai", "own5", "rater_calibrated")}
corr = {f"{a}~{b}": round(float(np.corrcoef(err[a], err[b])[0, 1]), 3) for a, b in itertools.combinations(err, 2)}
print("error correlation between systems:", corr)
print(pd.DataFrame(rows).to_string(index=False))
full["calibration"] = calib
full["error_correlation"] = corr
Path(out_json).write_text(json.dumps(full, indent=2), encoding="utf-8")
print("wrote", out_json)
