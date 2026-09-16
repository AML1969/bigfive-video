"""Combine per-clip test predictions of several systems (equal weights) and evaluate on FIV2 labels.

Usage: ensemble_own.py LABELS_CSV OUT_JSON name1=pred1.csv name2=pred2.csv[:column-prefix] ...
  pred csv: columns Path + Openness..Non-Neuroticism (optionally prefixed, e.g. "scene:" for sslmepr variants)
Evaluates every single system and every pair / triple average.
"""
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

from bs_bigfive.evaluate import evaluate
from bs_bigfive.norms import OCEANAI_COLUMNS

labels = pd.read_csv(sys.argv[1])
out_json = sys.argv[2]
systems = {}
for spec in sys.argv[3:]:
    name, rest = spec.split("=", 1)
    path, prefix = (rest.split(":", 1) + [""])[:2] if ":" in rest and not rest[1:3] == ":\\" else (rest, "")
    df = pd.read_csv(path)
    df["video_name"] = df["Path"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    cols = {f"{prefix}{c}": c for c in OCEANAI_COLUMNS}
    systems[name] = df[["video_name"] + list(cols)].rename(columns=cols).set_index("video_name")

common = set.intersection(*(set(s.index) for s in systems.values()))
print("systems:", list(systems), "| common clips:", len(common))
common = sorted(common)
rows, full = [], {}


def ev(name, frame):
    frame = frame.copy()
    frame.insert(0, "Path", [n + ".mp4" for n in frame.index])
    r = evaluate(frame.reset_index(drop=True), labels)
    full[name] = r
    rows.append({"variant": name, "n": r["n"], "mACC": r["mACC"], "mCCC": r["mCCC"],
                 **{f"ccc_{k[:5]}": v["ccc"] for k, v in r["per_trait"].items()}})


for name, s in systems.items():
    ev(name, s.loc[common])
for k in (2, 3):
    for combo in itertools.combinations(systems, k):
        avg = sum(systems[n].loc[common] for n in combo) / k
        ev("mean(" + ",".join(combo) + ")", avg)
print(pd.DataFrame(rows).to_string(index=False))
Path(out_json).write_text(json.dumps(full, indent=2), encoding="utf-8")
print("wrote", out_json)
