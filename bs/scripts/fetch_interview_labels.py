"""Fetch the FIV2 "interview" label (and the five traits for cross-checking) from the Hugging Face dataset
lyfesan/chalearn_bigfive_faces, reading only the label columns of the parquet shards (no images).

Writes BS/bs/bs_bigfive/mm/interview_labels.csv with video_name, interview, and verifies the traits
against the MM-PSYCHE csv files.
Usage: fetch_interview_labels.py [--probe]   (--probe: read one shard and print columns / sample ids)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "lyfesan/chalearn_bigfive_faces"
SHARDS = {"train": 8, "validation": 3, "test": 3}
COLS = ["video_id", "interview", "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "bs_bigfive" / "mm" / "interview_labels.csv"
CSV_DIR = HERE.parents[2] / "MM-PSYCHE" / "data" / "fiv2"

ap = argparse.ArgumentParser()
ap.add_argument("--probe", action="store_true")
a = ap.parse_args()


def shard_url(split, i, n):
    return f"hf://datasets/{REPO}/data/{split}-{i:05d}-of-{n:05d}.parquet"


if a.probe:
    df = pd.read_parquet(shard_url("test", 0, 3), columns=None if False else ["video_id", "frame_id", "interview", "openness"])
    print(df.head(5).to_string())
    print("rows", len(df), "unique video_id", df["video_id"].nunique())
    sys.exit()

frames = []
for split, n in SHARDS.items():
    for i in range(n):
        d = pd.read_parquet(shard_url(split, i, n), columns=COLS)
        d = d.drop_duplicates("video_id")
        d["split"] = split
        frames.append(d)
        print(f"{split} shard {i}: {len(d)} unique clips", flush=True)
lab = pd.concat(frames, ignore_index=True).drop_duplicates("video_id")
print("total unique clips:", len(lab))

# cross-check against MM-PSYCHE labels (their 'non-neuroticism' should equal this 'neuroticism' column if it is the
# same inverted ChaLearn value; otherwise 1 - x)
mm = pd.concat([pd.read_csv(CSV_DIR / f"{s}_full_with_description.csv").assign(mm_split=s) for s in ("train", "dev", "test")])
m = mm.merge(lab, left_on="video_name", right_on="video_id", how="left", suffixes=("_mm", "_hf"))
print("MM-PSYCHE clips:", len(mm), "| matched in HF:", int(m["interview"].notna().sum()))
for t in ["openness", "conscientiousness", "extraversion", "agreeableness"]:
    d = (m[f"{t}_mm"] - m[f"{t}_hf"]).abs()
    print(f"  {t:18s} max |diff| = {d.max():.4f}")
d1 = (m["non-neuroticism"] - m["neuroticism"]).abs().max()
d2 = (m["non-neuroticism"] - (1 - m["neuroticism"])).abs().max()
print(f"  non-neuroticism vs hf neuroticism: max|diff| = {d1:.4f} ; vs 1 - neuroticism: {d2:.4f}")
print("interview stats:", m["interview"].describe().round(3).to_dict())
print("corr(interview, traits):", {t: round(float(np.corrcoef(m['interview'], m[f'{t}_mm'])[0, 1]), 3)
                                    for t in ["openness", "conscientiousness", "extraversion", "agreeableness"]},
      "non-neuroticism:", round(float(np.corrcoef(m["interview"], m["non-neuroticism"])[0, 1]), 3))
out = m[["video_name", "mm_split", "interview"]].rename(columns={"mm_split": "split"})
out.to_csv(OUT, index=False)
print("wrote", OUT, len(out), "rows; missing interview:", int(out["interview"].isna().sum()))
