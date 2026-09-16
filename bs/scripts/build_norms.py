"""Build fiv2_norms.json (101 quantiles per trait) from the MM-PSYCHE FIV2 train csv. Stdlib only."""
import csv, json, statistics, sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
csv_path = root / "MM-PSYCHE" / "data" / "fiv2" / "train_full_with_description.csv"
out = root / "bs" / "bs_bigfive" / "fiv2_norms.json"
cols = ["openness", "conscientiousness", "extraversion", "agreeableness", "non-neuroticism", "interview"]
keys = ["openness", "conscientiousness", "extraversion", "agreeableness", "emotional_stability", "interview"]
rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
# interview label (ChaLearn job-interview variable) from scripts/fetch_interview_labels.py
iv_path = root / "bs" / "bs_bigfive" / "mm" / "interview_labels.csv"
iv = {r["video_name"]: r["interview"] for r in csv.DictReader(open(iv_path, encoding="utf-8"))}
for r in rows:
    r["interview"] = iv[r["video_name"]]
norms = {"source": "FIV2 train (6000 clips), MM-PSYCHE data/fiv2/train_full_with_description.csv", "n": len(rows),
         "quantiles": {}, "mean": {}, "std": {}}
for c, k in zip(cols, keys):
    v = sorted(float(r[c]) for r in rows)
    n = len(v)
    norms["quantiles"][k] = [round(v[min(n - 1, int(round(p / 100 * (n - 1))))], 5) for p in range(101)]
    norms["mean"][k] = round(statistics.mean(v), 5)
    norms["std"][k] = round(statistics.pstdev(v), 5)
out.write_text(json.dumps(norms, indent=1), encoding="utf-8")
print("wrote", out, "n =", len(rows))
for k in keys:
    print(f"  {k:20s} mean={norms['mean'][k]:.3f} sd={norms['std'][k]:.3f} p10={norms['quantiles'][k][10]:.3f} p50={norms['quantiles'][k][50]:.3f} p90={norms['quantiles'][k][90]:.3f}")
