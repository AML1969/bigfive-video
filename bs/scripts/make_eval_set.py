"""Build an evaluation folder for OCEAN-AI from already-downloaded FIV2 test clips.

Copies N complete .mp4 files (no .part suffix) into DST and writes <stem>.txt with the
professional (Rev) transcript from the MM-PSYCHE csv, so OCEAN-AI's text branch can run
without ASR. Stdlib only, run with any python3 (WSL: /mnt/c paths).
"""
import argparse, csv, shutil, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="dir with FIV2 test mp4 files")
ap.add_argument("--csv", required=True, help="MM-PSYCHE test_full_with_description.csv")
ap.add_argument("--dst", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--offset", type=int, default=0)
a = ap.parse_args()

rows = {r["video_name"]: r for r in csv.DictReader(open(a.csv, encoding="utf-8"))}
src = Path(a.src); dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
names = sorted(p.stem for p in src.glob("*.mp4") if p.stem in rows and not (src / (p.name + ".part")).exists())
chosen = names[a.offset:a.offset + a.n]
labels = dst / "labels.csv"
with open(labels, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["video_name", "openness", "conscientiousness", "extraversion", "agreeableness", "non-neuroticism", "text"])
    for nm in chosen:
        r = rows[nm]
        tgt = dst / (nm + ".mp4")
        if not tgt.exists():
            shutil.copy2(src / (nm + ".mp4"), tgt)
        (dst / (nm + ".txt")).write_text((r["text"].strip() or " "), encoding="utf-8")
        w.writerow([nm, r["openness"], r["conscientiousness"], r["extraversion"], r["agreeableness"], r["non-neuroticism"], r["text"]])
print(f"eval set: {len(chosen)} clips in {dst} (available complete: {len(names)}); labels -> {labels}")
