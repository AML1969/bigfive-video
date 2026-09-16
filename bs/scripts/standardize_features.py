"""Z-score a feature modality with train statistics (for hand-crafted features such as eGeMAPS whose columns
have wildly different scales). Rewrites features/<split>/<modality>.pt for train/dev/test (idempotent: the
train file remembers that it was standardised) and stores the statistics next to the files.
Usage: standardize_features.py ROOT MODALITY
"""
import json
import sys
from pathlib import Path

import torch

root, mod = Path(sys.argv[1]), sys.argv[2]
train = torch.load(root / "features" / "train" / f"{mod}.pt", map_location="cpu")
if train.get("stats", {}).get("standardized"):
    print(f"{mod}: already standardised"); sys.exit(0)
x = train["x"]
mean, std = x.mean(0), x.std(0)
std[std < 1e-6] = 1.0
(root / "features" / f"{mod}_standardize.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}), encoding="utf-8")
for split in ("train", "dev", "test"):
    p = root / "features" / split / f"{mod}.pt"
    d = torch.load(p, map_location="cpu")
    d["x"] = ((d["x"] - mean) / std).clamp(-6, 6).float().contiguous()
    d["stats"]["standardized"] = True
    torch.save(d, p)
    print(f"{split}/{mod}: {tuple(d['x'].shape)} standardised")
