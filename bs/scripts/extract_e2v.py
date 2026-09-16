"""emotion2vec+ large features for FIV2 (runs in its own venv with funasr; no bs_bigfive import).
Frame-level embeddings [T, 768] -> mean‖std [1536], written in the same .pt format as the other modalities.
Usage: extract_e2v.py ROOT SPLIT [--limit N]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("split")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out-name", default="audio_e2v")
a = ap.parse_args()
root = Path(a.root)
names = torch.load(root / "features" / a.split / "audio.pt", map_location="cpu")["names"]
if a.limit:
    names = names[: a.limit]

from funasr import AutoModel  # noqa: E402

model = None
for kw in ({"model": "iic/emotion2vec_plus_large", "hub": "ms"}, {"model": "emotion2vec/emotion2vec_plus_large", "hub": "hf"}):
    try:
        model = AutoModel(disable_update=True, **kw)
        print("loaded", kw, flush=True)
        break
    except Exception as e:  # noqa: BLE001
        print("load failed", kw, str(e)[:160], flush=True)
if model is None:
    raise SystemExit("emotion2vec could not be loaded")


def load16k(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        wav = soxr.resample(wav, sr, 16000)
    return wav / (np.abs(wav).max() + 1e-8)


xs, done, failed, t0 = [], [], [], time.time()
for i, n in enumerate(names, 1):
    try:
        wav = load16k(root / "audio" / a.split / f"{n}.wav")
        res = model.generate(wav, granularity="frame", extract_embedding=True, disable_pbar=True)
        feats = torch.tensor(np.asarray(res[0]["feats"]), dtype=torch.float32)     # [T, 768]
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        xs.append(torch.cat([feats.mean(0), feats.std(0, unbiased=False)]))
        done.append(n)
    except Exception as e:  # noqa: BLE001
        failed.append(n)
        print("failed", n, str(e)[:120], flush=True)
    if i % 200 == 0:
        print(f"[{a.split}/{a.out_name}] {i}/{len(names)} ({(time.time() - t0) / i:.2f} s/clip)", flush=True)
x = torch.stack(xs) if xs else torch.zeros(0, 1536)
out = root / "features" / a.split / f"{a.out_name}.pt"
torch.save({"names": done, "x": x.float().contiguous(),
            "stats": {"failed": failed, "n": len(done), "seconds": round(time.time() - t0, 1)}}, out)
print(f"saved {tuple(x.shape)} -> {out}; failed {len(failed)}")
