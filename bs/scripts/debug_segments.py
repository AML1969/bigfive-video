"""Replay consecutive segments of a batch run in ONE process, member by member, to locate a sticky CUDA
device-side assert (run with CUDA_LAUNCH_BLOCKING=1 so the failing kernel is reported synchronously).
Usage: CUDA_LAUNCH_BLOCKING=1 debug_segments.py RUN_DIR SEG [SEG ...] [--lang ru] [--no-ollama]
Transcripts are taken from RUN_DIR/segments/timeline.json, exactly as the batch used them.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from bs_bigfive.backend_ensemble import EnsembleBackend, EnsembleConfig
from bs_bigfive.backend_mm import MMConfig
from bs_bigfive.backend_oceanai import BackendConfig

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("segs", nargs="+", type=int)
ap.add_argument("--lang", default="ru")
ap.add_argument("--no-ollama", action="store_true", help="reuse the stored behaviour description instead of Ollama")
a = ap.parse_args()
run = Path(a.run_dir)
tl = {t["segment"]: t for t in json.loads((run / "segments" / "timeline.json").read_text(encoding="utf-8"))}
be = EnsembleBackend(EnsembleConfig(members=("oceanai", "mm"), lang=a.lang, oceanai_cfg=BackendConfig(lang=a.lang),
                                    mm_cfg=MMConfig(lang=a.lang))).load()
print("loaded; CUDA_LAUNCH_BLOCKING =", __import__("os").environ.get("CUDA_LAUNCH_BLOCKING"), flush=True)
for n in a.segs:
    t = tl[n]
    seg = next(iter((run / "segments").glob(f"seg{n:02d}_*.mp4")), None)
    print(f"\n===== segment {n} {t['start']}-{t['end']}s file={seg and seg.name} chars={len(t['transcript'])} "
          f"stored_error={str(t.get('error'))[:60]}", flush=True)
    if seg is None:
        print("segment file missing"); continue
    for name, b in be.backends.items():
        t0 = time.time()
        try:
            kw = {}
            if name == "mm" and a.no_ollama and t.get("behavior_description"):
                kw["behavior"] = t["behavior_description"]
            r = b.predict_video(seg, asr=False, transcript=t["transcript"], **kw)
            torch.cuda.synchronize()
            print(f"  {name}: OK {round(time.time() - t0, 1)}s", {k: round(v, 3) for k, v in r["scores"].items()}, flush=True)
        except Exception:
            print(f"  {name}: FAILED after {round(time.time() - t0, 1)}s", flush=True)
            traceback.print_exc()
            print("  --- last transcript:", t["transcript"][:300], flush=True)
