"""OCEAN-AI only, verbose, CUDA_LAUNCH_BLOCKING recommended: locate the device-side assert on one segment.
Usage: CUDA_LAUNCH_BLOCKING=1 debug_segment_oceanai.py SEG_MP4 TRANSCRIPT_TXT [lang]
"""
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

from bs_bigfive.backend_oceanai import BackendConfig, OceanAIBackend

seg, txt = Path(sys.argv[1]), Path(sys.argv[2])
lang = sys.argv[3] if len(sys.argv) > 3 else "ru"
corpus = sys.argv[4] if len(sys.argv) > 4 else None
be = OceanAIBackend(BackendConfig(lang=lang, corpus=corpus)).load()
tmp = Path(tempfile.mkdtemp(prefix="bs_dbg_"))
shutil.copy2(seg, tmp / "clip.mp4")
(tmp / "clip.txt").write_text(txt.read_text(encoding="utf-8"), encoding="utf-8")
print("running OCEAN-AI verbosely on", tmp, flush=True)
try:
    df = be.predict_dir(tmp, asr=False, exts=[".mp4"], verbose=True)
    print("OK\n", df.to_string())
except Exception:
    traceback.print_exc()
