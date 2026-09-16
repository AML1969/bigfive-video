"""Reproduce a failing segment member by member with CUDA_LAUNCH_BLOCKING=1 to locate a device-side assert.
Usage: CUDA_LAUNCH_BLOCKING=1 debug_segment.py SEG_MP4 [TRANSCRIPT_TXT_OR_TEXT] [lang]
"""
import sys
import traceback
from pathlib import Path

from bs_bigfive.backend_mm import MMBackend, MMConfig
from bs_bigfive.backend_oceanai import BackendConfig, OceanAIBackend

seg = sys.argv[1]
text_arg = sys.argv[2] if len(sys.argv) > 2 else ""
lang = sys.argv[3] if len(sys.argv) > 3 else "ru"
transcript = Path(text_arg).read_text(encoding="utf-8") if text_arg and Path(text_arg).exists() else text_arg
print("segment:", seg, "| transcript chars:", len(transcript), "| lang:", lang)

for name, make in (("mm", lambda: MMBackend(MMConfig(lang=lang))), ("oceanai", lambda: OceanAIBackend(BackendConfig(lang=lang)))):
    print(f"===== {name} =====", flush=True)
    try:
        be = make().load()
        r = be.predict_video(seg, asr=False, transcript=transcript)
        print("OK", {k: round(v, 3) for k, v in r["scores"].items()})
    except Exception:
        traceback.print_exc()
