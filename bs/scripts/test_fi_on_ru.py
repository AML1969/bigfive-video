"""Does OCEAN-AI with FIV2 weights accept a Russian segment when run as lang=en with a pre-translated transcript?
Usage: test_fi_on_ru.py SEG_MP4 SEG_TXT_RU
"""
import shutil
import sys
import tempfile
from pathlib import Path

from bs_bigfive.backend_mm import MMBackend, MMConfig
from bs_bigfive.backend_oceanai import BackendConfig, OceanAIBackend

seg, txt = Path(sys.argv[1]), Path(sys.argv[2])
mm = MMBackend(MMConfig(lang="ru"))
en = mm.to_english(txt.read_text(encoding="utf-8"))
print("EN:", en[:200])
tmp = Path(tempfile.mkdtemp(prefix="bs_fi_"))
shutil.copy2(seg, tmp / "clip.mp4")
(tmp / "clip.txt").write_text(en, encoding="utf-8")
for lang in ("en", "ru"):
    be = OceanAIBackend(BackendConfig(lang=lang, corpus="fi")).load()
    try:
        df = be.predict_dir(tmp, asr=False, exts=[".mp4"])
        print(f"lang={lang} corpus=fi: OK\n", df.to_string())
    except Exception as e:  # noqa: BLE001
        print(f"lang={lang} corpus=fi: FAILED: {str(e)[:120]}")
