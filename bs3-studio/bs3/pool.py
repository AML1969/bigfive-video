"""Pool of processed videos per language: every analysed video with a primary system is registered once
(fingerprint of the file) with its main scores. The pool is only collected internally: nothing is compared with it
and no page, PDF or text shows numbers or words based on it (change of 2026-09-26)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

from .norms import TRAIT_KEYS

POOL_DIR = Path(os.environ.get("BS3_POOL_DIR", "~/bs3_data/pool")).expanduser()


def fingerprint(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
    return f"{p.stat().st_size}_{h.hexdigest()}"


def _dir(lang: str) -> Path:
    d = POOL_DIR / lang
    d.mkdir(parents=True, exist_ok=True)
    return d


def add(video: str | Path, scores: dict, lang: str, primary: str | None, name: str | None = None) -> int:
    """Register (or refresh) a video's main scores in the pool of `lang`. Returns the pool size."""
    d = _dir(lang)
    entry = {"name": name or Path(video).name, "lang": lang, "primary": primary,
             "scores": {k: round(float(scores[k]), 4) for k in TRAIT_KEYS if k in scores},
             "created_at": _dt.datetime.now().isoformat(timespec="seconds")}
    (d / f"{fingerprint(video)}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(list(d.glob("*.json")))


def entries(lang: str) -> list[dict]:
    out = []
    for p in sorted(_dir(lang).glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return out
