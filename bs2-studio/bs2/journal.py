"""Plain-text journal of the web service: who opened the page and when, which file was analysed and the outcome.

One UTF-8 text file, entries appended in order and never rewritten: ~/bs2_data/logs/journal.txt (BS2_JOURNAL
overrides). A result entry carries the main scores, the second opinion and the «Краткие выводы» text exactly as the
page shows them. Visitor details come from the reverse proxy: X-Forwarded-For (client address) and X-Remote-User
(login name from Caddy basic_auth); opened directly on the machine, the address is marked «локально».
Writing the journal never breaks the page: every failure is logged and swallowed.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path

log = logging.getLogger("bs2.journal")

PATH = Path(os.environ.get("BS2_JOURNAL", "~/bs2_data/logs/journal.txt")).expanduser()
_lock = threading.Lock()

TRAITS = (("openness", "открытость"), ("conscientiousness", "добросовестность"), ("extraversion", "экстраверсия"),
          ("agreeableness", "доброжелательность"), ("emotional_stability", "эмоциональная стабильность"))
MEMBERS = {"oceanai": "OCEAN-AI", "mm": "своя модель", "scene": "SSL-MEPR (сцена)"}
LANGS = {"ru": "русский", "en": "английский"}
_DEVICES = (("iPhone", "iPhone"), ("iPad", "iPad"), ("Android", "Android"), ("Windows", "Windows"),
            ("Macintosh", "Mac"), ("Linux", "Linux"))
_BROWSERS = (("YaBrowser", "Яндекс Браузер"), ("Edg", "Edge"), ("OPR/", "Opera"), ("Firefox", "Firefox"),
             ("FxiOS", "Firefox"), ("CriOS", "Chrome"), ("Chrome", "Chrome"), ("Safari", "Safari"))


def _mmss(sec: float) -> str:
    sec = int(round(sec or 0))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def _agent(ua: str) -> str:
    ua = ua or ""
    dev = next((name for key, name in _DEVICES if key in ua), "")
    br = next((name for key, name in _BROWSERS if key in ua), "")
    return ", ".join(x for x in (dev, br) if x) or (ua[:60] or "браузер не указан")


def _who(request) -> str:
    """«сеанс 3f9a1c  логин  адрес  устройство» for the head line of an entry."""
    try:
        headers = request.headers
        fwd = (headers.get("x-forwarded-for") or "").split(",")[0].strip()
        ip = fwd or "локально"
        user = (headers.get("x-remote-user") or "").strip() or "без логина"
        sess = (getattr(request, "session_hash", "") or "")[:6] or "------"
        return f"сеанс {sess}  {user}  {ip}  {_agent(headers.get('user-agent'))}"
    except Exception:  # noqa: BLE001
        return "сеанс ?"


def _write(kind: str, request, tail: str = "", body: list[str] | tuple = ()) -> None:
    head = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {kind:<10} {_who(request)}" + (f"  {tail}" if tail else "")
    text = head + "\n" + "".join(f"    {line}\n" for line in body) + ("\n" if body else "")
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock, open(PATH, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception:  # noqa: BLE001
        log.exception("journal write failed (%s)", PATH)


def _file(video) -> str:
    name = Path(str(video)).name
    try:
        return f"файл «{name}», {os.path.getsize(video) / 1e6:.0f} МБ"
    except OSError:
        return f"файл «{name}»"


def _scores(scores: dict) -> str:
    return ", ".join(f"{ru} {float(scores[k]):.2f}" for k, ru in TRAITS if k in scores)


def result_lines(rep: dict, narrative: str) -> list[str]:
    """Body of a result entry: main scores, second opinion (or both members for the mean), summary, job folder."""
    model = rep.get("model") or {}
    primary = model.get("primary")
    main = {k: (v.get("score") if isinstance(v, dict) else v) for k, v in (rep.get("traits") or {}).items()}
    if primary:
        src = MEMBERS.get(primary, primary) + (", веса MuPTA" if primary == "oceanai" and model.get("lang") == "ru" else "")
    else:
        src = "среднее двух систем"
    line = f"Итог ({src}): {_scores(main)}"
    iv = rep.get("interview")
    iv = iv.get("score") if isinstance(iv, dict) else iv
    if iv is not None:
        line += f"; «пригласил бы на собеседование» {float(iv):.2f}"
    lines = [line]
    for m, v in (rep.get("variant_scores") or {}).items():
        if m != primary:
            lines.append(f"{'Второе мнение' if primary else 'Участник'} ({MEMBERS.get(m, m)}): {_scores(v)}")
    text = re.sub(r"\s+", " ", narrative or "").strip()
    if text:
        lines.append(f"Краткие выводы: {text}")
    if rep.get("job_dir"):
        lines.append(f"Папка: {rep['job_dir']}")
    return lines


def visit(request) -> None:
    _write("ВХОД", request)


def start(request, video, lang: str, explain: bool) -> None:
    _write("СТАРТ", request, f"{_file(video)}, язык {LANGS.get(lang, lang)}, объяснения {'да' if explain else 'нет'}")


def result(request, rep: dict, narrative: str, wall_sec: float) -> None:
    name = rep.get("original_file_name") or Path(str(rep.get("input", ""))).name
    tail = f"файл «{name}», ролик {_mmss(rep.get('duration_sec', 0))}, обработка {_mmss(wall_sec)}"
    try:
        body = result_lines(rep, narrative)
    except Exception:  # noqa: BLE001
        log.exception("journal: could not format the result")
        body = ["(не удалось оформить итог, см. result.json в папке задачи)"]
    _write("РЕЗУЛЬТАТ", request, tail, body)


def failed(request, video, message: str, stopped: bool = False) -> None:
    _write("ОСТАНОВЛЕНО" if stopped else "ОШИБКА", request, f"{Path(str(video)).name}: {message}")
