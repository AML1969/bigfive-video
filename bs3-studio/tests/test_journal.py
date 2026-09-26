"""The journal entries of BS Profiler 3.1 (design 10.6; task T19) on the numeric copy of sample B: the start entry
names the chosen model, the result entry carries the clean scores of one model, its MBTI line and the paragraph
«Коротко»; no second opinion, no language, no «Краткие выводы». The entries are written into a temporary file,
never into the journal of the service."""
from __future__ import annotations

import tempfile
from pathlib import Path

from samples import rep

from bs3 import journal


class _Req:
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0) Chrome/120"}
    session_hash = "abcdef123"


def test_result_lines_b():
    r = rep("B")
    r["job_dir"] = "/tmp/job"
    lines = journal.result_lines(r)
    assert lines[0].startswith("Итог (OCEAN-AI, веса MuPTA): ") and "экстраверсия 0.73" in lines[0]
    assert lines[1] == "Тип MBTI (OCEAN-AI): ENFJ «Наставник»; нейротизм — средний уровень"
    short = [ln for ln in lines if ln.startswith("Характеристика (коротко): ")]
    assert len(short) == 1 and "ENFJ" in short[0] and "Вторая система" not in short[0]
    for ln in lines:
        for w in ("предварительн", "типичн", "опорн", "русских роликов", "Второе мнение", "второе мнение",
                  "своя модель", "Участник", "ISXX", "ISTP"):
            assert w not in ln, (w, ln)
    assert lines[-1] == "Папка: /tmp/job"
    assert len(lines) == 4                                        # scores, type, «Коротко», folder: one model
    assert not any("Краткие выводы" in ln for ln in lines)
    assert "mbti" not in r                                        # the entry never stores the computed section


def test_result_lines_own_model():
    """A job of AMLAI 1.0 (3.1) or an old job where OCEAN-AI gave nothing: the lines name AMLAI 1.0."""
    r = rep("B")
    del r["variant_scores"]["oceanai"]
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    lines = journal.result_lines(r)
    assert lines[0].startswith("Итог (AMLAI 1.0): ") and "MuPTA" not in lines[0]
    assert lines[1].startswith("Тип MBTI (AMLAI 1.0): ISXX, ближайший ISTP")
    r = rep("B")
    r["model"].update({"selected": "mm", "primary": "mm", "selected_title": "AMLAI 1.0"})
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    line = journal.result_lines(r)[0]
    assert line.startswith("Итог (AMLAI 1.0): открытость 0.") and "экстраверсия 0.25" in line


def test_start_and_result_write_entries():
    old = journal.PATH
    with tempfile.TemporaryDirectory() as d:
        journal.PATH = Path(d) / "journal.txt"
        try:
            journal.start(_Req(), "/tmp/video.mp4", "mm", True)
            journal.start(_Req(), "/tmp/video.mp4", "oceanai", False)
            journal.result(_Req(), rep("B"), 65.0)
            text = journal.PATH.read_text(encoding="utf-8")
        finally:
            journal.PATH = old
    assert "СТАРТ" in text and "модель AMLAI 1.0, объяснения да" in text and "модель OCEAN-AI, объяснения нет" in text
    assert "язык" not in text.lower()
    assert "РЕЗУЛЬТАТ" in text and "обработка 1:05" in text
    assert "    Тип MBTI (OCEAN-AI): ENFJ" in text
    assert "Характеристика (коротко): " in text and "Краткие выводы" not in text and "торое мнение" not in text
