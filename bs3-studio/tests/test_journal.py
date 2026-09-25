"""The journal entry of a finished analysis (design 10.6; task T19) on the numeric copy of sample B: clean main scores,
the second opinion, the MBTI lines of both systems with their agreement, the paragraph «Коротко»; no «Краткие
выводы». The entry is written into a temporary file, never into the journal of the service."""
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
    assert lines[1].startswith("Второе мнение (своя модель): ")
    assert ("Тип MBTI (OCEAN-AI, пороги предварительные): EXFJ, ближайший ESFJ «Попечитель», возможен ENFJ; "
            "нейротизм — ниже типичного") in lines
    assert ("Второе мнение MBTI (своя модель): ISXX, ближайший ISTJ; уверенно совпадают 0 осей из 4 "
            "(E–I ≠, S–N ≈, T–F ≈, J–P ≈)") in lines
    short = [ln for ln in lines if ln.startswith("Характеристика (коротко): ")]
    assert len(short) == 1 and "ESFJ" in short[0]
    assert lines[-1] == "Папка: /tmp/job"
    assert not any("Краткие выводы" in ln for ln in lines)
    assert "mbti" not in r                                        # the entry never stores the computed section


def test_result_writes_entry():
    old = journal.PATH
    with tempfile.TemporaryDirectory() as d:
        journal.PATH = Path(d) / "journal.txt"
        try:
            journal.result(_Req(), rep("B"), 65.0)
            text = journal.PATH.read_text(encoding="utf-8")
        finally:
            journal.PATH = old
    assert "РЕЗУЛЬТАТ" in text and "обработка 1:05" in text
    assert "    Тип MBTI (OCEAN-AI, пороги предварительные): EXFJ" in text
    assert "Характеристика (коротко): " in text and "Краткие выводы" not in text
