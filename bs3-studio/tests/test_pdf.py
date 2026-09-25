"""PDF of BS Profiler 3.0 (design 10.7; tasks T21, T27) on the numeric copies of samples A and B: the plan puts the
section «Тип MBTI» right after the Big Five section, the report builds without charts and media, its text has the
characterization, the MBTI section with the letter strip and «Как получены оценки» and nothing of the 2.0 summary;
the appendix «Значения по отрезкам» has the MBTI column with a dash on the segments without OCEAN-AI."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from samples import english, rep

from bs3 import caveats, characterization, mbti, pdf_mbti, pdf_report, scores


def _parts(r: dict):
    view = scores.clean_view(r)
    mb = mbti.get_mbti(r, view)
    return view, mb, characterization.build(view, mb)


def _build(r: dict) -> tuple[Path, dict]:
    view, mb, ch = _parts(r)
    out = Path(tempfile.mkdtemp(prefix="bs3_pdf_test_")) / "report.pdf"
    pdf_report.build_pdf(view, out, mbti=mb, character=ch)
    return out, mb


def _text(path: Path) -> str | None:
    if not shutil.which("pdftotext"):
        return None
    r = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, text=True, check=True)
    return re.sub(r"\s+", " ", r.stdout)


def test_plan_puts_mbti_after_profile():
    view, mb, _ = _parts(rep("B"))
    pdf = pdf_report.Report()
    pdf_report._plan(pdf, view, None, [], {}, mb)
    keys = list(pdf.plan)
    assert keys[:2] == ["profile", "mbti"], keys
    assert pdf.plan["mbti"] == 2
    pdf2 = pdf_report.Report()
    pdf_report._plan(pdf2, view, None, [], {}, None)            # no type: no section, numbers close up
    assert "mbti" not in pdf2.plan and list(pdf2.plan)[1] != "mbti"


def test_strip_geometry():
    assert pdf_mbti._strip_geometry(33) == (2, 17, 6.0, 4.8)       # 17 + 16, not 30 + 3
    assert pdf_mbti._strip_geometry(18)[:3] == (1, 18, 6.0)
    blocks, per, cell, row_h = pdf_mbti._strip_geometry(90)
    assert (blocks, per) == (3, 30) and abs(cell - (pdf_report.TEXT_W_MM - 18) / 30) < 1e-9 and row_h <= cell


def test_facts_start_with_type_card():
    view, mb, _ = _parts(rep("B"))
    facts = pdf_report._pdf_facts(view, False, mb)
    assert facts[0] == mbti.fact_card(mb)
    assert pdf_report._pdf_facts(view, False, None)[0] != facts[0]


def test_segment_types_by_start():
    _, mb, _ = _parts(rep("B"))
    t = pdf_mbti.segment_types_by_start(mb)
    assert len(t) == 33
    assert sum(1 for v in t.values() if v is None) == 7                 # the segments without OCEAN-AI
    assert {v for v in t.values() if v} >= {"EXFJ"}


def test_pdf_builds_and_reads():
    for name in ("A", "B"):
        path, mb = _build(rep(name))
        assert path.exists() and path.stat().st_size > 10_000, name
        text = _text(path)
        if text is None:                                               # poppler-utils not installed: only the build
            continue
        for s in ("Характеристика личности", "Коротко.", "Границы вывода.", "Ключевые факты", "Тип MBTI · OCEAN-AI",
                  "Big Five: профиль и оценки", "Как получены оценки", "2. Тип MBTI (перевод шкал Big Five)",
                  "Тип по ходу ролика", "Как читать тип MBTI", "Как читать результаты", "BS Profiler 3.0 · стр.",
                  mb["type"], mb["type_strict"]):
            assert s in text, (name, s)
        for code in ("C8", "C9", "C16", "C18", "C11", "C15"):
            assert caveats.text(code)[:60] in text, (name, code)
        for bad in ("Краткие выводы", "сегмент", "определяет тип личности"):
            assert bad not in text.replace("не определяет тип личности", ""), (name, bad)
        appx = text.split("Значения по отрезкам")[-1]
        assert " MBTI " in appx, name
    # sample B: the strip summary (design 13.2) and the MBTI column with the types of the segments
    path, mb = _build(rep("B"))
    text = _text(path)
    if text is not None:
        assert ("ESFJ в 16 из 26 отрезков с оценкой, ENFJ — в 10; ось S–N совпадает с итогом в 16 из 26 отрезков, "
                "остальные оси — во всех") in text
        assert "Совпадают 0 из 4 осей; расходится E–I; на границе у одной из систем: S–N, T–F, J–P." in text
        appx = text.split("Значения по отрезкам")[-1]
        assert "EXFJ" in appx and "ESFJ" in appx
        assert "«—» в столбцах Big Five и MBTI" in appx


def test_pdf_english():
    path, mb = _build(english("B"))
    assert path.exists() and mb["source"] == "mean"
    text = _text(path)
    if text is not None:
        assert "Итог — среднее двух систем" in text and "OCEAN-AI (веса First Impressions V2)" in text
        assert "пороги предварительные" not in text
        assert caveats.c6("en")[:60] in text and caveats.c7("en")[:60] in text


def test_build_pdf_computes_missing_parts():
    """A caller that passes only the raw result gets the same characterization and type."""
    r = rep("B")
    out = Path(tempfile.mkdtemp(prefix="bs3_pdf_test_")) / "raw.pdf"
    pdf_report.build_pdf(r, out)
    text = _text(out)
    assert out.exists()
    if text is not None:
        assert "Характеристика личности" in text and "EXFJ" in text
