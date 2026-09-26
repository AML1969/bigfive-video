"""PDF of BS Profiler 3.1 (design 10.7; tasks T21, T27) on the numeric copies of samples A and B: the plan puts the
section «Тип MBTI» right after the Big Five section, the report builds without charts and media, its text has the
characterization, the MBTI section of one model with the letter strip and «Как получены оценки», no second opinion
and nothing of the 2.0 summary; the appendix «Значения по отрезкам» has the MBTI column with a dash on the segments
without OCEAN-AI."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from samples import english, rep

from bs3 import caveats, characterization, mbti, pdf_mbti, pdf_report, scores
from bs3.norms import TRAIT_KEYS


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
    assert {v for v in t.values() if v} == {"ENFJ"}


def test_pdf_builds_and_reads():
    for name in ("A", "B"):
        path, mb = _build(rep(name))
        assert path.exists() and path.stat().st_size > 10_000, name
        text = _text(path)
        if text is None:                                               # poppler-utils not installed: only the build
            continue
        for s in ("Характеристика личности", "Коротко.", "Границы вывода.", "Ключевые факты", "Тип MBTI · OCEAN-AI",
                  "Big Five: профиль и оценки", "Как получены оценки", "2. Тип MBTI (перевод шкал Big Five)",
                  "Тип по ходу ролика", "Как читать тип MBTI", "Как читать результаты", "BS Profiler 3.1 · стр.",
                  mb["type"], mb["type_strict"]):
            assert s in text, (name, s)
        for code in ("C8", "C9", "C16", "C11", "C15"):
            assert caveats.text(code)[:60] in text, (name, code)
        assert caveats.text("C18")[:60] not in text, name                          # one model: no second strip
        for bad in ("Краткие выводы", "сегмент", "определяет тип личности", "опорн", "положени", "типичн",
                    "русских роликов", "обработанных системой", "предварительн", "большинства", "Согласие двух систем",
                    "Вторая система", "на русских роликах", "на русской речи её", "порядок черт", "торое мнение",
                    "Уверенных совпадений", "хотя бы у одной из систем", "BS Profiler 3.0", "ISXX", "IXXX", "ISTP",
                    "ISTJ"):
            assert bad not in text.replace("не определяет тип личности", ""), (name, bad)
        appx = text.split("Значения по отрезкам")[-1]
        assert " MBTI " in appx, name
    # sample B: the strip summary (design 13.2) and the MBTI column with the types of the segments
    path, mb = _build(rep("B"))
    text = _text(path)
    if text is not None:
        assert "ENFJ во всех 26 отрезках с оценкой; все четыре оси совпадают с итогом во всех отрезках" in text
        assert "F, отчётливо (0.74)" in text and "E, умеренно (0.46)" in text
        assert "Число в скобках после буквы — уверенность по оси" in text          # the number is labelled
        assert "Согласие" not in text
        appx = text.split("Значения по отрезкам")[-1]
        assert "ENFJ" in appx
        assert "«—» в столбцах Big Five и MBTI" in appx


def test_pdf_own_model():
    """A job of AMLAI 1.0 (3.1): the PDF names its type and no OCEAN-AI type; no second opinion."""
    r = rep("B")
    r["model"].update({"selected": "mm", "primary": "mm", "selected_title": "AMLAI 1.0"})
    mm = {k: r["variant_scores"]["mm"][k] for k in TRAIT_KEYS}
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    for t in r["timeline"]:                    # a 3.1 job: the segments carry the own model's scores
        t.update({"members_used": ["mm"], "primary_used": "mm", "variants": {"mm": dict(mm)}, "scores": dict(mm)})
    path, mb = _build(r)
    assert path.exists() and mb["source"] == "own_model" and mb["type"] == "ISXX"
    text = _text(path)
    if text is not None:
        assert "ISXX" in text and "ISTP" in text and "ENFJ" not in text
        assert "торое мнение" not in text and caveats.text("C20")[:40] not in text


def test_pdf_old_english_job():
    path, mb = _build(english("B"))
    assert path.exists() and mb["source"] == "ocean_ai"
    text = _text(path)
    if text is not None:
        assert "среднее двух систем" not in text and "торое мнение" not in text
        assert "пороги предварительные" not in text


def test_build_pdf_computes_missing_parts():
    """A caller that passes only the raw result gets the same characterization and type."""
    r = rep("B")
    out = Path(tempfile.mkdtemp(prefix="bs3_pdf_test_")) / "raw.pdf"
    pdf_report.build_pdf(r, out)
    text = _text(out)
    assert out.exists()
    if text is not None:
        assert "Характеристика личности" in text and "ENFJ" in text
