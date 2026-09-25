"""Caveats C1–C22 (design 11, task T13): every one has a text, the templates are filled, word forms agree with the
numbers, none says «сегмент», and none mentions a group of processed videos or statistics of the two systems on
them (change of 2026-09-26)."""
from __future__ import annotations

import re

from bs3 import caveats
from bs3.report import DISCLAIMER_RU, INTERVIEW_DISCLAIMER_RU

PLACEHOLDER = re.compile(r"\{[a-zA-Z_]+\}")


def _all_filled() -> dict:
    out = {}
    for code in caveats.CODES:
        if code in ("C6", "C7"):
            out[code + "-ru"] = getattr(caveats, code.lower())("ru")
            out[code + "-en"] = getattr(caveats, code.lower())("en")
        elif code == "C13":
            out[code] = caveats.c13(7, 33)
        else:
            out[code] = caveats.text(code)
    return out


def test_every_caveat_has_a_text():
    assert caveats.CODES == tuple(f"C{i}" for i in range(1, 23))
    for code, t in _all_filled().items():
        assert isinstance(t, str) and len(t) > 30, code
        assert not PLACEHOLDER.search(t), (code, t)
        assert t.endswith(".") or t.endswith("»."), code


def test_c1_c2_are_the_report_disclaimers():
    assert caveats.text("C1") == DISCLAIMER_RU
    assert caveats.text("C2") == INTERVIEW_DISCLAIMER_RU


def test_no_segment_word_and_no_type_claim():
    for code, t in _all_filled().items():
        assert "сегмент" not in t.lower(), code
        # C3 says the opposite on purpose: «Система не определяет тип личности»
        if "определяет тип" in t:
            assert code == "C3" and "не определяет тип личности" in t


def test_no_reference_group():
    for code, t in _all_filled().items():
        low = t.lower()
        for w in ("опорн", "положени", "процентил", "русских ролик", "роликов, обработанных", "предварительн",
                  "типичн", "плохо согласуются", "из 13", "медиан"):
            if code == "C6-en" and w == "процентил":        # FIV2 percentiles on the English bars do not set letters
                continue
            assert w not in low, (code, w)


def test_c5_c6_c9_absolute_scale():
    assert "от 0.36 до 0.64" in caveats.text("C5") and "0.5" in caveats.text("C5")
    for lang in ("ru", "en"):
        assert "середина шкалы 0.5" in caveats.c6(lang)
    assert "у края шкалы" in caveats.text("C9")
    assert "Основной считается OCEAN-AI" in caveats.c7("ru") and "граница" not in caveats.c7("ru")
    assert caveats.text("C20").endswith("построены по своей модели.")


def test_c13_word_forms():
    assert caveats.c13(7, 33).startswith("В 7 отрезках из 33 система OCEAN-AI не дала оценки")
    assert "эти отрезки не вошли" in caveats.c13(7, 33) and "показаны пропусками" in caveats.c13(7, 33)
    one = caveats.c13(1, 18)
    assert one.startswith("В 1 отрезке из 18") and "этот отрезок не вошёл" in one and "показан пропуском" in one
    assert caveats.c13(21, 40).startswith("В 21 отрезке из 40")


def test_text_fills_templates_by_hand():
    t = caveats.text("C13", k=2, segments_k="отрезках", n=5, these="эти отрезки не вошли", shown="показаны пропусками")
    assert t.startswith("В 2 отрезках из 5")
    assert caveats.text("C6", "en") == caveats.C6_EN and caveats.text("C7", "en") == caveats.C7_EN
