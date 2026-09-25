"""Caveats C1–C22 (design 11, task T13): every one has a text, the templates are filled, word forms agree with the
numbers, and none says «сегмент»."""
from __future__ import annotations

import re

from bs3 import caveats, refnorms
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


def test_c6_ru_numbers_from_the_frozen_group():
    t = caveats.c6("ru")
    ref = refnorms.describe(refnorms.reference_for("oceanai", "ru"))
    assert f"среди {ref['n']} русских роликов, обработанных системой до 25.09.2026" in t
    assert "First Impressions V2" in caveats.c6("en")


def test_c7_ru_numbers_from_the_norms_file():
    t = caveats.c7("ru")
    st = refnorms.agreement_stats()
    assert st["letters_same"] == {"EI": 5, "SN": 7, "TF": 3, "JP": 3}
    assert "На 13 русских роликах их оценки пока плохо согласуются" in t
    assert "по оси E–I в 5 роликах из 13, по S–N — в 7, по T–F — в 3, по J–P — в 3." in t
    one = caveats.c7("ru", {"n": 21, "letters_same": {"EI": 1, "SN": 2, "TF": 3, "JP": 4}})
    assert "На 21 русском ролике" in one and "в 1 ролике из 21" in one


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
