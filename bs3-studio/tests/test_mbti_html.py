"""The tab «Тип MBTI» (design 10.3, 5.5–5.7; task T18) on the numeric copies of samples A and B: panels, agreement
line and signs, neuroticism, the letter strip with gaps and its summary line, C18/C19, the reading guide with the
caveats, the short emotion paragraph; no «сегмент» and nothing about a group of processed videos anywhere (change of
2026-09-26: letters on the absolute scale, the tracks show the score itself)."""
from __future__ import annotations

import copy
import re

from samples import english, rep

from bs3 import caveats, mbti, mbti_html, scores

AN_B = {"voice": {"mean": {"arousal": 0.1536, "dominance": 0.2612, "valence": 0.3717}},
        "emotions_text": {"mean": {"joy": 0.0611, "surprise": 0.0049, "neutral": 0.8604, "sadness": 0.0233,
                                   "fear": 0.0036, "anger": 0.0281, "disgust": 0.0185}}}


def _mb(r: dict) -> dict:
    return mbti.get_mbti(r, scores.clean_view(r))


RELATIVE = ("опорн", "положени", "типичн", "русских роликов", "обработанных", "предварительн", "большинства")


def _cells(strip: str, axis: str) -> int:
    return len(re.findall(rf"title='[^']*· {axis}: [A-Z], ", strip))


def test_types_b():
    h = mbti_html.types_html(_mb(rep("B")))
    for s in ("OCEAN-AI (веса MuPTA) — основная оценка", "Своя модель — второе мнение",
              "«Наставник»", "С учётом границ: ENFJ (пограничных осей нет)", "С учётом границ: ISXX",
              "Нейротизм — средний уровень. В MBTI этой шкалы нет, поэтому он приводится отдельно.",
              "Уверенных совпадений нет; расходятся E–I, S–N; на границе хотя бы у одной из систем: T–F, J–P.",
              "I · интроверсия", "экстраверсия · E", "соответствие шкал r ≈ 0.74",
              "E · умеренно (уверенность 0.46) · экстраверсия 0.73 — выше среднего",
              "F · отчётливо (уверенность 0.74) · доброжелательность 0.87 — высокий уровень",
              "X (ближе к P) · на границе · добросовестность 0.48 — средний уровень"):
        assert s in h, s
    # the marker of extraversion at the score 0.73 on the track 0…1
    assert "left:calc(73.0% - 7px)" in h
    # signs in the second panel, one per axis: ≠ on E–I and S–N, ≈ on the others
    assert h.count("</b> расходится") == 2 and h.count("</b> на границе хотя бы у одной из систем") == 2
    assert "minmax(min(320px,100%),1fr)" in h                        # panels wrap one under the other on a phone
    assert "сегмент" not in h
    for w in RELATIVE:
        assert w not in h, w


def test_types_a_and_none():
    h = mbti_html.types_html(_mb(rep("A")))
    assert "«Наставник»" in h and "С учётом границ: XNFJ · возможен INFJ («Советник»)" in h
    assert "X (ближе к E) · на границе · экстраверсия 0.56 — средний уровень" in h
    assert "С учётом границ: IXXX · тип не выражен: 3 оси из 4 на границе" in h
    assert "Уверенных совпадений нет; на границе хотя бы у одной из систем: E–I, S–N, T–F, J–P." in h
    # the own model's agreeableness 0.47531: printed 0.48 as on the bars, not 0.47 through a stored 0.475
    assert "X (ближе к T) · на границе · доброжелательность 0.48 — средний уровень" in h
    assert mbti_html.types_html(None) == f"<p style='{mbti_html.TEXT14};margin:0'>{caveats.text('C21')}</p>"


def test_types_en():
    h = mbti_html.types_html(_mb(english("B")))
    for s in ("Итог: среднее двух систем", "OCEAN-AI (веса First Impressions V2)", "Своя модель (First Impressions V2)"):
        assert s in h, s
    assert "пороги предварительные" not in h
    assert re.search(r"(отчётливо|умеренно) \(уверенность 0\.\d\d\)", h)   # the confidence, labelled


def test_strip_b():
    s = mbti_html.strip_html(_mb(rep("B")))
    for ax in ("E–I", "S–N", "T–F", "J–P"):
        assert _cells(s, ax) == 26, ax
    assert s.count("нет оценки OCEAN-AI") == 7 * 4
    assert "Основная система: ENFJ во всех 26 отрезках с оценкой; все четыре оси совпадают с итогом во всех отрезках." in s
    assert caveats.text("C8") in s and caveats.text("C18") in s
    assert "overflow-x:auto" in s and "grid-template-columns:56px repeat(33,26px)" in s
    assert "сегмент" not in s


def test_strip_a_short_and_new_jobs():
    s = mbti_html.strip_html(_mb(rep("A")))
    # the strict letters never change, but no segment has a confident ENFJ: the line says so
    assert ("Основная система: строгий тип ENFJ во всех 17 отрезках с оценкой; строгие буквы всех четырёх осей "
            "совпадают с итогом во всех отрезках; ось E–I на границе во всех 17 отрезках, S–N — в 7, T–F — в 6, "
            "J–P — в 6.") in s
    assert "ENFJ во всех 17 отрезках с оценкой; все четыре оси" not in s
    r = rep("B")
    r["timeline"] = []
    r["segments"] = 1
    assert caveats.text("C19") in mbti_html.strip_html(_mb(r))
    # a 3.0 job keeps the own model per segment: its strip is shown and C18 is not
    r = rep("B")
    for t in r["timeline"]:
        t["variants"] = {"mm": copy.deepcopy(r["variant_scores"]["mm"])}
        if "oceanai" in t["members_used"]:
            t["variants"]["oceanai"] = copy.deepcopy(t["scores"])
    s = mbti_html.strip_html(_mb(r))
    assert "Своя модель — второе мнение" in s and caveats.text("C18") not in s
    assert "Своя модель: строгий тип ISTP во всех 33 отрезках с оценкой" in s


def test_read():
    h = mbti_html.read_html(_mb(rep("B")))
    for code in ("C3", "C4", "C5", "C9", "C16"):
        assert caveats.text(code) in h, code
    assert caveats.c6("ru") in h and caveats.c7("ru") in h
    for s in ("высокое, r ≈ 0.74", "высокое, r ≈ 0.72", "среднее, r ≈ 0.44", "среднее, r ≈ 0.49", "в MBTI не выражается",
              "McCrae, Costa, 1989"):
        assert s in h, s
    assert caveats.c6("en") in mbti_html.read_html(_mb(english("B")))
    for w in RELATIVE:
        assert w not in h, w


def test_emo_intro():
    r = rep("B")
    assert mbti_html.emo_intro_html(r) == ""
    r["analyses"] = copy.deepcopy(AN_B)
    h = mbti_html.emo_intro_html(r)
    assert "По содержанию речи" in h and "Голос (модель эмоций в речи, шкала 0…1)" in h and h.count("<p") == 1
