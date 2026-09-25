"""«Как получены оценки» and the score bars on the clean view (design 9, 6.2; task T15)."""
from __future__ import annotations

from samples import english, rep

from bs3 import narrative, scores, webparts

FOOT = ("Пять черт — относительно предварительной опорной группы: 13 русских роликов, обработанных системой до "
        "25.09.2026. Роликов в сравнении пока 13: этого мало для процентов, поэтому положение описано словами и риска "
        "не ставится.")


def test_method_notes_sample_b():
    v = scores.clean_view(rep("B"))
    t = narrative.method_notes(v, None)
    assert t.startswith("Основные оценки дала система OCEAN-AI на весах MuPTA, обученных на русскоязычных участниках; "
                        "своя модель MM-PSYCHE показана как второе мнение и в основные оценки не входит.")
    assert ("Уровни черт и буквы MBTI считаются по положению оценки среди 13 русских роликов, обработанных системой до "
            "25.09.2026, отдельно для каждой системы: это предварительная опорная группа.") in t
    assert "По ходу ролика (26 отрезков с оценкой OCEAN-AI) оценки устойчивы: разброс не больше ±0.02." in t
    assert "В 7 отрезках из 33 система OCEAN-AI не дала оценки" in t
    assert "не противоречие в выводах" not in t
    assert "Своя модель обучена на англоязычных роликах First Impressions V2, и на русской речи её числа в среднем на" in t
    assert "каждая система сравнивается со своей опорной группой, а не по числам." in t
    assert "сегмент" not in t


def test_method_notes_sample_a_and_english():
    a = narrative.method_notes(scores.clean_view(rep("A")))
    assert "(17 отрезков с оценкой OCEAN-AI)" in a and "В 1 отрезке из 18" in a
    en = narrative.method_notes(scores.clean_view(english("B")))
    assert en.startswith("Оценки — среднее двух систем, OCEAN-AI (веса First Impressions V2) и своей модели MM-PSYCHE")
    assert "обучающей выборке First Impressions V2 (6000 роликов)" in en
    assert "OCEAN-AI не дала оценки" not in en and "предварительная" not in en


def test_bars_on_clean_view_ru():
    v = scores.clean_view(rep("B"))
    h = webparts._bar_html(v["traits"], v.get("interview"))
    assert "0.73 · выше, чем у большинства из 13 русских роликов" in h
    assert "0.71 · примерно посередине среди 13 русских роликов" in h            # openness 0.712, p 0.429
    assert FOOT in h and h.count("Роликов в сравнении пока") == 1
    assert "относительно Пять черт" not in h


def test_second_opinion_ru_same_group():
    v = scores.clean_view(rep("B"))
    h = webparts._members_html(v)
    assert "Второе мнение: своя модель MM-PSYCHE (положение среди тех же русских роликов)" in h
    assert "0.25 · ниже, чем у большинства из 13 русских роликов" in h           # own model E 0.2515, p 0.214
    assert "Сравнивайте положение в группе и порядок черт, а не сами числа." in h
    assert "положение среди людей FIV2" not in h


def test_second_opinion_english_unchanged():
    h = webparts._members_html(scores.clean_view(english("B")))
    assert "Участники ансамбля: итоговая оценка — их среднее" in h


def test_pct_phrases_65_35():
    ref = "ref:ru_prov_2026-09-25"
    assert webparts._pct_phrase(92.9, ref) == ("выше, чем у большинства из 13 русских роликов", False)
    assert webparts._pct_phrase(64.0, ref)[0] == "примерно посередине среди 13 русских роликов"
    assert webparts._pct_phrase(35.0, ref)[0] == "ниже, чем у большинства из 13 русских роликов"
    assert webparts._pool_size(ref) == 13 and webparts._ref_ru(ref) == FOOT
    pool = "пула обработанных русских роликов (N=5)"
    assert webparts._pct_phrase(62, pool)[0] == "примерно посередине среди 5 русских роликов"
    assert webparts._pct_phrase(65, pool)[0] == "выше, чем у большинства из 5 русских роликов"
    assert narrative._pct_phrase({"percentile": 92.9, "percentile_ref": ref}) == \
        "выше, чем у большинства из 13 русских роликов"
    assert narrative._pct_phrase({"percentile": 62, "percentile_ref": pool}) == "примерно посередине среди 5 русских роликов"
    assert webparts._pct_phrase(72, "train FIV2") == ("выше, чем у 72% людей в FIV2", True)
