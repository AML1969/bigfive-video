"""«Как получены оценки» and the score bars on the clean view (design 9, 6.2; task T15; change of 2026-09-26: no
reference group, Russian bars show the score only)."""
from __future__ import annotations

from samples import english, rep

from bs3 import narrative, scores, webparts

RELATIVE = ("опорн", "положени", "русских роликов", "обработанных", "большинства", "предварительн", "типичн")


def test_method_notes_sample_b():
    v = scores.clean_view(rep("B"))
    t = narrative.method_notes(v, None)
    assert t.startswith("Основные оценки дала система OCEAN-AI на весах MuPTA, обученных на русскоязычных участниках; "
                        "своя модель MM-PSYCHE показана как второе мнение и в основные оценки не входит.")
    assert ("Уровни черт и буквы MBTI считаются по самой оценке системы на шкале от 0 до 1 с серединой 0.5, отдельно "
            "для каждой системы.") in t
    assert "По ходу ролика (26 отрезков с оценкой OCEAN-AI) оценки устойчивы: разброс не больше ±0.02." in t
    assert "В 7 отрезках из 33 система OCEAN-AI не дала оценки" in t
    assert "не противоречие в выводах" not in t
    assert "Своя модель обучена на англоязычных роликах First Impressions V2, и на русской речи её числа в среднем на" in t
    assert "это разница шкал двух систем, поэтому их оценки не усредняются, а тип MBTI каждой показан отдельно." in t
    assert "сегмент" not in t
    for w in RELATIVE:
        assert w not in t, w


def test_method_notes_sample_a_and_english():
    a = narrative.method_notes(scores.clean_view(rep("A")))
    assert "(17 отрезков с оценкой OCEAN-AI)" in a and "В 1 отрезке из 18" in a
    en = narrative.method_notes(scores.clean_view(english("B")))
    assert en.startswith("Оценки — среднее двух систем, OCEAN-AI (веса First Impressions V2) и своей модели MM-PSYCHE")
    assert "обучающей выборке First Impressions V2 (6000 роликов)" in en and "от 0 до 1 с серединой 0.5" in en
    assert "OCEAN-AI не дала оценки" not in en and "предварительная" not in en


def test_bars_on_clean_view_ru():
    r = rep("B")
    for k in r["traits"]:              # an older job: percentiles against the pool of processed videos
        r["traits"][k].update({"percentile": 80.0, "percentile_ref": "пула обработанных русских роликов (N=5)"})
    v = scores.clean_view(r)
    h = webparts._bar_html(v["traits"], v.get("interview"))
    assert "Экстраверсия</b> <span style='margin-left:auto;text-align:right;font-variant-numeric:tabular-nums'>0.73<" in h
    assert "0.71<" in h                                                        # openness 0.712
    assert webparts.SCALE_NOTE in h
    for w in RELATIVE + ("процентил", "риска", "сейчас их"):
        assert w not in h.lower(), w
    assert "left:calc(" not in h                                               # no percentile tick


def test_bars_english_fiv2_percentile():
    r = english("B")
    for k in r["traits"]:
        r["traits"][k].update({"percentile": 72.0, "percentile_ref": "train First Impressions V2 (6000 клипов)"})
    v = scores.clean_view(r)
    h = webparts._bar_html(v["traits"], v.get("interview"))
    assert "выше, чем у 72% людей в FIV2" in h and "процентиль в First Impressions V2" in h
    assert "Все процентили — относительно обучающей выборки First Impressions V2 (6000 клипов)." in h
    assert webparts.TICK_NOTE in h and "опорн" not in h


def test_second_opinion_ru_score_only():
    v = scores.clean_view(rep("B"))
    h = webparts._members_html(v)
    assert "Второе мнение: своя модель MM-PSYCHE (своя шкала, обучена на First Impressions V2)" in h
    assert "0.25<" in h                                                        # own model E 0.2515
    assert "Сравнивайте порядок черт, а не сами числа." in h
    for w in RELATIVE + ("процентил",):
        assert w not in h.lower(), w


def test_second_opinion_english_unchanged():
    h = webparts._members_html(scores.clean_view(english("B")))
    assert "Участники ансамбля: итоговая оценка — их среднее" in h


def test_pct_phrases_fiv2_only():
    for ref in ("ref:ru_prov_2026-09-25", "пула обработанных русских роликов (N=5)", ""):
        assert webparts._pct_phrase(92.9, ref) == ("", False)
        assert narrative._pct_phrase({"percentile": 92.9, "percentile_ref": ref}) == ""
    assert webparts._pct_phrase(None, "train FIV2") == ("", False)
    assert webparts._pct_phrase(30, "train FIV2") == ("ниже, чем у 70% людей в FIV2", True)
    assert narrative._pct_phrase({"percentile": 72, "percentile_ref": "train First Impressions V2 (6000 клипов)"}) == \
        "выше, чем у 72% людей в First Impressions V2"
    assert webparts._pct_phrase(72, "train FIV2") == ("выше, чем у 72% людей в FIV2", True)
