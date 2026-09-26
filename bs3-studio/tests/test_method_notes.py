"""«Как получены оценки» and the score bars on the clean view (design 9, 6.2; task T15; change of 2026-09-26: no
reference group, Russian bars show the score only)."""
from __future__ import annotations

import re

from samples import english, rep

from bs3 import narrative, scores, webparts

RELATIVE = ("опорн", "положени", "русских роликов", "обработанных", "большинства", "предварительн", "типичн",
            "на русской речи её", "на русских роликах", "порядок черт")


SECOND = ("второе мнение", "своя модель", "своей модели", "двух систем", "каждой системы", "не усредняются")


def test_method_notes_sample_b():
    v = scores.clean_view(rep("B"))
    t = narrative.method_notes(v, None)
    assert t.startswith("Оценки дала система OCEAN-AI на весах MuPTA, обученных на русскоязычных участниках. "
                        "Уровни черт и буквы MBTI считаются по самой оценке модели на шкале от 0 до 1 с серединой 0.5.")
    assert "По ходу ролика (26 отрезков с оценкой OCEAN-AI) оценки устойчивы: разброс не больше ±0.02." in t
    assert "В 7 отрезках из 33 система OCEAN-AI не дала оценки" in t
    assert "не противоречие в выводах" not in t and "MM-PSYCHE" not in t
    assert "сегмент" not in t
    for w in RELATIVE + SECOND:
        assert w not in t.lower(), w


def test_method_notes_own_model_and_old_english_job():
    a = narrative.method_notes(scores.clean_view(rep("A")))
    assert "(17 отрезков с оценкой OCEAN-AI)" in a and "В 1 отрезке из 18" in a
    r = rep("B")
    r["model"].update({"selected": "mm", "primary": "mm"})
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    own = narrative.method_notes(scores.clean_view(r))
    assert own.startswith("Оценки дала модель AMLAI 1.0, построенная по рецепту MM-PSYCHE и обученная на First "
                          "Impressions V2; транскрипт русской речи для неё переведён на английский.")
    assert "(33 отрезка с оценкой AMLAI 1.0)" in own and "OCEAN-AI" not in own
    for w in RELATIVE + SECOND:
        assert w not in own.lower(), w
    # OCEAN-AI recorded but gave nothing: C20 instead of the source sentence
    r = rep("B")
    del r["variant_scores"]["oceanai"]
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    fell = narrative.method_notes(scores.clean_view(r))
    assert fell.startswith("Основная система OCEAN-AI не дала оценок по этому ролику")
    en = narrative.method_notes(scores.clean_view(english("B")))
    assert en.startswith("Оценки дала система OCEAN-AI") and "среднее двух систем" not in en
    assert "обучающей выборке First Impressions V2 (6000 роликов)" in en and "от 0 до 1 с серединой 0.5" in en
    assert "предварительная" not in en


def test_build_narrative_names_one_model():
    """The stored 2.0-style summary of a new job names the model that ran and no second opinion."""
    r = rep("B")
    r["model"].update({"selected": "oceanai"})
    t = narrative.build_narrative(r, None)
    assert t.startswith("Оценки дала система OCEAN-AI на весах MuPTA")
    for w in SECOND + ("MM-PSYCHE", "разница шкал"):
        assert w not in t.lower() and w not in t, w
    r["model"].update({"selected": "mm", "primary": "mm"})
    r["interview"] = {"score": 0.41}
    t = narrative.build_narrative(r, None)
    assert t.startswith("Оценки дала модель AMLAI 1.0") and "по модели AMLAI 1.0: 0.41" in t


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


def test_no_second_opinion_block():
    """One model in the view (3.1): the framed block under the bars is empty for any job, old or new."""
    assert webparts._members_html(scores.clean_view(rep("B"))) == ""
    assert webparts._members_html(scores.clean_view(english("B"))) == ""
    r = rep("B")
    r["model"].update({"selected": "mm", "primary": "mm"})
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    assert webparts._members_html(scores.clean_view(r)) == ""


def test_pct_phrases_fiv2_only():
    for ref in ("ref:ru_prov_2026-09-25", "пула обработанных русских роликов (N=5)", ""):
        assert webparts._pct_phrase(92.9, ref) == ("", False)
        assert narrative._pct_phrase({"percentile": 92.9, "percentile_ref": ref}) == ""
    assert webparts._pct_phrase(None, "train FIV2") == ("", False)
    assert webparts._pct_phrase(30, "train FIV2") == ("ниже, чем у 70% людей в FIV2", True)
    assert narrative._pct_phrase({"percentile": 72, "percentile_ref": "train First Impressions V2 (6000 клипов)"}) == \
        "выше, чем у 72% людей в First Impressions V2"
    assert webparts._pct_phrase(72, "train FIV2") == ("выше, чем у 72% людей в FIV2", True)
