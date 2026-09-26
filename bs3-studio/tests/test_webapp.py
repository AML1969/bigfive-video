"""The page of BS Profiler 3.1 (change request 3.1, sections 2–4) built headless, without a GPU: the «Модель» radio
(OCEAN-AI / AMLAI 1.0, no info text) and no «Объяснения» checkbox; the block labels of one model («Оценки по чертам»,
«Тип MBTI», «Модель и время обработки», the AMLAI 1.0 modality block); the compact «Характеристика личности» window
(APP_CSS: a flex item with a zero basis and a minimum height, its html-container scrolling) with the key facts as a
row under the top pair; page_outputs of a synthetic OCEAN-AI job (the note of the tab «Объяснения», the model line)
and of a job of AMLAI 1.0. Gradio is imported here (a few seconds); the Studio loads nothing until an analysis."""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from samples import rep

import bs3
from bs3 import webapp
from bs3.narrative import NO_EXPLAIN_RU
from bs3.norms import TRAIT_KEYS


def _job(r: dict, base: Path, name: str) -> dict:
    job = base / name
    job.mkdir(parents=True)
    r["job_dir"] = str(job)
    (job / "result.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    return json.loads((job / "result.json").read_text(encoding="utf-8"))


def _own(name: str = "B") -> dict:
    r = rep(name)
    r["model"].update({"selected": "mm", "primary": "mm", "selected_title": "AMLAI 1.0", "backend": "mm"})
    mm = {k: r["variant_scores"]["mm"][k] for k in TRAIT_KEYS}
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    for t in r["timeline"]:
        t.update({"members_used": ["mm"], "primary_used": "mm", "variants": {"mm": dict(mm)}, "scores": dict(mm)})
    return r


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def test_page_builds_with_the_model_radio_and_no_checkbox():
    import gradio as gr
    from bs3.pipeline import Studio
    with tempfile.TemporaryDirectory() as d:
        demo = webapp.build_app(Studio(), Path(d))
    comps = list(demo.blocks.values())
    radios = [c for c in comps if isinstance(c, gr.Radio)]
    assert len(radios) == 1
    radio = radios[0]
    # AMLAI 1.0 is the left choice and is already selected; OCEAN-AI is there for whoever wants it
    assert radio.label == "Модель" and radio.value == bs3.DEFAULT_MODEL == "mm"
    assert [c[0] for c in radio.choices] == ["AMLAI 1.0", "OCEAN-AI"] and [c[1] for c in radio.choices] == ["mm", "oceanai"]
    assert not getattr(radio, "info", None)
    assert not [c for c in comps if isinstance(c, gr.Checkbox)]                  # explanations follow the model
    labels = {getattr(c, "label", None) for c in comps}
    for lab in ("Оценки по чертам", "Тип MBTI", "Тип по ходу ролика", "Как читать тип MBTI",
                "Модель и время обработки", "Вклад модальностей в оценку модели AMLAI 1.0", "Ключевые факты",
                "Характеристика личности", "Видео"):
        assert lab in labels, lab
    for lab in ("Оценки по чертам и второе мнение", "Тип MBTI по двум системам", "Участники ансамбля и время обработки",
                "Вклад модальностей в оценку своей модели", "Язык речи",
                "Объяснения (ключевые кадры, вклад модальностей, слова)"):
        assert lab not in labels, lab
    # the analyze handler takes (video, model) only
    fns = [f for f in demo.fns.values() if getattr(f, "fn", None) is not None and f.fn.__name__ == "analyze"]
    assert len(fns) == 1 and len(fns[0].inputs) == 2
    assert [c.label for c in fns[0].inputs] == ["Видео", "Модель"]
    # the compact window: the block class, the flex rules and the scrolling container in the page CSS
    char = [c for c in comps if isinstance(c, gr.HTML) and getattr(c, "label", None) == "Характеристика личности"]
    assert len(char) == 1 and "bs3-char" in char[0].elem_classes
    css = webapp.APP_CSS
    assert f".row.bs3-pair>.column>.bs3-char{{display:flex;flex-direction:column;flex:1 1 0;min-height:{webapp.CHAR_MIN_PX}px}}" in css
    assert ".bs3-char>.html-container{flex:1 1 0;min-height:0;overflow-y:auto;overflow-x:hidden}" in css
    assert "max-height:none!important" not in css and 300 <= webapp.CHAR_MIN_PX <= 360
    # the key facts block is not inside the top pair (a row of its own under it)
    facts = [c for c in comps if isinstance(c, gr.HTML) and getattr(c, "label", None) == "Ключевые факты"][0]
    pairs = [c for c in comps if isinstance(c, gr.Row) and "bs3-pair" in (c.elem_classes or [])]
    inside = {id(x) for row in pairs for col in row.children for x in getattr(col, "children", [])}
    assert id(facts) not in inside and id(char[0]) in inside
    # the footer caveats: what holds for both models (C2 stands under the bars of a job that shows the label)
    assert webapp.FOOTER_CAVEATS == ("C1", "C10", "C3")


def test_page_outputs_oceanai_job():
    with tempfile.TemporaryDirectory() as d:
        r = rep("B")
        r["behavior_description_ru"] = "[0–20 с] Человек говорит спокойно."      # of the second model of 3.0
        r["interview"] = {"score": 0.4011, "name_ru": "впечатление «пригласить на собеседование»"}   # of the same
        r["narrative"] = "Оценки дала система OCEAN-AI …"                          # the stored summary of 2.0
        for t in r["timeline"]:
            if isinstance(t.get("scores"), dict):
                t["scores"]["interview"] = 0.4
        r = _job(r, Path(d), "20000101_000000")
        outs = webapp.page_outputs(r)
    assert len(outs) == webapp.N_PAGE == 27
    bars, facts, contrib, words, desc, members = outs[1], outs[2], outs[15], outs[16], outs[17], outs[18]
    assert _strip(contrib) == NO_EXPLAIN_RU and words == "" and desc == ""     # the one note of the tab «Объяснения»
    # the bars of one model: five traits once, no framed block («второе мнение» of 3.0) after the scale row, and no
    # label of the other model anywhere on the overview (bars, key facts, timeline chart)
    assert bars.count("<b>Экстраверсия</b>") == 1 and "второе мнение" not in bars.lower()
    assert "border-radius:8px'><div style='font-weight:600" not in bars
    for h in (bars, facts, outs[4]):
        assert "собеседовани" not in h.lower() and "AMLAI" not in h, h[:80]
    lines = members.split("\n")
    assert lines[0].startswith("Модель OCEAN-AI, веса MuPTA: открытость опыту 0.71")
    assert lines[1].startswith("Обработка заняла ")
    assert webapp.DATA_TRIMMED in lines                                          # an older job: percentiles left out
    assert "Ключевых кадров нет: модель OCEAN-AI не строит объяснений" in outs[14]
    page = " ".join(_strip(o) for o in outs if isinstance(o, str))
    for bad in ("второе мнение", "Второе мнение", "своя модель", "Своя модель", "своей модели", "MM-PSYCHE",
                "Участники ансамбля", "основная оценка", "Основная система", "среднее двух систем", "английской речи",
                "для русской речи", "язык речи"):
        assert bad not in page, bad
    assert "OCEAN-AI, веса MuPTA" in _strip(outs[24]) and "OCEAN-AI: ENFJ во всех 26 отрезках" in _strip(outs[25])


def test_page_outputs_own_model_job():
    from bs3 import caveats
    with tempfile.TemporaryDirectory() as d:
        r = _own("B")
        r["behavior_description_ru"] = "[0–20 с] Человек говорит спокойно."
        r["interview"] = {"score": 0.4011, "name_ru": "впечатление «пригласить на собеседование»"}
        r = _job(r, Path(d), "20000102_000000")
        outs = webapp.page_outputs(r)
    contrib, members, frames = outs[15], outs[18], outs[14]
    assert contrib == ""                                                        # no explanation on disk: empty, no note
    # the label of AMLAI 1.0 with its bar and C2 under the bars; a fresh job leaves nothing out of the tab «Данные»
    assert "Впечатление «пригласить на собеседование»" in outs[1] and caveats.text("C2") in outs[1]
    assert "Впечатление «собеседование»" in outs[2] and webapp.DATA_TRIMMED not in members
    assert outs[17] == "[0:00–0:20] Человек говорит спокойно."                    # the description of AMLAI 1.0 stays
    assert members.startswith("Модель AMLAI 1.0: открытость опыту 0.") and "MuPTA" not in members
    assert "Ключевые кадры не построены: лицо в кадре не найдено" in frames
    # the panel, the strip and the bars name AMLAI 1.0 only (C7 of the reading guide names both models on purpose)
    for i in (1, 24, 25):
        assert "OCEAN-AI" not in outs[i] and "MuPTA" not in outs[i], i
    assert ">AMLAI 1.0</div>" in outs[24] and "AMLAI 1.0: строгий тип ISTP во всех 33 отрезках" in _strip(outs[25])
    assert "Оценки дала модель AMLAI 1.0" in _strip(outs[22]) and "MBTI по AMLAI 1.0" in outs[3]
