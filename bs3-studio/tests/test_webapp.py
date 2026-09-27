"""The page of BS Profiler 3.1 (change request 3.1, sections 2–4) built headless, without a GPU: the «Модель» radio
(OCEAN-AI / AMLAI 1.0, no info text) and no «Объяснения» checkbox; the block labels of one model («Оценки по чертам»,
«Тип MBTI», «Модель и время обработки», the AMLAI 1.0 modality block); the compact «Характеристика личности» window
(APP_CSS: a flex item with a zero basis and a minimum height, its html-container scrolling) with the key facts as a
row under the top pair; page_outputs of a synthetic OCEAN-AI job (the note of the tab «Объяснения», the model line)
and of a job of AMLAI 1.0. Gradio is imported here (a few seconds); the Studio loads nothing until an analysis.
What Gradio serves (in-process through TestClient, no port): no file of a job folder by /gradio_api/file=, the PDF of
the button from a copy in Gradio's own temp folder, which Gradio deletes with its other temp files."""
from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from datetime import timedelta
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


@contextlib.contextmanager
def _gradio_temp(base: Path):
    """GRADIO_TEMP_DIR -> base/gradio while the test runs (Gradio reads it when the page and the app are built), the
    old value back afterwards."""
    old = os.environ.get("GRADIO_TEMP_DIR")
    up = base / "gradio"
    up.mkdir()
    os.environ["GRADIO_TEMP_DIR"] = str(up)
    try:
        yield up
    finally:
        if old is None:
            os.environ.pop("GRADIO_TEMP_DIR", None)
        else:
            os.environ["GRADIO_TEMP_DIR"] = old


JOB_FILES = ("input.mp4", "segments/seg01_0-20s.mp4", "segments/audio16k.wav", "segments/timeline.json", "result.json",
             "explain/explanation.json", "explain/key_01_frame10.jpg", "charts/chart_profile.png",
             f"{bs3.PRODUCT_SLUG}_report_video.pdf")


def _job_folder(work_dir: Path) -> Path:
    """A finished job folder with one file of every kind a real one holds (made-up content)."""
    job = work_dir / "20000101_000000"
    for name in JOB_FILES:
        p = job / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"%PDF-1.4\n%%EOF\n" if name.endswith(".pdf") else name.encode())
    return job


def _client(demo):
    """The app as launch() without allowed_paths serves it, in-process (no port)."""
    from fastapi.testclient import TestClient
    from gradio.routes import App
    demo.allowed_paths, demo.blocked_paths = [], []
    return TestClient(App.create_app(demo))


def test_file_route_serves_no_job_file():
    """/gradio_api/file=<path> gives 403 for every file of a job folder: the uploaded video, the segments, the
    transcript in result.json, the key frames, the charts and the PDF. The page needs none of them (key frames are
    data URIs, charts are srcdoc); the old launch(allowed_paths=[work dir]) served them to anyone with the path."""
    from bs3.pipeline import Studio
    with tempfile.TemporaryDirectory() as d, _gradio_temp(Path(d)):
        wd = Path(d) / "web_jobs"
        job = _job_folder(wd)
        demo = webapp.build_app(Studio(), wd)
        client = _client(demo)
        files = sorted(p for p in job.rglob("*") if p.is_file())
        assert len(files) == len(JOB_FILES)
        for f in files:
            r = client.get(f"/gradio_api/file={f}")
            assert r.status_code == 403, (str(f.relative_to(job)), r.status_code)
        # the control: the same request with the old allowed_paths gets the video, so the 403 above is the route's
        demo.allowed_paths = [str(wd)]
        assert client.get(f"/gradio_api/file={job / 'input.mp4'}").status_code == 200


def test_pdf_for_download():
    """The PDF button hands Gradio a copy of the stored PDF in Gradio's upload folder, in a new random folder per
    click, under the same file name; the stored PDF stays in the job folder. Gradio serves the copy and takes it as
    the button's output from any start directory, which the stored PDF itself does not pass without allowed_paths;
    Gradio's hourly pass (delete_cache) deletes the copy with its other temp files after 22 hours."""
    import gradio as gr
    from gradio import processing_utils, route_utils
    from gradio.context import LocalContext
    from gradio.exceptions import InvalidPathError
    from gradio.utils import get_upload_folder
    from bs3.pipeline import Studio
    with tempfile.TemporaryDirectory() as d, _gradio_temp(Path(d)) as up:
        base = Path(d)
        job = _job_folder(base / "web_jobs")
        pdf = job / JOB_FILES[-1]
        calls = []
        export_pdf = webapp.export_pdf
        webapp.export_pdf = lambda job_dir: calls.append(job_dir) or str(pdf)
        try:
            a, b = Path(webapp.pdf_for_download(job)), Path(webapp.pdf_for_download(str(job)))
        finally:
            webapp.export_pdf = export_pdf
        assert calls == [job, str(job)]
        for p in (a, b):
            assert p.parent.parent == Path(get_upload_folder()) == up and re.fullmatch(r"[0-9a-f]{32}", p.parent.name)
            assert p.name == pdf.name and p.read_bytes() == pdf.read_bytes() and job not in p.parents
        assert a.parent != b.parent and pdf.exists()

        demo = webapp.build_app(Studio(), base / "web_jobs")
        client = _client(demo)
        r = client.get(f"/gradio_api/file={a}")
        # Gradio 5.8 sends a file of its own folder that is not an image, audio, video, text or json as an attachment
        # of type application/octet-stream, the PDF too (it did the same with its cache copy of the stored PDF)
        assert r.status_code == 200 and r.content == pdf.read_bytes()
        disposition = r.headers["content-disposition"]
        assert disposition.startswith("attachment") and pdf.name in disposition

        # the button's output through Gradio's own check, as the launched app runs it after make_pdf, started from a
        # directory that holds neither the job folder nor the system temp folder
        btn = next(c for c in demo.blocks.values() if isinstance(c, gr.DownloadButton))
        elsewhere = base / "elsewhere"
        elsewhere.mkdir()
        cwd, tmpdir = os.getcwd(), tempfile.tempdir
        token = LocalContext.blocks.set(demo)
        demo.has_launched = True
        os.chdir(elsewhere)
        tempfile.tempdir = str(elsewhere)
        try:
            out = processing_utils.move_files_to_cache(btn.postprocess(str(a)), btn, postprocess=True)
            refused = False
            try:
                processing_utils.move_files_to_cache(btn.postprocess(str(pdf)), btn, postprocess=True)
            except InvalidPathError:
                refused = True
        finally:
            os.chdir(cwd)
            tempfile.tempdir = tmpdir
            demo.has_launched = False
            LocalContext.blocks.reset(token)
        assert refused, "the stored PDF passed without allowed_paths: the check above proves nothing"
        assert out["path"] == str(a) and out["orig_name"] == pdf.name
        assert client.get(out["url"]).status_code == 200

        # delete_cache: the hourly pass of Gradio deletes the copy once it is older than the age, and keeps it an hour
        # after the click. Gradio 5.8 compares timedelta.seconds (the part under a day), so the age stays under a day
        # with room for two passes; a full day (86400) would never delete anything
        frequency, age = demo.delete_cache
        assert (frequency, age) == (3600, 79200) and age <= 86400 - 2 * frequency
        real_datetime = route_utils.datetime
        shift = timedelta(0)

        class Later(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return real_datetime.now(tz) + shift

        route_utils.datetime = Later
        try:
            shift = timedelta(hours=1)
            route_utils.delete_files_created_by_app(demo, age)
            assert a.exists()
            shift = timedelta(seconds=age + frequency)
            route_utils.delete_files_created_by_app(demo, age)
            assert not a.exists()
        finally:
            route_utils.datetime = real_datetime
        assert pdf.exists()                                                     # the job folder is not Gradio's


def test_launch_without_allowed_paths():
    """The web app and the preview start Gradio without allowed_paths (what the two tests above rely on)."""
    root = Path(webapp.__file__).resolve().parents[1]
    for f in (root / "bs3" / "webapp.py", root / "scripts" / "ui_preview.py"):
        src = f.read_text(encoding="utf-8")
        assert ".launch(" in src and "allowed_paths=" not in src, f.name
