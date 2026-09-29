"""BS Profiler 3.1 web UI (Gradio): Big Five by the model chosen for the analysis (OCEAN-AI or AMLAI 1.0, Russian
speech) + emotions, voice, face and speech analytics with interactive charts.

Run:  bs3 web [--port 7880]      (inside WSL; open http://localhost:7880 on Windows). Независим от BS 2.0 (:7870).

Readability rules for the HTML blocks (both Gradio themes, see palette.py): text colours are inherited from the theme,
secondary text is the same colour at opacity .75 and at least 13 px, marks and outlines come from palette.HTML, and
every block shows its title (show_label=True, container=True).
"""
from __future__ import annotations

import logging
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

from . import DEFAULT_MODEL, MODEL_TITLES, PRODUCT, caveats, characterization, jobview, journal, settings
from .palette import (ACCENT, BUTTON_PRIMARY, BUTTON_PRIMARY_HOVER, BUTTON_STOP, BUTTON_STOP_HOVER, HTML as PAL,
                      PAGE_NOTE_OPACITY, SUBDUED_TEXT_LIGHT)
from .pdf import export_pdf
from .pipeline import Studio, run_analysis
from .textfmt import fmt_secs
from .web.page import N_PAGE, page_values

log = logging.getLogger("bs3.web")


def pdf_for_download(job_dir: str | Path) -> str:
    """The PDF button: the PDF of export_pdf (it stays in the job folder) copied into a new folder with a random name
    in Gradio's upload folder. Gradio serves its own folder without allowed_paths, from any start directory, and
    deletes the copy with its other temp files (delete_cache in build_app). The download keeps the PDF file name."""
    pdf = Path(export_pdf(job_dir))
    from gradio.utils import get_upload_folder
    d = Path(get_upload_folder()) / secrets.token_hex(16)
    d.mkdir(parents=True)
    out = d / pdf.name
    shutil.copy2(pdf, out)
    return str(out)


def analysis_error_ru(e: BaseException) -> str:
    """A failed analysis in words for the error dialog: the exceptions of the models are English (and show_error=True
    would print them as they are); the original goes to the server log."""
    import subprocess
    msg = str(e)
    low = msg.lower()
    if "out of memory" in low:
        return "Не хватило памяти видеокарты. Подождите минуту и запустите анализ заново."
    if "no segment could be analysed" in low or "no predictions for any file" in low or "no frames decoded" in low:
        return "В ролике не найдено ни лица, ни речи, поэтому оценить его нельзя. Проверьте файл."
    if "all ensemble members failed" in low:
        return ("Модель не смогла обработать ролик: чаще всего в кадре не найдено лицо или не слышна речь. "
                "Проверьте файл.")
    if isinstance(e, subprocess.CalledProcessError):
        return "Не удалось прочитать видеофайл: возможно, он повреждён или записан в неподдерживаемом формате."
    if re.search(r"[А-Яа-яЁё]", msg) and not re.search(r"[A-Za-z]", msg):
        return msg                                  # messages of this package are already Russian
    return "Не удалось обработать ролик из-за внутренней ошибки. Подробности записаны в журнал сервера."


STATUS_LABELS = {"running": "Идёт обработка", "done": "Готово", "stopped": "Остановлено", "error": "Ошибка"}


def _live_desc(state: dict) -> str:
    """«[1 мин 05 с] Отрезок 3/29 (0:40–1:00)»: the time since the start, counted anew on every refresh, so it keeps
    running during a long stage such as model loading (stage messages carry the time at which the stage began)."""
    e = time.time() - state["t0"]
    when = f"{int(e)} с" if e < 60 else fmt_secs(e)
    return f"[{when}] " + re.sub(r"^\[[^\]]*\]\s*", "", str(state["desc"]))


def _status_html(frac: float, desc: str, state: str = "running", label: str | None = None) -> str:
    """Progress line above the page. state: running (orange, «34% · …»), done (green), stopped or error (red).
    The text shows the real percentage (0% at the start); the bar keeps a 2% minimum width so it is visible.
    The other states fill the whole outlined track in their colour; the label and a colour dot name the state."""
    state = state if state in STATUS_LABELS else "running"
    pct = max(0, min(100, int(round(float(frac) * 100))))
    color = PAL[f"status_{state}"]
    head = label or STATUS_LABELS[state]
    if state == "running":
        width, txt = max(2, pct), f"{pct}% · {desc}"
    else:
        width, txt = 100, desc
    return (f"<div role='status' aria-live='polite' style='margin:4px 0 8px'>"
            "<div style='display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:2px 12px;"
            "font-size:14px;margin-bottom:5px'>"
            f"<b style='white-space:nowrap'><span aria-hidden='true' style='display:inline-block;width:10px;height:10px;"
            f"border-radius:50%;background:{color};margin-right:7px'></span>{head}</b> "
            f"<span style='text-align:right;font-variant-numeric:tabular-nums'>{txt}</span></div>"
            "<div role='progressbar' aria-valuemin='0' aria-valuemax='100' "
            # stopped and error fill the track without having got that far: no aria-valuenow, the text says what happened
            + (f"aria-valuenow='{pct}' " if state not in ("stopped", "error") else "") + f"aria-label='{head}' "
            f"style='height:10px;border-radius:6px;background:{PAL['track']};box-shadow:inset 0 0 0 1px {PAL['track_outline']}'>"
            f"<div style='width:{width}%;height:10px;border-radius:6px;background:{color};transition:width .5s'></div>"
            "</div></div>")


# the Code icon Gradio puts into every gr.HTML label chip reads as «code»: hide it on the result blocks.
# Chart blocks: charts.plot_html puts the chart title (bold 15 px div) and its subtitle above the iframe; the block chip
# already shows the same title, so the bold title line is hidden there and the subtitle (units, scale) stays.
APP_CSS = (".bs3-block > label[data-testid='block-label'] > span{display:none}"
           ".prose.bs3-chart > div:first-child[style*='font-weight:600']{display:none}")
COPYRIGHT = "© 2026 AMLAI"
# the copyright line under the title: 12 px, theme text colour dimmed (8.8:1 on the dark page, 5.6:1 on white), pulled
# up to the title so it reads as part of the heading, with the usual gap before the description line
APP_CSS += (f".gradio-container .prose p.bs3-copy{{font-size:12px;line-height:16px;letter-spacing:.03em;"
            f"opacity:{PAGE_NOTE_OPACITY};margin:-8px 0 12px}}")
# Gradio paints several marks with the theme accent, orange-500: the name and the underline of the selected tab (and
# the «…» button of the tabs that do not fit on a narrow screen), the ring of the upload progress and the icons of the
# video player. On white orange-500 is 2.8:1, so the whole page takes the accent of palette.ACCENT instead (orange-700
# in the light theme, as the checked boxes in _theme). The theme sets the variable on <html> and on body.dark, so a
# rule on the container wins for everything inside it in both themes.
APP_CSS += (f".gradio-container{{--color-accent:{ACCENT['light']}}}"
            f".dark .gradio-container{{--color-accent:{ACCENT['dark']}}}")
# result.json: the code viewer colours its tokens with fixed pale hues (2.0-2.8:1 on white), so the file is shown in the
# theme text colour. Both themes, although the dark hues do reach 4.5:1: one grey page, and the same block either way
APP_CSS += ".bs3-json .cm-content span{color:inherit!important}"
# height of the video window (gr.Video height). 3.1: the left column (video, model radio, buttons) sets the height of
# the compact «Характеристика личности» window next to it; at 240 px the column is about 420 px high and the tabs
# stay on the first screen of a 1080p display (with 300 px the pair alone was 480 px)
VIDEO_H = 240
# Two framed windows side by side (gr.Row with class bs3-pair): both columns get the height of the taller one, and in
# each column the window marked bs3-grow takes the extra height, so the two frames start and end on one line with no
# page background under the shorter one. A text box stretches its text area (the text fills the frame instead of
# scrolling in a small box), a chart block stretches its iframe (charts.plot_html fill=True redraws the figure at that
# height), the video window stretches its drop zone or player, a plain HTML block extends its frame under the content.
# Gradio's own equal_height grows every block of a column; here only the marked one grows. On a narrow screen the
# columns wrap onto separate lines, and a line is as tall as its only column, so nothing is stretched there.
APP_CSS += (
    ".gradio-container .row.bs3-pair{align-items:stretch}"
    ".row.bs3-pair>.column>.bs3-grow,.row.bs3-pair>.column>.form>.bs3-grow{flex-grow:1}"
    # a text box sits in a .form wrapper whose inline style (flex-grow:0, from scale) only !important overrides
    ".row.bs3-pair>.column>.form:has(>.bs3-grow){flex-grow:1!important;flex-wrap:nowrap}"
    # text box: block > label.container > title chip + .input-container > textarea
    ".row.bs3-pair .bs3-grow.block:has(textarea),.row.bs3-pair .bs3-grow>label.container,"
    ".row.bs3-pair .bs3-grow .input-container{display:flex;flex-direction:column;flex-grow:1}"
    ".row.bs3-pair .bs3-grow>label.container>[data-testid='block-info']{align-self:flex-start}"
    ".row.bs3-pair .bs3-grow textarea{flex-grow:1}"
    # chart: block > title chip + .html-container > .prose > subtitle + iframe (flex:1 0 auto from plot_html); a chart
    # that stops growing (the radar, max-height from the iframe script) is centred together with its subtitle
    ".row.bs3-pair>.column>.bs3-grow.bs3-chart,.row.bs3-pair .bs3-grow.bs3-chart>.html-container,"
    ".row.bs3-pair .prose.bs3-grow.bs3-chart{display:flex;flex-direction:column;flex-grow:1}"
    ".row.bs3-pair .prose.bs3-grow.bs3-chart{justify-content:center}"
    ".row.bs3-pair>.column>.bs3-grow.bs3-chart>label{align-self:flex-start}"
    # video: Gradio fixes the block height inline (height=VIDEO_H)
    f".row.bs3-pair>.column>.bs3-grow:has(.video-container){{height:auto!important;min-height:{VIDEO_H}px;"
    "display:flex;flex-direction:column}"
    ".row.bs3-pair .bs3-grow .video-container{flex-grow:1}")
# «Характеристика личности» (change request 3.1, section 4): a compact window like the 2.0 «Краткие выводы» box.
# The block is a flex item of its column with a zero flex basis, so the row's height comes from the left column
# (video, model radio, buttons, about 300 px of video plus the controls); align-items:stretch gives the right column
# that height and the block fills it. Inside, the label keeps its size and the html-container takes the rest and
# scrolls (min-height:0 lets it shrink below its content). CHAR_MIN_PX keeps the window readable when the columns
# wrap onto separate lines on a narrow screen: there the block is exactly this high and the text scrolls inside.
# 15 px text, line height 1.55, at most 75 characters per line, paragraphs 10 px apart with bold leads; the HTML of
# characterization.html() carries the same values inline, these rules keep Gradio's prose styles from overriding them.
CHAR_MIN_PX = 320
APP_CSS += (f".row.bs3-pair>.column>.bs3-char{{display:flex;flex-direction:column;flex:1 1 0;min-height:{CHAR_MIN_PX}px}}"
            ".bs3-char>.html-container{flex:1 1 0;min-height:0;overflow-y:auto;overflow-x:hidden}"
            ".bs3-char .bs3-char-text{font-size:15px;line-height:1.55;max-width:75ch}"
            ".bs3-char .bs3-char-text p{margin:0 0 10px}.bs3-char .bs3-char-text p b{font-weight:700}")


def _theme():
    """The BS 1.0 look: Gradio Default theme (zinc neutrals, Source Sans Pro, block titles in the block corner) with
    readable buttons and hints: white labels on orange-700 / red-700 (the Default orange-500 gives 2.8:1, red-500 3.8:1),
    zinc-600 hint text on white (zinc-400 gives 2.6:1), and checked boxes and radios in orange-700 on white, where the
    Default orange-500 is 2.8:1 (the dark theme keeps orange-500). The selected tab underline follows in APP_CSS."""
    import gradio as gr
    return gr.themes.Default().set(
        button_primary_background_fill=BUTTON_PRIMARY, button_primary_background_fill_dark=BUTTON_PRIMARY,
        button_primary_background_fill_hover=BUTTON_PRIMARY_HOVER, button_primary_background_fill_hover_dark=BUTTON_PRIMARY_HOVER,
        button_cancel_background_fill=BUTTON_STOP, button_cancel_background_fill_dark=BUTTON_STOP,
        button_cancel_background_fill_hover=BUTTON_STOP_HOVER, button_cancel_background_fill_hover_dark=BUTTON_STOP_HOVER,
        body_text_color_subdued=SUBDUED_TEXT_LIGHT, body_text_color_subdued_dark="*neutral_400",
        # placeholders have their own variable (light zinc-400 2.6:1 on white, dark zinc-500 3.1:1 on the block)
        input_placeholder_color=SUBDUED_TEXT_LIGHT, input_placeholder_color_dark="*neutral_400",
        checkbox_background_color_selected=ACCENT["light"], checkbox_background_color_selected_dark=ACCENT["dark"],
        checkbox_border_color_selected=ACCENT["light"], checkbox_border_color_selected_dark=ACCENT["dark"],
        checkbox_border_color_focus=ACCENT["light"], checkbox_border_color_focus_dark=ACCENT["dark"],
    )


ERROR_TITLE = "Ошибка"          # title of Gradio's error dialog (its default is the English "Error")
# Gradio's own texts (upload area «Перетащите видео сюда», buttons, footer) come from its translations and follow the
# browser language, so an English browser showed them in English. The page reports a Russian browser before the
# Gradio bundle reads navigator.language; this has to run before that bundle, which the `head` of gr.Blocks does not
# (Gradio inserts it later, after its translations are set up).
RU_LOCALE_JS = ("<script>try{['language','languages'].forEach(function(k){Object.defineProperty(Navigator.prototype,k,"
                "{configurable:true,get:function(){return k==='language'?'ru-RU':['ru-RU','ru'];}});});}catch(e){}</script>")


def force_russian_gradio() -> None:
    """Serve Gradio's index page with RU_LOCALE_JS at the top of <head> and lang="ru" (Gradio 5 renders the page
    from a Jinja template; its loader is wrapped once per process)."""
    import jinja2
    from gradio import routes
    env = routes.templates.env
    if getattr(env.loader, "bs3_russian", False):
        return
    base = env.loader

    class RussianIndexLoader(jinja2.BaseLoader):
        bs3_russian = True

        def get_source(self, environment, template):
            source, filename, uptodate = base.get_source(environment, template)
            if template.endswith("index.html") and RU_LOCALE_JS not in source:
                source = source.replace("<head>", "<head>" + RU_LOCALE_JS, 1).replace('lang="en"', 'lang="ru"', 1)
            return source, filename, uptodate

    env.loader = RussianIndexLoader()


def build_app(studio: Studio, work_dir: Path, preview_job: str | None = None):
    """preview_job: a finished job folder rendered on page load (UI testing without re-running the analysis)."""
    import gradio as gr
    from .longvideo import AnalysisCancelled

    N_REST = N_PAGE + 1           # page_outputs + the PDF button (design 10.5: 28)

    def render(jv: jobview.JobView) -> tuple:
        return page_values(jv) + (gr.update(interactive=True),)

    def analyze(video, member, request: gr.Request):
        """`member`: the model chosen on the page ("oceanai" | "mm"); the speech is Russian (pipeline). Explanations
        follow the model (change request 3.1, section 3): always for AMLAI 1.0, never for OCEAN-AI — no checkbox."""
        if not video:
            raise gr.Error("Загрузите видео", title=ERROR_TITLE)
        member = member if member in MODEL_TITLES else DEFAULT_MODEL
        journal.start(request, video, member)
        state = {"frac": 0.0, "desc": "запуск", "t0": time.time()}

        def cb(frac, desc=None, **kw):
            state["frac"] = float(frac)
            state["desc"] = str(desc if desc is not None else kw.get("desc", "")) or state["desc"]

        result: dict = {}

        def work():
            try:
                result["r"] = run_analysis(studio, work_dir, video, member=member, explain=(member == "mm"), progress=cb)
            except BaseException as e:  # noqa: BLE001
                result["e"] = e

        th = threading.Thread(target=work, daemon=True)
        th.start()
        while th.is_alive():
            th.join(1.0)
            yield (_status_html(state["frac"], _live_desc(state)),) + (gr.update(),) * N_REST
        if "e" in result:
            e = result["e"]
            # the bar must not stay on the orange «Идёт обработка» while the error dialog is shown (as in BS 1.0):
            # the outcome goes to the bar first, and the dialog on top of it explains what to do
            if isinstance(e, AnalysisCancelled):
                journal.failed(request, video, "остановлено пользователем", stopped=True)
                yield (_status_html(state["frac"], "по запросу пользователя", state="stopped"),) + (gr.update(),) * N_REST
                raise gr.Error("Обработка остановлена. Запустите анализ заново.", title="Остановлено")
            log.error("analysis failed", exc_info=(type(e), e, e.__traceback__))
            msg = analysis_error_ru(e)
            journal.failed(request, video, msg)
            yield (_status_html(state["frac"], msg, state="error"),) + (gr.update(),) * N_REST
            raise gr.Error(msg, title=ERROR_TITLE)
        rep = result["r"]
        jv = jobview.for_page(rep)
        outs = render(jv)
        journal.result(request, rep, time.time() - state["t0"], jv=jv)   # the page's view, type and summary
        yield (_status_html(1.0, f"обработано за {fmt_secs(time.time() - state['t0'])}", state="done"),) + outs

    # `from __future__ import annotations` keeps «gr.Request» as a string, and Gradio resolves it in the module namespace,
    # where gradio is not imported: hand it the class itself, before the handler is registered
    analyze.__annotations__["request"] = gr.Request

    def stop():
        studio.stop_event.set()
        return _status_html(0.0, "текущий отрезок дорабатывается, затем обработка прерывается (до ~20 с)", state="stopped")

    def make_pdf(job_dir):
        if not job_dir:
            raise gr.Error("Сначала проанализируйте видео", title=ERROR_TITLE)
        try:
            return pdf_for_download(job_dir)
        except Exception:  # noqa: BLE001
            log.exception("PDF export failed for %s", job_dir)
            raise gr.Error("Не удалось собрать PDF. Подробности записаны в журнал сервера.", title=ERROR_TITLE)

    def block(label: str, chart: bool = False, grow: bool = False, classes: tuple = (), value: str = ""):
        """Result block with a visible title in the block corner (gr.HTML hides its label and frame by default). Chart
        blocks (chart=True) use the chart's own title as the label; units and scales are in the chart subtitle under it.
        grow=True: the window of a bs3-pair row that takes the extra height (APP_CSS)."""
        return gr.HTML(value=value, label=label, show_label=True, container=True,
                       elem_classes=["bs3-block"] + (["bs3-chart"] if chart else []) + (["bs3-grow"] if grow else [])
                       + list(classes))

    force_russian_gradio()
    # delete_cache: every hour Gradio deletes its own temp files older than 22 hours (the uploaded videos, the PDF
    # copies of pdf_for_download), and all of them when the server stops; the job folders are not Gradio's and stay.
    # Not 24 hours: Gradio 5.8 compares timedelta.seconds, the part of the age under a day, so an age of a full day
    # never passes; 22 hours leaves two hourly passes before a file turns a day old
    with gr.Blocks(title=f"{PRODUCT} — характеристика личности, Big Five, MBTI, эмоции, голос, речь", theme=_theme(),
                   css=APP_CSS, delete_cache=(3600, 79200)) as demo:
        job_state = gr.State("")
        with gr.Row():
            with gr.Column(scale=4):
                # the title stays the first line; the copyright sits right under it in small dimmed type (APP_CSS)
                gr.Markdown(f"# {PRODUCT} — анализ человека по видео\n\n<p class='bs3-copy'>{COPYRIGHT}</p>\n\n"
                            "Характеристика личности, Big Five (первое впечатление) и перевод в нотацию MBTI, эмоции по "
                            "речи и по лицу, голос, манера речи, объяснения. Всё считается локально.")
            with gr.Column(scale=1, min_width=220):
                pdf_btn = gr.DownloadButton("Экспорт в PDF", variant="primary", interactive=False)
        status = gr.HTML(value="")
        # design 10.1 / change request 3.1, section 4: left — what was measured (video, model, buttons), right — what it
        # means: the characterization in a compact window the height of the left column, the text scrolling inside
        # (APP_CSS .bs3-char); the key facts follow as one row under the pair, so the tabs are on the first screen
        with gr.Row(elem_classes=["bs3-pair"]):
            with gr.Column(scale=1, min_width=320):
                video = gr.Video(label="Видео", sources=["upload"], height=VIDEO_H)
                # 3.1: the model is chosen by hand, one model runs per analysis; the speech is always Russian; the
                # explanations follow the model (AMLAI 1.0 only), so there is no checkbox for them. AMLAI 1.0 is the
                # left choice and is already selected (bs3.MODEL_TITLES keeps that order, bs3.DEFAULT_MODEL)
                model = gr.Radio(choices=[(t, m) for m, t in MODEL_TITLES.items()], value=DEFAULT_MODEL,
                                 label="Модель")
                with gr.Row():
                    btn = gr.Button("Анализировать", variant="primary")
                    stop_btn = gr.Button("Остановить обработку", variant="stop")
            with gr.Column(scale=2, min_width=480):
                character = block("Характеристика личности", classes=("bs3-char",),
                                  value=characterization.placeholder_html())
        facts = block("Ключевые факты")
        with gr.Tabs():
            with gr.Tab("Обзор"):
                # the radar window takes the height of the bars: the circle grows as far as the column width allows
                # and is centred with its subtitle
                with gr.Row(elem_classes=["bs3-pair"]):
                    with gr.Column(scale=1, min_width=360):
                        radar = block("Профиль Big Five", chart=True, grow=True)
                    with gr.Column(scale=1, min_width=360):
                        bars = block("Оценки по чертам", grow=True)
                method = block("Как получены оценки")
            with gr.Tab("Тип MBTI"):
                mbti_types = block("Тип MBTI")
                mbti_strip = block("Тип по ходу ролика")
                mbti_read = block("Как читать тип MBTI")
            with gr.Tab("Таймлайн"):
                traits_plot = block("Big Five по ходу ролика", chart=True)
                emo_plot = block("Эмоции по ходу ролика", chart=True)
                with gr.Row(elem_classes=["bs3-pair"]):
                    with gr.Column(scale=1, min_width=360):
                        voice_plot = block("Голос по ходу ролика", chart=True, grow=True)
                    with gr.Column(scale=1, min_width=360):
                        speech_plot = block("Речь по ходу ролика", chart=True, grow=True)
            with gr.Tab("Эмоции и голос"):
                emo_intro = block("Эмоции и голос: коротко")
                emo_bars = block("Средний профиль эмоций за ролик", chart=True)
                seg_table = block("Эмоции, голос и темп по отрезкам")
            with gr.Tab("Речь"):
                speech_html = block("Речь в цифрах")
                transcript = gr.Textbox(label="Транскрипт речи", lines=10, max_lines=14, autoscroll=False)
            with gr.Tab("Мимика и кадры"):
                face_html = block("Лицо: итоги по ролику")
                face_plot = block("Выражение лица за ролик", chart=True)
                gallery = block("Ключевые кадры")
            with gr.Tab("Объяснения"):
                # explanations exist for AMLAI 1.0 only; for an OCEAN-AI job the first block carries one note and the
                # other blocks stay empty (page_outputs)
                with gr.Row(elem_classes=["bs3-pair"]):
                    with gr.Column(scale=1, min_width=360):
                        contrib = block("Вклад модальностей в оценку модели AMLAI 1.0", grow=True)
                    with gr.Column(scale=1, min_width=360):
                        words_detail = gr.Textbox(label="Слова, на которые откликнулась модель", lines=8, max_lines=16,
                                                  autoscroll=False, elem_classes=["bs3-grow"])
                desc = gr.Textbox(label="Описание поведения по отрезкам", lines=8, max_lines=12, autoscroll=False)
            with gr.Tab("Данные"):
                members = gr.Textbox(label="Модель и время обработки", lines=3, max_lines=8, autoscroll=False)
                raw = gr.Code(label="result.json", language="json", lines=24, elem_classes=["bs3-json"])
                path = gr.Textbox(label="Сохранено в", interactive=False)
        # the caveats are the most important small print on the page: 13 px (gr.Markdown <small> gave 11 px);
        # design 11: C1, C10, C3, word for word from caveats.py (C2 follows the label it explains, see
        # caveats.PAGE_FOOTER)
        gr.HTML(f"<div style='font-size:13px;line-height:1.5;margin-top:6px;padding-top:10px;"
                f"border-top:1px solid {PAL['card_border']}'><b>Как читать результаты.</b> "
                + "<br>".join(caveats.text(c) for c in caveats.PAGE_FOOTER) + "</div>")
        outputs = [status, radar, bars, facts, character, traits_plot, emo_plot, voice_plot, speech_plot, emo_bars,
                   seg_table, speech_html, transcript, face_html, face_plot, gallery, contrib, words_detail, desc, members,
                   raw, path, job_state, method, emo_intro, mbti_types, mbti_strip, mbti_read, pdf_btn]
        assert len(outputs) == N_REST + 1
        run_ev = btn.click(analyze, inputs=[video, model], outputs=outputs, show_progress="hidden", api_name=False)
        stop_btn.click(stop, inputs=None, outputs=[status], cancels=[run_ev], show_progress="hidden", api_name=False)
        pdf_btn.click(make_pdf, inputs=[job_state], outputs=[pdf_btn], api_name=False)

        def on_visit(request: gr.Request):
            journal.visit(request)

        on_visit.__annotations__["request"] = gr.Request     # see the note under analyze()

        # outside the queue: a visit is written at once, not after the analysis running for someone else
        demo.load(on_visit, inputs=None, outputs=None, queue=False, show_progress="hidden", api_name=False)
        if preview_job:
            # the finished job becomes the initial value of every output (set before the page config is built), so
            # the page arrives filled; a demo.load event did not always reach the browser
            jv = jobview.load_job(preview_job)
            filled = (_status_html(1.0, "предпросмотр готового результата", state="done"),) + page_values(jv)
            for comp, value in zip(outputs, filled):
                comp.value = value
            pdf_btn.interactive = True
            # the radio shows the model the previewed job was processed with (an imported 2.0 job: OCEAN-AI)
            from .scores import recorded_model
            model.value = recorded_model(jv.rep) or DEFAULT_MODEL
    return demo


def main(port: int = settings.PORT, work_dir: str | None = None, share: bool = False,
         asr_model: str = settings.ASR_MODEL, ollama_model: str = settings.OLLAMA_MODEL, mm_ckpt: str | None = None,
         host: str = settings.HOST, models_dir: str | None = None):
    """The page picks the model per analysis (OCEAN-AI or AMLAI 1.0); the Studio loads each one on first use.
    `models_dir`: the OCEAN-AI weights cache (None = ~/bs/models)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings.apply_process_env()              # before build_app imports gradio
    wd = Path(work_dir or settings.JOBS_DIR)
    wd.mkdir(parents=True, exist_ok=True)
    studio = Studio(asr_model=asr_model, ollama_model=ollama_model, mm_ckpt=mm_ckpt, models_dir=models_dir)
    demo = build_app(studio, wd)
    # no allowed_paths: the page shows no file of the job folder (key frames are data URIs, charts are srcdoc), and the
    # PDF is handed to Gradio from its own temp folder (pdf_for_download); /gradio_api/file= serves no job file
    demo.queue(default_concurrency_limit=settings.QUEUE_CONCURRENCY).launch(
        server_name=host, server_port=port, share=share, show_api=False, show_error=True, quiet=False)
