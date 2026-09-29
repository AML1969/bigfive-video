"""BS Profiler 3.1 web UI (Gradio): Big Five by the model chosen for the analysis (OCEAN-AI or AMLAI 1.0, Russian
speech) + emotions, voice, face and speech analytics with interactive charts.

Run:  bs3 web [--port 7880]      (inside WSL; open http://localhost:7880 on Windows). Независим от BS 2.0 (:7870).

build_app assembles the page; the result blocks are built in bs3/web/page.py and the CSS and theme in bs3/web/style.py.
"""
from __future__ import annotations

import gc
import logging
import re
import secrets
import shutil
import sys
import threading
import time
from pathlib import Path

from .. import DEFAULT_MODEL, MODEL_TITLES, PRODUCT, caveats, characterization, jobview, journal, settings
from ..errors import AnalysisCancelled, user_message
from ..palette import HTML as PAL
from ..pdf import export_pdf
from ..pipeline import Studio, run_analysis
from ..textfmt import fmt_secs
from .page import N_PAGE, page_values
from .style import APP_CSS, COPYRIGHT, VIDEO_H, _theme, force_russian_gradio

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


ERROR_TITLE = "Ошибка"          # title of Gradio's error dialog (its default is the English "Error")


def build_app(studio: Studio, work_dir: Path, preview_job: str | None = None):
    """preview_job: a finished job folder rendered on page load (UI testing without re-running the analysis)."""
    import gradio as gr

    N_REST = N_PAGE + 1           # page_outputs + the PDF button (design 10.5: 28)

    def render(jv: jobview.JobView) -> tuple:
        return page_values(jv) + (gr.update(interactive=True),)

    def analyze(video, member, request: gr.Request):
        """`member`: the model chosen on the page ("oceanai" | "mm"); the speech is Russian (pipeline). Explanations
        follow the model (change request 3.1, section 3): always for AMLAI 1.0, never for OCEAN-AI — no checkbox."""
        if not video:
            raise gr.Error("Загрузите видео", title=ERROR_TITLE)
        try:
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
                if "out of memory" in str(e).lower():
                    gc.collect()                                      # release the failed run before anyone tries again
                    torch = sys.modules.get("torch")                 # only if it is already loaded — never import it here
                    if torch is not None:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:  # noqa: BLE001
                            pass
                log.error("analysis failed", exc_info=(type(e), e, e.__traceback__))
                msg = user_message(e, work_dir)
                journal.failed(request, video, msg)
                yield (_status_html(state["frac"], msg, state="error"),) + (gr.update(),) * N_REST
                raise gr.Error(msg, title=ERROR_TITLE)
            rep = result["r"]
            # the analysis is done and result.json is written; if the page cannot be built from it, say so calmly and
            # keep the exception off the page (FP5)
            try:
                jv = jobview.for_page(rep)
                outs = render(jv)
            except Exception:  # noqa: BLE001
                log.exception("page render failed after a finished analysis")
                msg = ("Анализ завершён и сохранён, но страницу с результатом построить не удалось. "
                       "Подробности записаны в журнал сервера.")
                journal.failed(request, video, msg)
                yield (_status_html(state["frac"], msg, state="error"),) + (gr.update(),) * N_REST
                raise gr.Error(msg, title=ERROR_TITLE)
            journal.result(request, rep, time.time() - state["t0"], jv=jv)   # the page's view, type and summary
            yield (_status_html(1.0, f"обработано за {fmt_secs(time.time() - state['t0'])}", state="done"),) + outs
        except gr.Error:
            raise                                                     # the calm dialogs above pass through unchanged
        except Exception as e:  # noqa: BLE001  — a last net so no raw exception text reaches the page
            log.exception("unexpected failure in analyze")
            raise gr.Error(user_message(e), title=ERROR_TITLE)

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
            from ..scores import recorded_model
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
