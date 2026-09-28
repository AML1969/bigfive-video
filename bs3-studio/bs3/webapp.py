"""BS Profiler 3.1 web UI (Gradio): Big Five by the model chosen for the analysis (OCEAN-AI or AMLAI 1.0, Russian
speech) + emotions, voice, face and speech analytics with interactive charts.

Run:  bs3 web [--port 7880]      (inside WSL; open http://localhost:7880 on Windows). Независим от BS 2.0 (:7870).

Readability rules for the HTML blocks (both Gradio themes, see palette.py): text colours are inherited from the theme,
secondary text is the same colour at opacity .75 and at least 13 px, marks and outlines come from palette.HTML, and
every block shows its title (show_label=True, container=True).
"""
from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
import threading
import time
from pathlib import Path, PurePosixPath

from . import (DEFAULT_MODEL, MODEL_TITLES, PRODUCT, PRODUCT_SLUG, caveats, characterization, jobfiles, jobview,
               journal, mbti_html, settings)
from .analyses_text import speech_description
from .charts import (fig_emotion_bars, fig_emotions_timeline, fig_face_expr, fig_radar, fig_speech_timeline,
                     fig_traits_timeline, fig_voice_timeline, plot_html as _plot_html)
from .facts import (FACTS_LEGEND, FER_NOTE, card_item, fact_cards, fact_label, head_motion_word, segment_cells,
                    speech_cards)
from .labels import EMO_RU
from .narrative import NO_EXPLAIN_RU, method_notes
from .palette import (ACCENT, BUTTON_PRIMARY, BUTTON_PRIMARY_HOVER, BUTTON_STOP, BUTTON_STOP_HOVER, CARD_TINT,
                      FACT_VALUE, HTML as PAL, PAGE_NOTE_OPACITY, SUBDUED_TEXT_LIGHT)
from .pipeline import Studio, run_analysis
from .ru_texts import transcript_shown, vocabulary_shown
from .scores import FACT_STATES, data_json, has_explanations
from .segments import representative
from .textfmt import clock, fmt_secs, mmss_labels, plural_ru, seg_label
from .webparts import NOTE, _bar_html, _contrib_html, _words_text, model_line, table_html, th_text

log = logging.getLogger("bs3.web")
# the note of the tab «Данные» when scores.data_json left something out of a Russian job: the percentiles and the
# stored 2.0 `narrative` of older jobs (imported 2.0, 3.0 and early 3.1 jobs), which the page does not use
DATA_TRIMMED = ("В result.json ниже не показаны поля прежних версий, которые 3.1 не использует (процентили, сводка "
                "версии 2.0); файл не изменён.")
# metric cards: 1 px outline 3:1 on every background, light tint (palette.CARD_TINT, the background check_palette.py
# measures the card text and the outline on) so label, value and note read as one card
CARDS = "display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px"
CARD = (f"padding:10px 14px;border:1px solid {PAL['card_border']};background:rgba(128,128,128,{CARD_TINT});"
        "border-radius:8px;min-width:0")
# A value is one line of 20 px text in a card 150 px wide at its narrowest, and a single long word («возбуждение
# 0.30») is wider than that: without overflow-wrap it paints over the card border and into the next card. Chrome's
# `hyphens: auto` was tried instead and dropped — it depends on hyphenation patterns the browser may not have, so the
# same card would break differently on two machines.
CARD_VALUE = ("font-size:20px;font-weight:600;line-height:1.25;font-variant-numeric:tabular-nums;"
              "overflow-wrap:anywhere")
# «Ключевые факты»: one class per state, so the value takes the colour of the theme the page is showing (the block
# HTML is built once for both themes, Gradio puts `dark` on an ancestor of the container). The rules travel with the
# block, so the same HTML also reads right outside the app (scripts/rerender_samples.py --html-dir).
# Gradio 5.8 gives everything inside an HTML block the body text colour with `.gradio-container-5-8-0 .prose *`,
# specificity (0,2,0). A rule of one class lost to it, and in the light theme every value was the plain text colour;
# the dark rule, two classes, won only because this <style> comes after Gradio's. So both rules sit under the class of
# the card grid (_cards): the light one, `div.bs3-facts .bs3-fact-…`, is (0,2,1) and beats Gradio's rule whatever the
# order of the style sheets, and the dark one, one class more, beats the light one (tests/test_key_facts.py).
FACTS_SCOPE = "bs3-facts"
FACTS_CSS = "<style>" + "".join(f"div.{FACTS_SCOPE} .bs3-fact-{s}{{color:{FACT_VALUE['light'][s]}}}"
                                f".dark div.{FACTS_SCOPE} .bs3-fact-{s}{{color:{FACT_VALUE['dark'][s]}}}"
                                for s in FACT_STATES) + "</style>"


def _cards(items, min_px: int = 180, value_first: bool = False) -> str:
    """(label, value, note[, state]) -> a grid of cards; note may be empty. `value_first`: the card reads value,
    then label, then note; a value with a state is painted by it (FACTS_CSS, under the grid's class FACTS_SCOPE) and
    its label line ends with the word of that state (facts.fact_label), so the card also reads without colour —
    «Ключевые факты» of 3.1."""
    html = ""
    for item in items:
        lab, val, note, state = card_item(item)
        cls = f" class='bs3-fact-{state}'" if value_first and state else ""
        value = (f"<div{cls} style='{CARD_VALUE};margin:"
                 + ("0 0 2px" if value_first else "3px 0") + f"'>{val if val not in (None, '') else '—'}</div>")
        label = f"<div style='{NOTE}'>{fact_label(lab, state) if value_first else lab}</div>"
        rest = f"<div style='{NOTE}'>{note}</div>" if note else ""
        html += f"<div style='{CARD}'>" + (value + label if value_first else label + value) + rest + "</div>"
    grid = CARDS.replace("minmax(180px", f"minmax({int(min_px)}px")
    scope = f" class='{FACTS_SCOPE}'" if value_first else ""
    return f"<div{scope} style='{grid}'>{html}</div>" if html else ""


def _facts_html(view: dict, mb: dict | None = None) -> str:
    """«Ключевые факты» in the left column (design 10.2): the MBTI type card first, then the cards of 2.0 from the clean
    view; 150 px minimum, two cards in a row in the 320 px column. 3.1: every card reads value, label, explanation; the
    value of a measured card is coloured by where it sits and its label says the same in a word; one line under the
    grid says what the colour and the word mean (facts.fact_cards, the same cards as the PDF)."""
    items, show_legend = fact_cards(view, mb)
    grid = _cards(items, min_px=150, value_first=True)
    if not grid:
        return ""
    legend = f"<p style='{NOTE};margin:8px 0 0'>{FACTS_LEGEND}</p>" if show_legend else ""
    return FACTS_CSS + grid + legend


def _segments_table(rep: dict) -> str:
    per = (rep.get("analyses") or {}).get("per_segment") or []
    if not per:
        return ""
    # one time format for the whole column («0:00–0:20 … 10:00–10:12»), the same as on the chart time axes; the
    # seconds are rounded as in the PDF and in the chart hover (640.0–651.8 is «10:40–10:52»)
    hours = max(float(r["end"]) for r in per) >= 3600
    head = [th_text("Отрезок", "ч:мин:с" if hours else "мин:с"), th_text("Эмоция", "по тексту речи"),
            th_text("Выражение", "лица"), th_text("Возбуждение", "голос, 0…1"), th_text("Уверенность", "голос, 0…1"),
            th_text("Позитивность", "голос, 0…1"), th_text("Темп", "слов в минуту"), th_text("Доля пауз", "в отрезке")]
    # the cells of the PDF appendix «Значения по отрезкам» (facts.segment_cells): «нет речи» / «нет текста» for a
    # segment with an empty transcript, «—» for one without a tempo
    rows = [[f"{clock(r['start'], hours)}–{clock(r['end'], hours)}", *segment_cells(r)] for r in per]
    return (f"<div style='{NOTE};margin-bottom:8px'>Для каждого отрезка: преобладающая эмоция по тексту речи и по лицу "
            "(с долей), три характеристики голоса от 0 до 1, темп речи и доля пауз. Шапка таблицы остаётся на месте "
            "при прокрутке.</div>" + table_html(head, rows, max_height=480))


def _speech_html(rep: dict) -> str:
    sp = (rep.get("analyses") or {}).get("speech") or {}
    if not sp:
        return ""
    items = speech_cards(sp, small_rate_words=False)             # the cards of the PDF; the page says «0 на 100 слов»
    vocab = ", ".join(f"{w} ({n})" for w, n in vocabulary_shown(rep)[:15])       # in Russian for any speech language
    return (_cards(items) + "<p style='font-size:15px;line-height:1.5;margin:12px 0 6px'>"
            f"{speech_description(sp)}</p>"
            + (f"<p style='font-size:14px;line-height:1.5;margin:0'><b>Частые слова</b> "
               f"<span style='opacity:.75'>(в скобках — сколько раз)</span>: {vocab}</p>" if vocab else ""))


def _face_html(rep: dict) -> str:
    """Cards with the face metrics; the distribution itself is drawn as a chart next to them."""
    fa = (rep.get("analyses") or {}).get("face") or {}
    if not fa:
        return ""
    m = fa.get("mean") or {}
    per = (rep.get("analyses") or {}).get("per_segment") or []
    frames = sum((r.get("face") or {}).get("frames", 0) for r in per)
    cards = []
    if m:
        k, v = max(m.items(), key=lambda kv: kv[1])
        cards.append(("Выражение лица чаще всего", EMO_RU.get(k, k), f"{v:.0%} кадров"))
    hm = fa.get("head_motion")
    if hm is not None:
        cards.append(("Движение головы", head_motion_word(hm, "motion"),
                      f"смещение между кадрами — {hm:.0%} ширины лица"))
    if fa.get("face_share") is not None:
        cards.append(("Лицо найдено", f"{fa['face_share']:.0%}", "доля разобранных кадров"))
    if frames:
        n = len(per)
        # «взяты из 31 отрезка», not «372 / из 31 отрезка», which reads like a fraction
        cards.append(("Кадров разобрано", f"{frames}", f"взяты из {n} {plural_ru(n, 'отрезка', 'отрезков', 'отрезков')}"))
    return (_cards(cards) + f"<p style='{NOTE};margin-top:10px'>{FER_NOTE.format(where=' (вкладка «Таймлайн»)')}</p>")


# key frames: the figure toggles .bs3-kf-big; enlarged, the image fills the window and the caption (moment of the
# video) stays readable on a dark plate at the bottom, so it is always clear which moment is shown
FRAMES_CSS = (
    "<style>"
    ".bs3-kf figure{margin:0!important;cursor:zoom-in}"
    # thumbnails keep the frame's own proportions (no empty letterbox bands); tall portrait frames stop at 320 px
    ".bs3-kf img{display:block;width:100%;height:auto;max-height:320px;object-fit:contain;border-radius:8px;"
    "background:rgba(128,128,128,.12);outline:1px solid " + PAL["card_border"] + ";outline-offset:-1px}"
    # one short line under the frame: the moment and what is visible; it never grows past two lines, the rest of
    # the story (expressions, what the frame did to the score) opens on hover as the figure's title
    ".bs3-kf figcaption{text-align:center;font-size:13px;line-height:1.35;margin-top:6px;opacity:.75;"
    "font-variant-numeric:tabular-nums;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;"
    "overflow:hidden}"
    ".bs3-kf figcaption b{font-weight:600}"
    ".bs3-kf figure.bs3-kf-big{cursor:zoom-out}"
    ".bs3-kf figure.bs3-kf-big img{position:fixed;inset:4vh 4vw;width:92vw;height:92vh;max-height:none;z-index:9999;"
    "border-radius:8px;background:rgba(0,0,0,.92);outline:0;box-shadow:0 0 0 100vmax rgba(0,0,0,.85)}"
    ".bs3-kf figure.bs3-kf-big figcaption{position:fixed;left:50%;bottom:calc(4vh + 14px);transform:translateX(-50%);"
    "z-index:10000;margin:0;padding:6px 14px;border-radius:8px;background:rgba(0,0,0,.8);color:#fff!important;"
    "font-size:16px;display:block;opacity:1;max-width:84vw;pointer-events:none}"
    ".bs3-kf figure:not(.bs3-kf-big) .bs3-kf-more{display:none}"
    "</style>")


NO_FRAMES_OCEANAI = ("Ключевых кадров нет: модель OCEAN-AI не строит объяснений, ключевые кадры есть только для модели "
                     "AMLAI 1.0.")
NO_FRAMES_MM = "Ключевые кадры не построены: лицо в кадре не найдено или объяснения не удалось посчитать."


def _frames_html(rep: dict, expl: dict | None = None, max_side: int = 640) -> str:
    """Key frames embedded as data-URI JPEGs. gr.Gallery depends on Gradio serving files from the job folder, which
    proved unreliable in this setup (images arrive broken); inline images always render. Click enlarges a frame.
    Key frames belong to AMLAI 1.0: an OCEAN-AI job shows one line instead (also an older job that carries the frames
    of the second model of 3.0, since the page shows one model), a job of AMLAI 1.0 without frames says why.

    Under every frame one short line — the moment of the video and, in a few words, what is visible there
    (frame_captions.build). Hovering the frame opens the rest in the figure's tooltip: the two strongest facial
    expressions of that frame and what the frame did to the score, with a direction. The enlarged frame shows both
    lines on its dark plate, so nothing is hidden from someone who never hovers."""
    import base64
    import io
    from html import escape
    from PIL import Image

    from . import frame_captions

    if not has_explanations(rep):
        return f"<p style='font-size:14px'>{NO_FRAMES_OCEANAI}</p>"
    # the frames of the job folder being shown, found by name (jobfiles); without a folder there are none
    paths = [str(p) for p in jobfiles.key_frame_paths(rep["job_dir"], rep)] if rep.get("job_dir") else []
    if not paths:
        return f"<p style='font-size:14px'>{NO_FRAMES_MM}</p>"
    shown, images = [], []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
        except Exception:  # noqa: BLE001
            continue
        shown.append(p)
        images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    if not shown:
        return f"<p style='font-size:14px'>{NO_FRAMES_MM}</p>"
    # frames come from the representative segment of a long video, otherwise from the whole video (no timeline): the
    # moment is then counted from 0, the same rule as in the PDF
    seg = representative(rep, among="all")
    entries = frame_captions.build(rep, shown, expl)
    tenths = frame_captions.has_tenths(rep, shown, None, expl)
    figs = []
    for b64, e in zip(images, entries):
        tip = e["tooltip"] or "Щёлкните, чтобы увеличить"
        more = f" · {escape(e['tooltip'])}" if e["tooltip"] else ""
        figs.append(
            f"<figure role='button' tabindex='0' title='{escape(tip)}' onclick=\"this.classList.toggle('bs3-kf-big')\" "
            "onkeydown=\"if(event.key==='Enter'||event.key===' '){event.preventDefault();this.classList.toggle('bs3-kf-big')}"
            "else if(event.key==='Escape'){this.classList.remove('bs3-kf-big')}\">"
            f"<img src='data:image/jpeg;base64,{b64}' alt='{escape(e['alt'])}'>"
            f"<figcaption><b>{escape(e['label'])}</b>"
            + (f" · {escape(e['tail'])}" if e["tail"] else "")
            + f"<span class='bs3-kf-more'>{more} · щелчок закрывает</span></figcaption></figure>")
    where = f" (отрезок {seg_label(seg['start'], seg['end'])})" if seg else ""
    # a job made before the captions has neither a phrase nor the expressions: the note promises only what the
    # page really shows, otherwise it sends the reader hunting for a description that is not there
    _, has_expr, has_eff = frame_captions.note_flags(entries)
    what = frame_captions.note_what(entries, tenths, "page")
    hover = [x for x, ok in (("выражение лица", has_expr), ("то, как кадр сдвинул оценку", has_eff)) if ok]
    return (FRAMES_CSS + "<div class='bs3-kf' style='display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));"
            f"gap:12px'>{''.join(figs)}</div>"
            f"<p style='{NOTE};margin-top:10px'>Кадры, сильнее всего повлиявшие на оценку модели AMLAI 1.0{where}. "
            + f"Рамкой на кадре отмечено найденное лицо, под кадром — {what}. "
            + (f"Наведите мышь на кадр — {'покажутся' if len(hover) > 1 else 'покажется'} {' и '.join(hover)}. "
               if hover else "")
            + "Щелчок по кадру увеличивает его, повторный щелчок закрывает.</p>")


N_PAGE = 27            # page_outputs values: 22 blocks kept in the places of the 2.0 page, then 5 added by 3.0


def model_text(view: dict, rep: dict, mb: dict | None) -> str:
    """«Модель и время обработки» (tab «Данные», 3.1): one line with the model that ran and its scores, one with the
    processing time, then the notes of the render (C22 for a type computed on display, DATA_TRIMMED)."""
    lines = [model_line(view), f"Обработка заняла {fmt_secs((rep.get('timings_sec') or {}).get('total_wall', 0))}."]
    if mb and mb.get("computed_on_render"):
        lines.append(caveats.text("C22"))
    if data_json(rep)[1]:
        lines.append(DATA_TRIMMED)
    return "\n".join(lines)


def _without_server_paths(data: dict) -> dict:
    """The result.json of the tab «Данные» without the server paths the file keeps: `job_dir`, `input`, every
    `key_frames` entry and every `timeline[].file` become the file or folder name alone. Works in place on the copy
    scores.data_json made; the file on disk keeps its paths."""
    def name(v):
        return PurePosixPath(v).name if isinstance(v, str) else v

    for key in ("job_dir", "input"):
        if key in data:
            data[key] = name(data[key])
    if isinstance(data.get("key_frames"), list):
        data["key_frames"] = [name(v) for v in data["key_frames"]]
    for t in data.get("timeline") if isinstance(data.get("timeline"), list) else []:
        if isinstance(t, dict) and "file" in t:
            t["file"] = name(t["file"])
    return data


def page_outputs(rep: dict) -> tuple:
    """Everything the result page shows for a finished job, in the order of the output blocks after the status line
    (without the PDF button). Jobs processed before the Russian texts existed get them here, stored back.

    Design 10.5: the numbers come from the clean view (scores.clean_view), the MBTI section from mbti.get_mbti (the
    saved one, or computed now and never written), the characterization from characterization.build, each built once
    (jobview.for_page; the app renders that JobView with page_values and hands it to the journal). The first 22
    values keep their places (index 2 — the key facts with the type card first, index 3 — the characterization), then
    five new ones: «Как получены оценки», «Эмоции и голос: коротко», the MBTI panel, the letter strip and «Как читать
    тип MBTI». One model everywhere (3.1): the bars, the panel and the strip are the model recorded in the job; the
    tab «Объяснения» of an OCEAN-AI job shows one note (narrative.NO_EXPLAIN_RU) in its first block. The «Данные» tab
    shows result.json as it lies on disk, except that for Russian speech the 2.0 fields that 3.x does not use
    (percentiles, the stored 2.0 summary) are left out, with a note (scores.data_json), and that the paths on the server
    are shown as file and folder names (_without_server_paths); «Сохранено в» is the name of the job folder. The last
    of the first 22 values, the job folder the PDF button reads, stays the full path: it goes into a gr.State, which
    Gradio keeps on the server and does not send to the browser with the results of an analysis.

    The files of the job (explanation.json, the key frames) are read from the folder `rep["job_dir"]` (jobfiles). A
    `rep` without it renders without them: no explanation, no key frames, and «Сохранено в» and the job folder of the
    PDF button stay empty."""
    return page_values(jobview.for_page(rep))


def page_values(jv: jobview.JobView) -> tuple:
    """The values of page_outputs for a JobView (jobview.for_page): the app renders a finished analysis from it and
    hands the same JobView to the journal entry of that analysis (journal.result)."""
    rep, view, mb, expl, job = jv.rep, jv.view, jv.mb, jv.expl, jv.job
    # explanations exist for AMLAI 1.0 only: an OCEAN-AI job says so once, in the first block of the tab, and the
    # other blocks stay empty (an older job may carry the explanation and the behaviour description of the second
    # model of 3.0; the page shows one model)
    own = has_explanations(view)
    contrib = _contrib_html(expl) if own else f"<p style='font-size:14px;margin:0'>{NO_EXPLAIN_RU}</p>"
    data, _ = data_json(rep)
    # the transcript block; an older job processed as English shows its Russian translation with a one-line note
    note, transcript = transcript_shown(rep)
    # fill=True: charts in a row of two windows grow to the height of the window next to them (see APP_CSS)
    out = (_plot_html(fig_radar, view, fill=True),
           _bar_html(view["traits"], view.get("interview")),
           _facts_html(view, mb), jv.character.html(), _plot_html(fig_traits_timeline, view),
           _plot_html(fig_emotions_timeline, view), _plot_html(fig_voice_timeline, view, fill=True),
           _plot_html(fig_speech_timeline, view, fill=True), _plot_html(fig_emotion_bars, view),
           _segments_table(view), _speech_html(view),
           "\n\n".join(t for t in (note, transcript) if t), _face_html(view), _plot_html(fig_face_expr, view),
           _frames_html(view, expl if own else None), contrib,
           _words_text(expl) if (expl and own) else "",
           mmss_labels(rep.get("behavior_description_ru") or "") if own else "", model_text(view, rep, mb),
           json.dumps(_without_server_paths(data), ensure_ascii=False, indent=2), job.name if job else "",
           str(job) if job else "",
           # the five blocks after the first 22 (design 10.3, 10.5)
           mbti_html.method_html(method_notes(view)), mbti_html.emo_intro_html(view),
           mbti_html.types_html(mb), mbti_html.strip_html(mb), mbti_html.read_html(mb))
    assert len(out) == N_PAGE
    return out


def export_pdf(job_dir: str | Path) -> str:
    from .pdf.charts import save_pdf_charts
    from .media import probe_media
    from .pdf_report import build_pdf
    job = Path(job_dir)
    # the files of this folder, whatever paths result.json stores; jobs processed before the Russian texts are
    # translated once and stored back; the same clean numbers (design 6.1), the saved MBTI section or one computed now
    # and never written (design 7.2) and the same characterization as the page
    jv = jobview.load_job(job)
    view = jv.view
    view["chart_files"] = save_pdf_charts(view, job / jobfiles.CHARTS_DIR, jv.expl)
    frames = [str(p) for p in jobfiles.key_frame_paths(job, jv.rep)]
    media = view.get("media")
    if not media or "error" in media:
        inp = jobfiles.input_file(job)
        media = probe_media(inp) if inp else None
    stem = re.sub(r"[^A-Za-z0-9А-Яа-яЁё._-]+", "_", Path(view.get("original_file_name") or "video").stem)[:60]
    return build_pdf(view, job / f"{PRODUCT_SLUG}_report_{stem}.pdf", explanation=jv.expl, media=media,
                     key_frames=frames, mbti=jv.mb, character=jv.character)


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
