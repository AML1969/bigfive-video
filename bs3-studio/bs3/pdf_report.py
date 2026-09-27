"""PDF report for one analysed video (fpdf2).

The main part reads top down (design 10.7; 3.1: one model): a passport of the file and the analysis, «Характеристика
личности» (its header line and paragraphs, pdf_mbti.characterization_block), the key facts with the MBTI type card
first, the Big Five profile (radar beside «Как получены оценки», the score bars of the one model that ran), «Тип MBTI
(перевод шкал Big Five)» (pdf_mbti.mbti_section), then one numbered section per topic — Big Five over time, emotions
and facial expression, voice and speech, what drove the score of AMLAI 1.0 (for an OCEAN-AI job one line under the
voice section says that it builds no explanations) — and «Как читать результаты». The appendices follow: file and
analysis parameters, the values of every segment (with the MBTI type of the segment), behaviour descriptions of the
notable segments, the transcript. Every chart of the web page has a print version (pdf_charts.py). The report is built
from the clean view (scores.clean_view). The original file name of the video is printed exactly as it is, also in the
footer of every page, so pages of two reports cannot be mixed up. The page itself (class Report) is pdf/document.py;
the per-segment data come from segments.py.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path

from . import MODALITIES, MODEL_TITLES, PRODUCT, caveats, frame_captions, jobview, segments, settings
from .analyses_text import analyses_parts
from .facts import FACTS_LEGEND, FER_NOTE, fact_cards, head_motion_word, speech_cards
from .labels import EMO_RU, EMOTION_ORDER, VOICE_RU, model_title
from .narrative import NO_EXPLAIN_RU
from .norms import RU_SHORT, RU_TITLES, TRAIT_KEYS
from .pdf.document import RADAR_W_MM, ROW_GAP_MM, Report
from .pdf_charts import save_modalities_chart
from .pdf_mbti import characterization_block, mbti_section, segment_types_by_start
from .ru_texts import transcript_shown, vocabulary_shown
from .scores import has_explanations, scored, shown_model
from .segments import (behavior_by_segment, dominant_emotion, empty_text, odd_segments, representative, seg_words,
                       segment_rows)
from .textfmt import clock, fix_counts, fmt_secs, plural_ru, seg_label

# «Значения по отрезкам» has 14 columns: the short trait names are broken over two lines where one line is wider than
# its column of numbers; the legend under the table joins the halves back («Добро-жел.» -> «Доброжел.»)
SEG_HEAD = {**RU_SHORT, "agreeableness": "Добро-\nжел.", "emotional_stability": "Эм.\nстаб.", "interview": "Собе-\nсед."}
MODALITY_TITLES = {"audio": "голос", "video": "видео", "text": "речь", "face": "лицо", "behavior": "описание поведения"}
# container tags from media.probe_media (ffprobe names) -> row titles of the «Файл» table
MEDIA_TAGS = {"creation_time": "Записан (метка в файле)", "encoder": "Программа записи",
              "com.apple.quicktime.make": "Производитель камеры", "com.apple.quicktime.model": "Модель камеры",
              "title": "Название"}
# training data of each model, for appendix А
TRAINED_ON = {"oceanai": "MuPTA (русская речь)", "mm": "First Impressions V2"}
BEHAVIOR_MAX = 6                    # appendix «Описание поведения»: at most this many notable segments


def _when(v) -> str:
    """'2026-08-13T15:37:12.000000Z' -> '2026-08-13 15:37:12 по всемирному времени'; anything that is not an ISO time
    stays as it is."""
    s = str(v or "")
    m = re.match(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?$", s)
    if not m:
        return s
    tz = m.group(3)
    return f"{m.group(1)} {m.group(2)}" + (" по всемирному времени" if tz == "Z" else (f" (часовой пояс {tz})" if tz else ""))


# ffprobe codec names -> the usual names of the formats
CODEC_NAMES = {"h264": "H.264", "hevc": "H.265 (HEVC)", "h265": "H.265", "vp8": "VP8", "vp9": "VP9", "av1": "AV1",
               "mpeg4": "MPEG-4", "mjpeg": "Motion JPEG", "prores": "ProRes", "aac": "AAC", "mp3": "MP3", "opus": "Opus",
               "vorbis": "Vorbis", "flac": "FLAC", "ac3": "AC-3", "eac3": "E-AC-3", "alac": "ALAC"}


def _codec(v) -> str:
    s = str(v or "")
    return CODEC_NAMES.get(s.lower(), "PCM" if s.lower().startswith("pcm_") else s or "—")


def _encoder(v) -> str:
    """'Lavf60.16.100' (the container writer of FFmpeg) -> 'FFmpeg (libavformat 60.16.100)'."""
    s = str(v or "")
    m = re.match(r"^Lav([fc])(\d[\d.]*)$", s)
    return f"FFmpeg (libav{'format' if m.group(1) == 'f' else 'codec'} {m.group(2)})" if m else s


def _asr_ru(name) -> str:
    """'openai/whisper-large-v3-turbo' -> 'Whisper large-v3-turbo': the model name without the hub prefix."""
    m = re.match(r"^(?:[\w.-]+/)?whisper-(.+)$", str(name or ""), re.I)
    return f"Whisper {m.group(1)}" if m else str(name or "")


def _version_ru(v) -> str:
    """'3.0.0a1' -> '3.0.0, альфа-версия 1' (PEP 440 pre-release suffixes a/b/rc in words)."""
    m = re.match(r"^(\d+(?:\.\d+)*)(a|b|rc)(\d+)$", str(v or ""))
    if not m:
        return str(v or "—")
    return f"{m.group(1)}, {({'a': 'альфа', 'b': 'бета', 'rc': 'предрелизная'})[m.group(2)]}-версия {m.group(3)}"


def _seg(report: dict, start, end) -> str:
    """«1:20–1:40» as textfmt.seg_label, but h:mm:ss from an hour on, like the time axis of the charts and the
    «Отрезок» column of the web table: one report never prints «1:05:00» on a chart and «65:00» in a table."""
    if float(report.get("duration_sec") or 0) < 3600:
        return seg_label(start, end)
    return f"{clock(start, hours=True)}–{clock(end, hours=True)}"


def _hms_text(text: str, dur: float) -> str:
    """From an hour on the report writes times as h:mm:ss (as the charts do); the narrative is shared with the web
    page and writes «отрезок 37:20–39:40», so its segment ranges are rewritten for the PDF."""
    if float(dur or 0) < 3600:
        return text

    def one(m) -> str:
        def f(mm: str, ss: str) -> str:
            return f"{int(mm) // 60}:{int(mm) % 60:02d}:{ss}"
        return f"{f(m.group(1), m.group(2))}–{f(m.group(3), m.group(4))}"
    return re.sub(r"\b(\d{1,3}):(\d\d)–(\d{1,3}):(\d\d)\b", one, text)


def _dash(text) -> str:
    """Russian punctuation in a description written by a model: a hyphen between spaces is an em dash."""
    return re.sub(r"(?<=\s)-(?=\s)", "—", str(text or ""))


def _one_line(head: str) -> str:
    """A two-line column header as one word for the legend: «Добро-\\nжел.» -> «Доброжел.», «Эм.\\nстаб.» -> «Эм. стаб.»."""
    return str(head).replace("-\n", "").replace("\n", " ")


# ---------------------------------------------------------------- plan: which sections and appendices are printed
def _plan(pdf: Report, report: dict, explanation, frames, charts: dict, mb: dict | None = None) -> None:
    an = report.get("analyses") or {}
    te, fa = an.get("emotions_text") or {}, an.get("face") or {}
    has = {"profile": True,
           "mbti": bool(mb),             # right after the Big Five section, when at least the main type is computed
           "timeline": bool(charts.get("traits")) and len(segments.scored(report)) >= 2,
           "emotions": bool(te.get("mean") or fa.get("mean") or charts.get("emotions")),
           "voice_speech": bool(an.get("voice") or an.get("speech") or charts.get("voice") or charts.get("speech")),
           # explanations exist for AMLAI 1.0 only (3.1): an OCEAN-AI job gets one line under section 4 instead
           "explain": bool(explanation or frames) and has_explanations(report),
           "how_to_read": True}          # numbered like the rest: between the sections and the lettered appendices
    n = 0
    for key, ok in has.items():
        if ok:
            n += 1
            pdf.plan[key] = n
    note, transcript = transcript_shown(report)
    # the behaviour description is written for AMLAI 1.0 (its video-language model); an older OCEAN-AI job that
    # carries the description of the second model of 3.0 does not print it: one model in the report
    appx = {"file": True, "segments": len(segment_rows(report)) >= 2,
            "behavior": bool(report.get("behavior_description_ru")) and has_explanations(report),
            "transcript": bool(note or transcript)}
    letters = iter("АБВГДЕ")
    for key, ok in appx.items():
        if ok:
            pdf.appx[key] = next(letters)


# ---------------------------------------------------------------- main part
def _passport(pdf: Report, report: dict, media: dict | None, fname: str) -> None:
    """Four short lines instead of the file and analysis tables (those are in appendix А)."""
    m = report.get("model") or {}
    md = media or report.get("media") or {}
    parts = [fname]
    dur = md.get("duration_sec") or report.get("duration_sec")
    if dur:
        parts.append(fmt_secs(dur))
    if md.get("width") and md.get("height"):
        w, h = md["width"], md["height"]
        if str(md.get("rotation") or "0").lstrip("-") in ("90", "270"):
            w, h = h, w
        size = f"{w}×{h}"
        try:
            size += f", {float(md['fps']):g} кадр/с" if md.get("fps") else ""
        except (TypeError, ValueError):
            pass
        parts.append(size)
    if md.get("size_mb") is not None:
        parts.append(f"{md['size_mb']} МБ")
    created = str(report.get("created_at") or "")
    mt = re.match(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})", created)
    analysis = [f"{mt.group(1)} {mt.group(2)}" if mt else created or None]
    n_seg = int(report.get("segments") or 0)
    analysis.append(f"{n_seg} {plural_ru(n_seg, 'отрезок', 'отрезка', 'отрезков')} по ~20 с" if n_seg > 1
                    else "ролик оценён целиком")
    t = report.get("timings_sec") or {}
    if t.get("total_wall", t.get("total")) is not None:
        analysis.append(f"время обработки {fmt_secs(t.get('total_wall', t.get('total')))}")
    # one model per analysis (3.1): the model of the view, «OCEAN-AI, веса MuPTA» or «AMLAI 1.0»
    rows = [("Файл", " · ".join(parts)), ("Анализ", " · ".join(p for p in analysis if p)),
            ("Модель", model_title(shown_model(report))),
            ("Отчёт", f"создан {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}; технические сведения о файле и анализе — "
                      f"в приложении {pdf.appx.get('file', 'А')}")]
    pdf.set_font("ui", "B", 8.5)
    w1 = max(pdf.get_string_width(k) for k, _ in rows) + 3
    pdf.kv_table(rows, w1=w1, size=8.5, lh=4.6)
    pdf.ln(2)


def _profile_section(pdf: Report, report: dict, explanation, charts: dict) -> None:
    """1. Radar beside «Как получены оценки» (narrative.method_notes on the clean view, design 9; its overflow
    continues under the row), then the score bars of the one model that ran (3.1)."""
    try:
        from .narrative import method_notes
        narrative = _hms_text(fix_counts(method_notes(report)), report.get("duration_sec"))
    except Exception:  # noqa: BLE001
        narrative = ""
    radar = charts.get("profile")
    text_x = pdf.l_margin + RADAR_W_MM + ROW_GAP_MM
    text_w = pdf.l_margin + pdf.epw - text_x
    radar_h = pdf.chart_height(radar, RADAR_W_MM) if radar else 0.0
    interview = scored(report.get("interview"))       # an entry without a numeric score is not shown
    bars_h = pdf.score_bars_height(report["traits"], interview)

    def layout(size: float):
        lh = size * 0.5
        pdf.set_font("ui", "", size)
        lines = pdf.multi_cell(text_w, lh, narrative, dry_run=True, output="LINES") if (narrative and radar) else []
        beside = int(max(0.0, radar_h - 7) // lh)
        row_h = max(radar_h, 7 + min(len(lines), beside) * lh)
        rest = " ".join(s.strip() for s in lines[beside:])
        return size, lh, lines, beside, row_h, rest

    def fits(v) -> bool:
        """The whole score-bar block still starts on this page with the explanation laid out this way."""
        return pdf.get_y() + 12 + v[4] + 1 + (pdf.para_height(v[5], v[0]) if v[5] else 0) + 6 + bars_h \
            <= pdf.page_break_trigger

    # the explanation is 8.5 pt; 8 pt when that keeps the whole score-bar block on the first page, or when it ends
    # the explanation beside the radar instead of leaving a tail of it across the page under the row
    cur = layout(8.5)
    alt = layout(8.0) if (radar and narrative) else None
    if alt and fits(alt) and (not fits(cur) or (cur[5] and not alt[5])):
        cur = alt
    size, lh, lines, beside, row_h, rest = cur
    pdf.section("Big Five: профиль и оценки", "profile", keep_mm=row_h if radar else 40)
    if radar:
        y0 = pdf.get_y() + 1
        pdf.image(radar, x=pdf.l_margin, y=y0, w=RADAR_W_MM, h=radar_h)
        if narrative:
            pdf.set_xy(text_x, y0)
            pdf.set_font("ui", "B", 9); pdf.cell(text_w, 6, "Как получены оценки", new_x="LEFT", new_y="NEXT")
            pdf.set_xy(text_x, y0 + 7)
            pdf.set_font("ui", "", size)
            pdf.multi_cell(text_w, lh, "\n".join(lines[:beside]), align="L", new_x="LEFT", new_y="NEXT")
        pdf.set_xy(pdf.l_margin, y0 + row_h + 1)
        if rest:
            pdf.para(rest, size)
    elif narrative:
        pdf.h3("Как получены оценки")
        pdf.para(narrative, size)
    # ---- the score bars of the one model (3.1: no second opinion, no table of averaged members)
    pdf.h3("Оценки по чертам", keep_mm=bars_h)
    pdf.score_bars(report["traits"], interview)


def _timeline_section(pdf: Report, report: dict, charts: dict, has_expl: bool) -> None:
    if "timeline" not in pdf.plan:
        return
    std = report.get("scores_std_across_segments") or {}
    cap = ""
    if std:
        cap = "Разброс между отрезками: " + ", ".join(f"{RU_SHORT[k].lower()} ±{std.get(k, 0):.2f}" for k in TRAIT_KEYS) + "."
    seg = representative(report, min_scored=2)
    if seg:
        cap += (f" Рамка и ★ — отрезок {_seg(report, seg['start'], seg['end'])}, ближайший к среднему профилю"
                + (f": по нему построены объяснения (раздел {pdf.plan['explain']})." if has_expl and "explain" in pdf.plan
                   else "."))
    if "segments" in pdf.appx:
        cap += f" Значения по каждому отрезку — в приложении {pdf.appx['segments']}."
    cap = cap.strip()
    # the heading does not repeat the title inside the chart («Big Five по ходу ролика»): the two stood 5 mm apart
    pdf.section("Как менялись оценки по ходу ролика", "timeline",
                keep_mm=pdf.chart_block_height(charts["traits"], cap))
    pdf.chart_block(charts["traits"], cap)


def _emotions_section(pdf: Report, report: dict, charts: dict) -> None:
    """3. Averages first (speech vs face, then the facial expression), then emotions over time."""
    if "emotions" not in pdf.plan:
        return
    an = report.get("analyses") or {}
    parts = analyses_parts(report)
    intro = " ".join(p for p in (parts["text_emotion"], parts["face"]) if p)
    per = an.get("per_segment") or []
    fa = an.get("face") or {}
    blocks = []                          # (chart key, caption, message when the chart is missing)
    prof_cap = ("Средняя доля каждой эмоции за ролик. Речь — модель эмоций текста по переводу транскрипта на английский; "
                "лицо — модель выражений по кадрам.")
    # an отрезок whose own transcript came out empty is «нет текста» everywhere else in the report, but the stored
    # average counts it as «нейтрально 100%»: say so, so the two views do not read as contradicting each other
    no_text = [r for r in per if empty_text(r)]
    if no_text and (an.get("emotions_text") or {}).get("mean"):
        n = len(no_text)
        prof_cap += (" Один отрезок без распознанного текста учтён в средней доле по речи как нейтральный." if n == 1
                     else f" {n} {plural_ru(n, 'отрезок', 'отрезка', 'отрезков')} без распознанного текста учтены "
                          "в средней доле по речи как нейтральные.")
    blocks.append(("emotion_profile", prof_cap,
                   "График среднего профиля эмоций не построен: данных об эмоциях нет."))
    if fa:
        hm = fa.get("head_motion")
        frames = sum(int((r.get("face") or {}).get("frames") or 0) for r in per)
        cap = []
        if hm is not None:
            cap.append(f"Движение головы {head_motion_word(hm, 'motion')}: смещение между кадрами — {hm:.0%} ширины "
                       "лица.")
        found = f"Лицо найдено в {fa['face_share']:.0%} кадров" if fa.get("face_share") is not None else ""
        if frames:
            n = len(per)
            found += ("; разобрано " if found else "Разобрано ") + (f"{frames} {plural_ru(frames, 'кадр', 'кадра', 'кадров')} "
                                                                   f"из {n} {plural_ru(n, 'отрезка', 'отрезков', 'отрезков')}")
        if found:
            cap.append(found + ".")
        cap.append(FER_NOTE.format(where=""))
        blocks.append(("face_expr", " ".join(cap), "График выражения лица не построен: лицо в кадре не найдено."))
    if len(per) >= 2:                    # one segment = the whole video: the averages above already say everything
        # hatched gaps. A segment with an empty transcript is not always silent: its own recognition may return nothing
        # while the whole-video transcript still has words in that window (tempo and pauses come from there), so
        # «нет речи» is said only when there are no words at all
        text_gaps = [r for r in per if empty_text(r) or not r.get("emotions_text")]
        gaps = []
        if text_gaps:
            gaps.append("по речи — " + ("в отрезке нет речи" if all(seg_words(r) == 0 for r in text_gaps)
                                        else "для отрезка нет распознанного текста"))
        if any(not (r.get("face") or {}).get("expressions") for r in per):
            gaps.append("по лицу — лицо не найдено")
        cap = ("Пустая светлая клетка — доля меньше 5 %; числа — доля эмоции в отрезке в процентах (от 15 %, если "
               "помещаются), жирное число — преобладающая эмоция отрезка"
               + (f", как в приложении {pdf.appx['segments']}" if "segments" in pdf.appx else "")
               + ". Полоска у названия строки — цвет этой эмоции."
               + (f" Штриховка — нет данных ({'; '.join(gaps)})." if gaps else ""))
        blocks.append(("emotions", cap, "График эмоций по ходу ролика не построен."))
    first = next((b for b in blocks if charts.get(b[0])), None)
    keep = pdf.para_height(intro, 9) + (pdf.chart_block_height(charts[first[0]], first[1]) if first else 10)
    pdf.section("Эмоции и мимика", "emotions", keep_mm=keep)
    pdf.para(intro, 9)
    for key, cap, missing in blocks:
        if charts.get(key):
            pdf.chart_block(charts[key], cap)
        else:
            pdf.para(missing, 8)


def _voice_speech_section(pdf: Report, report: dict, charts: dict) -> None:
    """4. Voice and speech together, like the web row: intro, the two time charts, «Речь в цифрах», frequent words."""
    if "voice_speech" not in pdf.plan:
        return
    an = report.get("analyses") or {}
    parts = analyses_parts(report)
    intro = " ".join(p for p in (parts["voice"], parts["speech"]) if p)
    vo, sp = an.get("voice") or {}, an.get("speech") or {}
    voice_cap = ("Средние за ролик: " + ", ".join(
        f"{VOICE_RU[d]} {vo['mean'].get(d, 0):.2f} (±{(vo.get('std') or {}).get(d, 0):.2f})" for d in VOICE_RU)
        + "; после ± — разброс между отрезками." if vo.get("mean") else "")
    speech_cap = ("Столбики — темп внутри речи, слов в минуту (левая шкала); линия — доля времени отрезка без речи, паузы "
                  "от 0.5 с (правая шкала).")
    first = "voice" if charts.get("voice") else ("speech" if charts.get("speech") else None)
    keep = pdf.para_height(intro, 9) + (pdf.chart_block_height(charts[first], voice_cap if first == "voice" else speech_cap)
                                        if first else 30)
    pdf.section("Голос и речь", "voice_speech", keep_mm=keep)
    pdf.para(intro, 9)
    # a chart without any data is not drawn; when the other one is there, one line says why this one is missing
    if charts.get("voice"):
        pdf.chart_block(charts["voice"], voice_cap)
    elif charts.get("speech"):
        pdf.para("График голоса не построен: данных о голосе нет.", 8)
    if charts.get("speech"):
        pdf.chart_block(charts["speech"], speech_cap)
    elif charts.get("voice"):
        pdf.para("График речи не построен: данных о темпе и паузах нет.", 8)
    if sp:
        items = speech_cards(sp, small_rate_words=True)     # the cards of the page, «меньше 1 на 100 слов» in words
        pdf.ln(1)
        pdf.h3("Речь в цифрах", keep_mm=pdf.cards_height(items))
        pdf.cards(items)
        vocab = vocabulary_shown(report)          # Russian words; for English speech their translations
        if vocab:
            pdf.para("Частые слова (в скобках — сколько раз): " + ", ".join(f"{w} ({n})" for w, n in vocab[:15]), 8)


def _frame_rows(pdf: Report, frames: list):
    """Rows of key frames: portrait frames all in one row (5 × 34.9 mm), landscape or square ones 3 per row."""
    from PIL import Image
    sizes = []
    for p in frames:
        with Image.open(p) as im:
            sizes.append(im.size)
    portrait = all(h > w for w, h in sizes)
    per_row = min(len(frames), 5) if portrait else 3
    gap, max_h = 4.0, 62.0
    cell_w = (pdf.epw - gap * (per_row - 1)) / per_row
    rows = []
    for i in range(0, len(frames), per_row):
        row = []
        for p, (iw, ih) in zip(frames[i:i + per_row], sizes[i:i + per_row]):
            scale = min(cell_w / iw, max_h / ih)
            row.append((p, iw * scale, ih * scale))
        rows.append(row)
    return rows, cell_w, gap


# caption under a key frame: 0.8 mm of air, two lines of 7 pt (3 mm each) and 1.7 mm before the next row
FRAME_CAP_SIZE = 7
FRAME_CAP_SIZE_NARROW = 6          # five frames in a row (34.9 mm): 7 pt leaves one word of the phrase
FRAME_CAP_NARROW = 40.0            # mm, below which the caption drops to FRAME_CAP_SIZE_NARROW
FRAME_CAP_LINE = 3.0
FRAME_CAP_H = 8.5


def _clip_words(pdf: Report, text: str, width: float, size: float) -> str:
    """`text` shortened by whole words until it fits `width`, with «…» where it was cut; "" when even the first
    word plus the ellipsis is too wide."""
    if not text:
        return ""
    pdf.set_font("ui", "", size)
    if pdf.get_string_width(text) <= width:
        return text
    words = text.split(" ")
    while len(words) > 1:
        words.pop()
        s = " ".join(words).rstrip(" ·,") + "…"
        if s != "…" and pdf.get_string_width(s) <= width:
            return s
    return ""


def _frame_second(pdf: Report, entry: dict | None, width: float) -> str:
    """The second line this cell really gets: the first candidate of frame_captions.pdf_second_line that fits its
    width, or "" when none of them does. The note under the frames is written from what this returns, so it never
    announces an expression the narrow cell had to drop."""
    if not entry:
        return ""
    pdf.set_font("ui", "", FRAME_CAP_SIZE if width >= FRAME_CAP_NARROW else FRAME_CAP_SIZE_NARROW)
    return next((c for c in frame_captions.pdf_second_line(entry) if pdf.get_string_width(c) <= width), "")


def _frame_caption(pdf: Report, entry: dict | None, x: float, y: float, width: float) -> None:
    """Two centred lines under one key frame: «2:14 · улыбается, смотрит в камеру» and «радость 62% · повысил
    экстраверсию». The second line falls back to its shorter forms («повысил экстраверсию», «повысил эм. стаб.»,
    «радость 62%») and is dropped when none of them fits; the first line, which carries the moment, is cut by
    whole words with «…» instead. Five frames in a row leave only 34.9 mm, where 7 pt keeps one word of the
    phrase: such a cell gets 6 pt, which buys about a sixth more characters on both lines."""
    if not entry:
        return
    size = FRAME_CAP_SIZE if width >= FRAME_CAP_NARROW else FRAME_CAP_SIZE_NARROW
    pdf.set_font("ui", "", size)
    first = _clip_words(pdf, entry.get("caption") or "", width, size) or (entry.get("label") or "")
    pdf.set_xy(x, y)
    pdf.cell(width, FRAME_CAP_LINE, first, align="C")
    second = _frame_second(pdf, entry, width)
    if second:
        pdf.set_xy(x, y + FRAME_CAP_LINE)
        pdf.cell(width, FRAME_CAP_LINE, second, align="C")


def _no_explain_note(pdf: Report, report: dict) -> None:
    """An OCEAN-AI job (3.1): one line under section 4 in place of section 5 — no empty section, no heading."""
    if "explain" in pdf.plan or has_explanations(report):
        return
    pdf.ln(1)
    pdf.caption(NO_EXPLAIN_RU, 8)


def _explain_section(pdf: Report, report: dict, explanation, frames: list, charts: dict, media: dict | None) -> None:
    """5. What drove the score of AMLAI 1.0: key frames, modality contributions, words."""
    if "explain" not in pdf.plan:
        return
    seg = representative(report, min_scored=2)
    tl_all = report.get("timeline") or []
    if seg and "timeline" in pdf.plan:
        intro = (f"Объяснения построены для модели AMLAI 1.0 по отрезку {_seg(report, seg['start'], seg['end'])}, "
                 "ближайшему к среднему профилю (★ на графике «Big Five по ходу ролика»).")
    else:
        intro = "Объяснения построены для модели AMLAI 1.0 по всему ролику."
    rows, cell_w, gap = _frame_rows(pdf, frames) if frames else ([], 0.0, 0.0)
    first_h = (max(h for _, _, h in rows[0]) + FRAME_CAP_H + 6) if rows else 20
    pdf.section("Что повлияло на оценку модели AMLAI 1.0", "explain", keep_mm=pdf.para_height(intro, 8.5) + first_h)
    pdf.para(intro, 8.5)
    if rows:
        # frames come from the clip the explanations were computed on: the representative segment of a long video,
        # otherwise the whole video; file names carry the frame index inside that clip (key_<i>_frame<N>.jpg).
        # The captions are the ones of the page (frame_captions): the moment and, in a few words, what is visible;
        # the second line names the expression and what the frame did to the score. The cell is narrow (34.9 mm
        # with five frames in a row), so there the caption is set in 6 pt, the first line is clipped by whole
        # words and the second falls back to shorter forms — the last of them, «повысил эм. стаб.», still carries
        # the direction — and is dropped only when even that does not fit.
        entries = {e["path"]: e for e in frame_captions.build(report, frames, explanation, media)}
        tenths = frame_captions.has_tenths(report, frames, media, explanation)
        # a job made before the captions has neither a phrase nor the expressions: the note promises only the
        # parts that are actually printed, and it names the second line, which the PDF cannot show on hover
        what = frame_captions.note_what(entries.values(), tenths, "pdf")
        # what the second line ended up carrying in this layout, not what the captions could have offered
        chosen = [(e, _frame_second(pdf, e, cell_w)) for e in entries.values()]
        has_expr = any(s and e["expr_line"] and s.startswith(e["expr_line"]) for e, s in chosen)
        has_eff = any(s and s != e["expr_line"] for e, s in chosen)
        second = [x for x, ok in (("выражение лица", has_expr), ("то, как кадр сдвинул оценку", has_eff)) if ok]
        note = ("Кадры, сильнее всего повлиявшие на оценку модели AMLAI 1.0. Рамкой отмечено найденное лицо; "
                f"под кадром — {what}"
                + (f"; во второй строке — {' и '.join(second)}." if second else "."))
        pdf.h3("Ключевые кадры", keep_mm=max(h for _, _, h in rows[0]) + FRAME_CAP_H)
        per_row = len(rows[0])
        for row in rows:
            row_h = max(h for _, _, h in row)
            if pdf.get_y() + row_h + FRAME_CAP_H > pdf.page_break_trigger:
                pdf.add_page()
            y0 = pdf.get_y()
            # a last row shorter than the others is centred: an empty cell at the right edge reads as a missing frame
            x_row = pdf.l_margin + (per_row - len(row)) * (cell_w + gap) / 2
            for i, (q, iw, ih) in enumerate(row):
                x = x_row + i * (cell_w + gap) + (cell_w - iw) / 2
                try:
                    pdf.image(q, x=x, y=y0 + (row_h - ih), w=iw, h=ih)
                except Exception:  # noqa: BLE001
                    continue
                _frame_caption(pdf, entries.get(q), x_row + i * (cell_w + gap), y0 + row_h + 0.8, cell_w)
            pdf.set_xy(pdf.l_margin, y0 + row_h + FRAME_CAP_H)
        pdf.caption(note, 8)
    if charts.get("modalities"):
        cap = ("Какая доля оценки модели AMLAI 1.0 пришлась на каждую модальность (по градиенту оценки: насколько "
               "признаки модальности сдвигают результат); в каждой строке — 100%. «<1%» — модальность почти не влияет "
               "на оценку этого ролика: модель, обученная на First Impressions V2, опирается в основном на лицо и голос.")
        pdf.chart_block(charts["modalities"], cap)
    rw_all = (explanation or {}).get("readable_words") or {}
    if rw_all:
        try:
            from .narrative import words_summary
            paras = words_summary(rw_all, explanation, RU_TITLES)
        except Exception as e:  # noqa: BLE001  (never let the words block break the whole PDF)
            paras = [f"Список слов недоступен: {str(e)[:120]}"]
        if paras:
            pdf.ln(1)
            # the heading needs its first lines beside it, not the whole block: these paragraphs may split, and
            # holding them together would leave the page before the appendix nearly empty
            pdf.h3("Слова, на которые откликнулась модель", keep_mm=min(14, sum(pdf.para_height(p, 8) for p in paras)))
            for para in paras:
                pdf.para(para, 8)
    # lists without Russian words (translation failed) are not printed: the raw English tokens stay in the JSON


def _how_to_read(pdf: Report, report: dict) -> None:
    """«Как читать результаты»: the caveats caveats.PDF_HOW_TO_READ, word for word from caveats.py. C2 explains the
    label «собеседование» and is printed only when the report carries that label (a job of AMLAI 1.0)."""
    texts = [caveats.text(c) for c in caveats.PDF_HOW_TO_READ if c != "C2" or scored(report.get("interview"))]
    # 7.5 pt like the caveats of the MBTI section: six caveats instead of the three of 2.0
    pdf.section("Как читать результаты", "how_to_read", keep_mm=sum(pdf.para_height(t, 7.5) for t in texts))
    for t in texts:
        pdf.para(t, 7.5)


# ---------------------------------------------------------------- appendices
def _file_rows(report: dict, media: dict | None) -> list:
    rows = []
    if media:
        ch = media.get("channels")
        rows += [("Имя файла", media.get("file_name") or report.get("original_file_name")),
                 ("Размер", f"{media.get('size_mb')} МБ ({media.get('size_bytes')} байт)"),
                 ("Длительность", fmt_secs(media.get("duration_sec"))),
                 ("Контейнер", media.get("container")),
                 ("Видео", f"кодек {_codec(media.get('video_codec'))}, {media.get('width')}×{media.get('height')}, "
                           f"{media.get('fps')} кадра/с" + (f", поворот {media.get('rotation')}°" if media.get("rotation") else "")),
                 ("Аудио", f"кодек {_codec(media.get('audio_codec'))}, {media.get('sample_rate')} Гц, "
                           + ("моно" if ch == 1 else "стерео" if ch == 2 else f"каналов: {ch}")),
                 ("Битрейт", f"{media.get('bitrate_kbps')} кбит/с"), ("Файл изменён", _when(media.get("modified")))]
        for k, v in media.items():
            if k.startswith("tag_"):
                rows.append((MEDIA_TAGS.get(k[4:], k[4:]), _when(v) if k == "tag_creation_time" else
                             _encoder(v) if k == "tag_encoder" else v))
        if media.get("sha256"):
            rows.append(("Контрольная сумма SHA-256", media["sha256"]))
    else:
        rows.append(("Имя файла", report.get("original_file_name") or Path(report.get("input", "")).name))
    return rows


def _analysis_rows(report: dict) -> list:
    """Appendix А, «Параметры анализа»: the one model of the report (3.1), speech recognition, training data, the
    modalities the model looked at (a job of 3.1 records them; a job of 3.0 recorded the member names instead, and
    those are replaced by the modalities of the shown model, bs3.MODALITIES), version, segments, time. The speech is
    always Russian, so no language row."""
    m = report.get("model") or {}
    main = shown_model(report)
    mods = [x for x in report.get("modalities_used") or [] if x not in MODEL_TITLES] or list(MODALITIES.get(main, ()))
    rows = [("Модель", model_title(main)),
            ("Распознавание речи", _asr_ru(m.get("asr_model")) if m.get("asr_model") else "готовый транскрипт"),
            ("Обучающие данные", TRAINED_ON.get(main) or str(m.get("trained_on") or "—")),
            ("Модальности", ", ".join(MODALITY_TITLES.get(x, x) for x in mods) or "—"),
            ("Версия", _version_ru(m.get("version")))]
    if report.get("segments"):
        n_seg = int(report["segments"])
        rows.append(("Отрезки", f"{n_seg} по ~20 с; итог — среднее с весом по длительности" if n_seg > 1
                     else "один отрезок (весь ролик)"))
    t = report.get("timings_sec") or {}
    if t:
        rows.append(("Время обработки", fmt_secs(t.get("total_wall", t.get("total")))))
    return rows


def _dominant_text(r: dict | None, source: str) -> str:
    """«нейтрально 99%» as on the web; «нет речи» / «нет текста» for an empty transcript, «—» without data."""
    if not r:
        return "—"
    if source == "text" and empty_text(r):
        return "нет речи" if seg_words(r) == 0 else "нет текста"
    d = dominant_emotion(r, source)
    # only the seven text emotions are named (the face labels come aliased to them); any other label stays as it is
    return f"{EMO_RU[d[0]] if d[0] in EMOTION_ORDER else d[0]} {d[1]:.0%}" if d else "—"


def _segments_table(pdf: Report, report: dict, mb: dict | None = None) -> None:
    rows_in = segment_rows(report)
    tl_scored = segments.scored(report)
    keys = TRAIT_KEYS + (["interview"] if tl_scored and all("interview" in t["scores"] for t in tl_scored) else [])
    seg = representative(report, min_scored=2)
    star = bool(seg)
    # the MBTI type of the main system on every segment, with X on the borderline axes (design 10.7, task T27)
    seg_types = segment_types_by_start(mb)
    mbti_col = bool(seg_types) and any(seg_types.values())
    # the three voice columns hold «0.46» but were titled «Возбуж-/дение»: their headers alone took 43 of the 190 mm
    # and pushed the whole table below 7.5 pt in the longer videos. They are shortened like the trait columns and
    # spelled out in the legend under the table
    header = (["Отрезок"] + [SEG_HEAD[k] for k in keys] + (["MBTI"] if mbti_col else [])
              + ["по речи", "по лицу", "Возб.", "Увер.", "Позит.", "Темп", "Паузы"])
    rows, any_no_text, any_skipped, any_no_primary = [], False, False, False
    for s, e, t, r in rows_in:
        label = _seg(report, s, e) + (" ★" if seg and t is seg else "")
        if t and t.get("scores"):
            scores = [f"{float(t['scores'][k]):.2f}" if k in t["scores"] else "—" for k in keys]
        elif t:
            # a segment without scores of the main system (clean view) or not scored at all: a dash in every column
            # (the word «пропущен» was wider than its column and ran over the next one)
            scores = ["—"] * len(keys)
            if t.get("no_primary"):
                any_no_primary = True
            else:
                any_skipped = True
        else:
            scores = ["—"] * len(keys)
        if mbti_col:
            scores.append(seg_types.get(int(round(float(s)))) or "—")
        v = (r or {}).get("voice") or {}
        sp = (r or {}).get("speech") or {}
        wpm = sp.get("words_per_min_speech")
        speech = _dominant_text(r, "text")
        any_no_text = any_no_text or speech == "нет текста"
        rows.append([label] + scores + [speech, _dominant_text(r, "face")]
                    + [f"{float(v[d]):.2f}" if v.get(d) is not None else "—" for d in ("arousal", "dominance", "valence")]
                    + [f"{float(wpm):.0f}" if wpm is not None else "—",
                       f"{float(sp.get('pause_share') or 0):.0%}" if sp else "—"])
    # column widths from the content: the widest header line (bold) or cell (regular) of each column plus the margins;
    # the cell margins are narrower than elsewhere (0.6 mm). When 14 columns still do not fit the text width, the
    # whole table goes one step smaller instead of being squeezed: squeezing wraps the titles one line further, and a
    # cell wider than its column runs over the rule next to it
    c_margin = pdf.c_margin
    pdf.c_margin = 0.6

    def col_widths(size: float) -> list:
        ws = []
        for j, h in enumerate(header):
            pdf.set_font("ui", "B", size)
            w = max(pdf.get_string_width(line) for line in str(h).split("\n"))
            pdf.set_font("ui", "", size)
            w = max([w] + [pdf.get_string_width(str(r[j])) for r in rows])
            ws.append(w + 2 * pdf.c_margin + 0.4)
        return ws

    for size in (7.5, 7.2, 7.0):
        widths = col_widths(size)
        if sum(widths) <= pdf.epw:
            break
    groups = [("", 1), (("Big Five и «собеседование», 0…1" if "interview" in keys else "Big Five, 0…1"), len(keys))]
    groups += ([("Тип", 1)] if mbti_col else []) + [("Преобладающая эмоция", 2), ("Голос, 0…1", 3), ("Речь", 2)]
    pdf.table(header, rows, widths, size=size, groups=groups, row_h=4.8,
              cont_title=f"Приложение {pdf.appx.get('segments', 'Б')}. Значения по отрезкам (продолжение)")
    pdf.c_margin = c_margin
    legend = ", ".join(f"{_one_line(SEG_HEAD[k])} — {RU_TITLES[k].lower() if k != 'interview' else 'впечатление «собеседование»'}"
                       for k in keys) + ". "
    if mbti_col:
        legend += ("MBTI — тип отрезка, X — ось на границе (подробнее — в разделе "
                   f"{pdf.plan['mbti']}). " if "mbti" in pdf.plan else "MBTI — тип отрезка, X — ось на границе. ")
    if any_no_primary:
        legend += ("«—» в столбцах Big Five" + (" и MBTI" if mbti_col else "")
                   + f" — модель {MODEL_TITLES.get(shown_model(report), 'OCEAN-AI')} не дала оценки отрезка, он не "
                   "вошёл в основные оценки. ")
    legend += ("Возб. — возбуждение, Увер. — уверенность, Позит. — позитивность. "
               "Эмоция — преобладающая в отрезке и её доля. Темп — слов в минуту речи; «—» — речи в отрезке меньше 3 с. "
               "Паузы — доля времени отрезка, занятая паузами от 0.5 с.")
    if any_no_text:
        legend += (" «нет текста» — для отрезка не распознан текст, поэтому эмоция речи не оценена; темп и паузы берутся "
                   "из транскрипта всего ролика.")
    if any_skipped:
        legend += " «—» во всех столбцах Big Five — отрезок не оценён."
    if star:
        legend += " ★ — отрезок для объяснений."
    pdf.caption(legend, 7)


def _notable(report: dict, has_expl: bool) -> dict:
    """{start of a segment: [reasons]} of the notable segments, at most BEHAVIOR_MAX, chosen in this order: the segment
    for the explanations, segments with unusual scores (largest deviation first), segments whose dominant speech
    emotion or facial expression differs from the one of the whole video (largest share first)."""
    from .palette import EMO_ALIAS
    an = report.get("analyses") or {}
    per = an.get("per_segment") or []
    cands: list = []                    # (start, reason) in priority order
    seg = representative(report, min_scored=2)
    if seg and has_expl:
        cands.append((float(seg["start"]), "отрезок для объяснений ★"))
    for t, z in sorted(odd_segments(report), key=lambda tz: -abs(tz[1])):
        cands.append((float(t["start"]), f"оценки {'выше' if z > 0 else 'ниже'} остального ролика"))
    te_dom = (an.get("emotions_text") or {}).get("dominant")
    speech = []
    for r in per:
        d = None if empty_text(r) else dominant_emotion(r, "text")
        if d and te_dom and d[0] != te_dom:
            speech.append((d[1], float(r["start"]), "в речи нейтральный тон" if d[0] == "neutral" else
                           f"в речи преобладает {EMO_RU.get(d[0], d[0])}"))
    face_dom = EMO_ALIAS.get((an.get("face") or {}).get("dominant") or "", (an.get("face") or {}).get("dominant"))
    face = []
    for r in per:
        d = dominant_emotion(r, "face")
        if d and face_dom and d[0] != face_dom:
            face.append((d[1], float(r["start"]), "лицо нейтральное" if d[0] == "neutral" else
                         f"на лице преобладает {EMO_RU.get(d[0], d[0])}"))
    for group in (speech, face):
        cands += [(s, f"{why} ({p:.0%})") for p, s, why in sorted(group, key=lambda g: -g[0])]
    chosen: dict = {}
    for s, _ in cands:
        if s not in chosen and len(chosen) < BEHAVIOR_MAX:
            chosen[s] = []
    for s, why in cands:
        if s in chosen and why not in chosen[s]:
            chosen[s].append(why)
    return chosen


def _behavior_appendix(pdf: Report, report: dict, has_expl: bool) -> None:
    entries = behavior_by_segment(report)
    letter = pdf.appx["behavior"]
    n_all = max(len(entries), len(segment_rows(report)))
    if not entries:                      # a description without segment labels is printed as it is
        pdf.h2(f"Приложение {letter}. Описание поведения", keep_mm=20)
        pdf.caption("Описание строит видеоязыковая модель по кадрам ролика.", 7.5)
        pdf.para(_dash(report.get("behavior_description_ru")), 8.5)
        return
    reasons = _notable(report, has_expl)
    if n_all <= BEHAVIOR_MAX:
        shown = entries
        title = f"Приложение {letter}. Описание поведения по отрезкам"
        note = "Описание строит видеоязыковая модель по кадрам каждого отрезка."
    else:
        starts = sorted(reasons)
        shown = [en for en in entries if any(abs(en[0] - s) < 1.5 for s in starts)]
        title = f"Приложение {letter}. Описание поведения в заметных отрезках"
        note = ("Описание строит видеоязыковая модель по кадрам каждого отрезка. Здесь — отрезок для объяснений, отрезки "
                "с необычными оценками и отрезки, где эмоция речи или выражение лица отличаются от преобладающих. "
                f"Описания всех {n_all} {plural_ru(n_all, 'отрезка', 'отрезков', 'отрезков')} — на веб-странице "
                "(вкладка «Объяснения») и в result.json.")
    first = shown[0] if shown else None
    pdf.h2(title, keep_mm=pdf.para_height(note, 7.5) + (pdf.para_height(first[2], 8.5) + 6 if first else 0))
    pdf.caption(note, 7.5)
    for s, e, body in shown:
        why = next((w for st, w in reasons.items() if abs(st - s) < 1.5), [])
        head = f"[{_seg(report, s, e)}]" + (" " + "; ".join(why) if why else "")
        if pdf.get_y() + 5 + min(pdf.para_height(body, 8.5), 13) > pdf.page_break_trigger:
            pdf.add_page()
        pdf.set_x(pdf.l_margin)
        pdf.set_font("ui", "B", 8.5); pdf.multi_cell(0, 4.4, head, new_x="LMARGIN", new_y="NEXT", align="L")
        pdf.para(_dash(body), 8.5)
        pdf.ln(0.5)


TRANSCRIPT_WIDOW_MM = 30            # below this much on the last page the transcript is set one step smaller


def _transcript_size(pdf: Report, text: str, extra_mm: float = 0.0) -> float:
    """8.5 pt, or 8 pt when that keeps the tail of the transcript off a page of its own. The transcript is the last
    thing in the report and flows freely, so a few of its lines can land alone on a sheet that is then 90% white.
    extra_mm: what follows the text (the line «Распознавание речи оборвалось…»), which must not be left alone either."""
    def widow(size: float) -> float:
        """mm of the text that would end up on a page with nothing else on it (0 = it ends on a shared page)."""
        h = pdf.para_height(text, size) + extra_mm
        left = pdf.page_break_trigger - pdf.get_y()
        if h <= left:
            return 0.0
        return (h - left) % (pdf.page_break_trigger - pdf.t_margin)
    w85 = widow(8.5)
    if not 0 < w85 < TRANSCRIPT_WIDOW_MM:
        return 8.5
    w80 = widow(8.0)
    return 8.0 if w80 == 0 or w80 > w85 else 8.5


CUT_NOTE = "Распознавание речи оборвалось на этом месте."
KV_LH = 8.5 * 0.61                  # line height of the two tables of appendix А (kv_table default at 8.5 pt)
KV_LH_TIGHT = 4.6                   # … set tighter so that a short transcript stays on the page of appendix А
SHORT_APPENDIX_MM = 30              # a transcript appendix below this is «short»: it never opens a page of its own


def _transcript_parts(report: dict) -> tuple[str, str, bool]:
    """(note, text, cut) of the transcript appendix. The recogniser can stop in the middle of a sentence; an ellipsis
    and one line (CUT_NOTE) say that the text really ends there, so the last page does not read as a fault."""
    note, transcript = transcript_shown(report)
    t = str(transcript or "").rstrip()
    cut = bool(t) and t[-1] not in ".!?…»)"
    return note or "", t + ("…" if cut else ""), cut


def _appendix_layout(pdf: Report, report: dict, media: dict | None, gap_top: float = 3.0) -> dict:
    """How appendix А and a transcript appendix that follows it directly are set, from the current position.

    Normal: the gaps and table lines of the report. A one-line transcript used to open a page of its own when
    appendix А ended a few millimetres too low (the heading rule of h2), leaving an A4 page with two lines on it. So
    when the transcript appendix is short (under SHORT_APPENDIX_MM) and does not fit under appendix А as it is,
    the two are measured tightened — smaller gaps above and between the tables, tighter table lines
    (KV_LH_TIGHT), the transcript at 8 pt — and set that way when that keeps them on one page (`tight`); when even
    that is not enough, the appendices start on the next page together (`new_page`), so the short appendix never
    stands alone. Returns {"tight", "new_page", "gap_top", "gap_mid", "lh", "size"}."""
    normal = dict(tight=False, new_page=False, gap_top=gap_top, gap_mid=1.0, lh=KV_LH, size=8.5)
    if "transcript" not in pdf.appx or "segments" in pdf.appx or "behavior" in pdf.appx:
        return normal
    note, t, cut = _transcript_parts(report)
    cut_h = (0.5 + pdf.para_height(CUT_NOTE, 7.5)) if cut else 0.0
    file_rows, analysis_rows = _file_rows(report, media), _analysis_rows(report)

    def heights(lay: dict) -> tuple[float, float]:
        """(appendix А with the title «Приложения», the transcript appendix) as _appendices prints them."""
        a = (lay["gap_top"] + 8 + 10 + 6 + pdf.kv_table_height(file_rows, lh=lay["lh"]) + lay["gap_mid"] + 6
             + pdf.kv_table_height(analysis_rows, lh=lay["lh"]))
        tr = 10 + ((0.5 + pdf.para_height(note, 7.5)) if note else 0.0) + pdf.para_height(t, lay["size"]) + cut_h
        return a, tr

    a, tr = heights(normal)
    left = pdf.page_break_trigger - pdf.get_y()
    if a + tr <= left or tr > SHORT_APPENDIX_MM:
        return normal
    tight = dict(tight=True, new_page=False, gap_top=min(gap_top, 1.0), gap_mid=0.0, lh=KV_LH_TIGHT, size=8.0)
    a2, tr2 = heights(tight)
    if a2 + tr2 <= left:
        return tight
    return {**normal, "new_page": True, "gap_top": 0.0}


def _appendices(pdf: Report, report: dict, media: dict | None, has_expl: bool, mb: dict | None = None) -> None:
    # the appendices do not start a page of their own: a forced break left the page before them three quarters empty
    # in every report. They begin here when the title and the first rows of appendix А still fit (8 mm for the title,
    # 10 mm for the heading of the appendix, 40 mm of its table), otherwise on the next page
    gap_top = 3.0
    if pdf.get_y() + gap_top + 8 + 10 + 40 > pdf.page_break_trigger:
        pdf.add_page()
        gap_top = 0.0
    lay = _appendix_layout(pdf, report, media, gap_top)
    if lay["new_page"]:                 # a short transcript that would stand alone on the last page otherwise
        pdf.add_page()
    pdf.ln(lay["gap_top"])
    pdf.set_font("ui", "B", 14); pdf.cell(0, 8, "Приложения", new_x="LMARGIN", new_y="NEXT")
    pdf.h2(f"Приложение {pdf.appx['file']}. Файл и параметры анализа", keep_mm=40)
    pdf.h3("Файл")
    pdf.kv_table(_file_rows(report, media), lh=lay["lh"])
    pdf.ln(lay["gap_mid"])
    pdf.h3("Параметры анализа")
    pdf.kv_table(_analysis_rows(report), lh=lay["lh"])
    if "segments" in pdf.appx:
        pdf.h2(f"Приложение {pdf.appx['segments']}. Значения по отрезкам", keep_mm=30)
        _segments_table(pdf, report, mb)
    if "behavior" in pdf.appx:
        _behavior_appendix(pdf, report, has_expl)
    if "transcript" in pdf.appx:
        note, t, cut = _transcript_parts(report)
        size = lay["size"] if lay["tight"] else 8.5
        pdf.h2(f"Приложение {pdf.appx['transcript']}. Транскрипт речи",
               keep_mm=pdf.para_height(note, 7.5) + min(20, pdf.para_height(t, size)))
        if note:
            pdf.caption(note, 7.5)
        if t:
            if not lay["tight"]:
                size = _transcript_size(pdf, t, (0.5 + pdf.para_height(CUT_NOTE, 7.5)) if cut else 0.0)
            pdf.para(t, size)
            if cut:
                pdf.caption(CUT_NOTE, 7.5)


# ---------------------------------------------------------------- the report
FACT_COLS = {7: 4, 8: 4}            # the type card makes 7 key facts: 4 + 3 cards in two rows instead of 3 + 3 + 1


def _render(report: dict, explanation, media, frames: list, charts: dict, fname: str, total: int | None,
            mb: dict | None, ch) -> Report:
    pdf = Report(file_label=fname, total_pages=total)
    _plan(pdf, report, explanation, frames, charts, mb)
    has_expl = "explain" in pdf.plan
    pdf.add_page()
    pdf.h1(f"{PRODUCT} — отчёт по видео: характеристика личности, Big Five, MBTI, эмоции, голос, речь")
    _passport(pdf, report, media, fname)
    if ch is not None:
        characterization_block(pdf, ch)
    # the key facts of the page (facts.fact_cards); «Речь в цифрах» of section 4 prints the pauses and the fillers
    speech_follows = "voice_speech" in pdf.plan and bool((report.get("analyses") or {}).get("speech"))
    facts, legend = fact_cards(report, mb, speech_cards_follow=speech_follows)
    if facts:
        cols = FACT_COLS.get(len(facts), 3)
        # the legend runs to two lines at this width, so the heading keeps the grid and both of them together
        pdf.h3("Ключевые факты", keep_mm=pdf.cards_height(facts, cols, value_first=True) + (8 if legend else 0))
        pdf.cards(facts, cols, value_first=True)
        if legend:
            pdf.caption(FACTS_LEGEND)
    if "timeline" not in pdf.plan:
        dur = float(report.get("duration_sec") or 0)
        pdf.para(("Ролик короче 30 с оценивается целиком" if 0 < dur <= settings.SINGLE_CLIP_MAX_SEC
                  else "Ролик оценён целиком, одним отрезком")
                 + ", поэтому графиков по ходу ролика нет.", 8)
    _profile_section(pdf, report, explanation, charts)
    mbti_section(pdf, report, mb)
    _timeline_section(pdf, report, charts, has_expl)
    _emotions_section(pdf, report, charts)
    _voice_speech_section(pdf, report, charts)
    _no_explain_note(pdf, report)
    _explain_section(pdf, report, explanation, frames, charts, media)
    _how_to_read(pdf, report)
    _appendices(pdf, report, media, has_expl, mb)
    return pdf


def build_pdf(report: dict, out_path: str | Path, explanation: dict | None = None, media: dict | None = None,
              key_frames: list[str] | None = None, mbti: dict | None = None, character=None) -> str:
    """The PDF of a clean view (scores.clean_view; a raw result.json is cleaned here). `mbti` — the section of
    mbti.get_mbti, `character` — characterization.build; both are computed here when the caller does not pass them
    (jobview.from_report: the view, its section and its characterization, each built once)."""
    if mbti is None and character is None:
        jv = jobview.from_report(report)
        report, mbti, character = jv.view, jv.mb, jv.character
    elif not report.get("view_meta"):
        from .scores import clean_view
        report = clean_view(report)
    if character is None:                           # the characterization of the caller's own section
        from . import characterization
        character = characterization.build(report, mbti)
    fname = report.get("original_file_name") or (media or {}).get("file_name") or Path(report.get("input", "")).name
    frames = [p for p in (key_frames or []) if os.path.exists(p)]
    charts = dict(report.get("chart_files") or {})
    if explanation and not charts.get("modalities") and charts:
        # save_pdf_charts called without the explanation (an older caller): the modality chart is drawn here
        try:
            p = save_modalities_chart(explanation, Path(next(iter(charts.values()))).parent)
            if p:
                charts["modalities"] = p
        except Exception:  # noqa: BLE001
            pass
    # the page count of the footer («стр. 3 из 7») comes from a first layout pass; the second one is written
    total = _render(report, explanation, media, frames, charts, fname, None, mbti, character).pages_count
    pdf = _render(report, explanation, media, frames, charts, fname, total, mbti, character)
    out_path = Path(out_path)
    pdf.output(str(out_path))
    return str(out_path)
