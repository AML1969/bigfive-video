"""PDF report for one analysed video (fpdf2): file metadata, analysis settings, scores with percentiles,
per-segment timeline, key frames as images, modality contributions and words, behaviour description, transcript.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path

from fpdf import FPDF

from .norms import RU_SHORT, TRAIT_KEYS
from .palette import SCORE_BAR_PDF, TRAIT_BAR_PDF, emo_pdf
from .report import DISCLAIMER_RU, INTERVIEW_DISCLAIMER_RU, fmt_secs, mmss_labels, seg_label
from .ru_texts import transcript_shown, vocabulary_shown

TITLES = {
    "openness": "Открытость опыту", "conscientiousness": "Добросовестность", "extraversion": "Экстраверсия",
    "agreeableness": "Доброжелательность", "emotional_stability": "Эмоциональная стабильность",
    "interview": "Впечатление «собеседование»",
}
MEMBERS = {"oceanai": "OCEAN-AI", "mm": "Своя модель (MM-PSYCHE)", "scene": "SSL-MEPR (сцена)",
           "face": "лицо", "audio": "голос\n(CLAP)", "audio_whisper": "голос\n(Whisper)", "audio_xlsr": "голос\n(XLS-R)",
           "audio_w2v_emo": "голос\n(wav2vec2)", "text": "речь", "behavior": "описание\nповедения"}
# two-line column headers: long Russian words do not fit narrow table columns
TITLES_2L = {"openness": "Открытость\nопыту", "conscientiousness": "Добросовест-\nность", "extraversion": "Экстра-\nверсия",
             "agreeableness": "Доброжела-\nтельность", "emotional_stability": "Эмоц.\nстабильность",
             "interview": "Собесе-\nдование"}
SYSTEM_TITLES = {"oceanai": "OCEAN-AI", "mm": "своя модель (MM-PSYCHE)", "scene": "SSL-MEPR (сцена)",
                 "ensemble": "ансамбль", "sslmepr": "SSL-MEPR"}
MODALITY_TITLES = {**SYSTEM_TITLES, "audio": "голос", "video": "видео", "text": "речь", "face": "лицо",
                   "behavior": "описание поведения"}
# container tags from media.probe_media (ffprobe names) -> row titles of the «Файл» table
MEDIA_TAGS = {"creation_time": "Записан (метка в файле)", "encoder": "Программа записи",
              "com.apple.quicktime.make": "Производитель камеры", "com.apple.quicktime.model": "Модель камеры",
              "title": "Название"}


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


SMALL_POOL = 20      # below this many processed videos a percentage only looks precise (same rule as the web page)


def _small_pool_phrase(pct: float, ref: str, group: str):
    """«выше, чем у большинства из 10 русских роликов» for a small pool, otherwise None."""
    import re as _re
    m = _re.search(r"N\s*=\s*(\d+)", ref or "")
    if "пула" not in (ref or "") or not m or int(m.group(1)) >= SMALL_POOL:
        return None
    n = int(m.group(1))
    if pct > 60:
        return f"выше, чем у большинства из {n} {group}"
    if pct < 40:
        return f"ниже, чем у большинства из {n} {group}"
    return f"примерно посередине среди {n} {group}"


def _small_pool_n(ref: str):
    """Size of the pool when it is below SMALL_POOL (the position is then given in words), otherwise None."""
    m = re.search(r"N\s*=\s*(\d+)", ref or "")
    return int(m.group(1)) if "пула" in (ref or "") and m and int(m.group(1)) < SMALL_POOL else None


def _trained_on(m: dict, members: list) -> str:
    """model.trained_on names one corpus for the whole ensemble; say which member learned from what."""
    oc = {"ru": "MuPTA (русская речь)", "en": "First Impressions V2"}.get(m.get("lang"))
    if m.get("backend") == "ensemble" and members:
        parts = [f"OCEAN-AI — {oc}" if x == "oceanai" and oc else "своя модель — First Impressions V2" if x == "mm"
                 else None for x in members]
        if all(parts):
            return "; ".join(parts)
    return str(m.get("trained_on") or "—")


def pct_phrase(pct, ref: str = "") -> str:
    """'выше, чем у 83% русских роликов' / 'ниже, чем у 95% людей FIV2' / 'посередине среди …' / 'пул пока мал'."""
    if pct is None:
        return "мало роликов для сравнения"
    group = "русских роликов" if "пула" in (ref or "") else "людей в FIV2"
    pct = max(0.0, min(100.0, float(pct)))
    small = _small_pool_phrase(pct, ref, group)
    if small:
        return small
    if round(pct) == 50:          # «выше, чем у 50%» reads as "above average" although it is exactly the middle
        return f"посередине среди {group}"
    return f"выше, чем у {pct:.0f}% {group}" if pct > 50 else f"ниже, чем у {100 - pct:.0f}% {group}"


def _ref_ru(ref: str) -> str:
    """percentile_ref from result.json (genitive, reads after «относительно») without technical English words."""
    r = re.sub(r",\s*своя модель\s*$", "", ref or "")
    r = re.sub(r"\(N\s*=\s*(\d+)\)", r"(сейчас их \1)", r)          # pool size, as on the web page
    return r.replace("train First Impressions V2", "обучающей выборки First Impressions V2").replace(
        "train FIV2", "обучающей выборки FIV2")


def _where_ru(ref: str) -> str:
    """The reference group as on the web page: «среди обработанных русских роликов (сейчас их 5)» for the pool of
    processed videos, «относительно обучающей выборки First Impressions V2 (6000 клипов)» otherwise."""
    r = _ref_ru(ref)
    m = re.match(r"^пула\s+(обработанных\s.+)$", r)
    return f"среди {m.group(1)}" if m else f"относительно {r}"


def _asr_ru(name) -> str:
    """'openai/whisper-large-v3-turbo' -> 'Whisper large-v3-turbo': the model name without the hub prefix."""
    m = re.match(r"^(?:[\w.-]+/)?whisper-(.+)$", str(name or ""), re.I)
    return f"Whisper {m.group(1)}" if m else str(name or "")


def _version_ru(v) -> str:
    """'2.0.0a1' -> '2.0.0, альфа-версия 1' (PEP 440 pre-release suffixes a/b/rc in words)."""
    m = re.match(r"^(\d+(?:\.\d+)*)(a|b|rc)(\d+)$", str(v or ""))
    if not m:
        return str(v or "—")
    return f"{m.group(1)}, {({'a': 'альфа', 'b': 'бета', 'rc': 'предрелизная'})[m.group(2)]}-версия {m.group(3)}"


def _group_name(keys: list[str]) -> str:
    """Which score rows a reference group applies to, for the note under the score bars."""
    if set(keys) == set(TRAIT_KEYS):
        return "пять черт"
    return ", ".join("«собеседование»" if k == "interview" else TITLES[k].lower() for k in keys)


MARGIN_MM = 10                      # left, right and top page margin (the fpdf default, made explicit)
TEXT_W_MM = 210 - 2 * MARGIN_MM     # A4 text width: every chart (pdf_charts draws them this wide), table and row of frames

FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/mnt/c/Windows/Fonts/arial.ttf", "/mnt/c/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
]


class Report(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(MARGIN_MM, MARGIN_MM, MARGIN_MM)
        self.set_auto_page_break(auto=True, margin=15)
        for reg, bold in FONT_CANDIDATES:
            if os.path.exists(reg):
                self.add_font("ui", "", reg)
                self.add_font("ui", "B", bold if os.path.exists(bold) else reg)
                break
        else:
            raise RuntimeError("no Unicode TTF font found for the PDF (DejaVu or Arial)")
        self.set_font("ui", "", 10)
        self.sections = 0                        # numbered sections so far: a section without data takes no number
        self.section_no: dict = {}               # section key -> its number, for references («★ в таблице раздела 4»)

    def footer(self):
        self.set_y(-10)
        self.set_font("ui", "", 8)
        self.set_text_color(85)                 # #555555, 7.46:1 on white
        self.cell(0, 5, f"BS 2.0 · стр. {self.page_no()}", align="R")
        self.set_text_color(0)

    def h1(self, text):
        # a title longer than the line wraps instead of running past the right edge of the page
        self.set_font("ui", "B", 16); self.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT", align="L"); self.ln(2)

    def h2(self, text):
        if self.get_y() > self.h - self.b_margin - 25:      # never leave a heading alone at the bottom of a page
            self.add_page()
        self.ln(2); self.set_font("ui", "B", 12); self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.set_font("ui", "", 10)

    def section(self, title: str, key: str | None = None) -> int:
        """Numbered section heading «4. Таймлайн по отрезкам»; numbers follow the sections actually printed."""
        self.sections += 1
        if key:
            self.section_no[key] = self.sections
        self.h2(f"{self.sections}. {title}")
        return self.sections

    def para_height(self, text, size=10) -> float:
        """Height in mm that para(text, size) takes, without printing it."""
        self.set_font("ui", "", size)
        lines = self.multi_cell(self.epw, max(3.6, size * 0.5), str(text), align="L", dry_run=True, output="LINES")
        return len(lines) * max(3.6, size * 0.5)

    def h3(self, text, keep_mm: float = 20):
        """Bold sub-heading kept on the same page as at least `keep_mm` of what follows it."""
        if self.get_y() + 6 + keep_mm > self.h - self.b_margin:
            self.add_page()
        self.set_x(self.l_margin)
        self.set_font("ui", "B", 9); self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.set_font("ui", "", 10)

    def para(self, text, size=10):
        self.set_x(self.l_margin)
        self.set_font("ui", "", size)
        # break words longer than the line (hashes, URLs) so fpdf can wrap them
        text = " ".join(w if len(w) < 60 else " ".join(w[i:i + 60] for i in range(0, len(w), 60)) for w in str(text).split(" "))
        # leading follows the font size (small notes stay close); left-aligned: justified Russian lines get wide gaps
        self.multi_cell(0, max(3.6, size * 0.5), text, align="L")
        self.ln(1)

    def chart(self, path, keep_mm: float = 0):
        """Chart PNG across the full text width, at the size pdf_charts drew it (1 pt of the figure = 1 pt on paper).
        A chart that does not fit on the rest of the page, together with `keep_mm` of the caption under it, starts a
        new page: charts are never cut and a caption never ends up on the page after its chart."""
        from PIL import Image
        with Image.open(path) as im:
            w_px, h_px = im.size
        h = self.epw * h_px / w_px
        if self.get_y() + h + keep_mm > self.page_break_trigger:
            self.add_page()
        self.set_x(self.l_margin)
        self.image(path, w=self.epw, h=h)

    def kv_table(self, rows, w1=55):
        self.set_font("ui", "", 9)
        for k, v in rows:
            self.set_x(self.l_margin)
            self.set_font("ui", "B", 9); self.cell(w1, 5.5, str(k))
            self.set_font("ui", "", 9)
            v = str(v)
            if len(v) > 48 and " " not in v:            # e.g. sha256: split so it wraps
                v = " ".join(v[i:i + 32] for i in range(0, len(v), 32))
            self.multi_cell(0, 5.5, v, new_x="LMARGIN", new_y="NEXT", align="L")

    def score_bars(self, traits: dict, interview: dict | None):
        """One row per trait: the bar is the score itself (0…1, what a reader expects to see filled), the text
        gives the score and the position relative to the reference population in words."""
        items = [(k, traits[k]) for k in TRAIT_KEYS] + ([("interview", interview)] if interview else [])
        bar_w, bar_h = 52, 3.6
        c = SCORE_BAR_PDF
        # the label column is as wide as the longest title; the phrase after the bar must end at the right margin
        self.set_font("ui", "", 8.5)
        label_w = max(self.get_string_width(TITLES[k]) for k, _ in items) + 3
        groups: dict[str, list[str]] = {}          # reference group in words («среди …», «относительно …») -> item keys
        x = self.l_margin + label_w
        for k, t in items:
            pct = t.get("percentile", t.get("percentile_vs_fiv2")); score = float(t["score"])
            ref = t.get("percentile_ref", "train First Impressions V2 (6000 клипов)")
            groups.setdefault(_where_ru(ref), []).append(k)
            self.set_font("ui", "", 8.5)
            self.cell(label_w, 6, TITLES[k])
            x, y = self.get_x(), self.get_y() + 1.2
            # track: light fill with a grey outline, so the full 0…1 length is visible (outline 3.84:1 on white)
            self.set_fill_color(c["track"]); self.set_draw_color(c["outline"]); self.set_line_width(0.2)
            self.rect(x, y, bar_w, bar_h, style="DF")
            # each bar in the colour of its line on the charts (the interview bar stays brown)
            hexc = TRAIT_BAR_PDF.get(k)
            self.set_fill_color(*((int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)) if hexc else c["fill"]))
            self.rect(x, y, bar_w * max(0.01, min(1.0, score)), bar_h, style="F")
            # 0.5 reference: grey stubs outside the bar; inside it a white segment where the fill covers the middle,
            # otherwise a grey one on the light track (#555555 on #f2f2f2, 6.7:1), so the mark crosses every bar
            xm = x + bar_w / 2
            self.set_draw_color(c["mid_tick"])
            self.line(xm, y - 1.0, xm, y); self.line(xm, y + bar_h, xm, y + bar_h + 1.0)
            if score > 0.5:
                self.set_draw_color(255)
            self.line(xm, y, xm, y + bar_h)
            self.set_draw_color(0)
            self.set_x(x + bar_w + 2)
            text = f"{score:.2f}   {pct_phrase(pct, ref)}"
            self.set_font("ui", "", 8)
            if self.get_string_width(text) > self.l_margin + self.epw - self.get_x():
                self.set_font("ui", "", 7.5)
            self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        # scale under the bars: 0, 0.5, 1, each label centred on its point of the bar
        self.set_font("ui", "", 7.5); self.set_text_color(85)
        y = self.get_y()
        for val, pos in (("0", x), ("0.5", x + bar_w / 2), ("1", x + bar_w)):
            self.set_xy(pos - 5, y); self.cell(10, 3.6, val, align="C")
        self.set_xy(self.l_margin, y + 4.2)
        if len(groups) == 1:
            where = next(iter(groups))
        else:
            where = "; ".join(f"{_group_name(keys)} — {ref}" for ref, keys in groups.items())
        note = (f"Полоска — оценка от 0 до 1, цвет — как у линии этой черты на графиках, чёрточка — середина шкалы (0.5). "
                f"Рядом — положение: {where}.")
        pool_ref = traits[TRAIT_KEYS[0]].get("percentile_ref", "")
        n = _small_pool_n(pool_ref)
        if n is not None:
            note += f" Роликов в сравнении пока {n}, поэтому положение черт описано словами, а не в процентах."
        if interview:
            note += " Коричневая полоска — впечатление «собеседование» (своя модель, шкала FIV2)."
        self.set_font("ui", "", 8); self.set_text_color(85)                 # #555555, 7.46:1
        self.multi_cell(0, 4.2, note, new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_text_color(0)

    def _table_header(self, header, widths, size, chips=None):
        """Header cells may contain '\\n' (multi-line titles); all cells get the same height.
        chips: optional colour per column ('#rrggbb' or None), drawn as a small outlined strip above the title, so a
        table column is linked to the matching chart colour."""
        self.set_font("ui", "B", size)
        lines = max(str(h).count("\n") + 1 for h in header)
        lh = size * 0.5                      # line height in mm for this font size
        chip_h = 2.2 if chips and any(chips) else 0.0
        top = chip_h + 1.0 if chip_h else 0.0
        hh = lines * lh + 1.5 + top
        x0, y0 = self.l_margin, self.get_y()
        x = x0
        for i, (h, w) in enumerate(zip(header, widths)):
            self.rect(x, y0, w, hh)
            chip = chips[i] if chip_h and i < len(chips) else None
            if chip:
                r, g, b = (int(chip.lstrip("#")[j:j + 2], 16) for j in (0, 2, 4))
                self.set_fill_color(r, g, b); self.set_draw_color(51)       # outline #333333
                self.rect(x + w * 0.18, y0 + 1.0, w * 0.64, chip_h, style="DF")
                self.set_draw_color(0)
            n = str(h).count("\n") + 1
            self.set_xy(x, y0 + top + (hh - top - n * lh) / 2)
            self.multi_cell(w, lh, str(h), border=0, align="C")
            x += w
        self.set_xy(x0, y0 + hh)
        self.set_font("ui", "", size)

    def table(self, header, rows, widths, size=8, chips=None, zebra=True, first_left=False):
        """zebra: every second row on a very light grey (#f5f5f5, decorative) so long rows are easy to follow.
        first_left: left-align the first column (row names) even when it is narrow."""
        # every table spans the text width, as the charts and the rows of key frames do (and never runs past the margin)
        k = self.epw / sum(widths)
        widths = [w * k for w in widths]
        if self.get_y() > 255:                # header + at least two rows must fit on this page
            self.add_page()
        self._table_header(header, widths, size, chips)
        for i, r in enumerate(rows):
            if self.get_y() > 275:
                self.add_page()
                self._table_header(header, widths, size, chips)
            shade = zebra and i % 2 == 1
            if shade:
                self.set_fill_color(245)
            for j, (c, w) in enumerate(zip(r, widths)):
                align = "L" if (j == 0 and first_left) or w >= 40 else "C"
                self.cell(w, 5.2, str(c), border=1, align=align, fill=shade)
            self.ln()


def _empty_text(r: dict) -> bool:
    """The segment's own transcript is empty: the text-emotion model then answers "neutral 100%", which is no data."""
    return "text_en" in r and not str(r.get("text_en") or "").strip()


def _seg_words(r: dict) -> int:
    try:
        return int((r.get("speech") or {}).get("words") or 0)
    except (TypeError, ValueError):
        return 0


def _analyses_sections(pdf, report: dict) -> None:
    """BS 2.0: emotions (text / face), voice dimensions, speech analytics with charts and per-segment table."""
    an = report.get("analyses") or {}
    if not an:
        return
    from .charts import EMO_RU, VOICE_RU
    from .narrative2 import analyses_sentences, plural_ru
    charts = report.get("chart_files") or {}
    intro = analyses_sentences(report)
    # the heading, the intro and the table of averages start on this page when they fit above the table's own
    # page-break line (255 mm); a fixed «below 200 mm -> new page» left up to 60 mm empty at the bottom of a page
    if pdf.get_y() + 12 + pdf.para_height(intro, 9) > 255:
        pdf.add_page()
    pdf.section("Эмоции, голос и мимика", "emotions")
    pdf.para(intro, 9)
    per = an.get("per_segment") or []
    te, fa, vo = an.get("emotions_text") or {}, an.get("face") or {}, an.get("voice") or {}
    # hatched gaps on the emotions chart. A segment with an empty transcript is not always silent: its own recognition
    # may return nothing while the whole-video transcript still has words in that window (tempo and pauses come
    # from there), so "нет речи" is said only when there are no words at all
    text_gaps = [r for r in per if _empty_text(r) or not r.get("emotions_text")]
    parts = []
    if text_gaps:
        parts.append("по речи — " + ("в отрезке нет речи" if all(_seg_words(r) == 0 for r in text_gaps)
                                     else "для отрезка нет распознанного текста"))
    if any(not (r.get("face") or {}).get("expressions") for r in per):
        parts.append("по лицу — лицо не найдено")
    gap_note = f" Штриховка на графике — нет данных ({'; '.join(parts)})." if parts and charts.get("emotions") else ""
    if te.get("mean") or fa.get("mean"):
        order = ["joy", "surprise", "neutral", "sadness", "fear", "anger", "disgust"]
        face_map = {"joy": "happy", "sadness": "sad", "anger": "angry"}
        two_line = {"joy": "радость", "surprise": "удивле-\nние", "neutral": "нейтраль-\nно", "sadness": "грусть", "fear": "страх",
                    "anger": "злость", "disgust": "отвраще-\nние"}
        header = ["Эмоция"] + [two_line[k] for k in order]
        rows = []
        if te.get("mean"):
            rows.append(["по речи"] + [f"{te['mean'].get(k, 0):.0%}" for k in order])
        if fa.get("mean"):
            rows.append(["по лицу"] + [f"{fa['mean'].get(face_map.get(k, k), 0):.0%}" for k in order])
        pdf.ln(1); pdf.table(header, rows, [40] + [20] * 7, size=8, chips=[None] + [emo_pdf(k) for k in order])
        # the text-emotion model reads English: the transcript itself for English speech, its translation otherwise
        text_src = "по транскрипту" if (report.get("model") or {}).get("lang") == "en" else "по переводу транскрипта на английский"
        pdf.para("Средние доли за ролик; цветная полоска над названием — цвет этой эмоции на графике «Эмоции по ходу ролика». "
                 f"Речь — модель эмоций текста {text_src}; лицо — модель выражений по кадрам (обучена на фотографиях, завышает "
                 "«грусть» и «страх» у спокойного лица)." + gap_note, 7)
    # the averages come first and the full-width chart after them: the short table fills the page under the text
    # instead of being pushed after a chart that has to start a new page
    if charts.get("emotions"):
        pdf.ln(1)
        pdf.chart(charts["emotions"])
    # a chart without any data is not drawn; when the other one is there, one line says why this one is missing
    voice_caption = ("Голос (модель эмоций в речи, 0…1): " + ", ".join(
        f"{VOICE_RU[d]} {vo['mean'].get(d, 0):.2f} (±{vo.get('std', {}).get(d, 0):.2f})" for d in VOICE_RU) + "."
        if vo.get("mean") else "")
    if charts.get("voice"):
        pdf.ln(1)
        # the averages below are its caption: exactly their height is kept with the chart (a fixed 6 mm pushed a chart
        # that fitted onto the next page)
        pdf.chart(charts["voice"], keep_mm=pdf.para_height(voice_caption, 8) if voice_caption else 0)
    elif charts.get("speech"):
        pdf.para("График голоса не построен: данных о голосе нет.", 8)
    if voice_caption:
        pdf.para(voice_caption, 8)
    if charts.get("speech"):
        pdf.ln(1)
        pdf.chart(charts["speech"])
    elif charts.get("voice"):
        pdf.para("График речи не построен: данных о темпе и паузах нет.", 8)
    # filler words (hover text of the web speech chart) are not drawn on the PDF chart: they are the last column of the
    # per-segment table below, explained in its note
    if per:
        pdf.h3("По отрезкам", keep_mm=45)
        header = ["Отрезок", "Эмоция\nречи", "Выражение\nлица", "Возбуж-\nдение", "Уверен-\nность", "Позитив-\nность",
                  "Слов\nв минуту", "Паузы", "Заполни-\nтели"]
        rows, any_no_text = [], False
        for r in per:
            t_e = (r.get("emotions_text") or {}); f_e = ((r.get("face") or {}).get("expressions") or {}); v = r.get("voice") or {}
            sp = r.get("speech") or {}
            wpm = sp.get("words_per_min_speech")
            if _empty_text(r):              # scored "neutral" by the text model, hatched on the chart
                speech_emo = "нет речи" if _seg_words(r) == 0 else "нет текста"
                any_no_text = any_no_text or speech_emo == "нет текста"
            else:
                speech_emo = EMO_RU.get(max(t_e.items(), key=lambda kv: kv[1])[0], "—") if t_e else "—"
            rows.append([seg_label(r["start"], r["end"]),
                         speech_emo,
                         EMO_RU.get(max(f_e.items(), key=lambda kv: kv[1])[0], "—") if f_e else "—",
                         f"{v.get('arousal', 0):.2f}" if v else "—", f"{v.get('dominance', 0):.2f}" if v else "—",
                         f"{v.get('valence', 0):.2f}" if v else "—",
                         f"{wpm:.0f}" if wpm is not None else "—",
                         f"{sp.get('pause_share', 0):.0%}" if sp else "—",
                         f"{sp.get('fillers_per_100', 0):.1f}" if sp else "—"])
        pdf.table(header, rows, [23, 25, 25, 18, 18, 18, 19, 16, 18], size=8)
        pdf.para("«—» — нет данных (в столбце темпа — речи в отрезке меньше секунды)."
                 + (" «нет текста» — для отрезка не распознан текст, поэтому эмоция речи не оценена; темп и паузы берутся "
                    "из транскрипта всего ролика." if any_no_text else "")
                 + " «Слов в минуту» — темп внутри речи, без пауз. «Паузы» — доля времени отрезка без речи. «Заполнители» — "
                 "слов-заполнителей («ну», «вот», «как бы») на 100 слов.", 7)
    sp = an.get("speech") or {}
    if sp:
        if pdf.get_y() > 240:
            pdf.add_page()
        pdf.section("Речь", "speech")
        wpm = sp.get("words_per_min_speech")

        def words(n) -> str:          # «274 слова», «91 слово», «25 слов»
            n = int(round(float(n or 0)))
            return f"{n} {plural_ru(n, 'слово', 'слова', 'слов')}"
        long_p = int(sp.get("long_pauses") or 0)
        pdf.kv_table([("Слов всего / уникальных", f"{sp.get('words')} / {sp.get('unique_words')}"),
                      ("Темп", (f"{words(wpm)} в минуту речи" if wpm is not None else "не посчитан (речи меньше секунды)")
                               + f"; с учётом пауз — {sp.get('words_per_min_wall', 0):.0f} в минуту"),
                      ("Паузы", f"{sp.get('pause_share', 0):.0%} времени; "
                                + (f"длинных пауз (дольше 2 с) — {long_p}" if long_p else "длинных пауз (дольше 2 с) нет")),
                      ("Слова-заполнители", f"{sp.get('fillers', 0)} ({sp.get('fillers_per_100', 0):.1f} на 100 слов)"),
                      ("Средняя фраза", words(sp.get("mean_sentence"))), ("Разнообразие словаря", f"{sp.get('ttr') or 0:.2f}")])
        vocab = vocabulary_shown(report)          # Russian words; for English speech their translations
        if vocab:
            pdf.para("Частые слова: " + ", ".join(f"{w} ({n})" for w, n in vocab[:15]), 8)


def build_pdf(report: dict, out_path: str | Path, explanation: dict | None = None, media: dict | None = None,
              key_frames: list[str] | None = None) -> str:
    pdf = Report()
    pdf.add_page()
    pdf.h1("BS 2.0 — отчёт по видео: Big Five, эмоции, голос, речь")
    fname = report.get("original_file_name") or (media or {}).get("file_name") or Path(report.get("input", "")).name
    pdf.para(f"Создан: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}   ·   Файл: {fname}", 9)

    # ---- file metadata
    pdf.section("Файл")
    rows = []
    if media:
        ch = media.get("channels")
        rows += [("Имя файла", media.get("file_name")), ("Размер", f"{media.get('size_mb')} МБ ({media.get('size_bytes')} байт)"),
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
        rows.append(("Путь", report.get("input")))
    pdf.kv_table(rows)

    # ---- analysis settings
    pdf.section("Параметры анализа")
    m = report.get("model", {})
    members = [x for x in report.get("modalities_used", []) if x in SYSTEM_TITLES]
    if m.get("backend") == "ensemble" and members:
        system = "ансамбль: " + " + ".join(SYSTEM_TITLES[x] for x in members)
        if m.get("primary"):
            system += f"; основная оценка — {SYSTEM_TITLES.get(m['primary'], m['primary'])}"
            if m.get("scale"):
                sc = str(m["scale"])
                sc = "MuPTA для русской речи" if sc.startswith("MuPTA") else (
                    "First Impressions V2" if sc.startswith("FIV2") else sc)
                system += f", шкала {sc}"
    else:
        system = f"{SYSTEM_TITLES.get(m.get('backend'), m.get('backend'))} ({m.get('corpus')})"
    rows = [("Система", system), ("Язык речи", {"ru": "русский", "en": "английский"}.get(m.get("lang"), m.get("lang"))),
            ("Распознавание речи", _asr_ru(m.get("asr_model")) if m.get("asr_model") else "готовый транскрипт"),
            ("Обучающие данные", _trained_on(m, members)), ("Модальности", ", ".join(MODALITY_TITLES.get(x, x) for x in report.get("modalities_used", []))),
            ("Версия", _version_ru(m.get("version")))]
    if report.get("segments"):
        n_seg = int(report["segments"])
        rows.append(("Отрезки", f"{n_seg} по ~20 с; итог — среднее с весом по длительности" if n_seg > 1
                     else "один отрезок (весь ролик)"))
    t = report.get("timings_sec", {})
    if t:
        rows.append(("Время обработки", fmt_secs(t.get("total_wall", t.get("total")))))
    pdf.kv_table(rows)

    # ---- scores
    try:
        from .narrative import build_narrative
        narrative = build_narrative(report, explanation)
    except Exception:  # noqa: BLE001
        narrative = ""
    if narrative:
        pdf.h2("Пояснение простыми словами")
        pdf.para(narrative, 9)

    pdf.section("Оценки")
    pdf.score_bars(report["traits"], report.get("interview"))
    std = report.get("scores_std_across_segments") or {}
    if std:
        pdf.para("Разброс между отрезками: " + ", ".join(f"{RU_SHORT[k].lower()} ±{std.get(k, 0):.2f}" for k in TRAIT_KEYS), 8)
    var = report.get("variant_scores") or {}
    if var:
        pdf.h3("Оценки участников ансамбля", keep_mm=27)      # the table itself moves to a new page below y=255
        header = ["Участник"] + [TITLES_2L[k] for k in TRAIT_KEYS]
        primary = m.get("primary")
        rows = [[MEMBERS.get(n, n) + (" (основная)" if n == primary else "")] + [f"{v.get(k, float('nan')):.3f}" for k in TRAIT_KEYS]
                for n, v in var.items()]
        pdf.table(header, rows, [50, 26, 26, 26, 26, 26])
        if primary:
            pdf.para("Основная оценка — участник, помеченный «(основная)»; остальные — второе мнение на другой шкале "
                     "(своя модель обучена на английских влогерах FIV2, OCEAN-AI MuPTA — на русскоязычных испытуемых).", 7)

    # ---- timeline
    tl = report.get("timeline") or []
    if tl:
        pdf.section("Таймлайн по отрезкам", "timeline")
        keys = TRAIT_KEYS + (["interview"] if any(s.get("scores") and "interview" in s["scores"] for s in tl) else [])
        header = ["Отрезок"] + [TITLES_2L.get(k, TITLES.get(k, k)) for k in keys]
        rows = []
        for s in tl:
            if s.get("scores"):
                rows.append([seg_label(s["start"], s["end"]) + (" ★" if s["segment"] == report.get("representative_segment") else "")]
                            + [f"{s['scores'][k]:.2f}" for k in keys])
            else:
                rows.append([seg_label(s["start"], s["end"]), "пропущен"] + [""] * (len(keys) - 1))
        pdf.table(header, rows, [30] + [150 / len(keys)] * len(keys), size=8)
        pdf.para("★ — отрезок, ближайший к среднему профилю; по нему построены объяснения (на графике ниже он выделен рамкой).", 7)
    charts = report.get("chart_files") or {}
    if charts.get("traits"):
        pdf.ln(2)
        pdf.chart(charts["traits"])
    _analyses_sections(pdf, report)

    # ---- key frames
    frames = [p for p in (key_frames or []) if os.path.exists(p)]
    if frames:
        from PIL import Image
        if pdf.get_y() > pdf.h - pdf.b_margin - 60:        # keep the heading together with the first row of frames
            pdf.add_page()
        pdf.section("Ключевые кадры")
        # frames come from the clip the explanations were computed on: the representative segment of a long video,
        # otherwise the whole video; file names carry the frame index inside that clip (key_<i>_frame<N>.jpg)
        tl_all = report.get("timeline") or []
        seg = next((s for s in tl_all if s.get("segment") == report.get("representative_segment")), None) if tl_all else None
        fps = float((media or {}).get("fps") or (report.get("media") or {}).get("fps") or 0)
        seg_start = float(seg["start"]) if seg else 0.0

        timed = []                      # captions that carry the moment of the video (the note below names it only then)

        def moment(path: str):
            m = re.search(r"_frame(\d+)", Path(path).stem)
            return seg_start + int(m.group(1)) / fps if (m and fps > 0 and (seg or not tl_all)) else None

        # frames a fraction of a second apart would share «0:37»: then every caption shows tenths («0:37,2»)
        secs = [int(t) for t in map(moment, frames) if t is not None]
        tenths = len(set(secs)) < len(secs)

        def caption(n: int, path: str) -> str:
            t = moment(path)
            if t is None:
                return f"кадр {n}"
            timed.append(n)
            if tenths:
                d = int(t * 10)
                return f"кадр {n} · {d // 600}:{d // 10 % 60:02d},{d % 10}"
            return f"кадр {n} · {int(t) // 60}:{int(t) % 60:02d}"

        max_h, gap, per_row = 62.0, 4.0, 3
        cell_w = (pdf.w - pdf.l_margin - pdf.r_margin - gap * (per_row - 1)) / per_row
        row, y0, n_done = [], pdf.get_y(), 0
        for p in frames + [None]:
            if p is not None:
                row.append(p)
                if len(row) < per_row:
                    continue
            if not row:
                break
            # size every image of the row to fit cell_w × max_h keeping its aspect ratio
            sizes = []
            for q in row:
                with Image.open(q) as im:
                    iw, ih = im.size
                scale = min(cell_w / iw, max_h / ih)
                sizes.append((iw * scale, ih * scale))
            row_h = max(h for _, h in sizes)
            if y0 + row_h + 8 > pdf.h - pdf.b_margin:
                pdf.add_page(); y0 = pdf.get_y()
            for i, (q, (iw, ih)) in enumerate(zip(row, sizes)):
                x = pdf.l_margin + i * (cell_w + gap) + (cell_w - iw) / 2
                try:
                    pdf.image(q, x=x, y=y0 + (row_h - ih), w=iw, h=ih)
                except Exception:  # noqa: BLE001
                    continue
                pdf.set_xy(pdf.l_margin + i * (cell_w + gap), y0 + row_h + 1)
                pdf.set_font("ui", "", 8); pdf.cell(cell_w, 4, caption(n_done + i + 1, q), align="C")
            n_done += len(row)
            y0 += row_h + 8
            pdf.set_y(y0)
            row = []
        tl_no = pdf.section_no.get("timeline")
        where = (f" (отрезок {seg_label(seg['start'], seg['end'])}" + (f", ★ в таблице раздела {tl_no})" if tl_no else ")")
                 if seg else "")
        pdf.para(f"Кадры, сильнее всего повлиявшие на оценку своей модели{where}. Рамкой отмечено найденное лицо; "
                 + (("под кадром — его номер и момент ролика (мин:с, после запятой — десятые доли секунды)." if tenths
                     else "под кадром — его номер и момент ролика (мин:с).") if timed else "под кадром — его номер."), 8)

    # ---- explanations
    if explanation:
        pdf.section("Вклад модальностей в оценку своей модели")
        ixg = explanation["modalities"]["input_x_gradient"]
        mods = list(next(iter(ixg.values())).keys())
        header = ["Черта"] + [MEMBERS.get(x, x) for x in mods]
        def pct(s):
            v = float(s) * 100
            return "<1%" if v < 0.95 else f"{v:.0f}%"
        rows = [[TITLES.get(k, k)] + [pct(row[x]["share"]) for x in mods] for k, row in ixg.items()]
        pdf.table(header, rows, [60] + [int(120 / len(mods))] * len(mods))
        pdf.para("Доля вклада каждой модальности в оценку своей модели (по градиенту оценки: насколько признаки модальности "
                 "сдвигают результат). «<1%» — модальность почти не влияет на "
                 "оценку этого ролика: модель, обученная на FIV2, опирается в основном на лицо и голос.", 7)
        loo = explanation["modalities"].get("leave_one_out_delta") or {}
        if loo:
            pdf.ln(2)
            pdf.h3("Как изменятся оценки, если убрать одну модальность", keep_mm=36)
            keys_l = [k for k in list(TRAIT_KEYS) + ["interview"] if any(k in v for v in loo.values())]
            header = ["Без модальности"] + [TITLES_2L.get(k, TITLES.get(k, k)) for k in keys_l]
            def delta(v):         # '-0.00' / '+0.00' would suggest a direction where there is none
                v = float(v or 0)
                return "0.00" if abs(v) < 0.005 else f"{v:+.2f}"
            rows = [[MEMBERS.get(x, x).replace("\n", " ")] + [delta(v.get(k, 0)) for k in keys_l] for x, v in loo.items()]
            # at 8 pt «Добросовест-» / «стабильность» need 24.2 mm and «описание поведения» 33 mm
            pdf.table(header, rows, [34] + [146 / len(keys_l)] * len(keys_l), size=8, first_left=True)
            pdf.para("Положительное число — без этой модальности оценка была бы выше, отрицательное — ниже.", 7)
        rw_all = explanation.get("readable_words") or {}
        lang = (report.get("model") or {}).get("lang", "en")
        if rw_all:
            pdf.ln(1)
            pdf.h3("Слова, на которые откликнулась модель", keep_mm=14)
            try:
                from .narrative import words_summary
                paras = words_summary(rw_all, explanation, TITLES, lang)
            except Exception as e:  # noqa: BLE001  (never let the words block break the whole PDF)
                paras = [f"Список слов недоступен: {str(e)[:120]}"]
            for para in paras:
                pdf.para(para, 8)
        # lists without Russian words (translation failed) are not printed: the raw English tokens stay in the JSON

    # ---- texts: Russian only (export_pdf fills the translations of older jobs before building the PDF)
    if report.get("behavior_description_ru"):
        pdf.section("Описание поведения")
        pdf.para("Описание строит видеоязыковая модель по кадрам каждого отрезка.", 7)
        pdf.para(mmss_labels(report["behavior_description_ru"]), 9)
    note, transcript = transcript_shown(report)
    if note or transcript:
        pdf.section("Транскрипт речи")
        if note:
            pdf.para(note, 7)
        if transcript:
            pdf.para(transcript, 9)

    pdf.h2("Ограничения")
    pdf.para(DISCLAIMER_RU, 8)
    pdf.para(INTERVIEW_DISCLAIMER_RU, 8)
    out_path = Path(out_path)
    pdf.output(str(out_path))
    return str(out_path)
