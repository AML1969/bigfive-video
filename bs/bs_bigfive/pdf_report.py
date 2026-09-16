"""PDF report for one analysed video (fpdf2): file metadata, analysis settings, scores with percentiles,
per-segment timeline, key frames as images, modality contributions and words, behaviour description, transcript.

The report is meant for white paper and office printers: body text is black, secondary text is grey 85 (7.5:1 on
white), nothing smaller than 8 pt, and every colour cue (blue traits vs brown interview) is repeated in words.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path

from fpdf import FPDF

from .norms import TRAIT_KEYS
from .report import DISCLAIMER_RU, INTERVIEW_DISCLAIMER_RU, clean_word, fmt_secs, seg_label

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
CORPUS_TITLES = {"fi": "веса First Impressions V2", "mupta": "веса MuPTA"}
TAG_TITLES = {"creation_time": "Дата съёмки", "encoder": "Кодировщик", "title": "Название",
              "com.apple.quicktime.make": "Производитель камеры", "com.apple.quicktime.model": "Модель камеры"}
# first line of a segment error (longvideo.py) -> reason in Russian; unknown errors get no stated reason
SKIP_REASONS = {"no frames decoded": "не удалось прочитать кадры", "empty audio": "в отрезке нет звука",
                "all ensemble members failed": "ни одна из моделей не дала оценки"}

# greys and colours for white paper (contrast on #ffffff in comments)
INK = 0                          # body text, table borders: 21:1
INK_SOFT = 85                    # legends, scale labels, footer: 7.5:1
BAR_TRACK = 242                  # 0…1 track under the fill
BAR_EDGE = 118                   # outline of the whole track (where 1.0 is): 4.5:1
BAR_TICK = 90                    # midpoint (0.5) stubs above and below the bar: 6.9:1
TRAIT_FILL = (29, 78, 216)       # #1d4ed8: 6.7:1 on white, 6.0:1 on the track (fallback)
INTERVIEW_FILL = (183, 121, 31)  # #b7791f: same colour as the interview bar and line on the web page
# per-trait bar colours: the same hues as the web score bars and chart lines, darkened for white paper
# (each >= 3.2:1 on the #f2f2f2 track and >= 3.6:1 on white)
TRAIT_FILLS = {"openness": (29, 78, 216), "conscientiousness": (4, 120, 87), "extraversion": (194, 65, 12),
               "agreeableness": (126, 34, 206), "emotional_stability": (190, 18, 60), "interview": INTERVIEW_FILL}
SKIPPED_FILL = 242               # merged "segment skipped" cell (grey 85 text on it: 6.7:1)
OUTLIER_FILL = 225               # row that differs from the rest of the video (black on it: 15.5:1)


def _pool_group(ref: str) -> str:
    """'пула обработанных русских роликов (N=9)' -> 'русских роликов'."""
    m = re.search(r"обработанных\s+(.+?)\s*(\(N=\d+\))?\s*$", ref or "")
    return m.group(1) if m else "роликов пула"


def _pool_n(ref: str) -> int | None:
    m = re.search(r"\(N=(\d+)\)", ref or "")
    return int(m.group(1)) if m else None


def pct_phrase(pct, ref: str = "") -> str:
    """'выше, чем у 83% русских роликов' / 'ниже, чем у 95% клипов First Impressions V2' / 'мало роликов для сравнения'."""
    if pct is None:
        return "мало роликов для сравнения"
    group = _pool_group(ref) if "пула" in (ref or "") else "клипов First Impressions V2"
    pct = float(pct)
    return f"выше, чем у {pct:.0f}% {group}" if pct >= 50 else f"ниже, чем у {100 - pct:.0f}% {group}"


def signed(d) -> str:
    """Signed change with a real minus sign; values that round to zero carry no sign."""
    d = float(d)
    return "0.00" if abs(d) < 0.005 else f"{d:+.2f}".replace("-", "\u2212")


def _fmt_dt(value) -> str:
    """ISO timestamps as '16.09.2026 12:36' (with 'UTC' when the value is in UTC); anything else unchanged."""
    s = str(value or "")
    try:
        d = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s
    out = d.strftime("%d.%m.%Y %H:%M")
    if d.tzinfo is not None:
        out += " UTC" if d.utcoffset() == _dt.timedelta(0) else d.strftime(" %z")
    return out


def _keep_dash(text: str) -> str:
    """A dash never starts a line: the space before ' — ' becomes non-breaking (fpdf prints it as a normal space)."""
    return str(text).replace(" — ", " — ")


def _thousands(n) -> str:
    try:
        return f"{int(n):,}".replace(",", "\u00a0")
    except (TypeError, ValueError):
        return str(n)


def _mmss(sec: float) -> str:
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def score_source(model: dict) -> str:
    """Who produced the five trait scores, for legends: 'основная оценка OCEAN-AI'."""
    primary, backend = model.get("primary"), model.get("backend")
    if primary:
        return "основная оценка " + {"oceanai": "OCEAN-AI", "mm": "своей модели (MM-PSYCHE)"}.get(primary, primary)
    if backend == "ensemble":
        return "среднее OCEAN-AI и своей модели"
    return {"oceanai": "оценка OCEAN-AI", "mm": "оценка своей модели (MM-PSYCHE)"}.get(backend, f"оценка {backend}")


def _scale_title(scale: str) -> str:
    s = str(scale or "")
    if s.startswith("MuPTA"):
        return "шкала MuPTA (русская речь)"
    if s.startswith("FIV2") or "First Impressions" in s:
        return "шкала First Impressions V2"
    return f"шкала {s}"


FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/mnt/c/Windows/Fonts/arial.ttf", "/mnt/c/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
]
# fonts with ★ for when the text font lacks it (Arial has no U+2605)
SYMBOL_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/mnt/c/Windows/Fonts/seguisym.ttf",
                "C:/Windows/Fonts/seguisym.ttf"]


class Report(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        for reg, bold in FONT_CANDIDATES:
            if os.path.exists(reg):
                self.add_font("ui", "", reg)
                self.add_font("ui", "B", bold if os.path.exists(bold) else reg)
                break
        else:
            raise RuntimeError("no Unicode TTF font found for the PDF (DejaVu or Arial)")
        self.file_label = ""
        # marker of the representative segment: ★ from the text font or a symbol fallback font, else a bullet
        self.star = "★"
        if not self._has_glyph("ui", self.star):
            for p in SYMBOL_FONTS:
                if os.path.exists(p):
                    self.add_font("sym", "", p)
                    if self._has_glyph("sym", self.star):
                        self.set_fallback_fonts(["sym"], exact_match=False)
                        break
            else:
                self.star = "•"
        self.set_font("ui", "", 10)

    def _has_glyph(self, fontkey: str, ch: str) -> bool:
        try:
            return ord(ch) in self.fonts[fontkey].ttfont.getBestCmap()
        except Exception:  # noqa: BLE001
            return False

    # ---- page furniture
    def footer(self):
        self.set_y(-10)
        self.set_font("ui", "", 8)
        self.set_text_color(INK_SOFT)
        label = self.file_label if len(self.file_label) <= 50 else self.file_label[:49] + "…"
        text = "BS Big Five" + (f" · {label}" if label else "") + f" · стр. {self.page_no()} из "
        # '{nb}' becomes the page count only when the file is written, so right-align by the width of a number
        # (as many digits as the current page) instead of the width of the placeholder
        width = self.get_string_width(text) + self.get_string_width("0" * len(str(self.page_no())))
        self.set_x(max(self.l_margin, self.w - self.r_margin - width - self.c_margin))
        self.cell(0, 5, text + "{nb}")
        self.set_text_color(INK)

    def keep(self, h: float):
        """Start a new page unless `h` mm still fit on this one (keeps headings with what follows). At the top of a
        page nothing is gained by breaking, so no empty page is ever added."""
        if self.get_y() + h > self.page_break_trigger and self.get_y() > self.t_margin + 1:
            self.add_page()

    def h1(self, text):
        self.set_font("ui", "B", 16); self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT"); self.ln(1)

    def h2(self, text, need: float = 20):
        self.keep(10 + need)
        if self.get_y() > self.t_margin + 1:
            self.ln(2)
        self.set_font("ui", "B", 12); self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.set_font("ui", "", 10)

    def h3(self, text, need: float = 20):
        self.keep(8 + need)
        if self.get_y() > self.t_margin + 1:
            self.ln(2)
        self.set_font("ui", "B", 10); self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.set_font("ui", "", 10)

    def para(self, text, size=10, h=None, align="L", after=1.0):
        """Left-aligned paragraph, line height follows the size (4 mm at 8 pt, 5 mm at 10 pt). The text is never
        altered: words longer than the line (hashes, file names) are broken by fpdf itself."""
        self.set_x(self.l_margin)
        self.set_font("ui", "", size)
        self.multi_cell(0, h or round(size * 0.5, 1), _keep_dash(text), align=align, new_x="LMARGIN", new_y="NEXT")
        if after:
            self.ln(after)

    def note(self, text, size=8):
        """Explanatory note under a table or chart."""
        self.para(text, size, after=1.5)

    def labelled_line(self, title, text, size=8.5, sep=": "):
        """'Title: text' with a bold title, wrapped at the right margin."""
        self.set_x(self.l_margin)
        h = round(size * 0.5, 1)
        self.set_font("ui", "B", size); self.write(h, f"{title}{sep}")
        self.set_font("ui", "", size); self.write(h, _keep_dash(text))
        self.ln(h + 0.8)

    def text_blocks(self, text, size=9):
        """Multi-paragraph text; a leading '[0–20 с]' segment label is printed bold so segments are easy to find.
        Labels in seconds ('[100–120 с]') are shown as in the timeline table ('[1:40–2:00]')."""
        for line in str(text).split("\n"):
            line = line.strip()
            if not line:
                continue
            mt = re.match(r"^(\[[^\]]{1,24}\])\s*(.*)$", line, re.S)
            if mt:
                label = mt.group(1)
                sec = re.match(r"^\[(\d+(?:[.,]\d+)?)\s*[–-]\s*(\d+(?:[.,]\d+)?)\s*(?:с|s)\]$", label)
                if sec:
                    label = "[" + seg_label(float(sec.group(1).replace(",", ".")),
                                            float(sec.group(2).replace(",", "."))) + "]"
                self.labelled_line(label, mt.group(2), size, sep=" ")
            else:
                self.para(line, size, after=0.8)

    def kv_table(self, rows, w1=55):
        lh = 4.6
        for k, v in rows:
            self.set_x(self.l_margin)
            self.set_font("ui", "B", 9); self.cell(w1, lh, str(k))
            self.set_font("ui", "", 9)
            v = "—" if v is None or v == "" else str(v)
            self.multi_cell(0, lh, v, new_x="LMARGIN", new_y="NEXT", align="L",
                            wrapmode="WORD" if " " in v else "CHAR")
            self.ln(0.9)

    def fit_cell(self, w, h, text, size, style="", pads=2, **kw):
        """cell() whose text never crosses the border: shrink by 0.5 pt down to 6.5 pt, then cut with '…'.
        `pads`: how many inner cell margins must stay free (1 for a last cell that ends at the page margin)."""
        text = str(text)
        s = size
        self.set_font("ui", style, s)
        while self.get_string_width(text) + pads * self.c_margin > w + 0.01 and s > 6.5:
            s -= 0.5
            self.set_font("ui", style, s)
        if self.get_string_width(text) + pads * self.c_margin > w + 0.01:
            while len(text) > 1 and self.get_string_width(text + "…") + pads * self.c_margin > w:
                text = text[:-1]
            text = text.rstrip() + "…"
        self.cell(w, h, text, **kw)
        self.set_font("ui", style, size)

    def _score_legend(self, traits: dict, interview: dict | None, std: dict, model: dict) -> list[str]:
        """Legend lines built from the data: which rows, which model, which reference group."""
        lines = ["Полоска — оценка от 0 до 1, риска — середина шкалы (0.5)"
                 + ("; ± — разброс оценки между отрезками ролика." if std else ".")]
        tref = traits[TRAIT_KEYS[0]].get("percentile_ref", "")
        if "пула" in tref:
            n = _pool_n(tref)
            where = f"среди обработанных {_pool_group(tref)}" + (f" (N={n})" if n else "")
            if n is not None and n < 30:
                where += "; группа пока мала, поэтому проценты приблизительные"
        else:
            where = "среди 6000 клипов обучающей выборки First Impressions V2"
        lines.append(f"Пять черт (синие полоски) — {score_source(model)}; процент — положение {where}.")
        if interview:
            iref = interview.get("percentile_ref", "")
            iwhere = (f"среди обработанных {_pool_group(iref)}" if "пула" in iref
                      else "среди 6000 клипов обучающей выборки First Impressions V2")
            lines.append(f"«Собеседование» (коричневая полоска, отдельно снизу) — своя модель, шкала First Impressions V2; "
                         f"процент — положение {iwhere}.")
        return lines

    SCORE_ROW_H = 6.0

    def score_bars_height(self, traits: dict, interview: dict | None, std: dict | None = None,
                          model: dict | None = None) -> float:
        """Height of the whole score block (rows, scale, legend), to keep it on one page with its heading."""
        self.set_font("ui", "", 8)
        legend = sum(self.multi_cell(0, 4, _keep_dash(line), dry_run=True, output="HEIGHT")
                     for line in self._score_legend(traits, interview, std or {}, model or {}))
        rows = (len(TRAIT_KEYS) + (1 if interview else 0)) * self.SCORE_ROW_H + (1.5 if interview else 0)
        return rows + 4.5 + legend + 1

    def score_bars(self, traits: dict, interview: dict | None, std: dict | None = None, model: dict | None = None):
        """One row per trait: the bar is the score itself (0…1), then the score ± spread across segments and the
        position relative to the reference population in words. A legend under the bars says which rows use which
        model and which reference group."""
        std = std or {}
        items = [(k, traits[k]) for k in TRAIT_KEYS] + ([("interview", interview)] if interview else [])
        # 'Впечатление «собеседование»' at 9 pt needs 54.2 mm (+ a visible gap before the bar); the longest FIV2
        # phrase needs ~75 mm
        label_w, bar_w = 56, 36.5
        row_h, bar_h = self.SCORE_ROW_H, 3.6
        x_bar = self.l_margin + label_w
        nums = {k: f"{float(t['score']):.2f}" + (f" ±{float(std[k]):.2f}" if k in std else "") for k, t in items}
        # the number column is as wide as its longest value, so the phrase follows the number without a gap
        self.set_font("ui", "", 9)
        num_w = min(20.0, max(self.get_string_width(s) for s in nums.values()) + 2 * self.c_margin + 1.5)
        self.keep(self.score_bars_height(traits, interview, std, model))
        for k, t in items:
            pct = t.get("percentile", t.get("percentile_vs_fiv2")); score = float(t["score"])
            ref = t.get("percentile_ref", "train First Impressions V2 (6000 клипов)")
            if k == "interview":
                self.ln(1.5)                  # the interview row comes from another model: keep it apart
            y_row = self.get_y()
            self.set_x(self.l_margin)
            self.fit_cell(label_w, row_h, TITLES[k], 9)
            y = y_row + (row_h - bar_h) / 2
            self.set_fill_color(BAR_TRACK); self.rect(x_bar, y, bar_w, bar_h, style="F")
            self.set_fill_color(*TRAIT_FILLS.get(k, TRAIT_FILL))
            fill_w = bar_w * max(0.01, min(1.0, score))
            self.rect(x_bar, y, fill_w, bar_h, style="F")
            self.set_line_width(0.3)
            self.set_draw_color(BAR_EDGE); self.rect(x_bar, y, bar_w, bar_h, style="D")
            xm = x_bar + bar_w / 2
            self.set_draw_color(BAR_TICK); self.line(xm, y - 0.8, xm, y + bar_h + 0.8)
            # the grey stub vanishes inside the dark fill: a white segment there, but only when the fill clearly
            # passes the middle (a fill ending right at 0.5 marks the middle itself; a white line would split it)
            if fill_w > bar_w / 2 + 0.8:
                self.set_draw_color(255); self.line(xm, y + 0.45, xm, y + bar_h - 0.45)
            self.set_draw_color(INK); self.set_line_width(0.2)
            self.set_xy(x_bar + bar_w + 2, y_row)
            self.fit_cell(num_w, row_h, nums[k], 9)
            self.fit_cell(self.w - self.r_margin - self.get_x(), row_h, pct_phrase(pct, ref), 9, pads=1)
            self.set_xy(self.l_margin, y_row + row_h)
        # scale under the last bar
        self.set_font("ui", "", 8); self.set_text_color(INK_SOFT)
        y = self.get_y() + 0.3
        for label, x, align in (("0", x_bar - 1, "L"), ("0.5", x_bar + bar_w / 2 - 6, "C"), ("1", x_bar + bar_w - 11, "R")):
            self.set_xy(x, y); self.cell(12, 3.6, label, align=align)
        self.set_xy(self.l_margin, y + 4.2)
        self.set_text_color(INK_SOFT)
        for line in self._score_legend(traits, interview, std, model or {}):
            self.para(line, 8, h=4, after=0)
        self.set_text_color(INK)
        self.ln(1)

    def _table_header(self, header, widths, size):
        """Header cells may contain '\\n' (two-line titles); all cells get the same height."""
        self.set_font("ui", "B", size)
        lines = max(str(h).count("\n") + 1 for h in header)
        lh = size * 0.5                      # line height in mm for this font size
        hh = lines * lh + 1.5
        x0, y0 = self.l_margin, self.get_y()
        x = x0
        for h, w in zip(header, widths):
            self.rect(x, y0, w, hh)
            n = str(h).count("\n") + 1
            self.set_xy(x, y0 + (hh - n * lh) / 2)
            self.multi_cell(w, lh, str(h), border=0, align="C")
            x += w
        self.set_xy(x0, y0 + hh)
        self.set_font("ui", "", size)

    def table(self, header, rows, widths, size=8, row_h=5.2):
        """rows: lists of cells, or dicts {"cells": [...], "bold": bool, "fill": grey or None, "span": text}; with
        "span" only the first cell is used and the text fills one merged grey cell over the remaining columns.
        The first column is left-aligned, the others centred."""
        avail = self.w - self.l_margin - self.r_margin
        if sum(widths) > avail + 0.1:        # never run past the right margin
            k = avail / sum(widths)
            widths = [w * k for w in widths]
        lines = max(str(h).count("\n") + 1 for h in header)
        self.keep(lines * size * 0.5 + 1.5 + 2 * row_h)     # header + at least two rows on this page
        self._table_header(header, widths, size)
        for r in rows:
            r = r if isinstance(r, dict) else {"cells": r}
            if self.get_y() + row_h > self.page_break_trigger:
                self.add_page()
                self._table_header(header, widths, size)
            style = "B" if r.get("bold") else ""
            fill = r.get("fill")
            if fill is not None:
                self.set_fill_color(fill)
            cells = list(r["cells"])
            self.set_x(self.l_margin)
            if r.get("span") is not None:
                self.fit_cell(widths[0], row_h, cells[0], size, style, border=1, align="L", fill=fill is not None)
                self.set_fill_color(SKIPPED_FILL); self.set_text_color(INK_SOFT)
                self.fit_cell(sum(widths[1:]), row_h, r["span"], size, "", border=1, align="C", fill=True)
                self.set_text_color(INK)
            else:
                for i, (c, w) in enumerate(zip(cells, widths)):
                    self.fit_cell(w, row_h, c, size, style, border=1, align="L" if i == 0 else "C",
                                  fill=fill is not None)
            self.ln(row_h)
        self.set_font("ui", "", size)
        self.ln(1)


def build_pdf(report: dict, out_path: str | Path, explanation: dict | None = None, media: dict | None = None,
              key_frames: list[str] | None = None) -> str:
    pdf = Report()
    pdf.add_page()
    num = [0]

    def section(title):
        num[0] += 1
        return f"{num[0]}. {title}"

    pdf.h1("Отчёт: Big Five по видео")
    fname = report.get("original_file_name") or (media or {}).get("file_name") or Path(report.get("input", "")).name
    pdf.file_label = str(fname or "")
    pdf.para(f"Создан: {_dt.datetime.now().strftime('%d.%m.%Y %H:%M')}   ·   Файл: {fname}", 9)
    m = report.get("model", {}) or {}
    primary = m.get("primary")

    # ---- file metadata
    pdf.h2(section("Файл"))
    rows = []
    if media and "error" not in media:
        video = ", ".join(str(x) for x in (media.get("video_codec"),
                                           f"{media.get('width')}×{media.get('height')}" if media.get("width") else None,
                                           f"{media.get('fps')} кадр/с" if media.get("fps") else None) if x)
        if media.get("rotation"):
            video += f", поворот {media.get('rotation')}°"
        rows += [("Имя файла", media.get("file_name")),
                 ("Размер", f"{media.get('size_mb')} МБ ({_thousands(media.get('size_bytes'))} байт)"),
                 ("Длительность", f"{fmt_secs(media.get('duration_sec'))} ({media.get('duration_sec')} с)"),
                 ("Контейнер", media.get("container")),
                 ("Видео", video),
                 ("Аудио", f"{media.get('audio_codec')}, {_thousands(media.get('sample_rate'))} Гц, каналов: {media.get('channels')}"
                  if media.get("audio_codec") else "нет звуковой дорожки"),
                 ("Битрейт", f"{_thousands(media.get('bitrate_kbps'))} кбит/с"),
                 ("Изменён", _fmt_dt(media.get("modified")))]
        for k, v in media.items():
            if k.startswith("tag_"):
                tag = k[4:]
                rows.append((TAG_TITLES.get(tag, tag), _fmt_dt(v) if tag == "creation_time" else v))
        if media.get("sha256"):
            rows.append(("SHA-256", media["sha256"]))
    else:
        rows.append(("Путь", report.get("input")))
    pdf.kv_table(rows)

    # ---- analysis settings
    pdf.h2(section("Параметры анализа"))
    used = report.get("modalities_used", []) or []
    members = [x for x in used if x in SYSTEM_TITLES]
    if m.get("backend") == "ensemble" and members:
        system = "ансамбль: " + " + ".join(SYSTEM_TITLES[x] for x in members)
        if primary:
            system += f"; основная оценка — {SYSTEM_TITLES.get(primary, primary)}"
            if m.get("scale"):
                system += f", {_scale_title(m['scale'])}"
        else:
            system += "; оценка — среднее участников"
    else:
        corpus = CORPUS_TITLES.get(m.get("corpus"), m.get("corpus"))
        system = f"{SYSTEM_TITLES.get(m.get('backend'), m.get('backend'))}" + (f", {corpus}" if corpus else "")
    trained_on = m.get("trained_on")
    if m.get("backend") == "ensemble" and members:
        # model.trained_on names one corpus for the whole ensemble; say which member learned from what
        oc = {"ru": "MuPTA (русская речь)", "en": "First Impressions V2"}.get(m.get("lang"))
        parts = [f"OCEAN-AI — {oc}" if x == "oceanai" and oc else "своя модель — First Impressions V2" if x == "mm" else None
                 for x in members]
        if all(parts):
            trained_on = "; ".join(parts)
    rows = [("Система", system), ("Язык речи", {"ru": "русский", "en": "английский"}.get(m.get("lang"), m.get("lang"))),
            ("Распознавание речи", m.get("asr_model") or "готовый транскрипт"),
            ("Обучающие данные", trained_on),
            ("Участники" if used and len(members) == len(used) else "Модальности",
             ", ".join(MODALITY_TITLES.get(x, x) for x in used)),
            ("Версия", m.get("version"))]
    if report.get("segments"):
        rows.append(("Отрезки", f"{report['segments']} по ~20 с; итог — среднее с весом по длительности"))
    t = report.get("timings_sec", {})
    if t:
        rows.append(("Время обработки", fmt_secs(t.get("total_wall", t.get("total")))))
    pdf.kv_table(rows)

    # ---- plain-language summary
    try:
        from .narrative import build_narrative
        narrative = build_narrative(report, explanation)
    except Exception:  # noqa: BLE001
        narrative = ""
    if narrative:
        pdf.h2("Пояснение простыми словами")
        pdf.para(narrative, 9)

    # ---- scores
    std = report.get("scores_std_across_segments") or {}
    pdf.h2(section("Оценки"), need=pdf.score_bars_height(report["traits"], report.get("interview"), std, m))
    pdf.score_bars(report["traits"], report.get("interview"), std, m)
    var = report.get("variant_scores") or {}
    if var:
        pdf.h3("Оценки участников ансамбля", need=25)
        header = ["Участник"] + [TITLES_2L[k] for k in TRAIT_KEYS]

        def val(v, k):
            x = v.get(k)
            return f"{float(x):.2f}" if isinstance(x, (int, float)) and x == x else "—"
        rows = [{"cells": [MEMBERS.get(n, n) + (" (основная)" if n == primary else "")] + [val(v, k) for k in TRAIT_KEYS],
                 "bold": n == primary} for n, v in var.items()]
        # bold 8 pt 'Своя модель (MM-PSYCHE) (основная)' needs 64.2 mm, bold 'Добросовест-' 24.2 mm (DejaVu Sans)
        pdf.table(header, rows, [66] + [24.8] * 5)
        if primary:
            pdf.note("Основная оценка — оценка участника с пометкой «основная» (строка выделена жирным); остальные — второе мнение"
                     + (" на другой шкале: своя модель обучена на англоязычных влогерах First Impressions V2, OCEAN-AI "
                        "(веса MuPTA) — на русскоязычных испытуемых." if m.get("lang") == "ru" else "."))

    # ---- timeline
    tl = report.get("timeline") or []
    rep_seg = next((s for s in tl if s.get("segment") == report.get("representative_segment")), None)
    if tl:
        pdf.h2(section("Оценки по ходу ролика"), need=25)
        keys = TRAIT_KEYS + (["interview"] if any(s.get("scores") and "interview" in s["scores"] for s in tl) else [])
        header = ["Отрезок"] + [TITLES_2L.get(k, TITLES.get(k, k)) for k in keys]
        # segments whose mean over the five traits is > 2 SD from the rest (as in the narrative text)
        scored = [s for s in tl if s.get("scores") and all(k in s["scores"] for k in TRAIT_KEYS)]
        odd = set()
        if len(scored) >= 4:
            means = [sum(s["scores"][k] for k in TRAIT_KEYS) / len(TRAIT_KEYS) for s in scored]
            mu = sum(means) / len(means)
            sd = (sum((x - mu) ** 2 for x in means) / len(means)) ** 0.5
            if sd > 0:
                odd = {s["segment"] for s, x in zip(scored, means) if abs(x - mu) / sd > 2.0}
        other_scale = False
        rows, skipped = [], False
        for s in tl:
            label = seg_label(s["start"], s["end"])
            if s.get("scores"):
                is_rep = rep_seg is not None and s["segment"] == rep_seg["segment"]
                missing_primary = bool(primary and s.get("members_used") and primary not in s["members_used"])
                other_scale = other_scale or missing_primary
                label += (f" {pdf.star}" if is_rep else "") + (" *" if missing_primary else "")
                rows.append({"cells": [label] + [f"{s['scores'][k]:.2f}" if k in s["scores"] else "—" for k in keys],
                             "bold": is_rep, "fill": OUTLIER_FILL if s["segment"] in odd else None})
            else:
                skipped = True
                err = str(s.get("error") or "")
                reason = next((ru for en, ru in SKIP_REASONS.items() if err.startswith(en)), "не удалось оценить")
                rows.append({"cells": [label], "span": f"отрезок пропущен — {reason}"})
        pdf.table(header, rows, [40] + [150 / len(keys)] * len(keys), size=8)
        notes = [f"Черты — {score_source(m)}"
                 + ("; «Собеседование» — своя модель, шкала First Impressions V2" if "interview" in keys else "")
                 + ". Все значения — от 0 до 1."]
        if rep_seg is not None:
            built = [x for x, ok in (("ключевые кадры", any(os.path.exists(p) for p in key_frames or [])),
                                     ("разбор вклада модальностей", bool(explanation))) if ok]
            notes.append(f"{pdf.star} (жирная строка) — отрезок, ближайший к среднему профилю ролика"
                         + (f": по нему {'построены' if 'ключевые кадры' in built else 'построен'} {' и '.join(built)}."
                            if built else "."))
        if other_scale:
            notes.append(f"* — отрезок оценён без {SYSTEM_TITLES.get(primary, primary)} (основная система не дала оценки), "
                         "поэтому его значения даны по другой шкале и не сравнимы с остальными строками.")
        if len(odd) == 1:
            notes.append("Серым фоном выделен отрезок, оценки которого заметно отличаются от остального ролика.")
        elif odd:
            notes.append("Серым фоном выделены отрезки, оценки которых заметно отличаются от остального ролика.")
        if skipped:
            notes.append("Пропущенные отрезки не входят в итоговые оценки.")
        for line in notes:
            pdf.para(line, 8, after=0)
        pdf.ln(1.5)

    # ---- key frames
    frames = [p for p in (key_frames or []) if os.path.exists(p)]
    if frames:
        from PIL import Image
        fps = float((media or {}).get("fps") or (report.get("media") or {}).get("fps") or 0) or None
        # frame numbers count inside the explained clip: the representative segment, or the whole short video
        start = rep_seg["start"] if rep_seg is not None else (0.0 if not tl else None)
        max_h, gap, per_row = 62.0, 4.0, 3
        cell_w = (pdf.w - pdf.l_margin - pdf.r_margin - gap * (per_row - 1)) / per_row

        def fitted(q):
            try:
                with Image.open(q) as im:
                    iw, ih = im.size
            except Exception:  # noqa: BLE001
                iw, ih = 16, 9
            scale = min(cell_w / iw, max_h / ih)
            return iw * scale, ih * scale

        def moment(q):
            mt = re.search(r"_frame(\d+)", Path(q).stem)
            return start + int(mt.group(1)) / fps if mt and start is not None and fps else None

        # the moment of the video as m:ss, with tenths ('0:42,1') when two frames fall into the same second;
        # the same format as the key frames on the web page (charts.frames_html)
        secs = [moment(q) for q in frames]
        whole = [_mmss(s) for s in secs if s is not None]
        tenths = len(set(whole)) < len(whole)
        captions = {}
        for q, s in zip(frames, secs):
            if s is None:
                mt = re.search(r"_frame(\d+)", Path(q).stem)
                captions[q] = (f"кадр №{mt.group(1)}" + (" отрезка" if rep_seg is not None else "")) if mt else "кадр"
            else:
                captions[q] = _mmss(s) + (f",{int(s * 10) % 10}" if tenths else "")

        first_h = max(fitted(q)[1] for q in frames[:per_row])
        timed = start is not None and fps is not None
        where = f"отрезка {seg_label(rep_seg['start'], rep_seg['end'])}" if rep_seg is not None else "ролика"
        note = (f"Кадры {where}, сильнее всего повлиявшие на оценку своей модели. Жёлтая рамка — найденное лицо; "
                + (("подпись — момент ролика (минуты:секунды, после запятой — десятые доли секунды)." if tenths
                    else "подпись — момент ролика (минуты:секунды).") if timed
                   else f"подпись — номер кадра в {'отрезке' if rep_seg is not None else 'ролике'}."))
        # heading + note + the first row of frames with captions stay on one page
        pdf.set_x(pdf.l_margin); pdf.set_font("ui", "", 8)
        note_h = pdf.multi_cell(0, 4, _keep_dash(note), dry_run=True, output="HEIGHT") + 1.5
        pdf.h2(section("Ключевые кадры"), need=note_h + first_h + 8)
        pdf.note(note)
        y0 = pdf.get_y()
        for r0 in range(0, len(frames), per_row):
            row = frames[r0:r0 + per_row]
            sizes = [fitted(q) for q in row]
            row_h = max(h for _, h in sizes)
            if y0 + row_h + 8 > pdf.page_break_trigger:
                pdf.add_page(); y0 = pdf.get_y()
            for i, (q, (iw, ih)) in enumerate(zip(row, sizes)):
                x = pdf.l_margin + i * (cell_w + gap) + (cell_w - iw) / 2
                try:
                    pdf.image(q, x=x, y=y0 + (row_h - ih), w=iw, h=ih)
                except Exception:  # noqa: BLE001
                    continue
                pdf.set_xy(pdf.l_margin + i * (cell_w + gap), y0 + row_h + 1)
                pdf.set_font("ui", "", 9); pdf.cell(cell_w, 4.5, captions[q], align="C")
            y0 += row_h + 8
            pdf.set_y(y0)

    # ---- explanations (own model, representative segment)
    if explanation:
        ixg = explanation["modalities"]["input_x_gradient"]
        mods = list(next(iter(ixg.values())).keys())
        pdf.h2(section("Вклад модальностей в оценку своей модели"), need=40)
        if rep_seg is not None:
            pdf.note(f"Этот раздел построен по одному отрезку — {seg_label(rep_seg['start'], rep_seg['end'])} "
                     f"({pdf.star} в таблице выше), а не по всему ролику.")
        header = ["Черта"] + [MEMBERS.get(x, x) for x in mods]

        def pct(s):
            v = float(s) * 100
            return "<1%" if v < 0.95 else f"{v:.0f}%"
        rows = [[TITLES.get(k, k)] + [pct(row[x]["share"]) for x in mods] for k, row in ixg.items()]
        pdf.table(header, rows, [60] + [130 / len(mods)] * len(mods))
        share = {x: sum(float(row[x]["share"]) for row in ixg.values()) / len(ixg) for x in mods}
        main = [MEMBERS.get(x, x).replace("\n", " ") for x in sorted(mods, key=lambda x: -share[x]) if share[x] >= 0.1]
        main_txt = " и ".join([", ".join(main[:-1]), main[-1]]) if len(main) > 1 else "".join(main)
        pdf.note("Доля каждой модальности в оценке своей модели (метод Input×Gradient). «<1%» — модальность почти не "
                 "влияет на оценку." + (f" Здесь модель опиралась в основном на {main_txt}." if main else ""))
        loo = explanation["modalities"].get("leave_one_out_delta") or {}
        if loo:
            keys_l = [k for k in list(TRAIT_KEYS) + ["interview"] if any(k in v for v in loo.values())]
            pdf.h3("Как меняется оценка без одной модальности", need=12 + 2 * 5.2)
            header = ["Без модальности"] + [TITLES_2L.get(k, TITLES.get(k, k)) for k in keys_l]
            rows = [[MEMBERS.get(x, x).replace("\n", " ")] + [signed(v.get(k, 0)) for k in keys_l] for x, v in loo.items()]
            pdf.table(header, rows, [36] + [154 / len(keys_l)] * len(keys_l), size=8)
            pdf.note(f"Своя модель заново оценивает {'отрезок' if rep_seg is not None else 'ролик'} без этой модальности "
                     "(метод leave-one-out). Плюс — без неё оценка была бы выше, минус — ниже; 0.00 — оценка не меняется.")
        from .narrative import words_sentences
        rw_all = explanation.get("readable_words") or {}
        lang = m.get("lang", "en")
        if rw_all:
            pdf.h3("Слова, повлиявшие на каждую черту", need=20)
            for line in words_sentences(rw_all, TITLES, lang):
                if line.endswith(":"):
                    pdf.keep(12)
                    pdf.set_font("ui", "B", 9); pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
                elif ": " in line:
                    title, rest = line.split(": ", 1)
                    pdf.labelled_line(title, rest)
                elif line:
                    pdf.para(line, 8.5)
            pdf.note("Слова из речи и из описания поведения, сильнее всего сдвинувшие оценку своей модели: «повышали» — "
                     "сдвигали оценку черты вверх, «понижали» — вниз. Служебные слова отброшены"
                     + ("; для русской речи слово показано так, как оно прозвучало." if lang == "ru" else "."))
        old = [(key, title) for key, title in (("transcript_words", "Слова речи, повлиявшие на оценку"),
                                               ("behavior_words", "Слова описания поведения, повлиявшие на оценку"))
               if key not in rw_all and key in explanation]
        for key, title in old:          # explanation without the readable lists (old run): English tokens
            pdf.h3(title, need=15)
            for k, d in explanation[key]["per_output"].items():
                pdf.labelled_line(TITLES.get(k, k), ", ".join(clean_word(w["word"]) for w in d["top_words"][:6]))
        if old:
            pdf.note("Слова в том виде, в каком их видит модель (на английском), в порядке силы влияния на оценку своей "
                     "модели; в какую сторону слово сдвигало оценку, здесь не показано.")

    # ---- texts
    if report.get("behavior_description"):
        vlm_note = f"Описание строит видеоязыковая модель по кадрам {'каждого отрезка' if tl else 'ролика'}."
        if report.get("behavior_description_ru"):
            pdf.h2(section("Описание поведения"))
            pdf.note(vlm_note)
            pdf.text_blocks(report["behavior_description_ru"], 9)
        else:
            pdf.h2(section("Описание поведения (на английском — перевод не получен)"))
            pdf.note(vlm_note)
            pdf.text_blocks(report["behavior_description"], 9)
    if report.get("transcript"):
        pdf.h2(section("Транскрипт речи"))
        pdf.para(report["transcript"], 9)

    pdf.h2("Ограничения")
    pdf.para(DISCLAIMER_RU, 8)
    pdf.para(INTERVIEW_DISCLAIMER_RU, 8)
    out_path = Path(out_path)
    pdf.output(str(out_path))
    return str(out_path)
