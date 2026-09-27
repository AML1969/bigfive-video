"""The page of the PDF report of BS Profiler 3.1 (fpdf2): class Report — A4 with the margins and the font of the
report, the footer with the file name and «стр. 3 из 7», headings, paragraphs, charts with their captions, key-value
tables, the card grid of «Ключевые факты» and «Речь в цифрах», the score bars and the tables whose header repeats on
every page. The sections that fill it are in pdf_report.py and pdf_mbti.py; pdf_charts.py draws the charts.

The layout constants live in pdf/layout.py and are re-exported here: `from bs3.pdf.document import Report, TEXT_W_MM,
NOTE_GREY` works.
"""
from __future__ import annotations

import os

from fpdf import FPDF

from .. import PRODUCT
from ..facts import card_item, fact_label
from ..norms import RU_TITLES, TRAIT_KEYS
from ..palette import CARD_PDF, FACT_VALUE_PDF, SCORE_BAR_PDF, TRAIT_BAR_PDF
from ..textfmt import fiv2_ref_ru, pct_phrase
from .layout import FONT_CANDIDATES, MARGIN_MM, NOTE_GREY, RADAR_W_MM, ROW_GAP_MM, TEXT_W_MM

__all__ = ["Report", "FONT_CANDIDATES", "MARGIN_MM", "NOTE_GREY", "RADAR_W_MM", "ROW_GAP_MM", "TEXT_W_MM"]


def _group_name(keys: list[str]) -> str:
    """Which score rows a FIV2 percentile applies to, for the note under the score bars."""
    if set(keys) == set(TRAIT_KEYS):
        return "пять черт"
    return ", ".join("«собеседование»" if k == "interview" else RU_TITLES[k].lower() for k in keys)


def _rgb(hexc: str) -> tuple:
    return tuple(int(hexc.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


class Report(FPDF):
    def __init__(self, file_label: str = "", total_pages: int | None = None):
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
        self.file_label = file_label             # original file name of the video, in the footer of every page
        self.total_pages = total_pages           # «стр. 3 из 7»: known from a first layout pass (build_pdf)
        self.plan: dict = {}                     # section key -> number, fixed before printing (references forward)
        self.appx: dict = {}                     # appendix key -> letter

    def footer(self):
        self.set_y(-10)
        self.set_font("ui", "", 8)
        self.set_text_color(NOTE_GREY)
        right = f"{PRODUCT} · стр. {self.page_no()}" + (f" из {self.total_pages}" if self.total_pages else "")
        y = self.get_y()
        if self.file_label:
            self.set_x(self.l_margin)
            self.cell(110, 5, self._middle_cut(self.file_label, 110))
        self.set_xy(self.l_margin, y)
        self.cell(0, 5, right, align="R")
        self.set_text_color(0)

    def _middle_cut(self, text: str, max_w: float) -> str:
        """A long file name loses characters in the middle («начало…конец.mp4»): both ends tell files apart."""
        if self.get_string_width(text) <= max_w:
            return text
        for keep in range(len(text) - 1, 1, -1):
            s = text[:keep // 2] + "…" + text[len(text) - (keep - keep // 2):]
            if self.get_string_width(s) <= max_w:
                return s
        return text[:1] + "…"

    def h1(self, text):
        # a title longer than the line wraps instead of running past the right edge of the page
        self.set_font("ui", "B", 16); self.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT", align="L"); self.ln(2)

    def h2(self, text, keep_mm: float = 25):
        """Section heading kept on the same page as at least `keep_mm` of what follows (its intro and first block)."""
        if self.get_y() + 10 + keep_mm > self.page_break_trigger:
            self.add_page()
        self.ln(2); self.set_font("ui", "B", 12); self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.set_font("ui", "", 10)

    def section(self, title: str, key: str, keep_mm: float = 25) -> int:
        """Numbered section heading «2. Big Five по ходу ролика»; numbers follow the sections actually printed and are
        fixed in self.plan before printing, so a caption can refer to a later section."""
        n = self.plan[key]
        self.h2(f"{n}. {title}", keep_mm)
        return n

    def h3(self, text, keep_mm: float = 20, size: float = 9):
        """Bold sub-heading kept on the same page as at least `keep_mm` of what follows it."""
        if self.get_y() + 6 + keep_mm > self.page_break_trigger:
            self.add_page()
        self.set_x(self.l_margin)
        self.set_font("ui", "B", size); self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.set_font("ui", "", 10)

    @staticmethod
    def _lh(size: float) -> float:
        return max(3.6, size * 0.5)          # leading follows the font size (small notes stay close)

    @staticmethod
    def _breakable(text) -> str:
        # break words longer than the line (hashes, URLs) so fpdf can wrap them
        return " ".join(w if len(w) < 60 else " ".join(w[i:i + 60] for i in range(0, len(w), 60)) for w in str(text).split(" "))

    def para_height(self, text, size=9, w: float = 0) -> float:
        """Height in mm that para(text, size) takes, without printing it."""
        if not text:
            return 0.0
        self.set_font("ui", "", size)
        lines = self.multi_cell(w or self.epw, self._lh(size), self._breakable(text), align="L", dry_run=True,
                                output="LINES")
        return len(lines) * self._lh(size) + 1

    def para(self, text, size=9, color: int = 0):
        if not text:
            return
        self.set_x(self.l_margin)
        self.set_font("ui", "", size)
        self.set_text_color(color)
        # left-aligned: justified Russian lines get wide gaps
        self.multi_cell(0, self._lh(size), self._breakable(text), align="L")
        self.set_text_color(0)
        self.ln(1)

    def caption(self, text, size=7.5):
        """Caption under a chart: small, dark grey, close to the chart."""
        self.ln(0.5)
        self.para(text, size, color=51)

    def chart_height(self, path, w: float | None = None) -> float:
        from PIL import Image
        with Image.open(path) as im:
            w_px, h_px = im.size
        return (w or self.epw) * h_px / w_px

    def chart(self, path, keep_mm: float = 0, w: float | None = None, x: float | None = None) -> float:
        """Chart PNG at the size pdf_charts drew it (1 pt of the figure = 1 pt on paper), across the text width unless
        `w` says otherwise. A chart that does not fit on the rest of the page, together with `keep_mm` of the caption
        under it, starts a new page: charts are never cut and a caption never ends up on the page after its chart.
        Returns the printed height."""
        w = w or self.epw
        h = self.chart_height(path, w)
        if self.get_y() + h + keep_mm > self.page_break_trigger:
            self.add_page()
        y = self.get_y()
        self.image(path, x=self.l_margin if x is None else x, y=y, w=w, h=h)
        self.set_xy(self.l_margin, y + h)
        return h

    def chart_block_height(self, path, cap: str = "", cap_size: float = 7.5) -> float:
        """Height of chart_block(path, cap): the gap above, the chart and its caption."""
        return 1 + self.chart_height(path) + (0.5 + self.para_height(cap, cap_size) if cap else 0)

    def chart_block(self, path, cap: str = "", cap_size: float = 7.5) -> None:
        """Full-width chart with its caption; the two are never separated by a page break. A chart that opens a page
        carries its own title, so the section heading is not repeated above it: the six millimetres of that repeat
        pushed the next chart off the page and left a hole of 66 mm where it had stood."""
        self.ln(1)
        self.chart(path, keep_mm=(0.5 + self.para_height(cap, cap_size)) if cap else 0)
        if cap:
            self.caption(cap, cap_size)

    @staticmethod
    def _kv_value(v) -> str:
        v = str(v)
        if len(v) > 48 and " " not in v:                # e.g. sha256: split so it wraps
            v = " ".join(v[i:i + 32] for i in range(0, len(v), 32))
        return v

    def kv_table(self, rows, w1=55, size=8.5, lh: float | None = None):
        lh = lh or size * 0.61
        for k, v in rows:
            self.set_x(self.l_margin)
            self.set_font("ui", "B", size); self.cell(w1, lh, str(k))
            self.set_font("ui", "", size)
            self.multi_cell(0, lh, self._kv_value(v), new_x="LMARGIN", new_y="NEXT", align="L")

    def kv_table_height(self, rows, w1=55, size=8.5, lh: float | None = None) -> float:
        """Height in mm that kv_table(rows, w1, size, lh) takes, without printing it."""
        lh = lh or size * 0.61
        self.set_font("ui", "", size)
        return sum(len(self.multi_cell(self.epw - w1, lh, self._kv_value(v), dry_run=True, output="LINES")) * lh
                   for _, v in rows)

    # ---------------------------------------------------------------- cards (key facts, speech in numbers)
    def _card_layout(self, items, cols: int, gap: float, value_first: bool = False):
        w = (self.epw - gap * (cols - 1)) / cols
        pad, inner = 1.8, (self.epw - gap * (cols - 1)) / cols - 3.6
        heights = []
        for lab, val, note, state in (card_item(x) for x in items):
            self.set_font("ui", "", 7.5)
            # the label carries the state word in the same cases as in cards(), or the card would be measured short
            n_lab = len(self.multi_cell(inner, 3.4, fact_label(lab, state) if value_first else str(lab),
                                        dry_run=True, output="LINES"))
            self.set_font("ui", "B", 11)
            n_val = len(self.multi_cell(inner, 5.0, str(val if val not in (None, "") else "—"), dry_run=True, output="LINES"))
            n_note = 0
            if note:
                self.set_font("ui", "", 7.5)
                n_note = len(self.multi_cell(inner, 3.4, str(note), dry_run=True, output="LINES"))
            heights.append(pad + n_lab * 3.4 + 0.6 + n_val * 5.0 + (0.3 + n_note * 3.4 if n_note else 0) + pad - 0.4)
        rows = [list(range(i, min(i + cols, len(items)))) for i in range(0, len(items), cols)]
        row_h = [max(heights[i] for i in r) for r in rows]
        return w, pad, inner, rows, row_h

    def cards_height(self, items, cols: int = 3, gap: float = 4.0, value_first: bool = False) -> float:
        if not items:
            return 0.0
        _, _, _, _, row_h = self._card_layout(items, cols, gap, value_first)
        return sum(row_h) + gap * 0.75 * (len(row_h) - 1) + 1.5

    def cards(self, items, cols: int = 3, gap: float = 4.0, value_first: bool = False):
        """(label, value, note[, state]) -> a grid of outlined cards, as on the web page: label and note small and
        grey, the value large and bold. `value_first`: the card reads value, then label, then note; a value with a
        state is printed in the colour of that state (palette.FACT_VALUE_PDF) and its label ends with the word of
        that state (facts.fact_label), so the card also reads on a black-and-white printer — «Ключевые факты»
        of 3.1. The grid is never split between pages."""
        if not items:
            return
        w, pad, inner, rows, row_h = self._card_layout(items, cols, gap, value_first)
        if self.get_y() + self.cards_height(items, cols, gap, value_first) > self.page_break_trigger:
            self.add_page()
        y = self.get_y()
        for r, rh in zip(rows, row_h):
            for j, i in enumerate(r):
                lab, val, note, state = card_item(items[i])
                x = self.l_margin + j * (w + gap)
                self.set_fill_color(CARD_PDF["fill"]); self.set_draw_color(CARD_PDF["outline"]); self.set_line_width(0.2)
                self.rect(x, y, w, rh, style="DF", round_corners=True, corner_radius=1.2)
                self.set_xy(x + pad, y + pad)

                def small(text, lead=0.3):
                    if lead:
                        self.set_y(self.get_y() + lead); self.set_x(x + pad)
                    self.set_font("ui", "", 7.5); self.set_text_color(NOTE_GREY)
                    self.multi_cell(inner, 3.4, str(text), align="L", new_x="LEFT", new_y="NEXT")

                def big(lead=0.6):
                    if lead:
                        self.set_y(self.get_y() + lead); self.set_x(x + pad)
                    self.set_font("ui", "B", 11)
                    ink = _rgb(FACT_VALUE_PDF[state]) if (value_first and state) else (0, 0, 0)
                    self.set_text_color(*ink)
                    self.multi_cell(inner, 5.0, str(val if val not in (None, "") else "—"), align="L",
                                    new_x="LEFT", new_y="NEXT")
                # the same two gaps either way (0.6 + 0.3), so cards_height does not depend on the order
                if value_first:
                    big(lead=0)
                    small(fact_label(lab, state), lead=0.6)
                else:
                    small(lab, lead=0)
                    big()
                if note:
                    small(note)
            y += rh + gap * 0.75
        self.set_text_color(0); self.set_draw_color(0)
        self.set_xy(self.l_margin, y - gap * 0.75 + 1.5)

    # ---------------------------------------------------------------- score bars
    BAR_W, BAR_H = 52, 3.6

    def _bar_rows(self, traits: dict, interview: dict | None):
        return [(k, traits[k]) for k in TRAIT_KEYS] + ([("interview", interview)] if interview else [])

    def _bars_legend(self):
        """Legend items (kind, text) under the score bars."""
        # «как на графиках» with one reservation: the extraversion fill is a step darker than its line on the charts,
        # because the chart colour gives only 2.9:1 on the light track of the bar
        return [("traits", "оценка черты 0…1 (цвет — как на графиках, экстраверсия чуть темнее)"),
                ("tick", "середина шкалы")]

    def _legend_lines(self, items) -> list:
        """Legend items wrapped into lines of the text width: [[(kind, text, x_offset), …], …]."""
        self.set_font("ui", "", 8)
        lines, cur, x = [], [], 0.0
        for kind, text in items:
            w = 8 + self.get_string_width(text) + 1
            if cur and x + w > self.epw:
                lines.append(cur); cur, x = [], 0.0
            cur.append((kind, text, x)); x += w + 3
        if cur:
            lines.append(cur)
        return lines

    def _bars_note(self, traits: dict, interview: dict | None) -> str:
        groups: dict[str, list[str]] = {}          # FIV2 reference in words -> item keys (Russian speech: none)
        for k, t in self._bar_rows(traits, interview):
            pct = t.get("percentile", t.get("percentile_vs_fiv2"))
            ref = t.get("percentile_ref", "train First Impressions V2 (6000 клипов)" if pct is not None else "")
            if pct is not None and pct_phrase(pct, ref)[0]:
                groups.setdefault(f"относительно {fiv2_ref_ru(ref)}", []).append(k)
        parts = ["Длина полоски — оценка модели от 0 до 1; уровни черт и буквы MBTI считаются по этой же шкале, "
                 "середина — 0.5."]
        interview_named = False
        if groups:
            if len(groups) == 1:
                parts.append(f"Рядом с полоской — процентиль {next(iter(groups))}.")
            else:
                parts += [(f"«Собеседование» (коричневая полоска, модель AMLAI 1.0) — процентиль {ref}."
                           if keys == ["interview"] else f"{_group_name(keys).capitalize()} — процентиль {ref}.")
                          for ref, keys in groups.items()]
                interview_named = any(keys == ["interview"] for keys in groups.values())
        if interview and not interview_named:
            parts.append("Коричневая полоска — впечатление «собеседование» (метка модели AMLAI 1.0, шкала 0…1).")
        return " ".join(parts)

    def score_bars_height(self, traits: dict, interview: dict | None) -> float:
        n = len(TRAIT_KEYS)
        h = n * 6.0 + (8.0 if interview else 0.0) + 4.2
        h += len(self._legend_lines(self._bars_legend())) * 4.4 + 1
        self.set_font("ui", "", 8)
        h += len(self.multi_cell(self.epw, 4.0, self._bars_note(traits, interview), dry_run=True, output="LINES")) * 4.0
        return h + 1

    def score_bars(self, traits: dict, interview: dict | None):
        """One row per trait: the bar is the score itself (0…1, what a reader expects to see filled), the text gives the
        score and, for a FIV2 percentile (an older English job), its position in words. One model (3.1): no second
        opinion under the bars."""
        items = self._bar_rows(traits, interview)
        bar_w, bar_h = self.BAR_W, self.BAR_H
        c = SCORE_BAR_PDF
        # the label column is as wide as the longest title; the phrase after the bar must end at the right margin
        self.set_font("ui", "", 8.5)
        label_w = max(self.get_string_width(RU_TITLES[k]) for k, _ in items) + 3
        x = self.l_margin + label_w
        row_h = 6.0
        for k, t in items:
            pct = t.get("percentile", t.get("percentile_vs_fiv2")); score = float(t["score"])
            ref = t.get("percentile_ref", "train First Impressions V2 (6000 клипов)" if pct is not None else "")
            if k == "interview":        # a separate label of the own model: set off from the five traits (dashed rule)
                yl = self.get_y() + 1.0
                self.set_draw_color(c["outline"]); self.set_line_width(0.2); self.set_dash_pattern(dash=0.8, gap=0.8)
                self.line(self.l_margin, yl, self.l_margin + self.epw, yl)
                self.set_dash_pattern(); self.set_draw_color(0)
                self.set_y(yl + 1.0)
            self.set_x(self.l_margin)
            self.set_font("ui", "", 8.5)
            self.cell(label_w, row_h, RU_TITLES[k])
            x, y = self.get_x(), self.get_y() + (row_h - bar_h) / 2
            # track: light fill with a grey outline, so the full 0…1 length is visible (outline 3.84:1 on white)
            self.set_fill_color(c["track"]); self.set_draw_color(c["outline"]); self.set_line_width(0.2)
            self.rect(x, y, bar_w, bar_h, style="DF")
            # each bar in the colour of its line on the charts (the interview bar stays brown)
            hexc = TRAIT_BAR_PDF.get(k)
            self.set_fill_color(*(_rgb(hexc) if hexc else c["fill"]))
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
            text = f"{score:.2f}   {pct_phrase(pct, ref)[0]}".rstrip()
            # cell() insets the text by c_margin on both sides: the room for it is that much smaller
            room = self.l_margin + self.epw - self.get_x() - 2 * self.c_margin
            for pt in (8, 7.5, 7.2, 7.0):
                self.set_font("ui", "", pt)
                if self.get_string_width(text) <= room:
                    break
            self.cell(0, row_h, text, new_x="LMARGIN", new_y="NEXT")
        # scale under the bars: 0, 0.5, 1, each label centred on its point of the bar
        self.set_font("ui", "", 7.5); self.set_text_color(NOTE_GREY)
        y = self.get_y()
        for val, pos in (("0", x), ("0.5", x + bar_w / 2), ("1", x + bar_w)):
            self.set_xy(pos - 5, y); self.cell(10, 3.6, val, align="C")
        self.set_text_color(0)
        self.set_xy(self.l_margin, y + 4.2)
        self._draw_bars_legend(self._legend_lines(self._bars_legend()))
        self.set_font("ui", "", 8); self.set_text_color(NOTE_GREY)
        self.multi_cell(0, 4.0, self._bars_note(traits, interview), new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_text_color(0)
        self.ln(1)

    def _draw_bars_legend(self, lines):
        c = SCORE_BAR_PDF
        for line in lines:
            y = self.get_y()
            for kind, text, dx in line:
                x = self.l_margin + dx
                if kind == "traits":             # the five trait colours side by side
                    for i, k in enumerate(TRAIT_KEYS):
                        self.set_fill_color(*_rgb(TRAIT_BAR_PDF[k])); self.rect(x + i * 1.3, y + 1.0, 1.3, 2.6, style="F")
                else:
                    self.set_draw_color(c["mid_tick"]); self.set_line_width(0.3)
                    self.line(x + 3, y + 0.4, x + 3, y + 4.0); self.set_draw_color(0); self.set_line_width(0.2)
                self.set_xy(x + 8, y)
                self.set_font("ui", "", 8)
                self.cell(self.get_string_width(text) + 1, 4.4, text)
            self.set_xy(self.l_margin, y + 4.4)
        self.ln(0.8)

    # ---------------------------------------------------------------- tables
    def _header_lines(self, header, widths, size) -> list:
        """How many lines each header title takes in its own column: a title is wrapped by its '\\n' and, when the
        column came out narrower than the title, by the column width itself. Measured at the printed widths, so a
        header that wraps one line more than it was written still gets a tall enough row."""
        self.set_font("ui", "B", size)
        return [len(self.multi_cell(w, size * 0.5, str(h), dry_run=True, output="LINES")) for h, w in zip(header, widths)]

    def _header_height(self, header, widths, size, chips=None, groups=None) -> float:
        chip = (2.2 + 1.0) if chips and any(chips) else 0.0
        return max(self._header_lines(header, widths, size)) * size * 0.5 + 1.5 + chip \
            + ((size * 0.5 + 1.5) if groups else 0.0)

    def _table_header(self, header, widths, size, chips=None, groups=None):
        """Header cells may contain '\\n' (multi-line titles); all cells get the same height.
        chips: optional colour per column ('#rrggbb' or None), drawn as a small outlined strip above the title.
        groups: optional [(title, n_columns)] row above the header, spanning columns."""
        x0, y0 = self.l_margin, self.get_y()
        if groups:
            self.set_font("ui", "B", size)
            gh = size * 0.5 + 1.5
            x, i = x0, 0
            for title, span in groups:
                w = sum(widths[i:i + span])
                self.rect(x, y0, w, gh)
                self.set_xy(x, y0 + 0.75)
                self.cell(w, size * 0.5, str(title), align="C")
                x += w; i += span
            y0 += gh
        n_lines = self._header_lines(header, widths, size)
        self.set_font("ui", "B", size)
        lh = size * 0.5                      # line height in mm for this font size
        chip_h = 2.2 if chips and any(chips) else 0.0
        top = chip_h + 1.0 if chip_h else 0.0
        hh = max(n_lines) * lh + 1.5 + top
        x = x0
        for i, (h, w) in enumerate(zip(header, widths)):
            self.rect(x, y0, w, hh)
            chip = chips[i] if chip_h and i < len(chips) else None
            if chip:
                self.set_fill_color(*_rgb(chip)); self.set_draw_color(51)       # outline #333333
                self.rect(x + w * 0.18, y0 + 1.0, w * 0.64, chip_h, style="DF")
                self.set_draw_color(0)
            self.set_xy(x, y0 + top + (hh - top - n_lines[i] * lh) / 2)
            self.multi_cell(w, lh, str(h), border=0, align="C")
            x += w
        self.set_xy(x0, y0 + hh)
        self.set_font("ui", "", size)

    def table(self, header, rows, widths, size=8, chips=None, zebra=True, first_left=False, groups=None,
              row_h: float = 5.2, min_rows: int = 3, cont_title: str = ""):
        """zebra: every second row on a very light grey (#f5f5f5, decorative) so long rows are easy to follow.
        first_left: left-align the first column (row names) even when it is narrow. The header (with the group row)
        repeats on every page, and a table starts on the next page unless its header and `min_rows` rows fit here.
        cont_title: a quiet line above the repeated header, so a page of numbers says which table it continues."""
        # every table spans the text width, as the charts and the rows of key frames do (and never runs past the margin)
        k = self.epw / sum(widths)
        widths = [w * k for w in widths]
        hh = self._header_height(header, widths, size, chips, groups)
        if self.get_y() + hh + min(min_rows, len(rows)) * row_h > self.page_break_trigger:
            self.add_page()
        self._table_header(header, widths, size, chips, groups)
        for i, r in enumerate(rows):
            if self.get_y() + row_h > self.page_break_trigger:
                self.add_page()
                if cont_title:
                    self.set_font("ui", "B", 8); self.set_text_color(NOTE_GREY)
                    self.cell(0, 5, cont_title, new_x="LMARGIN", new_y="NEXT")
                    self.set_text_color(0)
                self._table_header(header, widths, size, chips, groups)
            shade = zebra and i % 2 == 1
            if shade:
                self.set_fill_color(245)
            self.set_x(self.l_margin)
            for j, (c, w) in enumerate(zip(r, widths)):
                align = "L" if (j == 0 and first_left) or w >= 40 else "C"
                self.cell(w, row_h, str(c), border=1, align=align, fill=shade)
            self.ln()
