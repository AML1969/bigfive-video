"""PDF report for one analysed video (fpdf2): file metadata, analysis settings, scores with percentiles,
per-segment timeline, key frames as images, modality contributions and words, behaviour description, transcript.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

from fpdf import FPDF

from .norms import RU_SHORT, TRAIT_KEYS
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


def pct_phrase(pct, ref: str = "") -> str:
    """'выше, чем у 83% русских роликов' / 'ниже, чем у 95% людей FIV2' / 'пул пока мал'."""
    if pct is None:
        return "положение: пул пока мал"
    group = "русских роликов" if "пула" in (ref or "") else "людей в FIV2"
    pct = float(pct)
    return f"выше, чем у {pct:.0f}% {group}" if pct >= 50 else f"ниже, чем у {100 - pct:.0f}% {group}"
FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/mnt/c/Windows/Fonts/arial.ttf", "/mnt/c/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
]


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
        self.set_font("ui", "", 10)

    def footer(self):
        self.set_y(-10)
        self.set_font("ui", "", 7)
        self.set_text_color(120)
        self.cell(0, 5, f"BS 2.0 · стр. {self.page_no()}", align="R")
        self.set_text_color(0)

    def h1(self, text):
        self.set_font("ui", "B", 16); self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT"); self.ln(1)

    def h2(self, text):
        self.ln(2); self.set_font("ui", "B", 12); self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.set_font("ui", "", 10)

    def para(self, text, size=10):
        self.set_x(self.l_margin)
        self.set_font("ui", "", size)
        # break words longer than the line (hashes, URLs) so fpdf can wrap them
        text = " ".join(w if len(w) < 60 else " ".join(w[i:i + 60] for i in range(0, len(w), 60)) for w in str(text).split(" "))
        self.multi_cell(0, 5, text)
        self.ln(1)

    def kv_table(self, rows, w1=55):
        self.set_font("ui", "", 9)
        for k, v in rows:
            self.set_x(self.l_margin)
            self.set_font("ui", "B", 9); self.cell(w1, 5.5, str(k))
            self.set_font("ui", "", 9)
            v = str(v)
            if len(v) > 48 and " " not in v:            # e.g. sha256: split so it wraps
                v = " ".join(v[i:i + 32] for i in range(0, len(v), 32))
            self.multi_cell(0, 5.5, v, new_x="LMARGIN", new_y="NEXT")

    def score_bars(self, traits: dict, interview: dict | None):
        """One row per trait: the bar is the score itself (0…1, what a reader expects to see filled), the text
        gives the score and the position relative to the reference population in words."""
        items = [(k, traits[k]) for k in TRAIT_KEYS] + ([("interview", interview)] if interview else [])
        label_w, bar_w = 58, 52
        refs = []
        for k, t in items:
            pct = t.get("percentile", t.get("percentile_vs_fiv2")); score = float(t["score"])
            ref = t.get("percentile_ref", "train First Impressions V2 (6000 клипов)")
            if ref not in refs:
                refs.append(ref)
            self.set_font("ui", "", 8.5)
            self.cell(label_w, 6, TITLES[k])
            x, y = self.get_x(), self.get_y() + 1.2
            self.set_fill_color(230); self.rect(x, y, bar_w, 3.6, style="F")
            self.set_fill_color(76, 139, 245) if k != "interview" else self.set_fill_color(138, 109, 59)
            self.rect(x, y, bar_w * max(0.01, min(1.0, score)), 3.6, style="F")
            self.set_x(x + bar_w + 2)
            self.set_font("ui", "", 8)
            self.cell(0, 6, f"{score:.2f}   {pct_phrase(pct, ref)}", new_x="LMARGIN", new_y="NEXT")
        self.set_font("ui", "", 7); self.set_text_color(110)
        self.multi_cell(0, 4, "Полоска — оценка от 0 до 1. Рядом — положение относительно опорной группы: "
                             + "; ".join(refs) + ".")
        self.set_text_color(0)

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

    def table(self, header, rows, widths, size=8):
        if sum(widths) > self.w - self.l_margin - self.r_margin + 0.1:      # never run past the right margin
            k = (self.w - self.l_margin - self.r_margin) / sum(widths)
            widths = [w * k for w in widths]
        if self.get_y() > 255:                # header + at least two rows must fit on this page
            self.add_page()
        self._table_header(header, widths, size)
        for r in rows:
            if self.get_y() > 275:
                self.add_page()
                self._table_header(header, widths, size)
            for c, w in zip(r, widths):
                self.cell(w, 5.2, str(c), border=1, align="C" if w < 40 else "L")
            self.ln()


def _analyses_sections(pdf, report: dict) -> None:
    """BS 2.0: emotions (text / face), voice dimensions, speech analytics with charts and per-segment table."""
    an = report.get("analyses") or {}
    if not an:
        return
    from .charts import EMO_RU, VOICE_RU
    from .narrative2 import analyses_sentences
    charts = report.get("chart_files") or {}
    if pdf.get_y() > 200:
        pdf.add_page()
    pdf.h2("5. Эмоции, голос и мимика")
    pdf.para(analyses_sentences(report), 9)
    if charts.get("emotions"):
        pdf.image(charts["emotions"], w=180)
    te, fa, vo = an.get("emotions_text") or {}, an.get("face") or {}, an.get("voice") or {}
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
        pdf.ln(1); pdf.table(header, rows, [40] + [20] * 7, size=8)
        pdf.para("Средние доли за ролик. Речь — модель эмоций текста по переводу транскрипта; лицо — модель выражений по кадрам "
                 "(обучена на фотографиях, завышает «грусть» и «страх» у спокойного лица).", 7)
    if charts.get("voice_speech"):
        if pdf.get_y() > 200:
            pdf.add_page()
        pdf.image(charts["voice_speech"], w=180)
    if vo.get("mean"):
        pdf.para("Голос (модель эмоций в речи, 0…1): " + ", ".join(f"{VOICE_RU[d]} {vo['mean'].get(d, 0):.2f} (±{vo.get('std', {}).get(d, 0):.2f})"
                                                            for d in VOICE_RU) + ".", 8)
    per = an.get("per_segment") or []
    if per:
        if pdf.get_y() > 230:
            pdf.add_page()
        pdf.set_font("ui", "B", 9); pdf.cell(0, 6, "По отрезкам", new_x="LMARGIN", new_y="NEXT")
        header = ["Отрезок", "Эмоция\nречи", "Выражение\nлица", "Возбуж-\nдение", "Уверен-\nность", "Позитив-\nность", "Слов/\nмин", "Паузы"]
        rows = []
        for r in per:
            t_e = (r.get("emotions_text") or {}); f_e = ((r.get("face") or {}).get("expressions") or {}); v = r.get("voice") or {}
            sp = r.get("speech") or {}
            rows.append([seg_label(r["start"], r["end"]),
                         EMO_RU.get(max(t_e.items(), key=lambda kv: kv[1])[0], "—") if t_e else "—",
                         EMO_RU.get(max(f_e.items(), key=lambda kv: kv[1])[0], "—") if f_e else "—",
                         f"{v.get('arousal', 0):.2f}" if v else "—", f"{v.get('dominance', 0):.2f}" if v else "—",
                         f"{v.get('valence', 0):.2f}" if v else "—",
                         f"{sp.get('words_per_min_speech') or 0:.0f}" if sp else "—", f"{sp.get('pause_share', 0):.0%}" if sp else "—"])
        pdf.table(header, rows, [26, 26, 26, 20, 20, 20, 20, 20], size=7)
    sp = an.get("speech") or {}
    if sp:
        if pdf.get_y() > 240:
            pdf.add_page()
        pdf.h2("6. Речь")
        pdf.kv_table([("Слов всего / уникальных", f"{sp.get('words')} / {sp.get('unique_words')}"),
                      ("Темп", f"{sp.get('words_per_min_speech') or 0:.0f} слов в минуту речи; {sp.get('words_per_min_wall', 0):.0f} по времени ролика"),
                      ("Паузы", f"{sp.get('pause_share', 0):.0%} времени; длинных пауз (более 2 с): {sp.get('long_pauses', 0)}"),
                      ("Слова-заполнители", f"{sp.get('fillers', 0)} ({sp.get('fillers_per_100', 0):.1f} на 100 слов)"),
                      ("Средняя фраза", f"{sp.get('mean_sentence') or 0:.0f} слов"), ("Разнообразие словаря", f"{sp.get('ttr') or 0:.2f}")])
        if sp.get("vocabulary"):
            pdf.para("Частые слова: " + ", ".join(f"{w} ({n})" for w, n in sp["vocabulary"][:15]), 8)


def build_pdf(report: dict, out_path: str | Path, explanation: dict | None = None, media: dict | None = None,
              key_frames: list[str] | None = None) -> str:
    pdf = Report()
    pdf.add_page()
    pdf.h1("BS 2.0 — отчёт по видео: Big Five, эмоции, голос, речь")
    fname = report.get("original_file_name") or (media or {}).get("file_name") or Path(report.get("input", "")).name
    pdf.para(f"Создан: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}   ·   Файл: {fname}", 9)

    # ---- file metadata
    pdf.h2("1. Файл")
    rows = []
    if media:
        rows += [("Имя файла", media.get("file_name")), ("Размер", f"{media.get('size_mb')} МБ ({media.get('size_bytes')} байт)"),
                 ("Длительность", f"{fmt_secs(media.get('duration_sec'))} ({media.get('duration_sec')} с)"),
                 ("Контейнер", media.get("container")),
                 ("Видео", f"{media.get('video_codec')} {media.get('width')}×{media.get('height')} @ {media.get('fps')} fps"
                           + (f", поворот {media.get('rotation')}°" if media.get("rotation") else "")),
                 ("Аудио", f"{media.get('audio_codec')} {media.get('sample_rate')} Гц, каналов: {media.get('channels')}"),
                 ("Битрейт", f"{media.get('bitrate_kbps')} кбит/с"), ("Изменён", media.get("modified"))]
        for k, v in media.items():
            if k.startswith("tag_"):
                rows.append((k[4:], v))
        if media.get("sha256"):
            rows.append(("SHA-256", media["sha256"]))
    else:
        rows.append(("Путь", report.get("input")))
    pdf.kv_table(rows)

    # ---- analysis settings
    pdf.h2("2. Параметры анализа")
    m = report.get("model", {})
    members = [x for x in report.get("modalities_used", []) if x in SYSTEM_TITLES]
    if m.get("backend") == "ensemble" and members:
        system = "ансамбль: " + " + ".join(SYSTEM_TITLES[x] for x in members)
        if m.get("primary"):
            system += f"; основная оценка — {SYSTEM_TITLES.get(m['primary'], m['primary'])}"
            if m.get("scale"):
                system += f" (шкала {m['scale']})"
    else:
        system = f"{SYSTEM_TITLES.get(m.get('backend'), m.get('backend'))} ({m.get('corpus')})"
    rows = [("Система", system), ("Язык речи", {"ru": "русский", "en": "английский"}.get(m.get("lang"), m.get("lang"))),
            ("Распознавание речи", m.get("asr_model") or "готовый транскрипт"),
            ("Обучающие данные", m.get("trained_on")), ("Модальности", ", ".join(report.get("modalities_used", []))),
            ("Версия", m.get("version"))]
    if report.get("segments"):
        rows.append(("Сегменты", f"{report['segments']} по ~20 с; итог — среднее с весом по длительности"))
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

    pdf.h2("3. Оценки")
    pdf.score_bars(report["traits"], report.get("interview"))
    std = report.get("scores_std_across_segments") or {}
    if std:
        pdf.para("Разброс между сегментами: " + ", ".join(f"{RU_SHORT[k].lower()} ±{std.get(k, 0):.2f}" for k in TRAIT_KEYS), 8)
    var = report.get("variant_scores") or {}
    if var:
        pdf.set_font("ui", "B", 9); pdf.cell(0, 6, "Оценки участников ансамбля", new_x="LMARGIN", new_y="NEXT")
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
        pdf.h2("4. Таймлайн по сегментам")
        keys = TRAIT_KEYS + (["interview"] if any(s.get("scores") and "interview" in s["scores"] for s in tl) else [])
        header = ["Сегмент"] + [TITLES_2L.get(k, TITLES.get(k, k)) for k in keys]
        rows = []
        for s in tl:
            if s.get("scores"):
                rows.append([seg_label(s["start"], s["end"]) + (" ★" if s["segment"] == report.get("representative_segment") else "")]
                            + [f"{s['scores'][k]:.2f}" for k in keys])
            else:
                rows.append([seg_label(s["start"], s["end"]), "пропущен"] + [""] * (len(keys) - 1))
        pdf.table(header, rows, [30] + [150 / len(keys)] * len(keys), size=7)
        pdf.para("★ — сегмент, ближайший к среднему профилю; по нему построены объяснения.", 7)
    charts = report.get("chart_files") or {}
    if charts.get("traits"):
        pdf.ln(2)
        pdf.image(charts["traits"], w=180)
    _analyses_sections(pdf, report)

    # ---- key frames
    frames = [p for p in (key_frames or []) if os.path.exists(p)]
    if frames:
        pdf.h2("7. Ключевые кадры")
        from PIL import Image
        max_h, gap, per_row = 62.0, 4.0, 3
        cell_w = (pdf.w - pdf.l_margin - pdf.r_margin - gap * (per_row - 1)) / per_row
        row, y0 = [], pdf.get_y()
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
                pdf.set_font("ui", "", 7); pdf.cell(cell_w, 4, Path(q).stem.replace("key_", "кадр "), align="C")
            y0 += row_h + 8
            pdf.set_y(y0)
            row = []

    # ---- explanations
    if explanation:
        pdf.h2("8. Вклад модальностей (Input×Gradient, своя модель)")
        ixg = explanation["modalities"]["input_x_gradient"]
        mods = list(next(iter(ixg.values())).keys())
        header = ["Черта"] + [MEMBERS.get(x, x) for x in mods]
        def pct(s):
            v = float(s) * 100
            return "<1%" if v < 0.95 else f"{v:.0f}%"
        rows = [[TITLES.get(k, k)] + [pct(row[x]["share"]) for x in mods] for k, row in ixg.items()]
        pdf.table(header, rows, [60] + [int(120 / len(mods))] * len(mods))
        pdf.para("Доля вклада модальности в оценку своей модели (Input×Gradient). «<1%» — модальность почти не влияет на "
                 "оценку этого ролика: модель, обученная на FIV2, опирается в основном на лицо и голос.", 7)
        loo = explanation["modalities"].get("leave_one_out_delta") or {}
        if loo:
            pdf.ln(2)
            if pdf.get_y() > 240:            # keep the heading together with its table
                pdf.add_page()
            pdf.set_font("ui", "B", 9); pdf.cell(0, 6, "Сдвиг оценок при удалении модальности (leave-one-out)", new_x="LMARGIN", new_y="NEXT")
            keys_l = [k for k in list(TRAIT_KEYS) + ["interview"] if any(k in v for v in loo.values())]
            header = ["Без модальности"] + [TITLES_2L.get(k, TITLES.get(k, k)) for k in keys_l]
            rows = [[MEMBERS.get(x, x).replace("\n", " ")] + [f"{v.get(k, 0):+.2f}" for k in keys_l] for x, v in loo.items()]
            pdf.table(header, rows, [42] + [138 / len(keys_l)] * len(keys_l), size=7)
            pdf.para("Положительное число — без этой модальности оценка была бы выше, отрицательное — ниже.", 7)
        from .narrative import words_summary
        from .words import WORDS_NOTE
        rw_all = explanation.get("readable_words") or {}
        lang = (report.get("model") or {}).get("lang", "en")
        shown = False
        if rw_all:
            pdf.ln(1)
            pdf.set_font("ui", "B", 9); pdf.cell(0, 6, "Слова, на которые откликнулась модель", new_x="LMARGIN", new_y="NEXT")
            for para in words_summary(rw_all, explanation, TITLES, lang):
                pdf.para(para, 8)
            shown = False           # the summary explains itself; no extra note needed
        for key, title in (("transcript_words", "Слова речи, повлиявшие на оценку"),
                           ("behavior_words", "Слова описания поведения, повлиявшие на оценку")):
            if key in rw_all:
                continue
            elif key in explanation:        # explanation without the readable lists (old run): English tokens
                pdf.set_font("ui", "B", 9); pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
                for k, d in explanation[key]["per_output"].items():
                    pdf.para(f"{TITLES.get(k, k)}: " + ", ".join(clean_word(w["word"]) for w in d["top_words"][:6]), 8)
                shown = True
        if shown:
            pdf.para(WORDS_NOTE, 7)

    # ---- texts
    if report.get("behavior_description"):
        if report.get("behavior_description_ru"):
            pdf.h2("9. Описание поведения")
            pdf.para("Описание строит видеоязыковая модель по кадрам каждого сегмента.", 7)
            pdf.para(report["behavior_description_ru"], 9)
        else:
            pdf.h2("9. Описание поведения (видеоязыковая модель)")
            pdf.para(report["behavior_description"], 9)
    if report.get("transcript"):
        pdf.h2("10. Транскрипт речи")
        pdf.para(report["transcript"], 9)

    pdf.h2("Ограничения")
    pdf.para(DISCLAIMER_RU, 8)
    pdf.para(INTERVIEW_DISCLAIMER_RU, 8)
    out_path = Path(out_path)
    pdf.output(str(out_path))
    return str(out_path)
