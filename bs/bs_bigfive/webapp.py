"""Web UI (Gradio): upload a video -> Big Five scores, percentiles, interview score, behaviour description,
transcript, key frames and modality contributions.

Run:  bs web [--port 7860] [--members oceanai,mm]      (inside WSL; open http://localhost:7860 on Windows)
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

from .norms import RU_NAMES, TRAIT_KEYS, percentile
from .charts import frames_html as _charts_frames_html, traits_timeline_html as _charts_traits_html
from .palette import (BUTTON_PRIMARY, BUTTON_PRIMARY_HOVER, BUTTON_STOP, BUTTON_STOP_HOVER, MUTED_OPACITY, OUTLINE,
                      SECOND_OPINION, SKIP_TEXT, STATUS, SUBDUED_TEXT_LIGHT, TABLE_RULE, TRAIT_COLORS)
from .report import DISCLAIMER_RU, INTERVIEW_DISCLAIMER_RU, build_report, clean_word, fmt_secs, seg_label

log = logging.getLogger("bs.web")

TRAIT_TITLES = {
    "openness": "Открытость опыту",
    "conscientiousness": "Добросовестность",
    "extraversion": "Экстраверсия",
    "agreeableness": "Доброжелательность",
    "emotional_stability": "Эмоциональная стабильность",
    "interview": "Впечатление «пригласить на собеседование»",
}
MEMBER_TITLES = {"oceanai": "OCEAN-AI", "mm": "Своя модель (MM-PSYCHE)", "scene": "SSL-MEPR сцена",
                 "face": "лицо", "audio": "голос (CLAP)", "audio_whisper": "голос (Whisper)", "audio_xlsr": "голос (XLS-R)",
                 "audio_w2v_emo": "голос (wav2vec2)", "text": "речь", "behavior": "описание поведения"}
# two-line column headers for narrow tables (long Russian words do not wrap by themselves)
TRAIT_TITLES_2L = {"openness": "Открытость<br>опыту", "conscientiousness": "Добросо-<br>вестность",
                   "extraversion": "Экстра-<br>версия", "agreeableness": "Доброжела-<br>тельность",
                   "emotional_stability": "Эмоц.<br>стабильность", "interview": "Собесе-<br>дование"}
# short trait names for plain-text lists (the members textbox); cutting TRAIT_TITLES gave «Эмоциональна»
TRAIT_SHORT = {"openness": "открытость", "conscientiousness": "добросовестность", "extraversion": "экстраверсия",
               "agreeableness": "доброжелательность", "emotional_stability": "эмоц. стабильность"}
# Tables: Gradio's prose CSS draws every cell border in the full text colour (a heavy grid on the dark theme), so the
# cells get thin neutral rules instead; the first column and the header row stay visible while the table scrolls.
RULE = f"border:0;border-bottom:1px solid {TABLE_RULE}"
# whole-pixel line heights: with fractional row heights the opaque sticky cells of the first column are snapped
# differently from the scrolling cells and cover the rule of the row above (seen at phone width)
TH = f"padding:3px 6px;font-size:12px;line-height:14px;text-align:center;vertical-align:bottom;{RULE}"
TD = f"text-align:center;padding:3px 6px;line-height:18px;{RULE}"
STICKY_COL = "position:sticky;left:0;z-index:1;background:var(--block-background-fill);text-align:left"
STICKY_HEAD = "position:sticky;top:0;z-index:2;background:var(--block-background-fill)"
TD_FIRST = f"padding:3px 8px 3px 6px;white-space:nowrap;line-height:18px;{RULE};{STICKY_COL}"
# long row names («Впечатление «пригласить на собеседование»», «Своя модель (MM-PSYCHE)») may wrap: kept on one
# line, the sticky column took almost the whole block on a phone and the values scrolled away under it;
# the minimum width keeps whole words together («OCEAN-AI», not «OCEAN-» / «AI»)
TD_FIRST_WRAP = TD_FIRST.replace("white-space:nowrap", "white-space:normal") + ";min-width:7.5em"
TH_FIRST = f"{TH};{STICKY_COL};z-index:3"
NOTE = f"font-size:13px;opacity:{MUTED_OPACITY}"            # notes under bars and tables: dimmed theme text, not grey
TRACK = f"border:1px solid {OUTLINE};box-sizing:border-box;overflow:hidden"   # hollow bar track, visible on both themes
TABLE = "border:0;border-collapse:separate;border-spacing:0;font-size:13px;margin:0"
# no single red reaches 4.5:1 on both white and #27272a, so the class switches with Gradio's `dark` class;
# !important because gr.HTML content sits under `.gradio-container .prose *{color:var(--body-text-color)}`
SKIP_STYLE = (f"<style>.bs-skip{{color:{SKIP_TEXT['light']}!important;font-style:italic}}"
              f".dark .bs-skip{{color:{SKIP_TEXT['dark']}!important}}</style>")


def _scroll(table: str, max_height: int | None = None) -> str:
    """Horizontal scroll on narrow screens (the sticky first column stays in view); optional height limit."""
    mh = f"max-height:{max_height}px;" if max_height else ""
    return f"<div style='overflow:auto;max-width:100%;{mh}'>{table}</div>"


def _mid_sentence(name: str) -> str:
    """«Своя модель (MM-PSYCHE)» -> «своя модель (MM-PSYCHE)»; acronyms («OCEAN-AI», «SSL-MEPR сцена») are kept."""
    return name[:1].lower() + name[1:] if name[:2].istitle() else name


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


class Engine:
    """Holds the loaded backends; loads lazily on the first request (one at a time)."""

    def __init__(self, members=("oceanai", "mm"), asr_model="openai/whisper-large-v3-turbo", ollama_model="qwen2.5vl:7b",
                 mm_ckpt=None):
        self.members = tuple(members)
        self.asr_model, self.ollama_model, self.mm_ckpt = asr_model, ollama_model, mm_ckpt
        self._lock = threading.Lock()
        self._en = None
        self._ru = None
        # «Остановить обработку»: set by the stop button, checked between segments by the running analysis
        self.stop_event = threading.Event()

    def _mm_cfg(self, lang="en"):
        from .backend_mm import MMConfig
        kw = dict(lang=lang, asr_model=self.asr_model, ollama_model=self.ollama_model)
        if self.mm_ckpt:
            kw["checkpoint"] = self.mm_ckpt
        return MMConfig(**kw)

    def backend(self, lang: str):
        """en: ensemble of OCEAN-AI (FIV2 weights) + own model. ru: OCEAN-AI (MuPTA weights) + own model with the
        transcript translated to English; face, voice and the behaviour description do not depend on language."""
        with self._lock:
            attr = "_ru" if lang == "ru" else "_en"
            if getattr(self, attr) is None:
                from .backend_ensemble import EnsembleBackend, EnsembleConfig
                from .backend_oceanai import BackendConfig
                setattr(self, attr, EnsembleBackend(EnsembleConfig(
                    members=self.members, lang=lang,
                    oceanai_cfg=BackendConfig(lang=lang, asr_model=self.asr_model),
                    mm_cfg=self._mm_cfg(lang))).load())
            return getattr(self, attr)

    def mm_backend(self, lang: str = "en"):
        be = self.backend(lang)
        return be.backends.get("mm") if hasattr(be, "backends") else None

    def analyzer(self, lang: str):
        """Long-video wrapper (segments of 20 s) around the ensemble; one Whisper instance per language."""
        from .longvideo import LongVideoAnalyzer
        key = f"_an_{lang}"
        if getattr(self, key, None) is None:
            setattr(self, key, LongVideoAnalyzer(self.backend(lang), lang=lang, asr_model=self.asr_model))
        return getattr(self, key)


def _pct_phrase(pct, ref: str | None) -> str:
    """Position relative to the reference group in words: «выше, чем у 72% русских роликов». The group is the
    pool of processed videos when `ref` names it («пула …»), otherwise the FIV2 train labels."""
    if pct is None:
        return "положение: пул пока мал"
    group = "русских роликов" if "пула" in (ref or "") else "людей в FIV2"
    p = max(0.0, min(100.0, float(pct)))
    return f"выше, чем у {p:.0f}% {group}" if p >= 50 else f"ниже, чем у {100 - p:.0f}% {group}"


def _has_chart(rep: dict | None) -> bool:
    """The full-width timeline chart is drawn (at least two segments with scores)."""
    return sum(1 for t in ((rep or {}).get("timeline") or []) if t.get("scores")) >= 2


def _bar_html(traits: dict, interview: dict | None, with_chart: bool = False) -> str:
    rows = []
    items = [(k, traits[k]) for k in TRAIT_KEYS] + ([("interview", interview)] if interview else [])
    refs = []
    for k, t in items:
        pct = t.get("percentile", t.get("percentile_vs_fiv2"))
        score = float(t["score"])
        color = TRAIT_COLORS[k]                       # the same colour as the trait's line on the chart
        ref = t.get("percentile_ref", "train FIV2")
        if ref not in refs:
            refs.append(ref)
        txt = f"{score:.2f} · {_pct_phrase(pct, ref)}"
        width = max(2, min(100, score * 100))        # bar length = the score itself (0…1)
        rows.append(
            f"<div style='margin:6px 0'><div style='display:flex;justify-content:space-between;gap:12px;font-size:14px'>"
            f"<b>{TRAIT_TITLES[k]}</b><span style='text-align:right'>{txt}</span></div>"
            f"<div style='{TRACK};border-radius:6px;height:14px'>"
            f"<div style='width:{width:.0f}%;height:100%;background:{color}'></div></div></div>")
    note = ("Длина полоски — оценка от 0 до 1 (вся рамка соответствует 1). Рядом — положение относительно опорной группы: "
            + "; ".join(refs) + ".")
    if with_chart:
        note += " Цвет полоски совпадает с цветом линии этой черты на графике ниже."
    return ("<div style='max-width:640px'>" + "".join(rows) +
            f"<div style='{NOTE};margin-top:8px'>{note}</div></div>")


def _members_html(rep: dict) -> str:
    """Visible block under the main bars: the second opinion (own model on the FIV2 scale) when one member is
    primary, otherwise the members that were averaged."""
    var = rep.get("variant_scores") or {}
    if not var:
        return ""
    primary = (rep.get("model") or {}).get("primary")
    rows = ""
    if primary:
        others = [m for m in var if m != primary]
        if not others:
            return ""
        # «Второе мнение — своя модель (MM-PSYCHE)»: lower-case inside the sentence, names like OCEAN-AI stay as they are
        title = ("Второе мнение — "
                 + ", ".join(f"<span style='white-space:nowrap'>{_mid_sentence(MEMBER_TITLES.get(m, m))}</span>" for m in others)
                 + f"<div style='{NOTE};font-weight:400;margin-top:2px'>Шкала FIV2, положение — относительно train FIV2</div>")
        for m in others:
            for k in TRAIT_KEYS:
                s = float(var[m][k]); pct = percentile(k, s)
                rows += (f"<div style='margin:3px 0'><div style='display:flex;justify-content:space-between;gap:12px;font-size:13px'>"
                         f"<span>{TRAIT_TITLES[k]}</span><span style='text-align:right'>{s:.2f} · {_pct_phrase(pct, 'train FIV2')}</span></div>"
                         f"<div style='{TRACK};border-radius:5px;height:8px'>"
                         f"<div style='width:{max(2, min(100, s * 100)):.0f}%;height:100%;background:{SECOND_OPINION}'></div></div></div>")
        note = (f"Основная оценка выше — {MEMBER_TITLES.get(primary, primary)}. Полоска — оценка от 0 до 1. Второе мнение "
                "считается на другой шкале (модель обучена на англоязычных влогерах FIV2), поэтому его значения на русских "
                "роликах систематически ниже; сравнивать нужно положение относительно группы и порядок черт, а не сами числа.")
    else:
        title = "Участники ансамбля (итог — среднее)"
        head = "".join(f"<th style='{TH}'>{TRAIT_TITLES_2L[k]}</th>" for k in TRAIT_KEYS)
        body = "".join(f"<tr><td style='{TD_FIRST_WRAP}'>{MEMBER_TITLES.get(m, m)}</td>"
                       + "".join(f"<td style='{TD}'>{float(v[k]):.2f}</td>" for k in TRAIT_KEYS)
                       + "</tr>" for m, v in var.items())
        rows = _scroll(f"<table style='{TABLE}'><tr><th style='{TH_FIRST}'>Модель</th>{head}</tr>{body}</table>")
        note = "Обе системы на шкале FIV2; итоговая оценка — их среднее."
    return (f"<div style='max-width:640px;margin-top:14px;padding:10px 12px;border:1px solid {OUTLINE};border-radius:8px'>"
            f"<div style='font-weight:600;font-size:14px;margin-bottom:6px'>{title}</div>{rows}"
            f"<div style='{NOTE};margin-top:6px'>{note}</div></div>")


def _words_text(expl: dict, rep: dict, lang: str, expl_path: Path | None = None) -> str:
    """Readable word attributions (content words, Russian, grouped by direction); computed once and stored in
    explanation.json under "readable_words" so the PDF shows the same lists."""
    from .words import WORDS_NOTE, format_words, readable_words
    rw_all = expl.get("readable_words") or {}
    changed = False
    for key in ("transcript_words", "behavior_words"):
        if key in expl and key not in rw_all:
            ctx = rep.get("transcript_en") if key == "transcript_words" else rep.get("behavior_description")
            src = rep.get("transcript") if (key == "transcript_words" and lang != "en") else None
            try:
                rw_all[key] = readable_words(expl, key, lang, transcript_ru=src, context_en=ctx)
                changed = True
            except Exception as e:  # noqa: BLE001
                log.warning("readable words failed for %s: %s", key, str(e).splitlines()[0][:120])
    if changed:
        expl["readable_words"] = rw_all
        if expl_path is not None:
            try:
                expl_path.write_text(json.dumps(expl, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
    from .narrative import words_sentences
    lines = words_sentences(rw_all, TRAIT_TITLES, lang)
    return ("\n".join(lines).strip() + "\n\n" + WORDS_NOTE) if lines else ""


def _timeline_html(rep: dict) -> str:
    tl = rep.get("timeline") or []
    if not tl:
        return ""
    rows = ""
    rep_i = rep.get("representative_segment")
    keys = TRAIT_KEYS + (["interview"] if any(t.get("scores") and "interview" in t["scores"] for t in tl) else [])
    head = "".join(f"<th style='{TH};{STICKY_HEAD}'>{TRAIT_TITLES_2L.get(k, TRAIT_TITLES[k])}</th>" for k in keys)
    for seg in tl:
        label = seg_label(seg["start"], seg["end"])
        if not seg.get("scores"):
            rows += (f"<tr><td style='{TD_FIRST}'>{label}</td>"
                     f"<td colspan='{len(keys)}' class='bs-skip' style='padding:3px 6px;line-height:18px;{RULE}'>⚠ пропущен: нет лица или речи</td></tr>")
            continue
        cells = "".join(f"<td style='{TD}'>{seg['scores'].get(k, float('nan')):.2f}</td>" for k in keys)
        mark = " ★" if seg["segment"] == rep_i else ""
        rows += f"<tr><td style='{TD_FIRST}'>{label}{mark}</td>{cells}</tr>"
    std = rep.get("scores_std_across_segments") or {}
    # dimmed through an inner span: opacity on the cell itself would also dim its rule
    std_cells = "".join(f"<td style='{TD}'><span style='font-style:italic;opacity:{MUTED_OPACITY}'>±{std.get(k, 0):.2f}</span></td>"
                        for k in keys)
    rows += (f"<tr><td style='{TD_FIRST}'><span style='font-style:italic;opacity:{MUTED_OPACITY}'>разброс</span></td>"
             f"{std_cells}</tr>")
    n = len(tl)
    table = f"<table style='{TABLE}'><tr><th style='{TH_FIRST};{STICKY_HEAD};z-index:3'>Отрезок</th>{head}</tr>{rows}</table>"
    return (SKIP_STYLE
            + "<div style='font-weight:600;font-size:14px;margin:14px 0 4px'>Оценки по отрезкам ролика (★ — отрезок объяснений)</div>"
            + _scroll(table, max_height=560)
            + f"<div style='{NOTE};margin-top:4px'>Ролик {fmt_secs(rep.get('duration_sec', 0))} разбит на {n} "
              f"{_plural(n, 'отрезок', 'отрезка', 'отрезков')}; итоговые оценки — среднее по отрезкам с весом по длительности. "
              "★ — отрезок, по которому построены объяснения; «разброс» — стандартное отклонение оценки между отрезками.</div>")


def _contrib_html(expl: dict | None) -> str:
    if not expl:
        return ""
    ixg = expl["modalities"]["input_x_gradient"]
    mods = list(next(iter(ixg.values())).keys())
    # header cells get the same side padding as the body cells, so names and values line up
    head = "".join(f"<th style='{TH};padding:3px 8px'>{MEMBER_TITLES.get(m, m)[:1].upper() + MEMBER_TITLES.get(m, m)[1:]}</th>"
                   for m in mods)
    body = ""
    for k, row in ixg.items():
        cells = "".join(f"<td style='{TD};padding:4px 8px'>{html.escape(_pct(row[m]['share']))}</td>" for m in mods)
        body += f"<tr><td style='{TD_FIRST_WRAP};padding:4px 8px'>{TRAIT_TITLES.get(k, k)}</td>{cells}</tr>"
    return (_scroll(f"<table style='{TABLE}'><tr><th style='{TH_FIRST};text-align:left;padding:3px 8px'>Черта</th>{head}</tr>"
                    f"{body}</table>")
            + f"<div style='{NOTE};margin-top:4px'>Доля вклада модальности в оценку своей модели (Input×Gradient). "
            "«&lt;1%» — модальность почти не влияет на оценку этого ролика: модель, обученная на FIV2, опирается в основном "
            "на лицо и голос; речь и описание поведения слабо меняют результат.</div>")


def _pct(share: float) -> str:
    v = float(share) * 100
    return "<1%" if v < 0.95 else f"{v:.0f}%"


def run_analysis(engine: Engine, work_dir: Path, video_path: str, lang: str = "en", explain: bool = True,
                 progress=None) -> dict:
    """The whole web request without Gradio: copies the upload into a job folder, runs the backend, optionally
    the explanations, writes result.json and returns everything the UI shows."""
    t0 = time.time()

    def step(frac, desc):
        if progress is not None:
            progress(frac, desc=f"[{fmt_secs(time.time() - t0)}] {desc}")
    engine.stop_event.clear()
    job = work_dir / time.strftime("%Y%m%d_%H%M%S")
    job.mkdir(parents=True, exist_ok=True)
    src = Path(video_path)
    if not src.is_file():
        raise RuntimeError("Файл загрузки не найден (загрузка не завершилась или временный файл удалён). "
                           "Загрузите видео заново и дождитесь конца загрузки перед запуском.")
    local = job / ("input" + src.suffix.lower())
    shutil.copy2(src, local)
    step(0.05, "Загрузка моделей (первый запуск до минуты)")
    be = engine.backend(lang)
    step(0.25, "Лицо, голос, распознавание речи, описание поведения")
    from .longvideo import LongVideoAnalyzer
    analyzer = engine.analyzer(lang)
    res = analyzer.analyze(local, job / "segments", progress=step, should_stop=engine.stop_event.is_set)
    primary = res.get("primary")
    if primary:
        # main score on a non-FIV2 scale (OCEAN-AI MuPTA): percentiles against the pool of processed videos
        from . import pool
        pool.add(local, res["scores"], lang, primary, name=src.name)
    rep = build_report(local, res, backend="ensemble", corpus=be.cfg.corpus, lang=lang,
                       asr_model=engine.asr_model, modalities=tuple(getattr(be.cfg, "members", ("audio", "video", "text"))),
                       pool_lang=lang if primary else None, primary=primary)
    if "variants" in res:
        rep["variant_scores"] = res["variants"]
    if res.get("transcript_en"):
        rep["transcript_en"] = res["transcript_en"]
    rep["duration_sec"] = res.get("duration_sec")
    rep["segments"] = res.get("segments", 1)
    if res.get("timeline"):
        rep["timeline"] = res["timeline"]
        rep["scores_std_across_segments"] = res.get("scores_std")
        rep["representative_segment"] = res.get("representative_segment")
    expl, frames = None, []
    if engine.stop_event.is_set():
        from .longvideo import AnalysisCancelled
        raise AnalysisCancelled("обработка остановлена пользователем")
    if explain and engine.mm_backend(lang) is not None:
        step(0.85, "Объяснения: вклад модальностей и ключевые кадры")
        mmb = engine.mm_backend(lang)
        # for long videos explain the segment whose profile is closest to the overall mean
        if res.get("timeline"):
            seg = res["timeline"][res["representative_segment"] - 1]
            x_video, x_text, x_beh = seg["file"], seg["transcript"], seg.get("behavior_description") or None
        else:
            x_video, x_text, x_beh = local, res.get("transcript", ""), res.get("behavior_description") or None
        expl = mmb.explain_video(x_video, job / "explain", asr=False, transcript=x_text, behavior=x_beh)
        frames = [(p, f"кадр {Path(p).stem.split('_frame')[-1]}") for p in expl.get("frames", {}).get("key_frame_files", [])]
    rep["timings_sec"]["total_wall"] = round(time.time() - t0, 1)
    rep["original_file_name"] = src.name
    try:
        from .media import probe_media
        rep["media"] = probe_media(local)
        rep["media"]["file_name"] = src.name
    except Exception as e:  # noqa: BLE001
        rep["media"] = {"error": str(e)[:200]}
    (job / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    members = rep.get("variant_scores", {})
    member_txt = "\n".join(f"{MEMBER_TITLES.get(m, m)}{' — основная оценка' if m == primary else (' — второе мнение' if primary else '')}: "
                           + ", ".join(f"{TRAIT_SHORT[k]} {v[k]:.2f}" for k in TRAIT_KEYS)
                           for m, v in members.items())
    if primary:
        member_txt += ("\nШкалы разные: OCEAN-AI (MuPTA) обучена на русскоязычных испытуемых, своя модель — на английских "
                       "влогерах FIV2, поэтому её оценки на русских роликах систематически ниже.")
    # interface language: the models work in English (text branch, VLM description, word attribution); for other
    # languages the description and the top words are translated back for display, originals stay in the JSON
    if lang != "en":
        try:
            from .translate import translate_text
            if rep.get("behavior_description"):
                rep["behavior_description_ru"] = translate_text(rep["behavior_description"], "en", lang)
        except Exception as e:  # noqa: BLE001
            log.warning("translation for display failed: %s", str(e).splitlines()[0][:160])
    words_detail = _words_text(expl, rep, lang, job / "explain" / "explanation.json") if expl else ""
    from .narrative import build_narrative
    try:
        words = build_narrative(rep, expl)          # plain-language explanation shown in the main column
    except Exception as e:  # noqa: BLE001
        log.warning("narrative failed: %s", str(e).splitlines()[0][:120])
        words = ""
    transcript_txt = rep.get("transcript", "")      # the English translation used by the text branch stays in result.json
    n_seg = int(rep.get("segments", 1) or 1)
    return {
        "bars_html": (_bar_html(rep["traits"], rep.get("interview"), with_chart=_has_chart(rep))
                      + _members_html(rep) + _timeline_html(rep)),
        "description": rep.get("behavior_description_ru") or rep.get("behavior_description", "") or "(описание не получено)",
        "transcript": transcript_txt,
        "frames": frames,
        "traits_plot": _charts_traits_html(rep),      # full-width Big Five timeline chart (as in BS 2.0)
        "frames_html": _charts_frames_html(rep, frames),
        "contrib_html": _contrib_html(expl),
        "words": words.strip(),
        "words_detail": words_detail.strip(),
        "members": member_txt,
        "timing": f"{fmt_secs(rep['timings_sec']['total_wall'])} (ролик {fmt_secs(rep.get('duration_sec') or 0)}, "
                  f"{n_seg} {_plural(n_seg, 'отрезок', 'отрезка', 'отрезков')}; модели: {be.cfg.corpus})",
        "json": json.dumps(rep, ensure_ascii=False, indent=2),
        "path": str(job / "result.json"),
        "report": rep,
    }


def export_pdf(job_dir: str | Path) -> str:
    """Build the PDF report for a finished job folder (result.json + explain/ + key frames)."""
    from .pdf_report import build_pdf
    job = Path(job_dir)
    rep = json.loads((job / "result.json").read_text(encoding="utf-8"))
    expl_path = job / "explain" / "explanation.json"
    expl = json.loads(expl_path.read_text(encoding="utf-8")) if expl_path.exists() else None
    lang = (rep.get("model") or {}).get("lang", "en")
    if lang != "en" and rep.get("behavior_description") and not rep.get("behavior_description_ru"):
        try:    # jobs analysed before display translation existed: translate now, keep it in the job
            from .translate import translate_text
            rep["behavior_description_ru"] = translate_text(rep["behavior_description"], "en", lang)
            (job / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("display translation for PDF failed: %s", str(e).splitlines()[0][:160])
    if expl:
        _words_text(expl, rep, lang, expl_path)     # makes sure explanation.json carries the readable word lists
    frames = sorted(str(p) for p in (job / "explain").glob("key_*.jpg")) if (job / "explain").exists() else []
    media = rep.get("media")
    if not media or "error" in media:
        from .media import probe_media
        inp = next(job.glob("input.*"), None)
        media = probe_media(inp) if inp else None
        if media:
            media["file_name"] = rep.get("original_file_name") or media["file_name"]
    stem = re.sub(r"[^A-Za-z0-9А-Яа-яЁё._-]+", "_", Path(rep.get("original_file_name") or "video").stem)[:60]
    out = job / f"BigFive_report_{stem}.pdf"
    return build_pdf(rep, out, explanation=expl, media=media, key_frames=frames)


STATUS_LABELS = {"running": "Идёт обработка", "done": "Готово", "stopped": "Остановлено", "error": "Ошибка"}


def _status_html(frac: float, desc: str, kind: str = "running", label: str | None = None) -> str:
    """The progress / result bar above the page (gr.HTML without a container, i.e. on the page background).
    kind: running (orange, width = progress), done (green), stopped / error (red); the last three fill the bar."""
    pct = max(0, min(100, int(round(frac * 100))))
    width = max(2, pct) if kind == "running" else 100      # the bar stays visible even at 0%
    color = STATUS.get(kind, STATUS["running"])
    label = label or STATUS_LABELS.get(kind, "")
    # the analyzer reports «Сегмент 3/29 (0:40–1:00)»; the page calls these parts «отрезки» everywhere
    desc = html.escape(re.sub(r"\bСегмент\b", "Отрезок", str(desc)))
    text = f"{pct}% · {desc}" if kind == "running" else desc       # «34% · [1 мин 21 с] Отрезок 3/29 (0:40–1:00)»
    return (f"<div style='margin:4px 0 8px'><div style='display:flex;justify-content:space-between;gap:12px;"
            f"font-size:14px;margin-bottom:4px'><b>{label}</b><span style='text-align:right'>{text}</span></div>"
            f"<div role='progressbar' aria-valuemin='0' aria-valuemax='100' aria-valuenow='{pct if kind == 'running' else 100}' "
            f"style='{TRACK};border-radius:6px;height:10px'><div style='width:{width}%;height:100%;background:{color};"
            f"transition:width .5s'></div></div></div>")


def _theme():
    """Gradio Default theme with readable buttons and hints: white labels on orange-700 / red-700 (the Default
    orange-500 gives 2.8:1, red-500 3.8:1) and zinc-600 hint text on white (zinc-400 gives 1.9:1)."""
    import gradio as gr
    return gr.themes.Default().set(
        button_primary_background_fill=BUTTON_PRIMARY, button_primary_background_fill_dark=BUTTON_PRIMARY,
        button_primary_background_fill_hover=BUTTON_PRIMARY_HOVER, button_primary_background_fill_hover_dark=BUTTON_PRIMARY_HOVER,
        button_cancel_background_fill=BUTTON_STOP, button_cancel_background_fill_dark=BUTTON_STOP,
        button_cancel_background_fill_hover=BUTTON_STOP_HOVER, button_cancel_background_fill_hover_dark=BUTTON_STOP_HOVER,
        body_text_color_subdued=SUBDUED_TEXT_LIGHT, body_text_color_subdued_dark="*neutral_400",
        # placeholders have their own variable (light zinc-400 1.9:1 on white, dark zinc-500 2.9:1 on the block)
        input_placeholder_color=SUBDUED_TEXT_LIGHT, input_placeholder_color_dark="*neutral_400",
    )


def build_app(engine: Engine, work_dir: Path):
    import gradio as gr

    from .longvideo import AnalysisCancelled

    # outputs after the status block: bars, traits_plot, desc, transcript, gallery, contrib, words, words_detail,
    # members, raw, path, job_state, pdf_btn — must match the `outputs=[...]` list of btn.click below
    N_REST = 13

    def analyze(video, lang, explain):
        """Generator: one status bar at the top is updated once a second while the analysis runs in a thread;
        Gradio's own per-component spinners are switched off (show_progress='hidden')."""
        if not video:
            raise gr.Error("Загрузите видео")
        state = {"frac": 0.0, "desc": "запуск", "t0": time.time()}

        def cb(frac, desc=None, **kw):
            state["frac"] = float(frac)
            state["desc"] = str(desc if desc is not None else kw.get("desc", "")) or state["desc"]

        result: dict = {}

        def work():
            try:
                result["r"] = run_analysis(engine, work_dir, video, lang, explain, progress=cb)
            except BaseException as e:  # noqa: BLE001
                result["e"] = e

        th = threading.Thread(target=work, daemon=True)
        th.start()
        while th.is_alive():
            th.join(1.0)
            yield (_status_html(state["frac"], state["desc"]),) + (gr.update(),) * N_REST
        if "e" in result:
            e = result["e"]
            # the status bar must not stay on the orange «Идёт обработка» after a failure: show the outcome first
            if isinstance(e, AnalysisCancelled):
                yield (_status_html(state["frac"], "по запросу пользователя", kind="stopped"),) + (gr.update(),) * N_REST
                raise gr.Error("Обработка остановлена. Проверьте язык речи и запустите заново.")
            reason = (str(e).strip().splitlines() or [type(e).__name__])[0][:120]
            yield (_status_html(state["frac"], f"обработка прервана: {reason}", kind="error"),) + (gr.update(),) * N_REST
            raise e
        r = result["r"]
        job_dir = str(Path(r["path"]).parent)
        # the run time («6 мин 05 с (ролик 6 мин 10 с, 19 отрезков; модели: …)») is shown on the «Готово» line and kept
        # in the members textbox inside the accordion
        members_txt = (r["members"] + "\n\n" if r["members"] else "") + f"Время обработки: {r['timing']}"
        yield (_status_html(1.0, f"обработано за {r['timing']}", kind="done"),
               r["bars_html"], r["traits_plot"], r["description"], r["transcript"], r["frames_html"], r["contrib_html"],
               r["words"], r["words_detail"], members_txt, r["json"], r["path"], job_dir, gr.update(interactive=True))

    def stop():
        engine.stop_event.set()
        return _status_html(0.0, "по запросу пользователя: текущий отрезок дорабатывается в фоне, затем обработка "
                                 "прервётся (до ~20 с)", kind="stopped")

    def make_pdf(job_dir):
        if not job_dir:
            raise gr.Error("Сначала проанализируйте видео")
        try:
            return export_pdf(job_dir)
        except Exception as e:  # noqa: BLE001
            raise gr.Error(f"Не удалось собрать PDF: {e}")

    with gr.Blocks(title="BS — Big Five по видео", theme=_theme()) as demo:
        job_state = gr.State("")
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("# Big Five по видео\nЗагрузите ролик, где человек говорит в камеру. Результат: пять черт "
                            "личности (первое впечатление), их положение относительно других людей, впечатление «собеседование», описание поведения, "
                            "транскрипт и объяснения.")
            with gr.Column(scale=1, min_width=220):
                pdf_btn = gr.DownloadButton("Экспорт в PDF", variant="primary", interactive=False)
        status = gr.HTML(value="")           # the only progress indicator on the page (elapsed time and % included)
        # every row below is either one full-width block or two equal columns, so nothing sits alone in half a row
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                video = gr.Video(label="Видео", sources=["upload"], height=320)
                lang = gr.Radio(choices=["en", "ru"], value="en", label="Язык речи",
                                info="en: среднее OCEAN-AI (веса FIV2) и своей модели, положение относительно людей FIV2; "
                                     "ru: основная оценка OCEAN-AI (веса MuPTA), положение относительно пула обработанных "
                                     "русских роликов, своя модель — второе мнение и объяснения")
                explain = gr.Checkbox(value=True, label="Объяснения (ключевые кадры, вклад модальностей, слова)")
                with gr.Row():
                    btn = gr.Button("Анализировать", variant="primary")
                    stop_btn = gr.Button("Остановить обработку", variant="stop")
            with gr.Column(scale=1, min_width=320):
                # scores, second opinion and the per-segment timeline; framed like the video block on the left
                bars = gr.HTML(label="Оценки", show_label=True, container=True)
        # Big Five over the video, full width right under the scores (same chart as BS 2.0)
        traits_plot = gr.HTML(label="Big Five по ходу ролика", show_label=True, container=True)
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                # fixed height, scrollable: long texts must not push the blocks below off the screen
                desc = gr.Textbox(label="Описание поведения", lines=8, max_lines=8)
            with gr.Column(scale=1, min_width=320):
                transcript = gr.Textbox(label="Транскрипт речи", lines=8, max_lines=8)
        words = gr.Textbox(label="Пояснение простыми словами", lines=6, max_lines=8)
        # key frames are embedded as images: gr.Gallery relies on Gradio serving files and showed broken images
        gallery = gr.HTML(label="Ключевые кадры", show_label=True, container=True)
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                contrib = gr.HTML(label="Вклад модальностей", show_label=True, container=True)
            with gr.Column(scale=1, min_width=320):
                words_detail = gr.Textbox(label="Слова, повлиявшие на каждую черту (своя модель)", lines=6, max_lines=10)
        with gr.Accordion("Оценки участников ансамбля и полный JSON", open=False):
            members = gr.Textbox(label="Участники ансамбля и время обработки", lines=4, max_lines=8)
            # gr.JSON is avoided: gradio 5.8 / gradio_client fail to build its API schema (additionalProperties=True)
            raw = gr.Code(label="result.json", language="json", lines=20)
            path = gr.Textbox(label="Сохранено в", interactive=False)
        gr.Markdown(f"<small>{DISCLAIMER_RU}<br>{INTERVIEW_DISCLAIMER_RU}</small>")
        run_ev = btn.click(analyze, inputs=[video, lang, explain],
                           outputs=[status, bars, traits_plot, desc, transcript, gallery, contrib, words, words_detail,
                                    members, raw, path, job_state, pdf_btn],
                           show_progress="hidden", api_name=False)
        # the stop button frees the page at once (cancels the queued event) and raises the flag that the running
        # analysis checks between segments, so the GPU is released within one segment
        stop_btn.click(stop, inputs=None, outputs=[status], cancels=[run_ev], show_progress="hidden", api_name=False)
        pdf_btn.click(make_pdf, inputs=[job_state], outputs=[pdf_btn], api_name=False)
    return demo


def main(port: int = 7860, members: str = "oceanai,mm", work_dir: str | None = None, share: bool = False,
         asr_model: str = "openai/whisper-large-v3-turbo", ollama_model: str = "qwen2.5vl:7b", mm_ckpt: str | None = None,
         host: str = "0.0.0.0"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Gradio checks its own URL through httpx; a proxy variable in the shell makes that check fail
    # ("When localhost is not accessible ..."), so keep localhost off any proxy.
    for var in ("no_proxy", "NO_PROXY"):
        cur = os.environ.get(var, "")
        os.environ[var] = ",".join(x for x in [cur, "localhost", "127.0.0.1", "0.0.0.0"] if x)
    wd = Path(work_dir or os.path.expanduser("~/bs/web_jobs"))
    wd.mkdir(parents=True, exist_ok=True)
    engine = Engine(members=tuple(m.strip() for m in members.split(",") if m.strip()), asr_model=asr_model,
                    ollama_model=ollama_model, mm_ckpt=mm_ckpt)
    demo = build_app(engine, wd)
    demo.queue(default_concurrency_limit=1).launch(server_name=host, server_port=port, share=share,
                                                   allowed_paths=[str(wd)], show_error=True, show_api=False)
