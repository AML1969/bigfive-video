"""HTML helpers for the BS Profiler 3.0 web UI: score bars, second opinion, modality table and the shared table style used by
webapp.py. Forked from bs 1.0 (the unused 1.0 page, engine and run_analysis were removed; the page is webapp.py).

Colours: bars, outlines and rules come from palette.HTML (>= 3:1 on the dark and the light Gradio theme). Text colours
are never hard-coded: Gradio's `.prose *` rule gives the body text colour of the current theme, secondary text is the
same colour at opacity .75 (>= 6:1 on every background). The HTML lives in the page DOM, not in an iframe, so Gradio
CSS variables such as --block-background-fill switch with the theme by themselves (used for the sticky table header).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .narrative2 import plural_ru
from .norms import TRAIT_KEYS, percentile
from .palette import HTML as PAL

log = logging.getLogger("bs3.web")

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
# short row names for tables (the interview title is too long for a first column)
ROW_TITLES = {**TRAIT_TITLES, "interview": "«Собеседование»"}
# names inside the second-opinion title: «Второе мнение: своя модель MM-PSYCHE (шкала First Impressions V2, …)»
SECOND_TITLES = {"mm": "своя модель MM-PSYCHE", "oceanai": "OCEAN-AI", "scene": "сцена SSL-MEPR"}
# modality columns of the contribution table: (column title, second line)
MODALITY_HEADS = {"face": ("Лицо", "кадры"), "audio": ("Голос", "CLAP"), "audio_whisper": ("Голос", "Whisper"),
                  "audio_xlsr": ("Голос", "XLS-R"), "audio_w2v_emo": ("Голос", "wav2vec2"), "text": ("Речь", "текст"),
                  "behavior": ("Поведение", "описание"), "scene": ("Сцена", "SSL-MEPR")}
NOTE = "font-size:13px;opacity:.75;line-height:1.45"           # footnotes and card notes (13 px minimum)
SUB = "display:block;font-size:13px;font-weight:400;opacity:.75"  # second line of a table header (units, model)


# ---------------------------------------------------------------- tables
def th_text(title: str, sub: str | None = None) -> str:
    """Header cell content: title plus an optional smaller second line with the unit or the model."""
    return title + (f"<span style='{SUB}'>{sub}</span>" if sub else "")


def table_html(head: list[str], rows: list[list[str]], *, max_height: int | None = None, wrap_first: bool = False) -> str:
    """Table on a gr.HTML block (container=True).

    Opaque sticky header (theme block colour under a grey tint, so scrolled rows never show through) with a 2 px rule;
    1 px rules between rows; first column = row name, left-aligned and bold; numbers in tabular figures. Rows are told
    apart by rules only, without zebra stripes: the #808080 rule is 3.8:1 on the dark block and 3.9:1 on white, and a
    tint under it would take that down.
    `border-collapse:separate` keeps the header rule attached to the sticky header while scrolling; inline `border:0`
    overrides Gradio's prose grid (a full 1 px grid in the text colour)."""
    rule = PAL["table_rule"]
    th = ("border:0;border-bottom:2px solid " + rule + ";background:linear-gradient(rgba(128,128,128,.18),"
          "rgba(128,128,128,.18)),var(--block-background-fill);font-size:14px;font-weight:600;line-height:1.2;"
          "padding:6px 8px;position:sticky;top:0;z-index:1;vertical-align:bottom;")
    td = ("border:0;border-bottom:1px solid " + rule + ";padding:5px 8px;font-variant-numeric:tabular-nums;"
          "vertical-align:middle;")
    first_td = td + "text-align:left;font-weight:600;" + ("min-width:120px" if wrap_first else "white-space:nowrap")
    other_td = td + "text-align:center;white-space:nowrap"
    head_html = "".join(f"<th style='{th}text-align:{'left' if i == 0 else 'center'}'>{h}</th>" for i, h in enumerate(head))
    body = "".join("<tr style='border:0'>" + "".join(f"<td style='{first_td if i == 0 else other_td}'>{c}</td>"
                                                     for i, c in enumerate(r)) + "</tr>" for r in rows)
    wrap = "overflow:auto" + (f";max-height:{max_height}px" if max_height else "")
    return (f"<div style='{wrap}'><table style='border-collapse:separate;border-spacing:0;border:0;margin:0;font-size:14px'>"
            f"<thead><tr style='border:0'>{head_html}</tr></thead><tbody>{body}</tbody></table></div>")


# ---------------------------------------------------------------- score bars
def _track(value: float, fill: str, height: int, radius: int, tick_pct=None, fill_extra: str = "") -> str:
    """Outlined track (transparent inside, 1 px palette.HTML track_outline) with the fill = score 0…1 and an optional tick
    (3 px, text colour, sticks out 4 px above and below) at the percentile position 0…100%."""
    width = max(0.0, min(100.0, float(value) * 100))
    tick = ""
    if tick_pct is not None:
        p = max(0.0, min(100.0, float(tick_pct)))
        tick = (f"<div style='position:absolute;left:calc({p:.1f}% - 1.5px);top:-4px;bottom:-4px;width:3px;"
                "border-radius:1px;background:currentColor'></div>")
    return (f"<div style='position:relative;height:{height}px;border-radius:{radius}px;background:{PAL['track']};"
            f"box-shadow:inset 0 0 0 1px {PAL['track_outline']}'>"
            f"<div style='width:{width:.1f}%;min-width:3px;height:{height}px;border-radius:{radius}px;background:{fill};"
            f"{fill_extra}'></div>{tick}</div>")


def _score_row(title: str, value: float, text: str, fill: str, tick_pct=None, *, height: int = 14, radius: int = 6,
               bold: bool = True, fill_extra: str = "") -> str:
    # when the name and the text do not fit on one line (long names, narrow screens) the text moves to its own line,
    # right-aligned, instead of being squeezed into a narrow column of 4-5 short lines
    name = f"<b>{title}</b>" if bold else f"<span>{title}</span>"
    return (f"<div style='margin:10px 0 12px'><div style='display:flex;flex-wrap:wrap;justify-content:space-between;"
            f"align-items:baseline;gap:2px 12px;font-size:14px;margin-bottom:6px'>{name} <span style='margin-left:auto;"
            f"text-align:right;font-variant-numeric:tabular-nums'>{text}</span></div>"
            f"{_track(value, fill, height, radius, tick_pct, fill_extra)}</div>")


def _scale_row() -> str:
    """0 / 0.5 / 1 under a group of bars (same width as the tracks), so bar lengths can be read and compared."""
    tick = "position:absolute;top:0;width:1px;height:5px;background:currentColor"
    lab = "position:absolute;top:6px"
    return ("<div aria-hidden='true' style='position:relative;height:24px;margin-top:6px;font-size:13px;opacity:.75;"
            "font-variant-numeric:tabular-nums'>"
            f"<span style='{tick};left:0'></span><span style='{tick};left:50%'></span><span style='{tick};right:0'></span>"
            f"<span style='{lab};left:0'>0</span><span style='{lab};left:50%;transform:translateX(-50%)'>0.5</span>"
            f"<span style='{lab};right:0'>1</span></div>")


def _swatch(fill: str, extra: str = "") -> str:
    return (f"<span aria-hidden='true' style='display:inline-block;width:18px;height:10px;border-radius:3px;"
            f"vertical-align:middle;margin-right:6px;background:{fill};box-shadow:inset 0 0 0 1px {PAL['track_outline']};"
            f"{extra}'></span>")


def _tick_swatch() -> str:
    return ("<span aria-hidden='true' style='display:inline-block;width:3px;height:16px;border-radius:1px;"
            "vertical-align:middle;margin:0 7px 0 7px;background:currentColor'></span>")


def _legend(items: list[str]) -> str:
    return ("<div style='display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;line-height:1.4;margin-top:2px'>"
            + "".join(f"<span>{it}</span>" for it in items) + "</div>")


def _is_fiv2(ref: str | None) -> bool:
    """A percentile against the First Impressions V2 norms (6000 clips of that dataset). Percentiles against the pool
    of processed videos ("пула …") or a group of them ("ref:…") are never shown (change of 2026-09-26)."""
    r = ref or ""
    return "First Impressions V2" in r or "FIV2" in r


def _pct_phrase(pct, ref: str | None) -> tuple[str, bool]:
    """(the FIV2 percentile in words, whether a percentile tick may be drawn): «выше, чем у 72% людей в FIV2»;
    ("", False) for anything that is not a FIV2 percentile (Russian speech shows the score only)."""
    if pct is None or not _is_fiv2(ref):
        return "", False
    p = max(0.0, min(100.0, float(pct)))
    group = "людей в FIV2"
    if 45 <= p <= 55:
        return f"примерно посередине среди {group}", True
    return (f"выше, чем у {p:.0f}% {group}" if p > 50 else f"ниже, чем у {100 - p:.0f}% {group}"), True


def _ref_ru(ref: str) -> str:
    """percentile_ref from result.json (genitive, reads after «относительно») without technical English words."""
    r = re.sub(r",\s*своя модель\s*$", "", ref or "")
    return r.replace("train First Impressions V2", "обучающей выборки First Impressions V2").replace(
        "train FIV2", "обучающей выборки FIV2")


TICK_NOTE = ("Риска на полоске — процентиль в First Impressions V2: у какой доли людей этого датасета оценка ниже; "
             "риска посередине — медиана датасета.")
SCALE_NOTE = "Длина полоски — оценка системы от 0 до 1; уровни черт и буквы MBTI считаются по этой же шкале, середина — 0.5."


def _bar_html(traits: dict, interview: dict | None) -> str:
    items = [(k, traits[k]) for k in TRAIT_KEYS] + ([("interview", interview)] if interview else [])
    groups: dict[str, list[str]] = {}          # FIV2 reference (display text) -> item keys, for the footnote
    any_tick = False
    rows = []
    for k, t in items:
        pct = t.get("percentile", t.get("percentile_vs_fiv2"))
        ref = t.get("percentile_ref", "train FIV2" if pct is not None else "")
        phrase, tick_ok = _pct_phrase(pct, ref)
        if phrase:
            groups.setdefault(_ref_ru(ref), []).append(k)
        any_tick = any_tick or tick_ok
        score = float(t["score"])
        fill = PAL["interview_fill"] if k == "interview" else PAL["main_fill"]
        text = f"{score:.2f}" + (f" · {phrase}" if phrase else "")
        row = _score_row(TRAIT_TITLES[k], score, text, fill, pct if tick_ok else None)
        if k == "interview":           # a separate label of the own model: set off from the five traits
            row = (f"<div style='margin-top:14px;padding-top:2px;border-top:1px dashed {PAL['track_outline']}'>{row}</div>")
        rows.append(row)
    legend = [_swatch(PAL["main_fill"]) + "черта, оценка 0…1"]
    if interview:
        legend.append(_swatch(PAL["interview_fill"]) + "«собеседование», оценка 0…1")
    if any_tick:
        legend.append(_tick_swatch() + "процентиль в First Impressions V2")
    notes = [SCALE_NOTE]
    for ref, keys in groups.items():
        if len(keys) == len(items) and len(groups) == 1:
            subject = "Все процентили"
        elif set(keys) == set(TRAIT_KEYS):
            subject = "Процентили пяти черт"
        elif keys == ["interview"]:
            subject = "Процентиль «собеседования»"
        else:
            subject = "Процентили: " + ", ".join(ROW_TITLES[k] for k in keys)
        notes.append(f"{subject} — относительно {ref}.")
    if any_tick:
        notes.append(TICK_NOTE)
    return ("<div style='max-width:640px'>" + "".join(rows) + _scale_row() + _legend(legend) +
            f"<div style='{NOTE};margin-top:8px'>{' '.join(notes)}</div></div>")


def _members_html(rep: dict) -> str:
    """Framed block under the main bars: the second opinion (own model on the FIV2 scale) when one member is
    primary, otherwise a table of the members that were averaged."""
    var = rep.get("variant_scores") or {}
    if not var:
        return ""
    model = rep.get("model") or {}
    primary = model.get("primary")
    # a clean view (scores.clean_view) names the system its main scores come from: the own model when OCEAN-AI gave none
    main = (rep.get("view_meta") or {}).get("main_system") or primary
    if primary:
        others = [m for m in var if m != main]
        if not others:
            return ""
        # Russian speech: the score only (no percentile of any group); otherwise the FIV2 percentile as in 2.0
        ru = model.get("lang") == "ru"
        where = (" (своя шкала, обучена на First Impressions V2)" if ru
                 else " (шкала First Impressions V2, сравнение с людьми из этого датасета)")
        title = "Второе мнение: " + ", ".join(SECOND_TITLES.get(m, MEMBER_TITLES.get(m, m)) for m in others) + where
        rows = ""
        any_tick = False
        for m in others:
            if len(others) > 1:
                rows += f"<div style='font-weight:600;font-size:14px;margin-top:8px'>{MEMBER_TITLES.get(m, m)}</div>"
            for k in TRAIT_KEYS:
                s = float(var[m][k])
                pct = None if ru else percentile(k, s)
                phrase, tick_ok = _pct_phrase(pct, "train FIV2")
                any_tick = any_tick or tick_ok
                # neutral fill (theme text colour at .55): blue stays reserved for the main score, as on the radar
                text = f"{s:.2f}" + (f" · {phrase}" if phrase else "")
                rows += _score_row(TRAIT_TITLES[k], s, text, PAL["second_fill"], pct if tick_ok else None,
                                   height=8, radius=5, bold=False, fill_extra="opacity:.55")
        legend = [_swatch(PAL["second_fill"], "opacity:.55") + "второе мнение, оценка 0…1"]
        if any_tick:
            legend.append(_tick_swatch() + "процентиль в First Impressions V2")
        body = rows + _scale_row() + _legend(legend)
        from .scores import SECOND_SCALE_RU, gap_sentence
        gap = gap_sentence(var[others[0]], var[main], others[0], main) if len(others) == 1 and main in var else ""
        note = (f"Основная оценка ({MEMBER_TITLES.get(main, main)}) — в полосках над этой рамкой. "
                + (SECOND_SCALE_RU if others == ["mm"] else
                   "Второе мнение считается на другой шкале, поэтому оценки двух систем не усредняются.")
                + (f" {gap}" if gap else ""))
    else:
        title = "Участники ансамбля: итоговая оценка — их среднее"
        head = ["Модель"] + [TRAIT_TITLES_2L[k] for k in TRAIT_KEYS]
        table_rows = [[MEMBER_TITLES.get(m, m)] + [f"{float(v[k]):.2f}" for k in TRAIT_KEYS] for m, v in var.items()]
        body = table_html(head, table_rows)
        who = "Обе модели" if len(var) == 2 else "Все модели"
        note = f"{who} на шкале FIV2, оценки от 0 до 1; итог в полосках выше — их среднее."
    return (f"<div style='max-width:640px;margin-top:18px;padding:10px 12px 8px;border:1px solid {PAL['card_border']};"
            f"border-radius:8px'><div style='font-weight:600;font-size:15px;margin-bottom:4px'>{title}</div>{body}"
            f"<div style='{NOTE};margin-top:8px'>{note}</div></div>")


def _words_text(expl: dict, rep: dict, lang: str, expl_path: Path | None = None) -> str:
    """Readable word attributions (content words in Russian for any speech language, grouped by direction); computed
    once and stored in explanation.json under "readable_words" so the PDF shows the same lists."""
    from .narrative import words_summary
    from .ru_texts import ensure_words, write_json
    if ensure_words(rep, expl) and expl_path is not None:
        write_json(expl_path, expl)
    return "\n\n".join(words_summary(expl.get("readable_words") or {}, expl, TRAIT_TITLES, lang))


# ---------------------------------------------------------------- modality contributions
def _pct(share: float) -> str:
    v = float(share) * 100
    return "&lt;1%" if v < 0.95 else f"{v:.0f}%"


def _share_cell(share: float, top: bool) -> str:
    """Share as text (bold for the largest in the row) with a small outlined bar under it."""
    w = max(0.0, min(100.0, float(share) * 100))
    return (f"<div style='font-weight:{700 if top else 400}'>{_pct(share)}</div>"
            f"<div style='width:56px;height:6px;margin:3px auto 0;border-radius:3px;"
            f"box-shadow:inset 0 0 0 1px {PAL['track_outline']}'><div style='width:{w:.1f}%;height:6px;border-radius:3px;"
            f"background:{PAL['second_fill']};opacity:.55'></div></div>")


def _contrib_html(expl: dict | None) -> str:
    if not expl:
        return ""
    ixg = (expl.get("modalities") or {}).get("input_x_gradient")
    if not ixg:
        return ""
    mods = list(next(iter(ixg.values())).keys())
    head = ["Черта"]
    for m in mods:
        title, sub = MODALITY_HEADS.get(m, (MEMBER_TITLES.get(m, m)[:1].upper() + MEMBER_TITLES.get(m, m)[1:], None))
        head.append(th_text(title, sub))
    rows = []
    for k, row in ixg.items():
        shares = [float(row[m]["share"]) for m in mods]
        top = max(shares) if shares else 0.0
        rows.append([ROW_TITLES.get(k, k)] + [_share_cell(s, s == top) for s in shares])
    return (f"<div style='{NOTE};margin-bottom:8px'>Какая доля оценки своей модели пришлась на каждую модальность "
            "(по градиенту оценки: насколько признаки каждой модальности сдвигают результат). В каждой строке доли в "
            "сумме дают 100%; самая большая выделена жирным.</div>"
            + table_html(head, rows, wrap_first=True) +
            f"<div style='{NOTE};margin-top:8px'>«&lt;1%» — модальность почти не влияет на оценку этого ролика: модель, "
            "обученная на FIV2, опирается в основном на лицо и голос; речь и описание поведения слабо меняют результат.</div>")
