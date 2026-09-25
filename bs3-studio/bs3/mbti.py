"""Big Five in MBTI notation (design 5, 7).

The customer's formula is kept (BORDERLINE = 0.15, confidence = |v − thr| / 0.5, X on a borderline axis in `type`,
four letters in `type_strict`), but in the default method "position" it is applied to the position p of a score in the
reference group of the same system (refnorms), not to the raw score: the raw scales of the two systems are not
comparable, and with the fixed threshold 0.5 on raw scores every Russian video got the same type. Method "raw" (p =
the raw score clipped to [0, 1], thresholds `raw_thresholds`) is exactly the customer's document; it is kept for tests,
the BFI-2 check and calibration.

Deliberate deviations from the document: p instead of the raw score; the confidence denominator for a threshold other
than 0.5 (thr below it, 1 − thr above it, so both ends reach 1); the difference is rounded to 9 digits so that
0.65 − 0.5 is not 0.15000000000000002; a missing trait gives the letter X with `missing: true` in both `type` and
`type_strict`; a value outside [0, 1] is clipped with `clipped: true`.

Nothing here writes to disk: `get_mbti(rep, view)` returns the section saved by the pipeline (schema 1) or computes it
on the fly; the caller never stores the computed one.
"""
from __future__ import annotations

import copy
import datetime as _dt
import itertools
import json
import math
from collections import Counter
from importlib import resources

from . import PRODUCT, __version__, refnorms
from .norms import TRAIT_KEYS
from .scores import level_phrase, plural_ru

AXES = ("EI", "SN", "TF", "JP")
AXIS_LABEL = {"EI": "E–I", "SN": "S–N", "TF": "T–F", "JP": "J–P"}
SOURCE_OF = {"oceanai": "ocean_ai", "mm": "own_model", "mean": "mean"}
SYSTEM_OF = {v: k for k, v in SOURCE_OF.items()}
SOURCE_RU = {"ocean_ai": "OCEAN-AI", "own_model": "своя модель", "mean": "среднее двух систем"}
WORD_CLEAR, WORD_MODERATE, WORD_BORDER, WORD_MISSING = "отчётливо", "умеренно", "на границе", "нет данных"
RELIABILITY_BASIS = ("соответствие шкал MBTI и NEO-PI в самоотчётах (McCrae, Costa, 1989); "
                     "не точность оценки по видео")
NEURO_NOTE = "шкала не имеет соответствия в MBTI, приводится отдельно"
SIGN = {"agree": "=", "differ": "≠", "border": "≈"}

_cfg: dict | None = None


def load_config() -> dict:
    """config/mbti.json (a copy: callers may change it for an experiment without touching the cache)."""
    global _cfg
    if _cfg is None:
        with resources.files("bs3").joinpath("config/mbti.json").open("r", encoding="utf-8") as f:
            _cfg = json.load(f)
    return copy.deepcopy(_cfg)


def _num(x) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def word_for(axis: dict) -> str:
    """Confidence of an axis in words: «отчётливо» (>= 0.7), «умеренно», «на границе» (X), «нет данных»."""
    if axis.get("missing"):
        return WORD_MISSING
    if axis.get("borderline"):
        return WORD_BORDER
    return WORD_CLEAR if round(axis.get("confidence") or 0.0, 9) >= 0.7 else WORD_MODERATE


# --------------------------------------------------------------------------------------------------------- core ---

def alternatives(type_strict: str, axes: dict, cfg: dict | None = None) -> list[str]:
    """All strict types obtained by flipping the borderline axes (missing ones are not flipped), except type_strict
    itself, in the axis order EI, SN, TF, JP: 1 X -> 1 type, 2 -> 3, 3 -> 7, 4 -> 15."""
    ax_cfg = (cfg or load_config())["axes"]
    options = []
    for i, ax in enumerate(AXES):
        a = axes.get(ax) or {}
        cur = type_strict[i]
        if a.get("borderline") and not a.get("missing"):
            _, high, low = ax_cfg[ax]
            options.append([cur, low if cur == high else high])
        else:
            options.append([cur])
    out = ["".join(t) for t in itertools.product(*options)]
    return [t for t in out if t != type_strict]


def bigfive_to_mbti(scores: dict, thresholds: dict | None = None, *, borderline: float = 0.15,
                    cfg: dict | None = None) -> dict:
    """The customer's formula on values in [0, 1] (positions p in the method "position").

    Returns {"type", "type_strict", "type_name", "alternatives", "x_count", "axes": {axis: {"trait", "value",
    "threshold", "letter", "confidence", "borderline"[, "missing"][, "clipped"]}}, "neuroticism": {"value"} | None}.
    """
    cfg = cfg or load_config()
    scores = scores if isinstance(scores, dict) else {}
    thresholds = thresholds or {}
    axes: dict = {}
    loose, strict = [], []
    for ax in AXES:
        trait, high, low = cfg["axes"][ax]
        thr = float(thresholds.get(ax, 0.5))
        v = _num(scores.get(trait))
        if v is None:
            axes[ax] = {"trait": trait, "value": None, "threshold": thr, "letter": None, "missing": True,
                        "confidence": 0.0, "borderline": True}
            loose.append("X")
            strict.append("X")
            continue
        a = {"trait": trait}
        if v < 0.0 or v > 1.0:
            v = min(max(v, 0.0), 1.0)
            a["clipped"] = True
        letter = high if v >= thr else low
        d = round(abs(v - thr), 9)
        denom = thr if v < thr else 1.0 - thr
        conf = min(d / denom, 1.0) if denom > 0 else 1.0
        a.update({"value": v, "threshold": thr, "letter": letter, "confidence": round(conf, 9),
                  "borderline": d < borderline})
        axes[ax] = a
        loose.append("X" if a["borderline"] else letter)
        strict.append(letter)
    type_loose, type_strict = "".join(loose), "".join(strict)
    es = _num(scores.get("emotional_stability"))
    neuro = None if es is None else {"value": round(1.0 - min(max(es, 0.0), 1.0), 9)}
    return {"type": type_loose, "type_strict": type_strict,
            "type_name": None if "X" in type_strict else cfg["type_names_ru"].get(type_strict),
            "alternatives": alternatives(type_strict, axes, cfg), "x_count": type_loose.count("X"),
            "axes": axes, "neuroticism": neuro}


def _thresholds(cfg: dict, system: str, lang: str) -> dict:
    if cfg.get("method") == "raw":
        return dict(cfg.get("raw_thresholds") or {})
    th = cfg.get("thresholds") or {}
    return {**(th.get("default") or {}), **(th.get(f"{system}@{lang}") or {})}


def _r(x, nd):
    return None if x is None else round(x, nd)


def mbti_for(system: str, raw_scores: dict, lang: str, cfg: dict | None = None) -> dict:
    """Type of one system ('oceanai' | 'mm' | 'mean') from its raw Big Five scores, in the format of result.json."""
    cfg = cfg or load_config()
    lang = "ru" if lang == "ru" else "en"
    raw_scores = raw_scores if isinstance(raw_scores, dict) else {}
    method = cfg.get("method", "position")
    thr = _thresholds(cfg, system, lang)
    ref_id = refnorms.reference_for(system, lang)

    def pos(trait):
        v = _num(raw_scores.get(trait))
        if v is None:
            return None
        return refnorms.position(system, lang, trait, v) if method == "position" else v

    ps = {k: pos(k) for k in TRAIT_KEYS}
    core = bigfive_to_mbti(ps, thr, borderline=cfg.get("borderline", 0.15), cfg=cfg)
    axes = {}
    for ax in AXES:
        a = dict(core["axes"][ax])
        trait = a["trait"]
        raw = _num(raw_scores.get(trait))
        if method == "position":
            thr_raw = refnorms.value_at(system, lang, trait, a["threshold"])
        else:
            thr_raw = a["threshold"]
        a["value"] = _r(a["value"], 3)
        a["raw_score"] = _r(raw, 3)
        a["threshold_raw"] = _r(thr_raw, 3)
        a["confidence"] = round(a["confidence"], 2)
        a["word"] = word_for(core["axes"][ax])
        axes[ax] = {k: a[k] for k in ("trait", "value", "raw_score", "threshold", "threshold_raw", "letter",
                                      "confidence", "borderline", "word", "missing", "clipped") if k in a}
    corr = cfg.get("correspondence") or {}
    es_raw, es_p = _num(raw_scores.get("emotional_stability")), ps.get("emotional_stability")
    if es_raw is None or es_p is None:
        neuro, neuro_note = None, None
    else:
        np_ = 1.0 - min(max(es_p, 0.0), 1.0)
        lv = level_phrase(np_)
        neuro = {"value": round(1.0 - es_raw, 3), "position": round(np_, 3), "level": lv, "note": NEURO_NOTE}
        neuro_note = f"Шкала нейротизма ({lv}) в MBTI не выражается, приводится отдельно"
    ref = refnorms.describe(ref_id)
    return {
        "source": SOURCE_OF.get(system, system),
        "type": core["type"], "type_strict": core["type_strict"], "type_name": core["type_name"],
        "alternatives": core["alternatives"], "x_count": core["x_count"],
        "axes": axes,
        "reliability": {ax: f"{corr[ax]['label']} (r≈{corr[ax]['r']})" for ax in AXES if ax in corr},
        "reliability_r": {ax: corr[ax]["r"] for ax in AXES if ax in corr},
        "reliability_basis": RELIABILITY_BASIS,
        "neuroticism": neuro, "neuroticism_note": neuro_note,
        "reference": {k: ref[k] for k in ("id", "kind", "n", "frozen_at", "label_ru")},
    }


def agreement(main: dict, second: dict) -> dict:
    """Axis by axis: "agree" — the same strict letter and neither system on the border; "border" — at least one system
    on the border or without data (a difference there is not a signal); "differ" — different letters, both sure."""
    res = {}
    for ax in AXES:
        a, b = (main.get("axes") or {}).get(ax) or {}, (second.get("axes") or {}).get(ax) or {}
        if a.get("missing") or b.get("missing") or a.get("borderline", True) or b.get("borderline", True):
            res[ax] = "border"
        elif a.get("letter") == b.get("letter"):
            res[ax] = "agree"
        else:
            res[ax] = "differ"
    return {"pair": [main.get("source"), second.get("source")], "axes": res,
            "n_agree": sum(v == "agree" for v in res.values())}


def agreement_line(agr: dict | None) -> str:
    """«Совпадают 0 из 4 осей; расходится E–I; на границе у одной из систем: S–N, T–F, J–P.»"""
    if not agr:
        return ""
    axes = agr.get("axes") or {}
    n = agr.get("n_agree", 0)
    if n == 4:
        return "Обе системы дают один и тот же тип."
    differ = [AXIS_LABEL[a] for a in AXES if axes.get(a) == "differ"]
    border = [AXIS_LABEL[a] for a in AXES if axes.get(a) == "border"]
    parts = [f"{'Совпадает' if n == 1 else 'Совпадают'} {n} из 4 осей"]
    if differ:
        parts.append(("расходится " if len(differ) == 1 else "расходятся ") + ", ".join(differ))
    if border:
        parts.append("на границе у одной из систем: " + ", ".join(border))
    return "; ".join(parts) + "."


# ------------------------------------------------------------------------------------------- segments, section ---

def _segment_scores(t: dict, system: str, main: str) -> dict | None:
    """Scores of `system` on one segment of a clean view (design 5.7): variants of 3.0 jobs first; otherwise the
    segment's own scores, but only for the main system and only where it scored."""
    variants = t.get("variants")
    if isinstance(variants, dict) and system in variants:
        v = variants.get(system)
        return v if isinstance(v, dict) else None
    if system == main and isinstance(t.get("scores"), dict) and not t.get("no_primary"):
        return t["scores"]
    return None


def segment_types(view: dict, system: str, cfg: dict | None = None) -> list[dict]:
    """Type of `system` on every segment, with the same thresholds and reference group as for the whole video."""
    cfg = cfg or load_config()
    meta = view.get("view_meta") or {}
    main, lang = meta.get("main_system"), meta.get("lang", "en")
    out = []
    for t in view.get("timeline") or []:
        if not isinstance(t, dict):
            continue
        e = {"segment": t.get("segment"), "start": t.get("start"), "end": t.get("end")}
        sc = _segment_scores(t, system, main)
        if sc is None:
            e.update({"type": None, "reason": "no_primary" if system == main else "no_data"})
        else:
            m = mbti_for(system, sc, lang, cfg)
            e.update({"type": m["type"], "type_strict": m["type_strict"],
                      "words": {ax: m["axes"][ax]["word"] for ax in AXES}})
        out.append(e)
    return out


def stability(entries: list[dict], type_strict: str, cfg: dict | None = None) -> tuple[dict | None, list]:
    """({axis: {"same": k, "of": n}} or None when fewer than `min_segments_for_stability` segments were typed,
    modal types [[type, count], ...] — up to three most frequent strict types)."""
    cfg = cfg or load_config()
    typed = [e for e in entries if e.get("type_strict")]
    counts = Counter(e["type_strict"] for e in typed)
    first = {}
    for i, e in enumerate(typed):
        first.setdefault(e["type_strict"], i)
    modal = [[t, c] for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], first[kv[0]]))[:3]]
    n = len(typed)
    if n < int(cfg.get("min_segments_for_stability", 4)) or not type_strict:
        return None, modal
    st = {ax: {"same": sum(e["type_strict"][i] == type_strict[i] for e in typed), "of": n}
          for i, ax in enumerate(AXES)}
    return st, modal


def is_stable(item: dict | None, cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    return bool(item and item.get("of")) and item["same"] / item["of"] >= float(cfg.get("stable_share", 0.75))


def _scores_of(view: dict, system: str) -> dict | None:
    if system == (view.get("view_meta") or {}).get("main_system"):
        tr = view.get("traits") or {}
        sc = {k: (tr.get(k) or {}).get("score") for k in TRAIT_KEYS}
    else:
        sc = (view.get("variant_scores") or {}).get(system)
    if not isinstance(sc, dict) or all(_num(sc.get(k)) is None for k in TRAIT_KEYS):
        return None
    return {k: sc.get(k) for k in TRAIT_KEYS}


def _with_timeline(view: dict, system: str, item: dict, cfg: dict, always: bool) -> dict:
    entries = segment_types(view, system, cfg)
    if not always and not any(e.get("type_strict") for e in entries):
        return item
    st, modal = stability(entries, item["type_strict"], cfg)
    item.update({"stability": st, "modal_types": modal, "timeline": entries})
    return item


def build_section(view: dict, *, computed_at: str | None = None, cfg: dict | None = None) -> dict | None:
    """The `mbti` section of result.json (design 7.1) from a clean view (scores.clean_view); None without Big Five."""
    cfg = cfg or load_config()
    meta = view.get("view_meta")
    if not meta:
        from .scores import clean_view
        view = clean_view(view)
        meta = view["view_meta"]
    main, lang = meta["main_system"], meta["lang"]
    main_scores = _scores_of(view, main)
    if main_scores is None:
        return None
    m = mbti_for(main, main_scores, lang, cfg)
    entries = segment_types(view, main, cfg)
    st, modal = stability(entries, m["type_strict"], cfg)
    timeline = view.get("timeline") or []
    total = len(timeline) or int(view.get("segments") or 1)
    used = sum(1 for e in entries if e.get("type_strict")) if timeline else 1
    if lang == "ru":
        second_systems = [s for s in ("oceanai", "mm") if s != main]
    else:
        second_systems = ["oceanai", "mm"]
    second = []
    for s in second_systems:
        sc = _scores_of(view, s)
        if sc is None:
            continue
        item = {**mbti_for(s, sc, lang, cfg), "role": "second_opinion"}
        second.append(_with_timeline(view, s, item, cfg, always=False))
    agr = None
    if lang == "ru" and second:
        agr = agreement(m, second[0])
    elif lang != "ru" and len(second) == 2:
        agr = agreement(second[0], second[1])
    sec = {
        "schema_version": 1,
        "computed_by": f"{PRODUCT} {__version__}",
        "computed_at": computed_at or _dt.datetime.now().isoformat(timespec="seconds"),
        "method": cfg.get("method", "position"),
        "borderline": cfg.get("borderline", 0.15),
        "source": m["source"], "role": "main",
    }
    sec.update({k: v for k, v in m.items() if k != "source"})
    sec.update({"segments_total": total, "segments_used": used, "stability": st, "modal_types": modal,
                "timeline": entries, "second": second, "agreement": agr, "llm": None})
    return sec


def get_mbti(rep: dict, view: dict | None = None) -> dict | None:
    """The section to show: the one saved in result.json (schema 1) as it is, otherwise computed now from the clean
    view with `computed_on_render: True`. Never changes `rep` and never writes anything."""
    saved = rep.get("mbti") if isinstance(rep, dict) else None
    if isinstance(saved, dict) and saved.get("schema_version") == 1:
        return copy.deepcopy(saved)
    if view is None:
        from .scores import clean_view
        view = clean_view(rep)
    sec = build_section(view)
    return None if sec is None else {**sec, "computed_on_render": True}


# -------------------------------------------------------------------------------------------- short texts ---

def _is_provisional(mb: dict) -> bool:
    return (mb.get("reference") or {}).get("kind") == "provisional"


def _name(t: str | None, cfg: dict | None = None) -> str | None:
    if not t:
        return None
    return (cfg or load_config())["type_names_ru"].get(t)


def type_title(mb: dict) -> str:
    """«Тип MBTI · OCEAN-AI» / «Тип MBTI · своя модель» / «Тип MBTI · среднее двух систем»."""
    return f"Тип MBTI · {SOURCE_RU.get(mb.get('source'), mb.get('source'))}"


def fact_card(mb: dict | None) -> tuple[str, str, str] | None:
    """The first card of «Ключевые факты»: (label, value, note); None when there is no type."""
    if not mb:
        return None
    x = mb.get("x_count", 0)
    strict, name = mb.get("type_strict"), mb.get("type_name")
    notes = []
    if x == 0:
        value = strict
        if name:
            notes.append(f"«{name}»")
    elif x <= 2:
        value = mb.get("type")
        notes.append(f"ближайший {strict}" + (f" «{name}»" if name else ""))
    else:
        value = mb.get("type")
        notes.append(f"тип не выражен: {x} оси из 4 на границе")
    if _is_provisional(mb):
        notes.append("пороги предварительные")
    for s in mb.get("second") or []:
        notes.append(f"{SOURCE_RU.get(s.get('source'), s.get('source'))}: {s.get('type')}")
    return type_title(mb), value, " · ".join(notes)


def _type_words(m: dict) -> str:
    """«EXFJ, ближайший ESFJ «Попечитель», возможен ENFJ» / «ESFJ «Попечитель»» / «XXXJ, тип не выражен …»."""
    x, strict, name, loose = m.get("x_count", 0), m.get("type_strict"), m.get("type_name"), m.get("type")
    q = f" «{name}»" if name else ""
    if x == 0:
        return f"{strict}{q}"
    if x >= 3:
        return f"{loose}, тип не выражен (формально ближайший {strict})"
    alts = m.get("alternatives") or []
    tail = (", возможен " if len(alts) == 1 else ", возможны ") + ", ".join(alts) if alts else ""
    return f"{loose}, ближайший {strict}{q}{tail}"


def journal_lines(mb: dict | None) -> list[str]:
    """Lines about the type for the journal (design 10.6)."""
    if not mb:
        return ["Тип MBTI: не рассчитан (нет оценок Big Five)"]
    head = SOURCE_RU.get(mb.get("source"), mb.get("source"))
    if _is_provisional(mb):
        head += ", пороги предварительные"
    line = f"Тип MBTI ({head}): {_type_words(mb)}"
    if mb.get("neuroticism"):
        line += f"; нейротизм — {mb['neuroticism']['level']}"
    lines = [line]
    agr = mb.get("agreement") or {}
    second = mb.get("second") or []
    for i, s in enumerate(second):
        t = s.get("type")
        if s.get("x_count", 0) == 0:
            words = f"{s.get('type_strict')}"
        elif s.get("x_count", 0) >= 3:
            words = f"{t}, тип не выражен (формально ближайший {s.get('type_strict')})"
        else:
            words = f"{t}, ближайший {s.get('type_strict')}"
        ln = f"Второе мнение MBTI ({SOURCE_RU.get(s.get('source'), s.get('source'))}): {words}"
        if agr and i == len(second) - 1:
            n = agr.get("n_agree", 0)
            marks = ", ".join(f"{AXIS_LABEL[a]} {SIGN[agr['axes'][a]]}" for a in AXES)
            ln += (f"; уверенно {'совпадает' if n == 1 else 'совпадают'} {n} "
                   f"{plural_ru(n, 'ось', 'оси', 'осей')} из 4 ({marks})")
        lines.append(ln)
    return lines
