"""One layer of clean scores for everything BS Profiler 3.0 shows (design 6.1, 5.2).

`clean_view(rep)` returns a deep copy of a result.json in the same shape (the original is not changed and nothing is
written), in which
- the main Big Five scores are those of the main system only (Russian speech: OCEAN-AI with the MuPTA weights,
  `variant_scores[primary]`), never a mix of two scales;
- a segment where the main system gave no score has `scores = None` and `no_primary = True` (charts already draw such a
  segment as a gap «нет оценки»), and the spread over segments is recomputed on the remaining ones;
- every trait gets its position `p` in the reference group of the main system (refnorms), Russian jobs also the
  percentile of that position with `percentile_ref = "ref:<group id>"`;
- `view_meta` says which system is the main one, the reference group and which segments were left out.

Level bands of p (the same cuts as the MBTI axis zones, so text and letters never contradict each other):
d = p − 0.5: d >= 0.35 high, 0.15 <= d < 0.35 above, |d| < 0.15 mid, −0.35 < d <= −0.15 below, d <= −0.35 low.
"""
from __future__ import annotations

import copy
import math
import statistics

from . import refnorms
from .norms import RU_NAMES, TRAIT_KEYS

LEVELS = ("high", "above", "mid", "below", "low")
LEVELS_RU = {"high": "заметно выше типичного", "above": "выше типичного", "mid": "около типичного",
             "below": "ниже типичного", "low": "заметно ниже типичного"}
SOURCE_OF = {"oceanai": "ocean_ai", "mm": "own_model", "mean": "mean"}
FALLBACK_ORDER = ("oceanai", "mm")

__all__ = ["clean_view", "segment_ok", "level", "level_phrase", "position_phrase", "LEVELS_RU", "plural_ru",
           "main_system"]


def plural_ru(n, one: str, few: str, many: str) -> str:
    """narrative2.plural_ru, imported on use (narrative2 pulls in the chart libraries)."""
    from .narrative2 import plural_ru as _plural
    return _plural(n, one, few, many)


def _num(x) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _has_scores(d) -> bool:
    return isinstance(d, dict) and any(_num(d.get(k)) is not None for k in TRAIT_KEYS)


def main_system(rep: dict) -> tuple[str, bool]:
    """(main system, primary_missing): the configured primary if it produced scores, else the first of oceanai, mm
    that did; 'mean' when the job has no primary (English speech: the mean of the two systems)."""
    primary = (rep.get("model") or {}).get("primary")
    if not primary:
        return "mean", False
    var = rep.get("variant_scores") or {}
    if _has_scores(var.get(primary)):
        return primary, False
    for s in FALLBACK_ORDER:
        if _has_scores(var.get(s)):
            return s, True
    return primary, True


def segment_ok(rep: dict, t: dict) -> bool:
    """Did the main system score this segment? Jobs of 3.0 say so in `primary_used`; older ones in `members_used`."""
    main, _ = main_system(rep)
    if main == "mean":
        return True
    if "primary_used" in t and main == (rep.get("model") or {}).get("primary"):
        return t.get("primary_used") is not None
    variants = t.get("variants")
    if isinstance(variants, dict) and main in variants:
        return _has_scores(variants.get(main))
    return main in (t.get("members_used") or [])


def level(p) -> str | None:
    """Level band of a position p in [0, 1] (None for a missing position)."""
    v = _num(p)
    if v is None:
        return None
    d = round(v - 0.5, 9)
    if d >= 0.35:
        return "high"
    if d >= 0.15:
        return "above"
    if d > -0.15:
        return "mid"
    if d > -0.35:
        return "below"
    return "low"


def level_phrase(p) -> str | None:
    """«заметно выше типичного» … «заметно ниже типичного»."""
    lv = level(p)
    return LEVELS_RU[lv] if lv else None


def position_phrase(p, ref) -> str | None:
    """Where the score lies in words: for the small Russian group «выше, чем у большинства из 13 русских роликов» /
    «ниже, …» / «примерно посередине среди 13 русских роликов» (cuts 65/35); for FIV2 «выше, чем у 72% людей в
    First Impressions V2» / «ниже, чем у …» / «примерно посередине» (45–55%). `ref`: a describe() dict or an id."""
    v = _num(p)
    if v is None:
        return None
    info = ref if isinstance(ref, dict) else refnorms.describe(str(ref))
    if info.get("kind") == "norm":
        pct = int(round(100 * v))
        if 45 <= pct <= 55:
            return "примерно посередине"
        if pct > 55:
            return f"выше, чем у {pct}% {info['group_ru']}"
        return f"ниже, чем у {100 - pct}% {info['group_ru']}"
    d = round(v - 0.5, 9)
    group = info["group_ru"]
    if d >= 0.15:
        return f"выше, чем у большинства из {group}"
    if d <= -0.15:
        return f"ниже, чем у большинства из {group}"
    return f"примерно посередине среди {group}"


def clean_view(rep: dict) -> dict:
    """Deep copy of `rep` with clean main scores, gaps for segments without the main system and positions (see the
    module docstring). Idempotent: clean_view(clean_view(rep)) == clean_view(rep)."""
    view = copy.deepcopy(rep)
    model = view.get("model") or {}
    lang = "ru" if model.get("lang") == "ru" else "en"
    main, primary_missing = main_system(view)
    var = view.get("variant_scores") or {}
    traits = view.get("traits")
    if not isinstance(traits, dict):
        traits = view["traits"] = {}

    # 1-2. main scores = the main system's own means
    if main != "mean" and _has_scores(var.get(main)):
        for k in TRAIT_KEYS:
            v = _num(var[main].get(k))
            if v is None:
                continue
            t = traits.setdefault(k, {"name_ru": RU_NAMES[k]})
            t["score"] = round(v, 4)

    # 3. segments without the main system: gaps
    timeline = view.get("timeline") or []
    dropped = []
    for t in timeline:
        if not isinstance(t, dict):
            continue
        if main != "mean" and not segment_ok(view, t):
            dropped.append(t.get("segment"))
            t["scores"] = None
            t["no_primary"] = True
    kept = [t for t in timeline if isinstance(t, dict) and isinstance(t.get("scores"), dict)]

    # 4. spread over the remaining segments (population SD, as longvideo computes it)
    if dropped:
        std = view.get("scores_std_across_segments")
        keys = list(std) if isinstance(std, dict) and std else list(TRAIT_KEYS)
        new = {}
        for k in keys:
            vals = [x for x in (_num(t["scores"].get(k)) for t in kept) if x is not None]
            new[k] = statistics.pstdev(vals) if vals else 0.0
        view["scores_std_across_segments"] = new

    # 5. positions in the reference group of the main system
    ref_id = refnorms.reference_for(main, lang)
    for k in TRAIT_KEYS:
        t = traits.get(k)
        if not isinstance(t, dict):
            continue
        p = refnorms.position(main, lang, k, t.get("score"))
        t["position"] = None if p is None else round(p, 4)
        if lang == "ru" and p is not None:
            t["percentile"] = round(100 * p, 1)
            t["percentile_ref"] = "ref:" + ref_id

    # 6. what the view is made of
    used = [t for t in timeline if isinstance(t, dict) and isinstance(t.get("scores"), dict)]
    view["view_meta"] = {
        "main_system": main, "main_source": SOURCE_OF.get(main, main), "lang": lang,
        "reference": refnorms.describe(ref_id),
        "segments_total": len(timeline), "segments_used": len(used),
        "segments_without_primary": dropped, "primary_missing": bool(primary_missing),
    }
    return view
