"""One layer of clean scores for everything BS Profiler 3.0 shows (design 6.1, 5.2; change of 2026-09-26).

`clean_view(rep)` returns a deep copy of a result.json in the same shape (the original is not changed and nothing is
written), in which
- the main Big Five scores are those of the main system only (Russian speech: OCEAN-AI with the MuPTA weights,
  `variant_scores[primary]`), never a mix of two scales;
- a segment where the main system gave no score has `scores = None` and `no_primary = True` (charts already draw such a
  segment as a gap «нет оценки»), and the spread over segments is recomputed on the remaining ones;
- Russian speech shows the scores themselves only: the percentiles against the pool of processed videos that older
  jobs carry are removed from the view (no group of processed videos is compared with anywhere); English speech
  keeps the percentile against the First Impressions V2 norms (6000 clips of that dataset, not our videos);
- `view_meta` says which system is the main one and which segments were left out.

Levels and MBTI letters use the absolute score of the system on [0, 1] (the customer's formula: threshold 0.5,
borderline zone |v − 0.5| < 0.15). The five level bands are aligned with that zone, so a value with a confident
letter is never «средний уровень» and a value in the borderline zone always is:
d = v − 0.5: d >= 0.30 high, 0.15 <= d < 0.30 above, |d| < 0.15 mid, −0.30 < d <= −0.15 below, d <= −0.30 low
(v >= 0.80 high, 0.65 <= v < 0.80 above, 0.35 < v < 0.65 mid, 0.20 < v <= 0.35 below, v <= 0.20 low).
"""
from __future__ import annotations

import copy
import math
import statistics

from .norms import RU_NAMES, TRAIT_KEYS

# the band edges, in one place: distance of the score from the middle of the scale 0.5
MIDDLE = 0.5
MID_HALF_WIDTH = 0.15          # = the MBTI borderline zone (config/mbti.json "borderline")
HIGH_DISTANCE = 0.30           # v >= 0.80 high, v <= 0.20 low
LEVELS = ("high", "above", "mid", "below", "low")
LEVELS_RU = {"high": "высокий уровень", "above": "выше среднего", "mid": "средний уровень",
             "below": "ниже среднего", "low": "низкий уровень"}
RELATIVE_KEYS = ("percentile", "percentile_ref", "percentile_vs_fiv2", "position")
SOURCE_OF = {"oceanai": "ocean_ai", "mm": "own_model", "mean": "mean"}
FALLBACK_ORDER = ("oceanai", "mm")

__all__ = ["clean_view", "segment_ok", "level", "level_phrase", "score_text", "LEVELS_RU", "plural_ru",
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


def level(v) -> str | None:
    """Level band of a score v on [0, 1] (None for a missing score); see the module docstring."""
    x = _num(v)
    if x is None:
        return None
    d = round(x - MIDDLE, 9)             # as the borderline test of mbti: 0.65 − 0.5 is 0.15, not 0.15000000000000002
    if d >= HIGH_DISTANCE:
        return "high"
    if d >= MID_HALF_WIDTH:
        return "above"
    if d > -MID_HALF_WIDTH:
        return "mid"
    if d > -HIGH_DISTANCE:
        return "below"
    return "low"


def level_phrase(v) -> str | None:
    """«высокий уровень» / «выше среднего» / «средний уровень» / «ниже среднего» / «низкий уровень»."""
    lv = level(v)
    return LEVELS_RU[lv] if lv else None


def score_text(v) -> str | None:
    """The score as the page prints it: two decimals («0.73»)."""
    x = _num(v)
    return None if x is None else f"{x:.2f}"


def clean_view(rep: dict) -> dict:
    """Deep copy of `rep` with clean main scores and gaps for segments without the main system (see the module
    docstring). Idempotent: clean_view(clean_view(rep)) == clean_view(rep)."""
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

    # 5. Russian speech: the scores only — no percentile against the pool of processed videos (older jobs carry one)
    if lang == "ru":
        items = [traits.get(k) for k in TRAIT_KEYS] + [view.get("interview")]
        for t in items:
            if isinstance(t, dict):
                for key in RELATIVE_KEYS:
                    t.pop(key, None)
    else:
        for k in TRAIT_KEYS:
            if isinstance(traits.get(k), dict):
                traits[k].pop("position", None)

    # 6. what the view is made of
    used = [t for t in timeline if isinstance(t, dict) and isinstance(t.get("scores"), dict)]
    view["view_meta"] = {
        "main_system": main, "main_source": SOURCE_OF.get(main, main), "lang": lang,
        "segments_total": len(timeline), "segments_used": len(used),
        "segments_without_primary": dropped, "primary_missing": bool(primary_missing),
    }
    return view
