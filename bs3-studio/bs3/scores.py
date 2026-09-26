"""One layer of clean scores for everything BS Profiler 3.1 shows (design 6.1, 5.2; changes of 2026-09-26).

`clean_view(rep)` returns a deep copy of a result.json in the same shape (the original is not changed and nothing is
written), in which
- the view holds ONE model (3.1): the one recorded in result.json (`model.selected` of a 3.1 job, `model.primary` of a
  3.0 or an imported 2.0 job — OCEAN-AI), see `main_system`; the Big Five scores are that model's own
  (`variant_scores[main]`), never a mix of two scales, and `variant_scores` (whole video and per segment) keeps that
  member only, so an older job that carried two members shows one;
- a segment where the model gave no score has `scores = None` and `no_primary = True` (charts already draw such a
  segment as a gap «нет оценки»), and the spread over segments is recomputed on the remaining ones;
- Russian speech shows the scores themselves only: the percentiles against the pool of processed videos that older
  jobs carry are removed from the view (no group of processed videos is compared with anywhere); an older English job
  keeps the percentile against the First Impressions V2 norms (6000 clips of that dataset, not our videos);
- `view_meta` says which model the view shows (`main_system`, `selected`, `selected_title`) and which segments were
  left out.

Levels and MBTI letters use the absolute score of the system on [0, 1] (the customer's formula: threshold 0.5,
borderline zone |v − 0.5| < 0.15). The five level bands are aligned with that zone, so a value with a confident
letter is never «средний уровень» and a value in the borderline zone always is. The report prints a score with two
decimals, and the bands and letters are decided on exactly that printed value (`shown`): the view keeps the scores
unrounded and they are rounded once, here, so a printed 0.65 never gets two readings and the same score never prints
two ways:
d = v − 0.5: d >= 0.30 high, 0.15 <= d < 0.30 above, |d| < 0.15 mid, −0.30 < d <= −0.15 below, d <= −0.30 low
(v >= 0.80 high, 0.65 <= v < 0.80 above, 0.35 < v < 0.65 mid, 0.20 < v <= 0.35 below, v <= 0.20 low).
"""
from __future__ import annotations

import copy
import math
import statistics

from . import DEFAULT_MODEL, MODEL_TITLES
from .norms import RU_NAMES, TRAIT_KEYS

# the band edges, in one place: distance of the score from the middle of the scale 0.5
MIDDLE = 0.5
MID_HALF_WIDTH = 0.15          # = the MBTI borderline zone (config/mbti.json "borderline")
HIGH_DISTANCE = 0.30           # v >= 0.80 high, v <= 0.20 low
LEVELS = ("high", "above", "mid", "below", "low")
LEVELS_RU = {"high": "высокий уровень", "above": "выше среднего", "mid": "средний уровень",
             "below": "ниже среднего", "low": "низкий уровень"}
RELATIVE_KEYS = ("percentile", "percentile_ref", "percentile_vs_fiv2", "position")
SOURCE_OF = {"oceanai": "ocean_ai", "mm": "own_model"}
FALLBACK_ORDER = ("oceanai", "mm")

__all__ = ["clean_view", "segment_ok", "level", "level_phrase", "score_text", "shown", "LEVELS_RU", "plural_ru",
           "main_system", "recorded_model", "data_json"]


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


def recorded_model(rep: dict) -> str | None:
    """The model result.json names: `model.selected` (3.1), else `model.primary` (3.0 and imported 2.0 jobs:
    OCEAN-AI), else a single-model backend of a CLI report; None when the job names none."""
    model = rep.get("model") or {}
    for key in ("selected", "primary", "backend"):
        if model.get(key) in MODEL_TITLES:
            return model[key]
    return None


def main_system(rep: dict) -> tuple[str, bool]:
    """(the model the view shows, primary_missing): the recorded model (`recorded_model`) when it produced scores, or
    when the job has no per-member scores at all (a single-model report keeps its traits); otherwise the first of
    oceanai, mm that did, with primary_missing = True. A job that names no model is read as OCEAN-AI (3.1 shows one
    model; there is no mean of two systems any more)."""
    recorded = recorded_model(rep)
    var = rep.get("variant_scores") or {}
    if recorded and (_has_scores(var.get(recorded)) or not any(_has_scores(v) for v in var.values())):
        return recorded, False
    for s in FALLBACK_ORDER:
        if _has_scores(var.get(s)):
            return s, recorded is not None
    return recorded or DEFAULT_MODEL, recorded is not None


def segment_ok(rep: dict, t: dict) -> bool:
    """Did the shown model score this segment? Jobs of 3.x say so in `primary_used`; older ones in `members_used`."""
    main, _ = main_system(rep)
    if "primary_used" in t and main == (rep.get("model") or {}).get("primary"):
        return t.get("primary_used") is not None
    variants = t.get("variants")
    if isinstance(variants, dict) and main in variants:
        return _has_scores(variants.get(main))
    members = t.get("members_used")
    if members is None and variants is None:
        return isinstance(t.get("scores"), dict)      # a job without per-member records: its scores are the model's
    return main in (members or [])


def shown(v) -> float | None:
    """The score as the report prints it and decides on it: rounded once to two decimals (None for a missing one)."""
    x = _num(v)
    return None if x is None else round(x, 2)


def level(v) -> str | None:
    """Level band of a score v on [0, 1], decided on its printed value `shown(v)` (None for a missing score); see
    the module docstring."""
    x = shown(v)
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
    """The score as the page prints it: two decimals («0.73»), the same number as `shown(v)`."""
    x = shown(v)
    return None if x is None else f"{x:.2f}"


def clean_view(rep: dict) -> dict:
    """Deep copy of `rep` with clean main scores and gaps for segments without the main system (see the module
    docstring). Idempotent: clean_view(clean_view(rep)) == clean_view(rep)."""
    view = copy.deepcopy(rep)
    model = view.get("model") or {}
    lang = "ru" if model.get("lang", "ru") == "ru" else "en"
    main, primary_missing = main_system(view)
    var = view.get("variant_scores") or {}
    traits = view.get("traits")
    if not isinstance(traits, dict):
        traits = view["traits"] = {}

    # 1-2. the scores = the shown model's own means
    if _has_scores(var.get(main)):
        for k in TRAIT_KEYS:
            v = _num(var[main].get(k))
            if v is None:
                continue
            t = traits.setdefault(k, {"name_ru": RU_NAMES[k]})
            t["score"] = v                  # unrounded: rounded once, at display (shown)

    # 3. segments without the shown model: gaps (decided before the other members are dropped from the segments)
    timeline = view.get("timeline") or []
    dropped = []
    for t in timeline:
        if not isinstance(t, dict):
            continue
        if not segment_ok(view, t):
            dropped.append(t.get("segment"))
            t["scores"] = None
            t["no_primary"] = True
    kept = [t for t in timeline if isinstance(t, dict) and isinstance(t.get("scores"), dict)]

    # 3a. one model in the view (3.1): the other members an older job carried are left out, whole video and segments
    if isinstance(view.get("variant_scores"), dict):
        view["variant_scores"] = {m: v for m, v in view["variant_scores"].items() if m == main}
    for t in timeline:
        if isinstance(t, dict) and isinstance(t.get("variants"), dict):
            t["variants"] = {m: v for m, v in t["variants"].items() if m == main}

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
        "selected": main, "selected_title": MODEL_TITLES.get(main, main),
        "segments_total": len(timeline), "segments_used": len(used),
        "segments_without_primary": dropped, "primary_missing": bool(primary_missing),
    }
    return view


SYSTEM_GEN = {"oceanai": "OCEAN-AI", "mm": "модели AMLAI 1.0"}
SECOND_SCALE_RU = ("Второе мнение — своя модель, обученная на англоязычных роликах First Impressions V2; у неё своя "
                   "шкала, поэтому оценки двух систем не усредняются.")


def gap_sentence(second: dict, main: dict, second_sys: str = "mm", main_sys: str = "oceanai") -> str:
    """«На этой записи оценки своей модели в среднем на 0.35 ниже, чем у OCEAN-AI.»: the mean difference of the two
    systems over the five traits of this recording only (no rule drawn from other videos); '' when a score is missing
    or the difference prints as 0.00. `second` / `main`: {trait: score} or {trait: {"score": …}}."""
    def val(d, k):
        x = (d or {}).get(k)
        return _num(x.get("score") if isinstance(x, dict) else x)
    diffs = [(val(second, k), val(main, k)) for k in TRAIT_KEYS]
    if any(a is None or b is None for a, b in diffs):
        return ""
    d = sum(a - b for a, b in diffs) / len(diffs)
    if round(abs(d), 2) == 0:
        return ""
    return (f"На этой записи оценки {SYSTEM_GEN.get(second_sys, second_sys)} в среднем на {abs(d):.2f} "
            f"{'ниже' if d < 0 else 'выше'}, чем у {SYSTEM_GEN.get(main_sys, main_sys)}.")


LEGACY_KEYS = ("narrative",)         # the 2.0 plain-language summary; 3.0 shows «Как получены оценки» instead


def data_json(rep: dict) -> tuple[dict, bool]:
    """(what the tab «Данные» shows, whether anything was left out): a deep copy of result.json; for Russian speech
    without the percentile keys of step 5 of clean_view (older jobs carry percentiles against the pool of processed
    videos) and without the stored 2.0 `narrative` built from them. The file on disk is not changed."""
    data = copy.deepcopy(rep)
    if (data.get("model") or {}).get("lang") != "ru":
        return data, False
    dropped = False
    traits = data.get("traits") if isinstance(data.get("traits"), dict) else {}
    for t in [traits.get(k) for k in TRAIT_KEYS] + [data.get("interview")]:
        if isinstance(t, dict):
            for key in RELATIVE_KEYS:
                if key in t:
                    t.pop(key)
                    dropped = True
    for key in LEGACY_KEYS:
        if key in data:
            data.pop(key)
            dropped = True
    return data, dropped
