"""Reference groups of BS Profiler 3.0: where a score lies among the scores of the same system (design 4.2, 4.3).

- `fiv2` (English speech, any system): p = norms.percentile(trait, score) / 100 — 101 quantiles of the observer labels
  of the First Impressions V2 training split (6000 clips).
- `ru_prov_<date>` (Russian speech, OCEAN-AI and the own model separately): the frozen provisional group
  data/norms_ru_provisional.json (written by scripts/freeze_ru_norms.py). Rank interpolation: sorted values
  v_1 <= ... <= v_N, equal values collapse into one point with their mean rank, points (v_i, rank_i / (N + 1)), linear
  interpolation between them and clamping outside, so p lies in [1/(N+1), N/(N+1)] — never more certain than the group.

Standard library only; nothing is written anywhere.
"""
from __future__ import annotations

import json
import math
from importlib import resources

from .norms import TRAIT_KEYS, percentile

RU_PROV_FILE = "data/norms_ru_provisional.json"
DEFAULT_REFERENCES = {"ru": {"oceanai": "ru_prov", "mm": "ru_prov"},
                      "en": {"oceanai": "fiv2", "mm": "fiv2", "mean": "fiv2"}}
FIV2_N = 6000

_ru_prov: dict | None = None
_cfg_refs: dict | None = None


def load_ru_prov() -> dict:
    """The frozen provisional Russian group (cached)."""
    global _ru_prov
    if _ru_prov is None:
        with resources.files("bs3").joinpath(RU_PROV_FILE).open("r", encoding="utf-8") as f:
            _ru_prov = json.load(f)
    return _ru_prov


def _references() -> dict:
    global _cfg_refs
    if _cfg_refs is None:
        try:
            with resources.files("bs3").joinpath("config/mbti.json").open("r", encoding="utf-8") as f:
                _cfg_refs = json.load(f).get("references") or DEFAULT_REFERENCES
        except (OSError, ValueError):
            _cfg_refs = DEFAULT_REFERENCES
    return _cfg_refs


def _num(x) -> float | None:
    """A finite float, or None for None / NaN / inf / anything that is not a number (bool included)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def rank_points(values) -> list[tuple[float, float]]:
    """(value, mean rank) of the sorted group; equal values share one point with their mean rank (1-based)."""
    vals = sorted(float(v) for v in values)
    pts: list[tuple[float, float]] = []
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        pts.append((vals[i], (i + 1 + j + 1) / 2.0))
        i = j + 1
    return pts


def rank_position(values, score: float) -> float | None:
    """Position of `score` in the group `values`: rank / (N + 1), linear between the points, clamped at the ends."""
    v = _num(score)
    if v is None or not values:
        return None
    pts = rank_points(values)
    n1 = len(values) + 1.0
    if v <= pts[0][0]:
        return pts[0][1] / n1
    if v >= pts[-1][0]:
        return pts[-1][1] / n1
    for (x0, r0), (x1, r1) in zip(pts, pts[1:]):
        if x0 <= v <= x1:
            frac = 0.0 if x1 == x0 else (v - x0) / (x1 - x0)
            return (r0 + frac * (r1 - r0)) / n1
    return pts[-1][1] / n1                                   # not reached


def reference_for(system: str, lang: str) -> str:
    """Id of the reference group of `system` ('oceanai' | 'mm' | 'mean') for speech language `lang`."""
    kind = (_references().get("ru" if lang == "ru" else "en") or {}).get(system)
    if kind is None:
        kind = "ru_prov" if lang == "ru" else "fiv2"
    return load_ru_prov()["id"] if kind == "ru_prov" else "fiv2"


def _is_ru_prov(ref: str) -> bool:
    ref = ref[4:] if ref.startswith("ref:") else ref
    return ref == "ru_prov" or ref.startswith("ru_prov_")


def position(system: str, lang: str, trait: str, score) -> float | None:
    """Position p in [0, 1] of a raw score of `system` among its reference group; None for a missing score."""
    v = _num(score)
    if v is None:
        return None
    ref = reference_for(system, lang)
    if ref == "fiv2":
        if trait not in TRAIT_KEYS:
            return None
        return percentile(trait, v) / 100.0
    group = (load_ru_prov().get("sources") or {}).get(system, {}).get(trait)
    if not group:
        return None
    # compared at the precision the group is stored with (4 digits), so a video of the group lands on its own rank
    return rank_position(group, round(v, 4))


def value_at(system: str, lang: str, trait: str, p: float) -> float | None:
    """Inverse of position(): the raw score whose position is `p` (for p = 0.5 — the median of the group); used to
    show which raw value a threshold on the position scale corresponds to."""
    q = _num(p)
    if q is None:
        return None
    ref = reference_for(system, lang)
    if ref == "fiv2":
        from .norms import load_norms
        qs = load_norms()["quantiles"].get(trait)
        if not qs:
            return None
        x = min(max(q * 100.0, 0.0), 100.0)
        i = min(int(x), 99)
        return qs[i] + (x - i) * (qs[i + 1] - qs[i])
    group = (load_ru_prov().get("sources") or {}).get(system, {}).get(trait)
    if not group:
        return None
    n1 = len(group) + 1.0
    pts = [(v, r / n1) for v, r in rank_points(group)]
    if q <= pts[0][1]:
        return pts[0][0]
    if q >= pts[-1][1]:
        return pts[-1][0]
    for (x0, p0), (x1, p1) in zip(pts, pts[1:]):
        if p0 <= q <= p1:
            return x0 if p1 == p0 else x0 + (q - p0) / (p1 - p0) * (x1 - x0)
    return pts[-1][0]


def date_ru(iso: str | None) -> str:
    """'2026-09-25' -> '25.09.2026'."""
    if not iso or len(iso) < 10:
        return iso or ""
    return f"{iso[8:10]}.{iso[5:7]}.{iso[0:4]}"


def ru_group_gen(n: int) -> str:
    """'13 русских роликов' / '21 русского ролика' — the group in the genitive case (after «из», «среди»)."""
    one = n % 10 == 1 and n % 100 != 11
    return f"{n} русского ролика" if one else f"{n} русских роликов"


def describe(reference_id: str) -> dict:
    """What the reference group is, in words for the page: id, kind, n, frozen_at, label_ru, group_ru, footnote_ru."""
    if _is_ru_prov(reference_id):
        d = load_ru_prov()
        n, date = int(d["n"]), date_ru(d.get("frozen_at"))
        group = ru_group_gen(n)
        return {"id": d["id"], "kind": d.get("kind", "provisional"), "n": n, "frozen_at": d.get("frozen_at"),
                "label_ru": d.get("label_ru") or f"{group}, обработанных системой до {date}",
                "group_ru": group,
                "footnote_ru": (f"Пять черт — относительно предварительной опорной группы: {group}, обработанных "
                                f"системой до {date}. Роликов в сравнении пока {n}: этого мало для процентов, поэтому "
                                "положение описано словами и риска не ставится.")}
    return {"id": "fiv2", "kind": "norm", "n": FIV2_N, "frozen_at": None,
            "label_ru": "оценки наблюдателей в обучающей выборке First Impressions V2 (6000 роликов)",
            "group_ru": "людей в First Impressions V2",
            "footnote_ru": ("Процентили — относительно оценок наблюдателей в обучающей выборке First Impressions V2 "
                            "(6000 роликов).")}


def agreement_stats() -> dict:
    """How the two systems agree on the frozen Russian group: {"n", "spearman": {trait: rho}, "letters_same": {axis: k}}."""
    a = load_ru_prov().get("agreement") or {}
    return {"n": a.get("n"), "spearman": dict(a.get("spearman") or {}), "letters_same": dict(a.get("letters_same") or {})}
