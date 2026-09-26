"""Captions under the key frames of AMLAI 1.0 (change request «подписи под ключевыми кадрами», 2026-09-26).

One place builds what the page and the PDF print under every key frame, so the same frame is never described two
ways:

- the moment of the whole video (m:ss, tenths when two frames share a second);
- a short phrase of the video-language model about that single frame (Qwen-VL, stored in explanation.json);
- the two strongest facial expressions of that frame (the same 7 classes as the rest of the report);
- what the frame did to the score, with a direction, from the signed frame attribution
  (mm/explain.frame_attribution: `signed` = sum(grad * embedding) per frame and output).

The page shows one short line and opens the rest on hover (`title` of the figure); the PDF prints the same caption in
two lines clipped to the width of the frame. Jobs made before this change carry neither `signed` nor the phrases:
they fall back to the wording they had («сильнее всего повлиял на оценку …») and to the expression alone.

The prompt of the frame phrase and its validation live here too (the explanation code asks for them): a phrase is
kept only if it stays a short list of visible things — no verdicts about character or mood, no digits, no sentence.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .norms import TRAIT_KEYS

# ---------------------------------------------------------------- trait names in the two cases the captions need
# «повысил оценку экстраверсии» (genitive) on the page, «повысил экстраверсию» (accusative) in the narrow PDF cell
TRAIT_GEN = {
    "openness": "открытости опыту",
    "conscientiousness": "добросовестности",
    "extraversion": "экстраверсии",
    "agreeableness": "доброжелательности",
    "emotional_stability": "эмоциональной стабильности",
}
TRAIT_ACC = {
    "openness": "открытость опыту",
    "conscientiousness": "добросовестность",
    "extraversion": "экстраверсию",
    "agreeableness": "доброжелательность",
    "emotional_stability": "эмоциональную стабильность",
}

# ---------------------------------------------------------------- the short phrase from the video-language model
FRAME_PROMPT = (
    "Опиши по-русски только то, что видно на этом одном кадре: выражение лица, направление взгляда, положение "
    "головы и плеч, руки. От трёх до шести слов, одной короткой фразой, без точки в конце, без имён, без цифр. "
    "Нельзя писать о характере, настроении или впечатлении («уверенный», «дружелюбный», «нервничает»), нельзя "
    "описывать фон и обстановку — только человека."
)
PHRASE_MAX = 40                     # characters, after trimming at the last comma
PHRASE_NUM_PREDICT = 48             # the answer is a few words: a short budget keeps the five requests fast
PHRASE_TRIES = 2                    # one retry, then the caption falls back to the expression
PHRASE_TIMEOUT = 45                 # seconds per request: a caption never holds the job for minutes
PHRASE_BUDGET = 30                  # seconds for all five frames together; what is left over keeps no phrase
# Verdicts about character or mood the phrase must not contain (the prompt forbids them; this is the safety net).
# The seven names of the expression classes («радость», «грусть», «страх», «злость», «отвращение», «удивление»,
# «нейтрально») are not verdicts: they are what the expression model itself reports, and the caption may repeat them.
PHRASE_VERDICTS = (
    "увере", "неувер", "дружелюб", "нервн", "тревож", "застенчив", "стеснит", "общитель", "замкнут", "харизм",
    "искренн", "агрессив", "враждеб", "доброжелат", "скучающ", "взволнован", "раздраж", "настроен", "эмоционал",
    "характер", "личност", "впечатлен", "похоже", "вероятно", "возможно",
    "спокой", "опустош", "подавлен", "угнет", "задумчив", "устал", "решительн", "мрачн",
)
PHRASE_BANNED = ("кажется", "выглядит как")
_RE_YA = re.compile(r"(?<![а-яёa-z])я(?![а-яёa-z])", re.I)
_RE_DIGIT = re.compile(r"\d")


def clean_phrase(raw: str | None) -> Optional[str]:
    """The model's answer as a caption phrase, or None when it must be dropped.

    Quotes are stripped, a too long answer is trimmed at its last comma; the phrase is dropped when it is still
    longer than PHRASE_MAX, or contains a digit, a line break, the pronoun «я», «кажется», «выглядит как» or any
    verdict about character or mood."""
    if raw is None:
        return None
    s = str(raw)
    if "\n" in s.strip().strip("«»\"'“”„ "):
        return None                                   # a whole answer in several lines is not a short phrase
    s = " ".join(s.split())
    s = s.strip().strip("«»\"'“”„ ").strip()
    s = s.rstrip(" .!?;:—-").strip()
    s = s.strip("«»\"'“”„ ").strip()
    if not s:
        return None
    while len(s) > PHRASE_MAX and "," in s:
        s = s.rsplit(",", 1)[0].strip().rstrip(" .!?;:—-").strip()
    if not s or len(s) > PHRASE_MAX:
        return None
    if _RE_DIGIT.search(s):
        return None
    low = s.lower()
    if _RE_YA.search(low) or any(b in low for b in PHRASE_BANNED) or any(v in low for v in PHRASE_VERDICTS):
        return None
    if s[:1].isupper() and s[:1].isalpha():
        s = s[:1].lower() + s[1:]
    return s


# ---------------------------------------------------------------- moments of the key frames
def _frame_index(path: str) -> Optional[int]:
    """The index of the frame inside the sampled sequence, from the file name `key_<i>_frame<N>.jpg`."""
    m = re.match(r"^\w+?_(\d+)_frame\d+$", Path(path).stem)
    return int(m.group(1)) if m else None


def _decoded_index(path: str) -> Optional[int]:
    """The number of the frame in the clip, from the file name `key_<i>_frame<N>.jpg`."""
    m = re.search(r"_frame(\d+)$", Path(path).stem)
    return int(m.group(1)) if m else None


def moments(report: dict, frames: Sequence[str], media: dict | None = None) -> Tuple[List[Optional[float]], bool]:
    """(seconds of every key frame in the whole video or None, whether captions must show tenths).

    Key frames come from the representative segment of a long video, otherwise from the whole video; the file name
    carries the frame number inside that clip. Without a frame rate there is no moment at all."""
    tl_all = report.get("timeline") or []
    seg = next((t for t in tl_all if t.get("segment") == report.get("representative_segment")), None) if tl_all else None
    fps = float((media or {}).get("fps") or (report.get("media") or {}).get("fps") or 0)
    start = float(seg["start"]) if seg else 0.0
    ok = fps > 0 and (seg is not None or not tl_all)
    out: List[Optional[float]] = []
    for p in frames:
        n = _decoded_index(p)
        out.append(start + n / fps if (ok and n is not None) else None)
    secs = [int(t) for t in out if t is not None]
    return out, len(set(secs)) < len(secs)


def moment_text(t: float, tenths: bool) -> str:
    """«2:14», or «2:14,3» when two key frames fall into the same second."""
    if not tenths:
        return f"{int(t) // 60}:{int(t) % 60:02d}"
    d = int(t * 10)
    return f"{d // 600}:{d // 10 % 60:02d},{d % 10}"


# ---------------------------------------------------------------- expressions and the signed effect
def _expr_of(info: dict | None) -> List[Tuple[str, float]]:
    """[(Russian class name, share)] of the frame, strongest first (at most two)."""
    out = []
    for e in (info or {}).get("expressions") or []:
        ru, share = e.get("ru"), e.get("share")
        if ru and share is not None:
            out.append((str(ru), float(share)))
    return out[:2]


def expr_text(expr: Sequence[Tuple[str, float]], n: int = 2) -> str:
    """«радость 62%, нейтрально 21%»."""
    return ", ".join(f"{ru} {round(share * 100):.0f}%" for ru, share in list(expr)[:n])


def _signed_pairs(fr: dict, idx: Optional[int]) -> List[Tuple[str, float]]:
    """[(trait, signed)] of one frame, |signed| descending; empty for a job made before the signed attribution."""
    if idx is None:
        return []
    eff = (fr.get("per_frame_effect") or [])
    if idx < len(eff) and eff[idx]:
        pairs = [(e["output"], float(e["signed"])) for e in eff[idx]
                 if isinstance(e, dict) and e.get("output") in TRAIT_GEN and e.get("signed") is not None]
        if pairs:
            return sorted(pairs, key=lambda kv: -abs(kv[1]))
    per = fr.get("per_output") or {}
    pairs = []
    for k in TRAIT_KEYS:
        arr = (per.get(k) or {}).get("signed")
        if isinstance(arr, list) and idx < len(arr):
            pairs.append((k, float(arr[idx])))
    return sorted(pairs, key=lambda kv: -abs(kv[1]))


def _legacy_pairs(fr: dict, idx: Optional[int]) -> List[Tuple[str, float]]:
    """The same ranking from the magnitudes of an old job (no direction is known there)."""
    if idx is None:
        return []
    per = fr.get("per_output") or {}
    pairs = []
    for k in TRAIT_KEYS:
        arr = (per.get(k) or {}).get("importance")
        if isinstance(arr, list) and idx < len(arr):
            pairs.append((k, float(arr[idx])))
    return sorted(pairs, key=lambda kv: -abs(kv[1]))


SECOND_SHARE = 0.5                  # a second trait is named when its |signed| is at least half of the first
TINY_SHARE = 0.2                    # below this share of the strongest key frame the direction is not printed


def effect_text(pairs: Sequence[Tuple[str, float]], top_all: float, *, signed: bool = True, short: bool = False) -> str:
    """«повысил оценку экстраверсии и доброжелательности» / «повысил экстраверсию» (short, for the PDF cell).

    `pairs` — the traits of this frame ranked by |signed|, `top_all` — the largest |signed| over all key frames."""
    if not pairs:
        return ""
    k1, v1 = pairs[0]
    if not signed:
        if short:
            return f"повлиял на {TRAIT_ACC[k1]}"
        rest = ""
        if len(pairs) > 1 and abs(pairs[1][1]) >= SECOND_SHARE * abs(v1):
            rest = f" и {TRAIT_GEN[pairs[1][0]]}"
        return f"сильнее всего повлиял на оценку {TRAIT_GEN[k1]}{rest}"
    if top_all > 0 and abs(v1) < TINY_SHARE * top_all:
        return f"повлиял на {TRAIT_ACC[k1]}" if short else f"заметно повлиял на оценку {TRAIT_GEN[k1]}"
    verb1 = "повысил" if v1 >= 0 else "понизил"
    if short:
        return f"{verb1} {TRAIT_ACC[k1]}"
    if len(pairs) > 1 and abs(pairs[1][1]) >= SECOND_SHARE * abs(v1):
        k2, v2 = pairs[1]
        verb2 = "повысил" if v2 >= 0 else "понизил"
        if verb2 == verb1:
            return f"{verb1} оценку {TRAIT_GEN[k1]} и {TRAIT_GEN[k2]}"
        return f"{verb1} оценку {TRAIT_GEN[k1]}, {verb2} — {TRAIT_GEN[k2]}"
    return f"{verb1} оценку {TRAIT_GEN[k1]}"


# ---------------------------------------------------------------- the captions themselves
def build(report: dict, frames: Sequence[str], explanation: dict | None, media: dict | None = None) -> List[dict]:
    """One dict per key frame, in the order of `frames`:

    `caption` — the line under the frame on the page («2:14 · улыбается, смотрит в камеру»);
    `tooltip` — the line the page opens on hover («радость 62%, нейтрально 21% · повысил оценку экстраверсии»);
    `expr_line` / `effect_short` — the pieces the PDF joins into its second line;
    `alt` — the text alternative of the image."""
    fr = ((explanation or {}).get("frames") or {})
    info_by_file = {str(i.get("file")): i for i in (fr.get("key_frame_info") or []) if isinstance(i, dict)}
    times, tenths = moments(report, frames, media)
    idxs = [_frame_index(p) for p in frames]
    signed_ok = bool(fr.get("per_frame_effect")) or any(
        isinstance(((fr.get("per_output") or {}).get(k) or {}).get("signed"), list) for k in TRAIT_KEYS)
    ranked = [(_signed_pairs(fr, i) if signed_ok else _legacy_pairs(fr, i)) for i in idxs]
    top_all = max((abs(r[0][1]) for r in ranked if r), default=0.0)
    out = []
    total = len(frames)
    for n, (p, t, idx, pairs) in enumerate(zip(frames, times, idxs, ranked), 1):
        info = info_by_file.get(Path(p).name) or {}
        phrase = clean_phrase(info.get("phrase")) if info.get("phrase") else None
        expr = _expr_of(info)
        label = moment_text(t, tenths) if t is not None else f"кадр {n}"
        tail = phrase or (expr[0][0] if expr else "")
        caption = f"{label} · {tail}" if tail else label
        eff_long = effect_text(pairs, top_all, signed=signed_ok)
        eff_short = effect_text(pairs, top_all, signed=signed_ok, short=True)
        tooltip = " · ".join(x for x in (expr_text(expr), eff_long) if x)
        alt = (f"Ключевой кадр, момент {label}" if t is not None else f"Ключевой кадр {n} из {total}")
        out.append({"path": str(p), "n": n, "total": total, "moment": t, "label": label, "phrase": phrase,
                    "expr": expr, "expr_line": expr_text(expr, 1), "effect": eff_long, "effect_short": eff_short,
                    "tail": tail, "caption": caption, "tooltip": tooltip,
                    "alt": alt if not tail else f"{alt}: {tail}"})
    return out


def has_tenths(report: dict, frames: Sequence[str], media: dict | None = None) -> bool:
    return moments(report, frames, media)[1]


def any_moment(entries: Iterable[dict]) -> bool:
    return any(e.get("moment") is not None for e in entries)


def pdf_second_line(entry: dict) -> List[str]:
    """Candidates for the second PDF line, widest first; the caller keeps the first one that fits the cell."""
    expr, eff = entry.get("expr_line") or "", entry.get("effect_short") or ""
    out = [x for x in (" · ".join(p for p in (expr, eff) if p), eff, expr) if x]
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res
