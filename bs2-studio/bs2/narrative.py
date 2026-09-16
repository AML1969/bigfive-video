"""Plain-language explanation of a result, built deterministically from the numbers (no LLM): what the scores are
based on, which traits stand out, how stable the video is, what the own model looked at, which words mattered."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .norms import RU_TITLES, TRAIT_KEYS
from .report import seg_label

MOD_RU = {"face": "лицо", "audio": "голос", "audio_whisper": "голос", "audio_xlsr": "голос", "audio_w2v_emo": "голос",
          "text": "содержание речи", "behavior": "описание поведения"}


def _pct_phrase(t: dict, what: str = "роликов") -> str:
    pct = t.get("percentile")
    if pct is None:
        return ""
    ref = t.get("percentile_ref", "")
    pool = "обработанных русских роликов" if "пула" in ref else "людей в датасете First Impressions V2"
    if pct >= 50:
        return f"выше, чем у {pct:.0f}% {pool}"
    return f"ниже, чем у {100 - pct:.0f}% {pool}"


def _name(k: str) -> str:
    return RU_TITLES[k].lower()


def build_narrative(rep: dict, expl: dict | None = None) -> str:
    traits = rep["traits"]
    model = rep.get("model") or {}
    lang, primary = model.get("lang", "en"), model.get("primary")
    parts: List[str] = []

    # 1. what the scores come from
    if primary == "oceanai":
        parts.append("Основные оценки дала система OCEAN-AI на весах MuPTA, обученных на русскоязычных участниках; своя модель "
                     "показана как второе мнение.")
    elif primary:
        parts.append(f"Основные оценки дала система {primary}.")
    else:
        parts.append("Оценки — среднее двух систем (OCEAN-AI и своей модели), обе на шкале First Impressions V2.")

    # 2. traits that stand out
    order = sorted(TRAIT_KEYS, key=lambda k: -traits[k]["score"])
    hi1, hi2, lo = order[0], order[1], order[-1]

    def tr(k):
        p = _pct_phrase(traits[k])
        return f"{_name(k)} ({traits[k]['score']:.2f}" + (f", {p}" if p else "") + ")"
    parts.append(f"Сильнее всего выражены {tr(hi1)} и {tr(hi2)}; слабее всего — {tr(lo)}.")

    # 3. interview impression
    iv = rep.get("interview")
    if iv:
        p = _pct_phrase(iv)
        parts.append(f"Впечатление «пригласить на собеседование» по своей модели: {iv['score']:.2f}"
                     + (f", {p}" if p else "") + ".")

    # 4. stability across segments
    tl = [t for t in (rep.get("timeline") or []) if t.get("scores")]
    std = rep.get("scores_std_across_segments") or rep.get("scores_std") or {}
    if tl and std:
        worst = max(TRAIT_KEYS, key=lambda k: std.get(k, 0))
        if std.get(worst, 0) <= 0.05:
            parts.append(f"По ходу ролика ({len(tl)} сегментов) оценки устойчивы: разброс не больше ±{std[worst]:.2f}.")
        else:
            parts.append(f"По ходу ролика ({len(tl)} сегментов) оценки в целом устойчивы, сильнее всего колеблется "
                         f"{_name(worst)} (±{std[worst]:.2f}).")
        means = np.array([np.mean([t["scores"][k] for k in TRAIT_KEYS]) for t in tl])
        if len(means) >= 4 and means.std() > 0:
            z = (means - means.mean()) / means.std()
            odd = [(t, z_) for t, z_ in zip(tl, z) if abs(z_) > 2.0]
            for t, z_ in odd[:2]:
                parts.append(f"Заметно отличается сегмент {seg_label(t['start'], t['end'])}: оценки "
                             f"{'выше' if z_ > 0 else 'ниже'} остального ролика.")

    # 5. what the own model looked at
    if expl and expl.get("modalities", {}).get("input_x_gradient"):
        ixg = expl["modalities"]["input_x_gradient"]
        mods = list(next(iter(ixg.values())).keys())
        share = {m: float(np.mean([row[m]["share"] for row in ixg.values()])) for m in mods}
        main = [m for m in sorted(mods, key=lambda m: -share[m]) if share[m] >= 0.01]
        weak = [m for m in mods if share[m] < 0.01]
        s = "Своя модель опиралась в основном на " + " и ".join(f"{MOD_RU.get(m, m)} ({share[m] * 100:.0f}%)" for m in main)
        if weak:
            s += "; " + " и ".join(MOD_RU.get(m, m) for m in weak) + " почти не повлияли (меньше 1%)"
        parts.append(s + ".")

    # 6. words (union over traits, largest effects)
    rw = (expl or {}).get("readable_words", {}).get("transcript_words")
    if rw:
        up, down = {}, {}
        for d in rw.values():
            for i in d["up"]:
                w = i.get("source") or i.get("ru") or i["en"]; up[w] = max(up.get(w, 0), abs(i["signed"]))
            for i in d["down"]:
                w = i.get("source") or i.get("ru") or i["en"]; down[w] = max(down.get(w, 0), abs(i["signed"]))
        ups = [w for w, _ in sorted(up.items(), key=lambda x: -x[1])[:4]]
        downs = [w for w, _ in sorted(down.items(), key=lambda x: -x[1])[:4]]
        if ups or downs:
            s = "Отдельные слова речи сдвигали оценки лишь незначительно"
            if ups:
                s += "; в сторону повышения — " + ", ".join(f"«{w}»" for w in ups)
            if downs:
                s += "; в сторону понижения — " + ", ".join(f"«{w}»" for w in downs)
            parts.append(s + ".")

    # 7. second opinion on another scale
    var = rep.get("variant_scores") or rep.get("variants") or {}
    if primary == "oceanai" and "mm" in var:
        diff = float(np.mean([var["oceanai"][k] - var["mm"][k] for k in TRAIT_KEYS]))
        parts.append(f"Своя модель, обученная на англоязычных влогерах, оценивает те же черты в среднем на {diff:.2f} ниже: "
                     "это разница шкал двух систем, а не противоречие в выводах.")
    return " ".join(parts)


def words_sentences(rw_all: Dict[str, dict], titles: Dict[str, str], lang: str) -> List[str]:
    """Per-trait sentences: 'Открытость опыту: повышали слова «простой»; понижали — «явление», «работе».'"""
    out = []
    for key, what in (("transcript_words", "речи"), ("behavior_words", "описания поведения")):
        rw = rw_all.get(key)
        if not rw:
            continue
        out.append(f"Слова {what}:")
        for trait in list(TRAIT_KEYS) + [k for k in rw if k not in TRAIT_KEYS]:
            if trait not in rw:
                continue

            def lab(i):
                return "«" + (i.get("source") or i.get("ru") or i["en"] if lang != "en" else i["en"]) + "»"
            ups = [lab(i) for i in rw[trait]["up"]]
            downs = [lab(i) for i in rw[trait]["down"]]
            if not ups and not downs:
                continue
            s = f"{titles.get(trait, trait)}: "
            if ups:
                s += "повышали " + ", ".join(ups)
            if downs:
                s += ("; понижали " if ups else "понижали ") + ", ".join(downs)
            out.append(s + ".")
        out.append("")
    return out
