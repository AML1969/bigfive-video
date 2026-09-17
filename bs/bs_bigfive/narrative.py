"""Plain-language explanation of a result, built deterministically from the numbers (no LLM): what the scores are
based on, which traits stand out, how stable the video is, what the own model looked at, which words mattered."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .norms import RU_TITLES, TRAIT_KEYS
from .report import seg_label
from .words import shown_word

MOD_RU = {"face": "лицо", "audio": "голос", "audio_whisper": "голос", "audio_xlsr": "голос", "audio_w2v_emo": "голос",
          "text": "содержание речи", "behavior": "описание поведения"}


SMALL_POOL = 20          # below this many processed videos a percentage only looks precise: say it in words
SYSTEM_RU = {"oceanai": "OCEAN-AI", "mm": "своя модель (MM-PSYCHE)", "scene": "SSL-MEPR (сцена)"}
# grammatical gender of the modality names, for «почти не повлиял / повлияло»
MOD_GENDER = {"лицо": "n", "голос": "m", "содержание речи": "n", "описание поведения": "n"}


def plural_ru(n: int, forms: tuple) -> str:
    """plural_ru(21, ("отрезок", "отрезка", "отрезков")) -> "отрезок"."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def _pool_size(ref: str):
    import re
    m = re.search(r"N\s*=\s*(\d+)", ref or "")
    return int(m.group(1)) if m else None


def _pct_phrase(t: dict, what: str = "роликов") -> str:
    """Same wording as the score bars: a percentage for large reference groups, words for a small pool."""
    pct = t.get("percentile")
    if pct is None:
        return ""
    ref = t.get("percentile_ref", "")
    if "пула" in ref:
        group = "русских роликов" if "русских" in ref else ("английских роликов" if "английских" in ref else "обработанных роликов")
    else:
        group = "людей в First Impressions V2"
    p = max(0.0, min(100.0, float(pct)))
    n = _pool_size(ref)
    if "пула" in ref and n is not None and n < SMALL_POOL:
        if p > 60:
            return f"выше, чем у большинства из {n} {group}"
        if p < 40:
            return f"ниже, чем у большинства из {n} {group}"
        return f"примерно посередине среди {n} {group}"
    if 45 <= p <= 55:
        return f"примерно посередине среди {group}"
    return f"выше, чем у {p:.0f}% {group}" if p > 50 else f"ниже, чем у {100 - p:.0f}% {group}"


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
        parts.append(f"Основные оценки дала система {SYSTEM_RU.get(primary, primary)}.")
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
            parts.append(f"По ходу ролика ({len(tl)} {plural_ru(len(tl), ('отрезок', 'отрезка', 'отрезков'))}) оценки устойчивы: "
                         f"разброс не больше ±{std[worst]:.2f}.")
        else:
            parts.append(f"По ходу ролика ({len(tl)} {plural_ru(len(tl), ('отрезок', 'отрезка', 'отрезков'))}) оценки в целом "
                         f"устойчивы, сильнее всего колеблется "
                         f"{_name(worst)} (±{std[worst]:.2f}).")
        means = np.array([np.mean([t["scores"][k] for k in TRAIT_KEYS]) for t in tl])
        if len(means) >= 4 and means.std() > 0:
            z = (means - means.mean()) / means.std()
            odd = [(t, z_) for t, z_ in zip(tl, z) if abs(z_) > 2.0]
            for t, z_ in odd[:2]:
                parts.append(f"Заметно отличается отрезок {seg_label(t['start'], t['end'])}: оценки "
                             f"{'выше' if z_ > 0 else 'ниже'} остального ролика.")

    # 5. what the own model looked at
    if expl and expl.get("modalities", {}).get("input_x_gradient"):
        ixg = expl["modalities"]["input_x_gradient"]
        mods = list(next(iter(ixg.values())).keys())
        share = {m: float(np.mean([row[m]["share"] for row in ixg.values()])) for m in mods}
        order = sorted(mods, key=lambda m: -share[m])
        # «в основном» from 10%, as the note under the PDF contribution table; 1–10% «повлияли слабо», <1% «почти не»
        main = [m for m in order if share[m] >= 0.10]
        minor = [m for m in order if 0.01 <= share[m] < 0.10]
        weak = [m for m in mods if share[m] < 0.01]

        def listed(items):
            # «лицо (62%), голос (35%) и содержание речи (2%)», not «… и … и …»
            return ", ".join(items[:-1]) + " и " + items[-1] if len(items) > 1 else "".join(items)

        def verb(names, what):          # «повлияли» / «повлиял» (голос) / «повлияло» (лицо, описание поведения)
            return (f"{what}и" if len(names) > 1 else
                    f"{what}" if MOD_GENDER.get(names[0]) == "m" else f"{what}о")
        s = "Своя модель опиралась в основном на " + listed([f"{MOD_RU.get(m, m)} ({share[m] * 100:.0f}%)" for m in main])
        if minor:
            s += ("; " + listed([f"{MOD_RU.get(m, m)} ({share[m] * 100:.0f}%)" for m in minor]) + " "
                  + verb([MOD_RU.get(m, m) for m in minor], "повлиял") + " слабо")
        if weak:
            names = list(dict.fromkeys(MOD_RU.get(m, m) for m in weak))
            s += "; " + " и ".join(names) + f" {verb(names, 'почти не повлиял')} (меньше 1%)"
        parts.append(s + ".")

    # 6. words (union over traits, largest effects)
    rw = (expl or {}).get("readable_words", {}).get("transcript_words")
    if rw:
        # one direction per word, from its effect summed over the traits: a word raising one trait and lowering
        # another must not be named on both sides
        net = {}
        for d in rw.values():           # Russian forms only, whatever the speech language (words.shown_word)
            for i in d["up"] + d["down"]:
                w = shown_word(i)
                if w:
                    net[w] = net.get(w, 0.0) + float(i["signed"])
        ups = [w for w, v in sorted(net.items(), key=lambda x: -x[1]) if v > 0][:4]
        downs = [w for w, v in sorted(net.items(), key=lambda x: x[1]) if v < 0][:4]
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
        lines = []
        for trait in list(TRAIT_KEYS) + [k for k in rw if k not in TRAIT_KEYS]:
            if trait not in rw:
                continue
            # the Russian form for any speech language; a word without one is left out (words.shown_word)
            ups = [f"«{shown_word(i)}»" for i in rw[trait]["up"] if shown_word(i)]
            downs = [f"«{shown_word(i)}»" for i in rw[trait]["down"] if shown_word(i)]
            if not ups and not downs:
                continue
            s = f"{titles.get(trait, trait)}: "
            if ups:
                s += "повышали " + ", ".join(ups)
            if downs:
                s += ("; понижали " if ups else "понижали ") + ", ".join(downs)
            lines.append(s + ".")
        if lines:                       # no heading over an empty list
            out += [f"Слова {what}:"] + lines + [""]
    return out
