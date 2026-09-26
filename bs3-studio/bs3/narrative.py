"""Plain-language explanation of a result, built deterministically from the numbers (no LLM): what the scores are
based on, which traits stand out, how stable the video is, what the own model looked at, which words mattered."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import DEFAULT_MODEL, MODEL_TITLES
from .norms import RU_TITLES, TRAIT_KEYS
from .report import seg_label

MOD_RU = {"face": "лицо", "audio": "голос", "audio_whisper": "голос", "audio_xlsr": "голос", "audio_w2v_emo": "голос",
          "text": "содержание речи", "behavior": "описание поведения"}


SYSTEM_RU = {**MODEL_TITLES, "scene": "SSL-MEPR (сцена)"}
# grammatical gender of the modality names, for «почти не повлиял / повлияло»
MOD_GENDER = {"лицо": "n", "голос": "m", "содержание речи": "n", "описание поведения": "n"}
# where the scores come from, by the model that ran (one model per analysis since 3.1)
SOURCE_RU = {
    "oceanai": "Оценки дала система OCEAN-AI на весах MuPTA, обученных на русскоязычных участниках.",
    "mm": ("Оценки дала модель AMLAI 1.0, построенная по рецепту MM-PSYCHE и обученная на First Impressions V2; "
           "транскрипт русской речи для неё переведён на английский."),
}
SCALE_RU = "Уровни черт и буквы MBTI считаются по самой оценке модели на шкале от 0 до 1 с серединой 0.5."
# the one note of the tab «Объяснения» and of the PDF (under section 4) for an OCEAN-AI job (change request 3.1,
# section 3, the owner's final decision): explanations exist for AMLAI 1.0 only
NO_EXPLAIN_RU = ("Модель OCEAN-AI не строит объяснений: ключевые кадры, вклад модальностей и слова, повлиявшие на "
                 "оценку, есть только для модели AMLAI 1.0.")


def plural_ru(n: int, forms: tuple) -> str:
    """plural_ru(21, ("отрезок", "отрезка", "отрезков")) -> "отрезок"."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def _pct_phrase(t: dict) -> str:
    """The FIV2 percentile in words, as on the score bars; '' for anything else (percentiles against the pool of
    processed videos are never worded, change of 2026-09-26)."""
    pct = t.get("percentile")
    ref = t.get("percentile_ref", "")
    if pct is None or not ("First Impressions V2" in ref or "FIV2" in ref):
        return ""
    group = "людей в First Impressions V2"
    p = max(0.0, min(100.0, float(pct)))
    if 45 <= p <= 55:
        return f"примерно посередине среди {group}"
    return f"выше, чем у {p:.0f}% {group}" if p > 50 else f"ниже, чем у {100 - p:.0f}% {group}"


def _name(k: str) -> str:
    return RU_TITLES[k].lower()


def word_effects(rw: Dict[str, dict]) -> tuple:
    """({word: largest |effect|} of the words that raised a score, {…} of those that lowered one) over all traits of
    a readable_words list; Russian words only (an item without a translation is left out)."""
    from .words import shown_word
    ups: Dict[str, float] = {}
    downs: Dict[str, float] = {}
    for d in rw.values():
        for items, acc in ((d["up"], ups), (d["down"], downs)):
            for i in items:
                w = shown_word(i)
                if w:
                    acc[w] = max(acc.get(w, 0.0), abs(i["signed"]))
    return ups, downs


def top_directions(ups: Dict[str, float], downs: Dict[str, float], k: int = 4) -> tuple:
    """The k strongest words of each direction. A word that raised some scores and lowered others names no direction
    and is left out of both lists («в сторону повышения — «продукт»; понижения — «продукт»» is no information)."""
    both = set(ups) & set(downs)
    top_up = [w for w, _ in sorted(ups.items(), key=lambda kv: -kv[1]) if w not in both][:k]
    top_down = [w for w, _ in sorted(downs.items(), key=lambda kv: -kv[1]) if w not in both][:k]
    return top_up, top_down


def odd_segments(rep: dict) -> List[tuple]:
    """[(timeline entry, z)] of the segments whose mean of the five scores lies more than 2 standard deviations from
    the other segments (at least 4 scored segments), in time order; z > 0 — the scores are higher."""
    tl = [t for t in (rep.get("timeline") or []) if t.get("scores")]
    means = np.array([np.mean([t["scores"][k] for k in TRAIT_KEYS]) for t in tl])
    if len(means) < 4 or means.std() <= 0:
        return []
    z = (means - means.mean()) / means.std()
    return [(t, float(z_)) for t, z_ in zip(tl, z) if abs(z_) > 2.0]


def build_narrative(rep: dict, expl: dict | None = None) -> str:
    traits = rep["traits"]
    model = rep.get("model") or {}
    lang, primary = model.get("lang", "en"), model.get("primary")
    parts: List[str] = []

    # 1. what the scores come from (one model per analysis; a report without a recorded model is read as OCEAN-AI,
    # as scores.main_system reads it)
    selected = model.get("selected") or primary or DEFAULT_MODEL
    parts.append(SOURCE_RU.get(selected) or f"Оценки дала система {SYSTEM_RU.get(selected, selected)}.")

    # 2. traits that stand out
    order = sorted(TRAIT_KEYS, key=lambda k: -traits[k]["score"])
    hi1, hi2, lo = order[0], order[1], order[-1]

    def tr(k):
        p = _pct_phrase(traits[k])
        return f"{_name(k)} ({traits[k]['score']:.2f}" + (f", {p}" if p else "") + ")"
    parts.append(f"Сильнее всего выражены {tr(hi1)} и {tr(hi2)}; слабее всего — {tr(lo)}.")

    # 3. interview impression (the FIV2 percentile is worded for English speech only: nothing of a Russian job
    # talks about FIV2 percentiles, change request 3.1, section 1)
    iv = rep.get("interview")
    if iv:
        p = _pct_phrase(iv) if lang == "en" else ""
        parts.append(f"Впечатление «пригласить на собеседование» по модели AMLAI 1.0: {iv['score']:.2f}"
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
        for t, z_ in odd_segments(rep)[:2]:
            parts.append(f"Заметно отличается отрезок {seg_label(t['start'], t['end'])}: оценки "
                         f"{'выше' if z_ > 0 else 'ниже'} остального ролика.")

    # 5. what the own model looked at
    if expl and expl.get("modalities", {}).get("input_x_gradient"):
        ixg = expl["modalities"]["input_x_gradient"]
        mods = list(next(iter(ixg.values())).keys())
        share = {m: float(np.mean([row[m]["share"] for row in ixg.values()])) for m in mods}
        main = [m for m in sorted(mods, key=lambda m: -share[m]) if share[m] >= 0.01]
        weak = [m for m in mods if share[m] < 0.01]
        named = [f"{MOD_RU.get(m, m)} ({share[m] * 100:.0f}%)" for m in main]
        # «на лицо (57%), голос (37%) и описание поведения (4%)», not «… и … и …»
        s = "Модель AMLAI 1.0 опиралась в основном на " + (", ".join(named[:-1]) + " и " + named[-1] if len(named) > 1
                                                            else "".join(named))
        if weak:
            names = list(dict.fromkeys(MOD_RU.get(m, m) for m in weak))
            verb = "почти не повлияли" if len(names) > 1 else (
                "почти не повлиял" if MOD_GENDER.get(names[0]) == "m" else "почти не повлияло")
            s += "; " + " и ".join(names) + f" {verb} (меньше 1%)"
        parts.append(s + ".")

    # 6. words (union over traits, largest effects), Russian only whatever the speech language; the same lists as the
    # block «Слова, на которые откликнулась модель» (words_summary)
    rw = (expl or {}).get("readable_words", {}).get("transcript_words")
    if rw:
        ups, downs = top_directions(*word_effects(rw))
        if ups or downs:
            s = "Отдельные слова речи сдвигали оценки лишь незначительно"
            if ups:
                s += "; в сторону повышения — " + ", ".join(f"«{w}»" for w in ups)
            if downs:
                s += "; в сторону понижения — " + ", ".join(f"«{w}»" for w in downs)
            parts.append(s + ".")
    return " ".join(parts)


def method_notes(view: dict, expl: dict | None = None) -> str:
    """«Как получены оценки» (design 9; one model since 3.1): which model gave the scores, the scale of levels and
    letters, how stable the scores are over the video and the segments without a score (C13). Built on the clean view
    (scores.clean_view); replaces the Big Five part of the 2.0 summary. `expl` is accepted for the page and the PDF,
    which call it with the explanation at hand."""
    from . import caveats
    meta = view.get("view_meta") or {}
    main = meta.get("main_system")
    parts: List[str] = []
    if meta.get("primary_missing"):
        parts.append(caveats.text("C20"))
    else:
        parts.append(SOURCE_RU.get(main) or f"Оценки дала система {SYSTEM_RU.get(main, main)}.")
    parts.append(SCALE_RU)

    # stability over the segments the model scored
    tl = [t for t in (view.get("timeline") or []) if t.get("scores")]
    std = view.get("scores_std_across_segments") or view.get("scores_std") or {}
    if tl and std:
        n = len(tl)
        count = f"{n} {plural_ru(n, ('отрезок', 'отрезка', 'отрезков'))} с оценкой {MODEL_TITLES.get(main, main)}"
        worst = max(TRAIT_KEYS, key=lambda k: std.get(k, 0))
        if std.get(worst, 0) <= 0.05:
            parts.append(f"По ходу ролика ({count}) оценки устойчивы: разброс не больше ±{std[worst]:.2f}.")
        else:
            parts.append(f"По ходу ролика ({count}) оценки в целом устойчивы, сильнее всего колеблется "
                         f"{_name(worst)} (±{std[worst]:.2f}).")
        for t, z_ in odd_segments(view)[:2]:
            parts.append(f"Заметно отличается отрезок {seg_label(t['start'], t['end'])}: оценки "
                         f"{'выше' if z_ > 0 else 'ниже'} остального ролика.")

    dropped = meta.get("segments_without_primary") or []
    if dropped:
        parts.append(caveats.c13(len(dropped), int(meta.get("segments_total") or len(dropped)), main))
    return " ".join(parts)


def words_sentences(rw_all: Dict[str, dict], titles: Dict[str, str], lang: str) -> List[str]:
    """Per-trait sentences: 'Открытость опыту: повышали слова «простой»; понижали — «явление», «работе».'"""
    from .words import shown_word
    out = []
    for key, what in (("transcript_words", "речи"), ("behavior_words", "описания поведения")):
        rw = rw_all.get(key)
        if not rw:
            continue
        out.append(f"Слова {what}:")
        for trait in list(TRAIT_KEYS) + [k for k in rw if k not in TRAIT_KEYS]:
            if trait not in rw:
                continue

            ups = [f"«{w}»" for w in map(shown_word, rw[trait]["up"]) if w]
            downs = [f"«{w}»" for w in map(shown_word, rw[trait]["down"]) if w]
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


def words_summary(rw_all: Dict[str, dict], expl: dict | None, titles: Dict[str, str], lang: str) -> List[str]:
    """Human paragraphs about the attributed words. When the text nodes barely matter (the usual case for the
    FIV2-trained model) one paragraph per source says so and names the few words the model reacted to; per-trait
    lists are shown only when traits really react to different words."""
    shares: Dict[str, float] = {}
    ixg = ((expl or {}).get("modalities") or {}).get("input_x_gradient") or {}
    if ixg:
        mods = next(iter(ixg.values())).keys()
        for m in mods:
            shares[m] = float(np.mean([row[m]["share"] for row in ixg.values()]))

    from .words import shown_word

    def q(ws):
        return ", ".join(f"«{w}»" for w in ws)

    out: List[str] = []
    weak_all = True
    for key, head, node in (("transcript_words", "Речь", "text"), ("behavior_words", "Описание поведения", "behavior")):
        rw = rw_all.get(key)
        if not rw:
            continue
        share = shares.get(node)
        per_trait = {}
        for trait, d in rw.items():
            # Russian words only: an item without a translation is left out
            u = [w for w in map(shown_word, d["up"]) if w]
            dn = [w for w in map(shown_word, d["down"]) if w]
            per_trait[trait] = (u[:3], dn[:3])
        top_up, top_down = top_directions(*word_effects(rw))       # the same lists as the plain-language paragraph
        agree = [1.0 if (set(u) <= set(top_up) and set(dn) <= set(top_down)) else 0.0 for u, dn in per_trait.values()]
        uniform = (sum(agree) / max(1, len(agree))) >= 0.7
        weak = share is not None and share < 0.02
        weak_all = weak_all and weak
        if share is None:
            s = f"{head}."
        elif weak:
            s = (f"{head}. " + ("Содержание речи почти не повлияло" if node == "text" else "Текст описания почти не повлиял")
                 + f" на оценки: вклад меньше {'1' if share < 0.01 else '2'}%.")
        else:
            s = f"{head}. Вклад в оценки около {share * 100:.0f}%."
        if uniform or weak:
            if len(top_up) == 1 and not top_down:
                s += f" Единственное слово, на которое модель заметно отреагировала, — {q(top_up)}: оно немного подняло все оценки."
            else:
                if top_up:
                    s += f" Немного поднимали все оценки слова {q(top_up)}"
                    s += f", немного снижали — {q(top_down)}." if top_down else "."
                elif top_down:
                    s += f" Немного снижали все оценки слова {q(top_down)}."
            if top_up and top_down:
                s += " Направление одинаково для всех черт: модель откликается на общий тон текста, а не на отдельные черты."
        else:
            lines = []
            for trait, (u, dn) in per_trait.items():
                if not u and not dn:
                    continue
                part = f"{titles.get(trait, trait)}: "
                if u:
                    part += f"поднимали {q(u)}"
                if dn:
                    part += (", снижали " if u else "снижали ") + q(dn)
                lines.append(part + ".")
            s += " Разные черты реагируют на разные слова. " + " ".join(lines)
        out.append(s)
    if out:
        out.append("Что это значит: модель AMLAI 1.0 судит в основном по лицу и голосу, а слова показывают, на какие "
                   "формулировки она откликается; итоговые оценки от слов почти не зависят." if weak_all else
                   "«Поднимали» — слово сдвигало оценку черты вверх, «снижали» — вниз; это реакция модели AMLAI 1.0, а не "
                   "смысл слов сам по себе.")
    return out
