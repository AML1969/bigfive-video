"""Plain-language sentences for the BS 2.0 analyses (emotions, voice, face, speech), appended to the Big Five
narrative. Deterministic templates over the numbers in result.json."""
from __future__ import annotations

from typing import List

from .charts import EMO_RU
from .report import seg_label


def _level(v: float, low: float = 0.4, high: float = 0.6) -> str:
    return "низкое" if v < low else ("высокое" if v > high else "среднее")


def analyses_sentences(rep: dict) -> str:
    an = rep.get("analyses") or {}
    parts: List[str] = []
    te = an.get("emotions_text")
    if te and te.get("mean"):
        m = te["mean"]
        top = sorted(m.items(), key=lambda kv: -kv[1])[:2]
        if top[0][0] == "neutral" and top[0][1] >= 0.6:
            s = f"По содержанию речи эмоциональная окраска в основном нейтральная ({top[0][1]:.0%})"
            if len(top) > 1 and top[1][1] >= 0.1:
                s += f", из выраженных эмоций заметнее всего {EMO_RU.get(top[1][0], top[1][0])} ({top[1][1]:.0%})"
            parts.append(s + ".")
        else:
            parts.append("По содержанию речи преобладает " + ", затем ".join(f"{EMO_RU.get(k, k)} ({v:.0%})" for k, v in top) + ".")
        dps = te.get("dominant_per_segment") or []
        changes = sum(1 for a, b in zip(dps, dps[1:]) if a and b and a != b)
        if len(dps) >= 4:
            parts.append("Эмоциональный тон речи " + ("ровный по всему ролику." if changes <= len(dps) // 4 else
                                                    f"меняется по ходу ролика ({changes} смен доминирующей эмоции)."))
    vo = an.get("voice")
    if vo and vo.get("mean"):
        m = vo["mean"]
        parts.append(f"Голос: возбуждение {_level(m.get('arousal', 0.5))} ({m.get('arousal', 0):.2f}), уверенность "
                     f"{_level(m.get('dominance', 0.5))} ({m.get('dominance', 0):.2f}), позитивность {_level(m.get('valence', 0.5))} "
                     f"({m.get('valence', 0):.2f}) по модели эмоций в речи; значения относительные, шкала английских записей.")
    fa = an.get("face")
    if fa and fa.get("mean"):
        m = fa["mean"]
        top = sorted(m.items(), key=lambda kv: -kv[1])[:2]
        s = "Выражение лица чаще всего читается как " + ", реже ".join(f"{EMO_RU.get(k, k)} ({v:.0%})" for k, v in top)
        if fa.get("head_motion") is not None:
            hm = fa["head_motion"]
            s += "; голова " + ("почти неподвижна" if hm < 0.05 else ("двигается умеренно" if hm < 0.15 else "двигается активно"))
        if fa.get("face_share") is not None and fa["face_share"] < 0.9:
            s += f"; лицо видно в {fa['face_share']:.0%} кадров"
        parts.append(s + ". Распознавание выражений обучено на фотографиях и склонно завышать «грусть» и «страх» у спокойного лица.")
    sp = an.get("speech")
    if sp and sp.get("description"):
        parts.append(sp["description"])
        if sp.get("vocabulary"):
            parts.append("Чаще всего звучат слова: " + ", ".join(f"«{w}»" for w, _ in sp["vocabulary"][:6]) + ".")
    return " ".join(parts)


def key_facts(rep: dict) -> List[tuple]:
    """(label, value, note) cards for the overview tab."""
    an = rep.get("analyses") or {}
    facts = []
    te = an.get("emotions_text")
    if te and te.get("mean"):
        k, v = max(te["mean"].items(), key=lambda kv: kv[1])
        facts.append(("Эмоция речи", EMO_RU.get(k, k), f"{v:.0%} времени"))
    fa = an.get("face")
    if fa and fa.get("mean"):
        k, v = max(fa["mean"].items(), key=lambda kv: kv[1])
        facts.append(("Выражение лица", EMO_RU.get(k, k), f"{v:.0%} кадров"))
    vo = an.get("voice")
    if vo and vo.get("mean"):
        m = vo["mean"]
        facts.append(("Голос", f"возбуждение {m.get('arousal', 0):.2f}", f"уверенность {m.get('dominance', 0):.2f}, позитивность {m.get('valence', 0):.2f}"))
    sp = an.get("speech")
    if sp and sp.get("words"):
        wpm = sp.get("words_per_min_speech")
        facts.append(("Речь", f"{wpm:.0f} слов/мин" if wpm else f"{sp['words']} слов", f"паузы {sp.get('pause_share', 0):.0%}, заполнители {sp.get('fillers_per_100', 0):.0f}/100"))
    if rep.get("interview"):
        facts.append(("«Собеседование»", f"{rep['interview']['score']:.2f}", "своя модель, шкала FIV2"))
    dur = rep.get("duration_sec")
    if dur:
        facts.append(("Ролик", seg_label(0, dur).split("–")[-1] if dur >= 60 else f"{dur:.0f} с", f"{rep.get('segments', 1)} сегментов"))
    return facts
