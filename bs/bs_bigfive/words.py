"""Human-readable version of the word attributions.

The text branch works on the English translation, so the raw attribution is a list of English tokens dominated by
function words ('to', 'your', 'on'). For the interface we keep content words only and show each one as the word the
reader can find in the Russian text shown next to the lists: the transcript for Russian speech, the Russian
translation of the transcript for English speech, and the Russian translation of the behaviour description. The word
is the one the LLM aligns with it in that text, or its dictionary translation (context-aware) matched on the stem.
Words are grouped by the direction of the effect; a word that does not occur in that Russian text is left out (it
would be a word the reader cannot find, or an English token)."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Dict, List, Tuple

from .norms import TRAIT_KEYS
from .report import clean_word

STOP_EN = set("""
a an the and or but if so as of to in on at by for from with without into onto over under about above below between
through during before after again further then once here there when where why how all any both each few more most
other some such no nor not only own same than too very can will just don should now is are was were be been being
have has had having do does did doing i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these
those am would could might must shall may also however therefore thus yes yeah okay ok well like um uh oh hmm
because while although though even still already yet ever never always often sometimes maybe perhaps quite rather
really actually basically literally something anything nothing everything someone anyone everyone one ones thing
things way lot lots much many little less least going get got gets getting go goes went come comes came make makes
made say says said tell told know knew think thought want wanted need let us gonna wanna kind sort bit
appear appears appeared seem seems seemed suggest suggests suggesting suggested indicate indicates indicating
indicated maintain maintains maintaining evident overall significant slightly possibly personality trait traits
""".split())            # last two lines: filler words of the VLM's descriptions ('appears calm', 'suggesting that',
                        # 'no strong personality traits evident')
_RU_WORD = re.compile(r"[А-Яа-яЁё]{3,}")
KEYS = ("transcript_words", "behavior_words")
# stored lists of another version, or matched against a Russian text that has changed since (a translation made
# again), are computed again
WORDS_VERSION = 2


def _content_word(w: str) -> bool:
    w = clean_word(w).lower()
    return len(w) >= 3 and w.isalpha() and w not in STOP_EN


def _source_word(ru_lemma: str, counts: Counter) -> str | None:
    """The word as the reader finds it in the Russian text: the most frequent word there sharing the lemma's stem
    (first 5 letters, 4 for short lemmas). 'сложный' -> 'сложная', 'ответ' -> 'ответить'."""
    lemma = clean_word(ru_lemma).lower().split()[0] if ru_lemma else ""
    if len(lemma) < 3:
        return None
    stem = lemma[: 5 if len(lemma) >= 6 else max(3, len(lemma) - 1)]
    best, best_n = None, 0
    for w, n in counts.items():
        if w.startswith(stem) and n > best_n:
            best, best_n = w, n
    return best


def _in_text(candidate: str, text_low: str, counts: Counter) -> str | None:
    """A Russian word or short phrase as it stands in the text: itself when the text contains it, otherwise the
    word of the text sharing its stem."""
    c = clean_word(candidate).lower()
    if c and re.search(rf"(?<![а-яё]){re.escape(c)}(?![а-яё])", text_low):
        return c
    return _source_word(c, counts)


def _vocab(expl: dict, key: str) -> List[str]:
    per = expl.get(key, {}).get("per_output", {})
    return sorted({clean_word(w["word"]) for d in per.values() for w in d["top_words"] if _content_word(w["word"])})


def russian_candidates(vocab: List[str], text_ru: str, context_en: str | None = None,
                       context_ru: str | None = None) -> Dict[str, List[str]]:
    """Russian forms to look for, per English word: first the word of the explained Russian text that renders it
    (LLM alignment of `context_en` with `context_ru`), then the wording the translation was asked to use for the
    VLM's stock words (translate.GLOSSARY_RU), then its dictionary translation (for words the alignment missed or
    changed). Empty when there is no Russian text or nothing could be translated."""
    if not vocab or not text_ru:
        return {}
    from .translate import GLOSSARY_RU, align_words, translate_words
    aligned: Dict[str, str] = {}
    dictionary: Dict[str, str] = {}
    try:
        aligned = align_words(vocab, context_en or "", context_ru or text_ru[:2500])
    except Exception:  # noqa: BLE001
        pass
    try:
        dictionary = translate_words(vocab, "en", "ru", context=context_en)
    except Exception:  # noqa: BLE001
        pass
    if not (aligned or dictionary):     # nothing translated: empty, so the lists are tried again later
        return {}
    # the glossary goes before the dictionary, which gives 'hesitation' -> «отлагательство»
    glossary = {en: ru.split(" (")[0] for en, ru in GLOSSARY_RU if " " not in en}
    out: Dict[str, List[str]] = {}
    for w in vocab:
        cands = [t for t in (aligned.get(w), glossary.get(w.lower()), dictionary.get(w)) if t]
        if cands:
            out[w] = cands
    return out


def readable_words(expl: dict, key: str, lang: str, text_ru: str | None = None, context_en: str | None = None,
                   context_ru: str | None = None, top_k: int = 5,
                   candidates: Dict[str, List[str]] | None = None) -> Dict[str, dict]:
    """{trait: {"up": [{"ru","en","source"}...], "down": [...]}} for expl[key]; content words only. `context_en`: the
    English text the words come from; `text_ru`: the Russian text shown to the reader, where each word is looked up
    ("source"; a word not found there is left out); `context_ru`: the part of it that renders `context_en`;
    `candidates`: russian_candidates() made by the caller."""
    per = expl.get(key, {}).get("per_output", {})
    if candidates is None:
        candidates = russian_candidates(_vocab(expl, key), text_ru or "", context_en, context_ru)
    text_low = (text_ru or "").lower()
    counts = Counter(m.group(0).lower() for m in _RU_WORD.finditer(text_ru or ""))
    out: Dict[str, dict] = {}
    for trait, d in per.items():
        best: Dict[str, dict] = {}          # displayed word -> item with the largest |effect| (several English
        for w in d["top_words"]:            # tokens may map to one Russian word: man / person -> человек)
            en = clean_word(w["word"])
            if not _content_word(en):
                continue
            cands = candidates.get(en) or []
            ru = cands[0] if cands else None
            src = next((f for f in (_in_text(c, text_low, counts) for c in cands) if f), None)
            if not src:                     # no translation, or not in the Russian text the reader sees
                continue
            item = {"en": en, "ru": ru, "source": src, "signed": float(w.get("signed", 0.0))}
            if src not in best or abs(item["signed"]) > abs(best[src]["signed"]):
                best[src] = item
        items = sorted(best.values(), key=lambda i: -abs(i["signed"]))
        up = [i for i in items if i["signed"] >= 0][:top_k]
        down = [i for i in items if i["signed"] < 0][:top_k]
        out[trait] = {"up": up, "down": down}
    return out


def _fingerprint(text: str | None) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def _russian_part(en_full: str, ru_full: str, en_part: str, limit: int = 2500) -> str:
    """The part of a Russian translation that renders `en_part` (the explained segment) of `en_full`: the line with
    the same number when the translation keeps the lines (behaviour description, one line per segment), otherwise
    the stretch at the same relative position."""
    en_part = (en_part or "").strip()
    if not en_part or not en_full:
        return ru_full[:limit]
    en_lines, ru_lines = en_full.split("\n"), ru_full.split("\n")
    if len(en_lines) == len(ru_lines):
        for e, r in zip(en_lines, ru_lines):
            if en_part[:200] in e:
                return re.sub(r"^\[[^\]]*\]\s*", "", r)[:limit]
    if len(ru_full) <= limit:
        return ru_full
    pos = en_full.find(en_part[:200])
    mid = (pos + len(en_part) / 2) / len(en_full) if pos >= 0 else 0.5
    start = max(0, min(len(ru_full) - limit, int(mid * len(ru_full)) - limit // 2))
    return ru_full[start:start + limit]


def _texts(expl: dict, rep: dict, key: str) -> Tuple[str, str, str, str]:
    """(English source text, English text the words come from, Russian text shown to the reader, its explained
    part) for one word list."""
    lang = (rep.get("model") or {}).get("lang", "en")
    if key == "behavior_words":
        en_full = rep.get("behavior_description") or expl.get("behavior_description") or ""
        ru_full = rep.get("behavior_description_ru") or ""
        return en_full, expl.get("behavior_description") or en_full, ru_full, _russian_part(
            en_full, ru_full, expl.get("behavior_description") or "")
    if lang != "en":            # Russian speech: the model read its English translation; the reader sees the speech
        ru_full = rep.get("transcript") or ""
        return (ru_full, expl.get("transcript_en") or rep.get("transcript_en") or "", ru_full,
                expl.get("transcript") or ru_full[:2500])
    en_full = rep.get("transcript") or expl.get("transcript") or ""
    ru_full = rep.get("transcript_ru") or ""
    return (en_full, expl.get("transcript") or en_full, ru_full,
            _russian_part(en_full, ru_full, expl.get("transcript") or ""))


def word_lists(expl: dict, rep: dict) -> dict:
    """readable_words for both word lists of an explanation, plus "meta": the version, a fingerprint of the Russian
    text each list was matched against, the number of attributed content words and whether everything needed was
    there (complete=False: a Russian text or the word translation was missing; the lists are computed again later)."""
    lang = (rep.get("model") or {}).get("lang", "en")
    out: dict = {}
    meta = {"version": WORDS_VERSION, "texts": {}, "content_words": 0, "complete": True}
    for key in KEYS:
        if key not in expl:
            continue
        source, context_en, text_ru, part_ru = _texts(expl, rep, key)
        vocab = _vocab(expl, key)
        candidates = russian_candidates(vocab, text_ru, context_en, part_ru)
        out[key] = readable_words(expl, key, lang, text_ru=text_ru, candidates=candidates)
        meta["texts"][key] = _fingerprint(text_ru)
        meta["content_words"] += len(vocab)
        if vocab and source.strip() and not candidates:     # no Russian text yet, or the words could not be
            meta["complete"] = False                          # translated: try again next time
    out["meta"] = meta
    return out


def lists_outdated(expl: dict, rep: dict) -> bool:
    """The stored word lists have to be computed (again): missing, of an older version, incomplete, or matched
    against a Russian text that has changed since."""
    meta = (expl.get("readable_words") or {}).get("meta") or {}
    if meta.get("version") != WORDS_VERSION or not meta.get("complete"):
        return any(k in expl for k in KEYS)
    return any(k in expl and meta.get("texts", {}).get(k) != _fingerprint(_texts(expl, rep, k)[2]) for k in KEYS)


def shown_word(item: dict) -> str | None:
    """The word as the interface shows it: as it stands in the Russian text next to the lists; None for items stored
    by older versions without that form (they are computed again before they are shown)."""
    return item.get("source") or None


def _label(item: dict, lang: str, show_en: bool) -> str:
    word = shown_word(item)
    return f"{word} ({item['en']})" if show_en else word


def format_words(rw: Dict[str, dict], titles: Dict[str, str], lang: str, show_en: bool = False) -> List[str]:
    """One line per trait: 'Открытость опыту — выше: сложная, ответ; ниже: подход'."""
    lines = []
    for trait in list(TRAIT_KEYS) + [k for k in rw if k not in TRAIT_KEYS]:
        if trait not in rw:
            continue
        parts = []
        up = [i for i in rw[trait]["up"] if shown_word(i)]
        down = [i for i in rw[trait]["down"] if shown_word(i)]
        if up:
            parts.append("выше: " + ", ".join(_label(i, lang, show_en) for i in up))
        if down:
            parts.append("ниже: " + ", ".join(_label(i, lang, show_en) for i in down))
        lines.append(f"{titles.get(trait, trait)} — " + ("; ".join(parts) if parts else "значимых слов нет"))
    return lines


def words_note(lang: str) -> str:
    """Note under the word lists (web page and PDF)."""
    return ("Слова из речи и из описания поведения, сильнее всего сдвинувшие оценку своей модели: «повышали» — сдвигали "
            "оценку черты вверх, «понижали» — вниз. Служебные слова отброшены. Каждое слово дано в той форме, в какой "
            "оно стоит в " + ("русском переводе транскрипта или описания поведения" if lang == "en"
                              else "транскрипте речи или в русском тексте описания поведения")
            + "; слова, которых там нет, не показаны.")


def words_missing_note(rw_all: dict | None) -> str:
    """Instead of the lists when the model did point at content words but none of them can be shown."""
    meta = (rw_all or {}).get("meta") or {}
    if not meta.get("content_words"):
        return ""
    if not meta.get("complete"):
        return ("Слова, сильнее всего повлиявшие на оценку своей модели, сейчас не удалось показать по-русски: не "
                "сработал перевод. Они появятся при следующем экспорте отчёта.")
    return ("Слова, сильнее всего повлиявшие на оценку своей модели, — служебные или не встречаются в русском тексте "
            "транскрипта и описания поведения, поэтому списка слов нет.")
