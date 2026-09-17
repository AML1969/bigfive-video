"""Translation helpers for showing results in the interface language. The models compute in English (text branch,
behaviour descriptions from the VLM, word attributions); the web page and the PDF show Russian translations instead
of the English originals, which stay in result.json. The interface is Russian for any speech language, so English
speech gets its transcript translated too.

Texts people read (behaviour description, transcript) are translated by the local LLM through Ollama
(translate_prose): Marian (Helsinki-NLP opus-mt) translates them word for word («спокойной и созидательной» for
'calm and composed'). Marian stays for the English input of the text branch (translate_text) and as the fallback when
Ollama cannot be reached."""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Tuple

import torch

log = logging.getLogger("bs.translate")
_models: dict = {}
_SENT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")
_PREFIX = re.compile(r"^(\[[^\]]*\]\s*)")          # "[0–20 с] " prefixes of segment descriptions
# Marian reads at most 512 tokens and translates long inputs badly well before that: a sentence longer than this is
# cut at commas, then between words (Whisper transcripts of fast speech can run on without a full stop)
MAX_PIECE_TOKENS = 150
LLM_MODEL = "qwen2.5vl:7b"          # the model that already writes the behaviour descriptions (no second model in memory)
# a transcript goes to the LLM in pieces of whole sentences up to this many characters, each with the piece before it
# as context (a 7B model shortens or stops translating long inputs)
LLM_PIECE_CHARS = 900
# one piece takes seconds; a request still waiting after this long (Ollama busy with other work) counts as "LLM
# unavailable": Marian translates the rest, the LLM is not asked again for LLM_RETRY_AFTER seconds, and the text is
# translated by the LLM again on a later render or export
LLM_TIMEOUT = 90
LLM_RETRY_AFTER = 300


def _get(src: str, tgt: str, device: str):
    key = (src, tgt)
    if key not in _models:
        from transformers import MarianMTModel, MarianTokenizer
        name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        tok = MarianTokenizer.from_pretrained(name)
        mdl = MarianMTModel.from_pretrained(name).to(device).eval()
        _models[key] = (tok, mdl)
        log.info("loaded %s on %s", name, device)
    return _models[key]


def _device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def translate_sentences(sentences: List[str], src="en", tgt="ru", device=None, batch: int = 16) -> List[str]:
    if not sentences:
        return []
    tok, mdl = _get(src, tgt, _device(device))
    out = []
    for i in range(0, len(sentences), batch):
        enc = tok(sentences[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=400).to(mdl.device)
        gen = mdl.generate(**enc, max_new_tokens=400, num_beams=2)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def _pieces(sentence: str, tok, limit: int = MAX_PIECE_TOKENS) -> List[str]:
    """A sentence as one piece, or cut into pieces of at most `limit` tokens: at commas first, then between words."""
    def fits(s):
        return len(tok(s, add_special_tokens=False)["input_ids"]) <= limit
    if fits(sentence):
        return [sentence]
    units = []                          # clauses, and groups of words for a clause that is too long by itself
    for clause in _CLAUSE.split(sentence):
        if fits(clause):
            units.append(clause)
            continue
        group = ""
        for w in clause.split():
            if group and not fits(group + " " + w):
                units.append(group)
                group = w
            else:
                group = (group + " " + w).strip()
        if group:
            units.append(group)
    out, cur = [], ""
    for u in units:                     # neighbouring clauses joined back while they fit
        if cur and not fits(cur + " " + u):
            out.append(cur)
            cur = u
        else:
            cur = (cur + " " + u).strip()
    return out + ([cur] if cur else [])


def translate_text(text: str, src="en", tgt="ru", device=None) -> str:
    """Line by line (keeps "[0–20 с]" prefixes and paragraph structure), sentence-batched; long sentences are cut
    into pieces that fit the model (see MAX_PIECE_TOKENS)."""
    lines = text.split("\n")
    tok, _ = _get(src, tgt, _device(device))
    pieces, plan = [], []              # plan: (prefix, start, count) per line
    for line in lines:
        m = _PREFIX.match(line)
        prefix = m.group(1) if m else ""
        body = line[len(prefix):].strip()
        sents = [p for s in _SENT.split(body) if s for p in _pieces(s, tok)] if body else []
        plan.append((prefix, len(pieces), len(sents)))
        pieces += sents
    tr = translate_sentences(pieces, src, tgt, device)
    return "\n".join(prefix + " ".join(tr[a:a + n]) for prefix, a, n in plan)


_YO = [(re.compile(r"\b([Ее])е\b"), r"\1ё"), (re.compile(r"\b([Нн])ее\b"), r"\1её"), (re.compile(r"\b([Ее])ще\b"), r"\1щё")]


def yo(text: str) -> str:
    """«ее», «нее», «еще» -> «её», «неё», «ещё» (these words are always written so; translators drop the dots)."""
    for pattern, repl in _YO:
        text = pattern.sub(repl, text)
    return text


class LLMUnavailable(RuntimeError):
    """Ollama could not be reached or has no such model (as opposed to an answer that fails the checks)."""


_LLM_SYSTEM = "Ты — профессиональный переводчик с английского языка на русский. Отвечай только по-русски, кириллицей."
# the VLM repeats a few stock phrases; their usual Russian wording keeps the translated description natural, and
# words.py looks for the single words among them in the Russian text (the 7B model does not always align them)
GLOSSARY_RU = (("calm and composed", "спокойный и собранный"), ("engaged with", "занят (чем-то)"),
               ("casual demeanor", "непринуждённая манера держаться"), ("demeanor", "манера держаться"),
               ("reflective or thoughtful", "задумчивый"), ("upright posture", "прямая осанка"),
               ("attentiveness", "внимательность"), ("neutrality", "нейтральность"), ("appears", "выглядит"),
               ("hesitation", "нерешительность"), ("ambivalence", "двойственность"), ("hand gestures", "жесты рук"),
               ("facial expression", "выражение лица"), ("mouth slightly open", "рот слегка приоткрыт"),
               ("detail-oriented", "внимательный к деталям"), ("openness", "открытость"),
               ("emotional cues", "эмоциональные сигналы"))
_LLM_TASK = {
    "description": ("Ниже — описание поведения человека на видео, на английском. Переведи его на естественный русский "
                    "язык, как написал бы русский психолог. Сохрани смысл точно: ничего не добавляй, не убирай и не "
                    "пересказывай. Используй привычные русские формулировки, например: "
                    + "; ".join(f"{en} — {ru}" for en, ru in GLOSSARY_RU)
                    + ". Пиши букву ё (её, неё, ещё). Ответь только русским переводом, без кавычек и пояснений."),
    "transcript": ("Ниже — автоматическая расшифровка речи человека из видео, на английском: разговорная речь, возможны "
                   "повторы, текст может обрываться на полуслове. Сначала пойми, о чём весь текст, затем переведи его на "
                   "естественный разговорный русский, выбирая значения слов по теме текста. Сохрани смысл и тон, ничего "
                   "не исправляй, не дописывай и не сокращай. Пиши букву ё. Ответь только русским переводом, без кавычек "
                   "и пояснений."),
}
_CJK = re.compile("[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")    # CJK, Hangul, full-width forms
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


_llm_down_until = 0.0


def _ollama(path: str, payload: dict) -> dict:
    """POST to the local Ollama. LLMUnavailable when it cannot be reached, times out or has no such model; for
    LLM_RETRY_AFTER seconds after that every request fails at once, so a stopped or busy Ollama costs one timeout,
    not one per text and word list."""
    global _llm_down_until
    import time
    import urllib.error
    import urllib.request

    from .backend_mm import default_ollama_url
    if time.time() < _llm_down_until:
        raise LLMUnavailable("did not answer a moment ago")
    req = urllib.request.Request(f"{default_ollama_url()}{path}", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
            body = r.read()
    except (urllib.error.URLError, OSError) as e:        # refused, timed out, HTTP 404 for a missing model
        _llm_down_until = time.time() + LLM_RETRY_AFTER
        raise LLMUnavailable(str(e)) from e
    return json.loads(body.decode("utf-8"))


def _llm_chat(model: str, user: str, temperature: float, num_predict: int) -> str:
    data = _ollama("/api/chat", {"model": model, "stream": False, "think": False, "keep_alive": "30m",
                                 "messages": [{"role": "system", "content": _LLM_SYSTEM},
                                              {"role": "user", "content": user}],
                                 "options": {"temperature": temperature, "num_predict": num_predict}})
    return str((data.get("message") or {}).get("content") or "")


def _usable(out: str, src: str) -> bool:
    """An LLM answer the page can show: Cyrillic, no Chinese characters (a Qwen habit), no English word except names
    kept from the source, and about as long as the source (not a summary, not a runaway)."""
    if not re.search(r"[А-Яа-яЁё]", out) or _CJK.search(out):
        return False
    names = {w for w in _LATIN_WORD.findall(src) if w[0].isupper() and w != "I"}
    if any(w not in names for w in _LATIN_WORD.findall(out)):
        return False
    return 0.6 <= len(out) / max(1, len(src)) <= 2.0


def _llm_piece(model: str, piece: str, kind: str, context: str = "") -> str | None:
    """One piece through the LLM; None when both attempts fail the checks. Raises LLMUnavailable."""
    user = _LLM_TASK[kind] + (f"\n\nПредыдущий фрагмент, только для понимания (его не переводи):\n{context}\n\n"
                              f"Фрагмент для перевода:\n{piece}" if context else f"\n\n{piece}")
    for temperature in (0.0, 0.3):          # the second attempt differs a little from a failed first answer
        try:
            out = _llm_chat(model, user, temperature, num_predict=min(4096, 300 + len(piece)))
        except ValueError:                  # not JSON
            continue
        out = re.sub(r"^\s*(?:русский\s+)?перевод\s*:\s*", "", out.strip(), flags=re.I)
        out = re.sub(r"\s+", " ", out.strip().strip("\"'«»“”")).strip()
        if _usable(out, piece):
            return out
    log.info("LLM translation of a %s piece failed the checks; Marian translates it", kind)
    return None


def _sentence_groups(text: str, limit: int) -> List[str]:
    """Whole sentences joined into pieces of at most `limit` characters (a longer sentence stays one piece)."""
    out, cur = [], ""
    for s in (x for x in _SENT.split(text) if x):
        if cur and len(cur) + 1 + len(s) > limit:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    return out + ([cur] if cur else [])


def translate_prose(text: str, kind: str = "description", fallback: bool = True,
                    model: str = LLM_MODEL) -> Tuple[str | None, str]:
    """English text people read -> natural Russian. kind="description": the behaviour description, line by line
    ("[0–20 с]" prefixes kept); kind="transcript": speech, in pieces of whole sentences with the previous piece as
    context. A piece whose LLM answer fails the checks is translated by Marian. Returns (translation, engine):
    engine "llm" when the LLM answered, "marian" when Ollama could not be reached and Marian translated the rest;
    with fallback=False that case returns (None, "") instead (the caller keeps what it has)."""
    llm_ok = True
    lines = []
    for line in text.split("\n"):
        m = _PREFIX.match(line)
        prefix = m.group(1) if m else ""
        body = line[len(prefix):].strip()
        if not body:
            lines.append(line.strip())
            continue
        pieces = [body] if kind == "description" else _sentence_groups(body, LLM_PIECE_CHARS)
        done = []
        for i, piece in enumerate(pieces):
            ru = None
            if llm_ok:
                try:
                    ru = _llm_piece(model, piece, kind, context=pieces[i - 1] if i else "")
                except LLMUnavailable as e:
                    log.warning("LLM translation unavailable (%s)%s", str(e).splitlines()[0][:80],
                                "; using Marian" if fallback else "")
                    llm_ok = False
                    if not fallback:
                        return None, ""
            done.append(ru if ru is not None else translate_text(piece, "en", "ru"))
        ru_line = " ".join(done)
        if kind == "transcript" and re.search(r"[,;:–—-]\s*$", body):     # speech recognised up to a cut-off
            ru_line = re.sub(r"[\s.,;:–—-]*$", "", ru_line) + "…"
        lines.append(prefix + yo(ru_line))
    return "\n".join(lines), ("llm" if llm_ok else "marian")


def _context_sentence(word: str, text: str) -> str:
    """First sentence of `text` containing the word (for sense disambiguation), or ''."""
    if not text:
        return ""
    core = re.escape(word.strip(" .,;:!?\"'()[]{}«»—-"))
    if not core:
        return ""
    for sent in _SENT.split(text):
        if re.search(rf"(?i)(?<![A-Za-z]){core}(?![A-Za-z])", sent):
            return sent.strip()[:200]
    return ""


def _ollama_dictionary(words: List[str], tgt: str, model: str = LLM_MODEL, context: str | None = None) -> Dict[str, str]:
    """Dictionary-form translations from the local Ollama model (a sentence MT model turns 'calm' into
    'успокойся'; an LLM asked for lemmas gives 'спокойный'). A context sentence per word disambiguates the sense."""
    lang_name = {"ru": "Russian"}.get(tgt, tgt)
    items = [{"word": w, "context": _context_sentence(w, context or "")} for w in words]
    examples = {"ru": '{"hesitation": "нерешительность", "calm": "спокойный", "his": "его", "gestures": "жесты", '
                      '"imagine": "представлять", "storage": "хранение", "work": "работа"}'}.get(tgt, "{}")
    prompt = (f"You are a bilingual English-{lang_name} dictionary. For each English word below give its {lang_name} "
              f"dictionary translation in base form (nouns: nominative singular; verbs: infinitive; adjectives: "
              f"masculine singular), choosing the sense used in the context sentence when one is given. Translate "
              f"the meaning of the word itself, never a different word from the sentence. One to three words, no "
              f"explanations. Example of the expected output: {examples}\nAnswer with a JSON object mapping each "
              f"English word exactly as written to its translation.\nItems: {json.dumps(items, ensure_ascii=False)}")
    data = _ollama("/api/generate", {"model": model, "prompt": prompt, "stream": False, "format": "json",
                                     "think": False, "keep_alive": "30m",
                                     "options": {"temperature": 0.0, "num_predict": 1500}})
    obj = json.loads(data.get("response") or data.get("thinking") or "{}")
    out = {}
    for w in words:
        t = obj.get(w) or obj.get(w.lower()) or obj.get(w.capitalize())
        if isinstance(t, str) and valid_translation(t, tgt):
            out[w] = t.strip().strip(".")
    return out


def valid_translation(t, tgt: str) -> bool:
    """A usable display translation: short, and for Russian written in Cyrillic without stray Latin letters
    (small models sometimes answer 'Ero' for 'His')."""
    if not isinstance(t, str) or not (0 < len(t.strip()) <= 40):
        return False
    if tgt == "ru":
        return re.search(r"[А-Яа-яЁё]", t) is not None and re.search(r"[A-Za-z]", t) is None
    return True


def align_words(words: List[str], text_en: str, text_ru: str, model: str = LLM_MODEL) -> Dict[str, str]:
    """{English word: the word (or short phrase) of `text_ru` that renders it in `text_en`, copied as written there}
    from the local LLM; words it finds no counterpart for are left out. A dictionary gives 'promising' ->
    «обещающий»; the speech said «перспективным», and that is the word the reader can find."""
    uniq = sorted({w.strip() for w in words if w and w.strip()})
    if not uniq or not text_en.strip() or not text_ru.strip():
        return {}
    prompt = ("Below are an English text and its Russian version. For each English word in the list, find the word in "
              "the Russian text that expresses its meaning, and copy that Russian word exactly as it is written in the "
              "Russian text (same form and ending). If the Russian text has no word for it, answer with an empty "
              "string. Answer with a JSON object mapping each English word exactly as written to the Russian word.\n\n"
              f"English text: {text_en.strip()}\n\nRussian text: {text_ru.strip()}\n\nWords: {json.dumps(uniq)}")
    data = _ollama("/api/generate", {"model": model, "prompt": prompt, "stream": False, "format": "json",
                                     "think": False, "keep_alive": "30m",
                                     "options": {"temperature": 0.0, "num_predict": 1500}})
    obj = json.loads(data.get("response") or "{}")
    out = {}
    for w in uniq:
        t = obj.get(w) or obj.get(w.lower()) or obj.get(w.capitalize())
        if isinstance(t, str) and valid_translation(t, "ru") and len(t.split()) <= 3:
            out[w] = t.strip().strip(".,;:«»\"")
    return out


def translate_words(words: List[str], src="en", tgt="ru", device=None, context: str | None = None) -> Dict[str, str]:
    """Single words/short phrases -> {word: translation}. Ollama dictionary first (base forms, sense taken from
    `context` = the text the words come from), Marian for the rest."""
    uniq = sorted({w.strip() for w in words if w and w.strip()})
    if not uniq:
        return {}
    out: Dict[str, str] = {}
    try:
        out = _ollama_dictionary(uniq, tgt, context=context)
    except Exception as e:  # noqa: BLE001
        log.warning("Ollama dictionary translation failed (%s); using Marian", str(e).splitlines()[0][:80])
    missing = [w for w in uniq if w not in out]
    if missing:
        tr = translate_sentences(missing, src, tgt, device)
        for w, t in zip(missing, tr):
            t = t.strip().strip(".").strip()
            if valid_translation(t, tgt) and len(t) <= 4 * max(3, len(w)):   # a runaway translation is worse than none
                out[w] = t
    return out
