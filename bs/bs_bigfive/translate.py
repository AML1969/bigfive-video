"""Marian (Helsinki-NLP opus-mt) helpers for showing results in the interface language. The models compute in
English (text branch, behaviour descriptions from the VLM, word attributions); the web page and the PDF show
Russian translations next to / instead of the English originals, which stay in result.json."""
from __future__ import annotations

import logging
import re
from typing import Dict, List

import torch

log = logging.getLogger("bs.translate")
_models: dict = {}
_SENT = re.compile(r"(?<=[.!?])\s+")
_PREFIX = re.compile(r"^(\[[^\]]*\]\s*)")          # "[0–20 с] " prefixes of segment descriptions


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


def translate_text(text: str, src="en", tgt="ru", device=None) -> str:
    """Line by line (keeps "[0–20 с]" prefixes and paragraph structure), sentence-batched."""
    lines = text.split("\n")
    pieces, plan = [], []              # plan: (prefix, start, count) per line
    for line in lines:
        m = _PREFIX.match(line)
        prefix = m.group(1) if m else ""
        body = line[len(prefix):].strip()
        sents = [s for s in _SENT.split(body) if s] if body else []
        plan.append((prefix, len(pieces), len(sents)))
        pieces += sents
    tr = translate_sentences(pieces, src, tgt, device)
    return "\n".join(prefix + " ".join(tr[a:a + n]) for prefix, a, n in plan)


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


def _ollama_dictionary(words: List[str], tgt: str, model: str = "qwen2.5vl:7b", context: str | None = None) -> Dict[str, str]:
    """Dictionary-form translations from the local Ollama model (a sentence MT model turns 'calm' into
    'успокойся'; an LLM asked for lemmas gives 'спокойный'). A context sentence per word disambiguates the sense."""
    import json
    import urllib.request

    from .backend_mm import default_ollama_url
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
    payload = {"model": model, "prompt": prompt, "stream": False, "format": "json", "think": False,
               "keep_alive": "30m", "options": {"temperature": 0.0, "num_predict": 1500}}
    req = urllib.request.Request(f"{default_ollama_url()}/api/generate", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read().decode("utf-8"))
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
