"""Russian texts of older English jobs (3.1): the speech is Russian, so nothing of English speech is made or translated
any more. The Russian texts an English job carries are only read: ru_texts.ensure_russian leaves them alone and calls
no translator, transcript_shown and vocabulary_shown return them as stored."""
from __future__ import annotations

import copy
import inspect

from samples import english

import bs3.translate
from bs3 import ru_texts
from bs3.analyses import speech_stats

TRANSCRIPT = "I work at a small company. My work is about new tools, and I like my work. The company is small."
TRANSCRIPT_RU = ("Я работаю в небольшой компании. Моя работа связана с новыми инструментами, и мне нравится моя работа. "
                 "Компания небольшая.")
VOCABULARY_RU = [["работа", 3, ["work"]], ["компания", 2, ["company"]], ["инструмент", 1, ["tools"]]]


def _english_job(by: str = "ollama") -> tuple:
    """(result.json, explanation.json) of an older English job with every Russian text stored; `by` marks the
    translated transcript and the vocabulary, the other texts are marked "ollama"."""
    r = english()
    r["transcript"] = TRANSCRIPT
    r["transcript_ru"], r["transcript_ru_by"] = TRANSCRIPT_RU, by
    r["behavior_description"] = "[0–20 с] The person speaks calmly and keeps eye contact."
    r["behavior_description_ru"] = "[0–20 с] Человек говорит спокойно и поддерживает зрительный контакт."
    r["behavior_description_ru_by"] = "ollama"
    r["analyses"] = {"speech": {"words": 20, "vocabulary": [["work", 3], ["company", 2]],
                                "vocabulary_ru": copy.deepcopy(VOCABULARY_RU), "vocabulary_ru_by": by}}
    per = {"per_output": {"extraversion": {"top_words": [{"word": "work", "signed": 0.2}]}}}
    shown = {"extraversion": {"up": [{"en": "work", "ru": "работа", "source": None, "signed": 0.2}], "down": []}}
    expl = {"transcript": TRANSCRIPT, "transcript_words": per, "behavior_words": copy.deepcopy(per),
            "readable_words": {"transcript_words": shown, "behavior_words": copy.deepcopy(shown)},
            "readable_words_by": "ollama"}
    return r, expl


def _ensure_without_translator(rep: dict, expl: dict) -> tuple:
    """ensure_russian with every public function of bs3.translate replaced by one that fails when called."""
    saved, called = {}, []
    for name, fn in list(vars(bs3.translate).items()):
        if name.startswith("_") or not inspect.isfunction(fn):
            continue

        def stub(*_a, _name=name, **_k):
            called.append(_name)
            raise AssertionError(f"bs3.translate.{_name} was called")
        saved[name] = fn
        setattr(bs3.translate, name, stub)
    assert {"ollama_available", "translate_description", "translate_words", "translate_sentences"} <= set(saved)
    try:
        changed = ru_texts.ensure_russian(rep, expl)
    finally:
        for name, fn in saved.items():
            setattr(bs3.translate, name, fn)
    return changed, called


def test_ensure_russian_translates_nothing_of_an_english_job():
    # "marian": made by Marian while Ollama was unreachable. Before 3.1 such a transcript and vocabulary were
    # translated again as soon as Ollama answered; now they are kept as they are, without asking Ollama
    for by in ("ollama", "marian"):
        rep, expl = _english_job(by)
        rep0, expl0 = copy.deepcopy(rep), copy.deepcopy(expl)
        changed, called = _ensure_without_translator(rep, expl)
        assert changed == (False, False) and called == [], (by, called)
        assert rep == rep0 and expl == expl0, by


def test_english_job_without_translation_stays_so():
    rep, expl = _english_job()
    del rep["transcript_ru"], rep["transcript_ru_by"], rep["analyses"]["speech"]["vocabulary_ru"]
    del rep["analyses"]["speech"]["vocabulary_ru_by"]
    changed, called = _ensure_without_translator(rep, expl)
    assert changed == (False, False) and called == []
    assert "transcript_ru" not in rep and "vocabulary_ru" not in rep["analyses"]["speech"]
    assert ru_texts.transcript_shown(rep) == (ru_texts.NO_TRANSCRIPT_TRANSLATION, "")
    assert ru_texts.vocabulary_shown(rep) == []


def test_shown_texts_are_the_stored_ones():
    rep, _ = _english_job()
    assert ru_texts.transcript_shown(rep) == (ru_texts.TRANSCRIPT_NOTE, TRANSCRIPT_RU)
    # words said at least MIN_FREQUENT times, as stored (the Russian word and its count)
    assert ru_texts.vocabulary_shown(rep) == [("работа", 3), ("компания", 2)]


def test_english_speech_generation_is_gone():
    for name in ("ensure_transcript", "ensure_vocabulary", "_speaker_note", "_title_words", "_NOT_NAMES", "_TITLE_GLUE"):
        assert not hasattr(ru_texts, name), name
    for name in ("translate_transcript", "translate_long", "_script_runs", "_LAT", "_REPEAT", "_patch_latin_runs"):
        assert not hasattr(bs3.translate, name), name
    assert list(inspect.signature(bs3.translate.llm_translate).parameters) == ["texts"]
    assert list(inspect.signature(bs3.translate._llm_answer_ok).parameters) == ["en", "ru"]
    for name in ("_en_content", "STOP_EN_SPEECH"):
        assert not hasattr(speech_stats, name), name
    assert "STOP_EN" not in inspect.getsource(speech_stats)
    assert "lang" not in inspect.signature(speech_stats.vocabulary).parameters
    assert "en" in speech_stats.FILLERS                # English fillers still count in Russian speech
