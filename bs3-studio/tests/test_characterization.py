"""«Характеристика личности» (design 8, 13.1 test_characterization; task T14).

The lexicon rules of 8.5, and on synthetic views (ru samples A and B, en, without analyses, without the second system,
all traits typical, all extreme, main system missing): length, forbidden words, no raw scores, letters agree with the
MBTI section, the closing paragraph, the provisional reference group, no type name when the type is not expressed,
determinism. The verbatim golden texts (tests/golden/char_A.txt, char_B.txt) are compared once they exist (task T25).
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

from samples import english, rep

from bs3 import characterization as C
from bs3 import mbti, refnorms, scores
from bs3.norms import TRAIT_KEYS

LEVELS = ("high", "above", "mid", "below", "low")
STOP = ("диагноз", "расстройств", "патолог", "норма", "нормальн", "отклонени", "плохо", "хорошо", "пригод", "рекоменд",
        "тревожн", "является", "склонен", "сегмент")
PRONOUNS = ("он", "она", "его", "её", "ее", "ему", "ей", "него", "неё", "нему", "ней")
GOLDEN = Path(__file__).resolve().parent / "golden"
# the verbatim design texts (8.4-8.6, caveats of 11) give about 720-770 words for the two samples, more than the
# 400-550 the design estimated; the upper bound leaves room for the longest combination (see the stage notes)
MIN_WORDS, MAX_WORDS = 150, 850

# numbers only (voice, speech, face and text-emotion means of the two samples), no identities
AN = {
    "A": {"speech": {"words_per_min_speech": 99.8, "pause_share": 0.166, "words": 498},
          "voice": {"mean": {"arousal": 0.3557, "dominance": 0.3981, "valence": 0.4287}},
          "face": {"mean": {"happy": 0.5067, "surprise": 0.0806, "neutral": 0.1034, "sad": 0.024, "fear": 0.1605,
                            "angry": 0.1049, "disgust": 0.0198}},
          "emotions_text": {"mean": {"joy": 0.2518, "surprise": 0.014, "neutral": 0.6873, "sadness": 0.02,
                                     "fear": 0.0027, "anger": 0.0129, "disgust": 0.0114}}},
    "B": {"speech": {"words_per_min_speech": 61.3, "pause_share": 0.144, "words": 570},
          "voice": {"mean": {"arousal": 0.1536, "dominance": 0.2612, "valence": 0.3717}},
          "face": {"mean": {"happy": 0.0359, "surprise": 0.0196, "neutral": 0.6512, "sad": 0.0188, "fear": 0.2452,
                            "angry": 0.0277, "disgust": 0.0015}},
          "emotions_text": {"mean": {"joy": 0.0611, "surprise": 0.0049, "neutral": 0.8604, "sadness": 0.0233,
                                     "fear": 0.0036, "anger": 0.0281, "disgust": 0.0185}}},
}


def _with_analyses(r: dict, name: str) -> dict:
    r["analyses"] = copy.deepcopy(AN[name])
    return r


def _build(r: dict):
    v = scores.clean_view(r)
    mb = mbti.get_mbti(r, v)
    return v, mb, C.build(v, mb)


def _group_values(system: str, trait: str) -> list:
    return refnorms.load_ru_prov()["sources"][system][trait]


def _cases() -> dict:
    out = {"A": _with_analyses(rep("A"), "A"), "B": _with_analyses(rep("B"), "B"),
           "en": _with_analyses(english("B"), "B"), "no_analyses": rep("B")}
    r = _with_analyses(rep("B"), "B")
    del r["variant_scores"]["mm"]
    out["no_second"] = r
    r = _with_analyses(rep("B"), "B")                 # every trait at the median of the group: all axes on the border
    for k in TRAIT_KEYS:
        vals = sorted(_group_values("oceanai", k))
        r["variant_scores"]["oceanai"][k] = vals[len(vals) // 2]
    out["all_mid"] = r
    r = _with_analyses(rep("A"), "A")                 # every trait above the whole group
    for k in TRAIT_KEYS:
        r["variant_scores"]["oceanai"][k] = max(_group_values("oceanai", k)) + 0.01
    out["all_high"] = r
    r = _with_analyses(rep("B"), "B")                 # OCEAN-AI gave nothing: the own model becomes the main system
    del r["variant_scores"]["oceanai"]
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    out["primary_missing"] = r
    return out


# --------------------------------------------------------------------------------------------------- lexicon ---

def test_lexicon_complete_and_follows_the_rules():
    lex = C.load_lexicon()
    assert lex["lexicon_version"] == 1
    assert set(lex["levels"]) == set(TRAIT_KEYS)
    for k in TRAIT_KEYS:
        assert set(lex["levels"][k]) == set(LEVELS), k
        assert set(lex["names"][k]) == {"title", "acc", "dat"}, k
        for lv in LEVELS:
            entry = lex["levels"][k][lv]
            where = f"{k}/{lv}"
            assert 2 <= len(entry) <= 4, where
            for s in entry:
                low = s.lower()
                assert s[-1] in ".!?", where
                assert not re.search(r"\d", s), where
                for w in STOP:          # at the start of a word: «проявляется» (openness/high) is not «является»
                    assert not re.search(r"(?<![а-яё])" + w, low), (where, w)
                tokens = re.findall(r"[а-яё]+", low)
                for p in PRONOUNS:
                    assert p not in tokens, (where, p)
            if lv in ("high", "low"):
                assert entry[-1].startswith("В беседе"), where


def test_short_phrases_poles():
    lex = C.load_lexicon()
    assert set(lex["short"]) == {"extraversion", "openness", "agreeableness", "conscientiousness"}
    assert lex["short"]["extraversion"] == {"high": "общительным и энергичным", "low": "сдержанным и немногословным"}
    assert lex["short"]["conscientiousness"]["low"] == "свободным и спонтанным в манере"
    for d in lex["short"].values():
        for s in d.values():
            for w in STOP:
                assert w not in s


# ------------------------------------------------------------------------------------------ synthetic views ---

def test_every_case_reads_within_the_rules():
    names16 = set(mbti.load_config()["type_names_ru"].values())
    for case, r in _cases().items():
        v, mb, ch = _build(r)
        text = ch.plain()
        n = ch.word_count()
        assert MIN_WORDS <= n <= MAX_WORDS, (case, n)
        low = text.lower()
        for w in ("сегмент", "определяет тип", "диагноз"):
            assert w not in low, (case, w)
        # «ключ: число» in the paragraphs (the header label «тип не выражен: 3 оси из 4 на границе» is fixed by 8.3)
        body_text = "\n".join(p["lead"] + " " + p["text"] for p in ch.paragraphs)
        assert not re.search(r"\b\w+:\s*\d", body_text), (case, re.search(r"\b\w+:\s*\d", body_text).group(0))
        for p in ch.paragraphs:
            body = p["lead"] + " " + p["text"]
            if p["key"] == "behavior":
                assert not re.search(r"0\.\d{2}", body), case
                rest = re.sub(r"\d+ (слово|слова|слов) в минуту", "", body)
                assert not re.findall(r"\d+(?![\d%])", rest), (case, rest)       # only shares with %
            else:
                assert not re.search(r"0\.\d{2}", body), (case, p["key"])
        assert ch.paragraphs[-1]["key"] == "limits" and ch.paragraphs[-1]["lead"] == "Границы вывода."
        assert ch.paragraphs[0]["key"] == "short" and ch.paragraphs[1]["key"] == "basis"
        if v["view_meta"]["lang"] == "ru":
            assert "предварительная" in text and "а не относительно населения" in text, case
            assert "пороги предварительные" in ch.header_plain(), case
        else:
            assert "предварительн" not in text, case
        # letters agree with the MBTI section
        x = mb["x_count"]
        shown = mb["type_strict"] if x <= 2 else mb["type"]
        assert ch.header["letters_text"] == shown, case
        typology = ch.paragraph("mbti")["text"]
        assert mb["type_strict"] in typology, case
        if x:
            assert mb["type"] in typology, case
        for word in re.findall(r"\b[EIXSNTFJP]{4}\b", text):
            allowed = {mb["type"], mb["type_strict"], *mb["alternatives"]}
            for s in mb.get("second") or []:
                allowed |= {s["type"], s["type_strict"]}
            assert word in allowed, (case, word)
        if x >= 3:
            assert not any(nm in text for nm in names16), case
        # deterministic
        v2, mb2, ch2 = _build(copy.deepcopy(r))
        assert ch2.plain() == text and ch2.html() == ch.html(), case


def test_sample_b_matches_the_design_example():
    _, mb, ch = _build(_with_analyses(rep("B"), "B"))
    assert ch.header_plain() == ("ESFJ «Попечитель» · MBTI по OCEAN-AI · ось S–N на границе · нейротизм: ниже "
                                 "типичного · пороги предварительные")
    short = ch.short_plain()
    assert short == ("По первому впечатлению от записи человек выглядит заметно собранным и организованным, а также "
                     "заметно общительным и энергичным. В нотации MBTI ближе всего тип ESFJ («Попечитель»), ось S–N на "
                     "границе. Нейротизм — ниже типичного. Вторая система расходится с основной по оси E–I (её тип — "
                     "ISXX).")
    assert [p["key"] for p in ch.paragraphs] == [
        "short", "basis", "trait:conscientiousness", "trait:extraversion", "trait:agreeableness", "trait:openness",
        "stability", "mbti", "behavior", "agreement", "limits"]
    assert "по 26 отрезкам записи" in ch.paragraph("basis")["text"]
    e = ch.paragraph("trait:extraversion")
    assert e["lead"] == "Экстраверсия — заметно выше типичного"
    assert e["text"].startswith("(выше, чем у большинства из 13 русских роликов; в MBTI — буква E, отчётливо). ")
    assert e["text"].endswith("Вторая система оценивает эту черту противоположно.")
    o = ch.paragraph("trait:openness")
    assert o["text"] == ("(примерно посередине среди 13 русских роликов; в MBTI ось S–N на границе). По открытости "
                         "опыту человек не выделяется из группы сравнения: интерес к новому и опора на привычное "
                         "выглядят уравновешенными.")
    for k in ("conscientiousness", "agreeableness"):
        assert "противоположно" not in ch.paragraph(f"trait:{k}")["text"]
    st = ch.paragraph("stability")
    assert st["lead"] == "Эмоциональная устойчивость — выше типичного; нейротизм, соответственно, — ниже типичного"
    typ = ch.paragraph("mbti")["text"]
    assert typ.startswith("В нотации MBTI ближе всего тип ESFJ («Попечитель»), но ось S–N на границе, поэтому точнее "
                          "записать EXFJ: возможен и тип ENFJ («Наставник»).")
    assert typ.endswith("По ходу записи буква по оси S–N совпадает с итоговой в 16 из 26 отрезков с оценкой; по "
                        "остальным осям — во всех.")
    beh = ch.paragraph("behavior")["text"]
    assert beh.startswith("Темп речи медленный (61 слово в минуту), паузы умеренные. Голос по модели эмоций в речи "
                          "ровный и спокойный. Выражение лица, которое модель распознаёт чаще всего, — нейтральное (65% "
                          "кадров). По содержанию речь в основном нейтральна по эмоциональной окраске. Сдержанный голос "
                          "и неторопливая речь не вполне согласуются с оценкой экстраверсии — стоит посмотреть запись.")
    agr = ch.paragraph("agreement")["text"]
    assert ("с основной почти не согласна: по открытости опыту, добросовестности, доброжелательности и эмоциональной "
            "устойчивости одна из систем даёт уровень около типичного; по экстраверсии системы расходятся в "
            "противоположные стороны — этот вывод наименее надёжен.") in agr
    assert ("По её оценкам тип — ISXX (ближайший ISTJ); ни одна буква не совпадает уверенно: по оси E–I буквы "
            "расходятся, по S–N, T–F и J–P хотя бы одна система на границе.") in agr
    assert agr.endswith("(проверка на 13 роликах), поэтому основной считается OCEAN-AI.")
    lim = ch.paragraph("limits")["text"]
    assert "В 7 отрезках из 33 система OCEAN-AI не дала оценки" in lim
    assert lim.endswith("а не пересказывают эпизоды этого ролика.")


def test_sample_a():
    _, mb, ch = _build(_with_analyses(rep("A"), "A"))
    assert ch.header_plain() == ("ISTP «Мастер» · MBTI по OCEAN-AI · нейротизм: заметно выше типичного · пороги "
                                 "предварительные")
    assert "В нотации MBTI это тип ISTP («Мастер»)." in ch.short_plain()
    assert ch.paragraph("mbti")["text"].startswith("В нотации MBTI профиль соответствует типу ISTP («Мастер»).")
    assert "во всех 17 отрезках с оценкой" in ch.paragraph("mbti")["text"]
    assert [p["key"] for p in ch.paragraphs][2:6] == ["trait:openness", "trait:extraversion", "trait:agreeableness",
                                                      "trait:conscientiousness"]
    for k in ("openness", "extraversion", "agreeableness", "conscientiousness"):
        assert ch.paragraph(f"trait:{k}")["lead"].endswith("заметно ниже типичного")
    assert "Сдержанный голос" not in ch.plain() and "Оживлённый голос" not in ch.plain()
    assert "радость (51% кадров)" in ch.paragraph("behavior")["text"]
    assert "XXXJ (ближайший ENFJ)" in ch.paragraph("agreement")["text"]
    assert "В 1 отрезке из 18" in ch.paragraph("limits")["text"]


def test_special_cases():
    cases = _cases()
    _, _, ch = _build(cases["no_analyses"])
    assert ch.paragraph("behavior") is None
    _, _, ch = _build(cases["no_second"])
    assert ch.paragraph("agreement")["text"] == "Второе мнение для этого ролика недоступно."
    assert "Вторая система" not in ch.short_plain()
    _, mb, ch = _build(cases["all_mid"])
    assert mb["type"] == "XXXX" and len(mb["alternatives"]) == 15
    assert "тип не выражен: 4 оси из 4 на границе" in ch.header_plain() and ch.header["name"] is None
    assert "Тип MBTI по этой записи не выражен (XXXX)." in ch.short_plain()
    assert ch.paragraph("mbti")["text"].startswith("По всем четырём осям значения близки к границе (XXXX)")
    assert "ни одна из четырёх черт, связанных с MBTI, не выделяется" in ch.short_plain()
    _, mb, ch = _build(cases["all_high"])
    assert mb["type"] == "ENFJ"
    assert all(ch.paragraph(f"trait:{k}")["lead"].endswith("заметно выше типичного")
               for k in ("openness", "conscientiousness", "extraversion", "agreeableness"))
    _, mb, ch = _build(cases["primary_missing"])
    assert mb["source"] == "own_model"
    assert "MBTI по своей модели" in ch.header_plain()
    assert "по оценкам своей модели MM-PSYCHE" in ch.paragraph("basis")["text"]
    assert "Основная система OCEAN-AI не дала оценок по этому ролику" in ch.paragraph("limits")["text"]
    _, mb, ch = _build(cases["en"])
    assert "MBTI по среднему двух систем" in ch.header_plain()
    assert "First Impressions V2 (6000 роликов)" in ch.paragraph("basis")["text"]
    assert "Системы по отдельности" in ch.paragraph("agreement")["text"]
    assert "проверка на" not in ch.paragraph("agreement")["text"]


def test_no_big_five():
    r = rep("B")
    r["traits"] = {}
    r["variant_scores"] = {}
    v = scores.clean_view(r)
    mb = mbti.get_mbti(r, v)
    assert mb is None
    ch = C.build(v, mb)
    assert "Тип MBTI не рассчитан: в результате нет оценок Big Five." in ch.plain()
    assert ch.paragraphs[-1]["key"] == "limits"


def test_html_and_pdf_forms():
    _, _, ch = _build(_with_analyses(rep("B"), "B"))
    h = ch.html()
    assert h.count("<p ") == len(ch.paragraphs)
    assert "<b>Коротко.</b>" in h and "font-size:36px" in h
    assert "E<span style='opacity:.55;text-decoration:underline 2px dashed #808080" in h     # the S on the border
    assert "border:1px dashed #808080" in h and "ось S–N на границе" in h
    assert "<script" not in h
    pdf = ch.pdf_paragraphs()
    assert pdf[0]["markdown"].startswith("**Коротко.** По первому впечатлению")
    hd = ch.pdf_header()
    assert hd["letters"] == [("E", False), ("S", True), ("F", False), ("J", False)] and hd["name"] == "Попечитель"
    assert "Здесь появится характеристика личности" in C.placeholder_html()


def test_golden_texts_when_present():
    """Verbatim comparison with tests/golden/char_A.txt, char_B.txt (saved after the owner's reading, task T25)."""
    for name in ("A", "B"):
        path = GOLDEN / f"char_{name}.txt"
        if not path.exists():
            continue
        _, _, ch = _build(_with_analyses(rep(name), name))
        assert ch.plain() == path.read_text(encoding="utf-8"), name
