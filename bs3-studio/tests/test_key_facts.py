"""«Ключевые факты» of BS Profiler 3.1: the card reads value, label, explanation, and the value of a measured card is
coloured by where it sits — green around neutral, blue below, dark orange above (change request «Ключевые факты» of
2026-09-26). Here: the three states of every card type at the edges of their bands, the cards that are never coloured
(«Тип MBTI», «Длительность ролика»), the row order on the page and in the PDF, and the line under the grid."""
from __future__ import annotations

import re

from bs3 import narrative2, palette, scores, webapp
from bs3.narrative2 import card_item, key_facts
from bs3.scores import ABOVE, BELOW, NEUTRAL, emotion_state, scale_state, tempo_state


def _rep(*, emotion="neutral", face="neutral", arousal=0.50, wpm=120.0, interview=0.50, words=300) -> dict:
    """A result.json-like dict with just the fields key_facts reads."""
    rep = {"duration_sec": 660.0, "segments": 33,
           "analyses": {"emotions_text": {"mean": {emotion: 0.9, "joy": 0.05}},
                        "face": {"mean": {face: 0.8, "joy": 0.1}},
                        "voice": {"mean": {"arousal": arousal, "dominance": 0.40, "valence": 0.49}},
                        "speech": {"words": words, "words_per_min_speech": wpm, "pause_share": 0.21,
                                   "fillers_per_100": 2.0}}}
    if interview is not None:
        rep["interview"] = {"score": interview}
    return rep


def _by_label(rep: dict) -> dict:
    return {card_item(f)[0]: card_item(f) for f in key_facts(rep)}


# --------------------------------------------------------------------------------- the three states of each card

def test_scale_state_edges_follow_the_level_bands():
    assert [scale_state(v) for v in (0.00, 0.20, 0.35)] == [BELOW, BELOW, BELOW]
    assert [scale_state(v) for v in (0.36, 0.50, 0.64)] == [NEUTRAL, NEUTRAL, NEUTRAL]
    assert [scale_state(v) for v in (0.65, 0.80, 1.00)] == [ABOVE, ABOVE, ABOVE]
    assert scale_state(None) is None and scale_state("нет") is None
    # the colour never contradicts the words the page prints for the same number
    for v in (0.35, 0.36, 0.64, 0.65):
        assert (scores.level(v) == "mid") == (scale_state(v) == NEUTRAL)
    # decided on the printed two decimals, as the levels are
    assert scale_state(0.6449) == NEUTRAL and scale_state(0.6451) == ABOVE


def test_tempo_state_edges():
    assert scores.TEMPO_BAND == (100, 160)
    assert [tempo_state(n) for n in (40, 99)] == [BELOW, BELOW]
    assert [tempo_state(n) for n in (100, 130, 160)] == [NEUTRAL, NEUTRAL, NEUTRAL]
    assert [tempo_state(n) for n in (161, 250)] == [ABOVE, ABOVE]
    assert tempo_state(None) is None
    assert tempo_state(99.6) == NEUTRAL and tempo_state(160.4) == NEUTRAL      # on the whole number the card prints


def test_emotion_state_of_every_class_and_of_the_face_labels():
    assert emotion_state("neutral") == NEUTRAL
    for k in ("sadness", "fear", "disgust"):
        assert emotion_state(k) == BELOW, k
    for k in ("joy", "surprise", "anger"):
        assert emotion_state(k) == ABOVE, k
    # the face expressions share palette.EMO_ALIAS with the text emotions
    for face, text in palette.EMO_ALIAS.items():
        assert emotion_state(face) == emotion_state(text), face
    assert emotion_state("что-то ещё") is None


# --------------------------------------------------------------------------------- the cards themselves

def test_key_facts_states_of_every_card():
    cards = _by_label(_rep(emotion="sadness", face="joy", arousal=0.70, wpm=90.0, interview=0.30))
    assert card_item(cards["Эмоция по тексту речи"])[3] == BELOW
    assert card_item(cards["Выражение лица"])[3] == ABOVE
    assert card_item(cards["Голос, шкала 0…1"])[3] == ABOVE
    assert card_item(cards["Темп речи"])[3] == BELOW
    assert card_item(cards["Впечатление «собеседование»"])[3] == BELOW
    cards = _by_label(_rep(emotion="neutral", face="neutral", arousal=0.50, wpm=130.0, interview=0.50))
    for lab in ("Эмоция по тексту речи", "Выражение лица", "Голос, шкала 0…1", "Темп речи",
                "Впечатление «собеседование»"):
        assert cards[lab][3] == NEUTRAL, lab
    cards = _by_label(_rep(emotion="joy", face="anger", arousal=0.65, wpm=161.0, interview=0.65))
    for lab in ("Эмоция по тексту речи", "Выражение лица", "Голос, шкала 0…1", "Темп речи",
                "Впечатление «собеседование»"):
        assert cards[lab][3] == ABOVE, lab


def test_the_cards_that_are_never_coloured():
    from bs3 import mbti
    cards = _by_label(_rep())
    # «Длительность ролика» is not a measurement of the person
    assert cards["Длительность ролика"][3] is None
    # «Тип MBTI» comes from mbti.fact_card, which carries three fields and no state
    mb = {"type": "ENFJ", "type_strict": "ENFJ", "type_name": "Наставник", "x_count": 0, "model": "mm"}
    card = mbti.fact_card(mb)
    assert len(card) == 3 and card_item(card)[3] is None


def test_a_value_without_a_band_is_not_coloured():
    rep = _rep(wpm=None)
    cards = _by_label(rep)
    assert cards["Темп речи"][1].endswith("слов") and cards["Темп речи"][3] is None   # a word count has no band
    rep = _rep()
    rep.pop("interview")
    assert "Впечатление «собеседование»" not in _by_label(rep)


# --------------------------------------------------------------------------------- the page

def _divs(card_html: str) -> list:
    return re.findall(r"<div(?: class='[^']*')? style='(?:font-size:20px|font-size:13px)[^']*'>([^<]*)</div>",
                      card_html)


def test_page_card_reads_value_label_explanation():
    html = webapp._cards([("Темп речи", "90 слов в минуту", "паузы — 21% времени", BELOW)], value_first=True)
    assert _divs(html) == ["90 слов в минуту", "Темп речи", "паузы — 21% времени"]
    assert "class='bs3-fact-below'" in html
    # a card without a note is two rows
    two = webapp._cards([("Длительность ролика", "11:00", "", None)], value_first=True)
    assert _divs(two) == ["11:00", "Длительность ролика"] and "bs3-fact-" not in two
    # the other card grids of the page keep label, value, note
    plain = webapp._cards([("Слов всего", "300", "без повторов")])
    assert _divs(plain) == ["Слов всего", "300", "без повторов"] and "bs3-fact-" not in plain


def test_facts_block_carries_the_colours_and_the_line_under_the_grid():
    html = webapp._facts_html(_rep(emotion="joy", arousal=0.20, wpm=200.0))
    for state in scores.FACT_STATES:
        assert f".bs3-fact-{state}{{color:{palette.FACT_VALUE['light'][state]}}}" in html
        assert f".dark .bs3-fact-{state}{{color:{palette.FACT_VALUE['dark'][state]}}}" in html
    assert "class='bs3-fact-above'" in html and "class='bs3-fact-below'" in html
    assert narrative2.FACTS_LEGEND in html and "зелёный — около нейтрального" in html
    # the line belongs under the grid, not above it
    assert html.index(narrative2.FACTS_LEGEND) > html.rindex("<div style='padding:10px")
    assert webapp._facts_html({}) == ""


# --------------------------------------------------------------------------------- the PDF

def _draw_cards(items, **kw):
    """(text, ink) of every line the card grid prints, ink as the (r, g, b) the card asked for."""
    from bs3 import pdf_report

    class Recorder(pdf_report.Report):
        _depth = 0                      # fpdf re-dispatches multi_cell to itself: record the outermost call only

        def __init__(self):
            super().__init__(file_label="t", total_pages=1)
            self.drawn, self.ink = [], (0, 0, 0)

        def set_text_color(self, r, g=None, b=None):
            self.ink = (r, r, r) if g is None else (r, g, b)
            super().set_text_color(r) if g is None else super().set_text_color(r, g, b)

        def multi_cell(self, *a, **kw):
            if not self._depth and not kw.get("dry_run"):
                text = kw.get("text", kw.get("txt", a[2] if len(a) > 2 else ""))
                self.drawn.append((str(text), self.ink))
            self._depth += 1
            try:
                return super().multi_cell(*a, **kw)
            finally:
                self._depth -= 1

    pdf = Recorder()
    pdf.add_page()
    pdf.cards(items, **kw)
    return pdf.drawn


def test_pdf_card_prints_the_value_first_and_in_colour():
    from bs3 import pdf_report
    drawn = _draw_cards([("Темп речи", "200 слов в минуту", "паузы — 21% времени", ABOVE)], cols=1, value_first=True)
    assert [t for t, _ in drawn] == ["200 слов в минуту", "Темп речи", "паузы — 21% времени"]
    orange = tuple(int(palette.FACT_VALUE_PDF[ABOVE].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    assert drawn[0][1] == orange and drawn[1][1] == drawn[2][1] == (pdf_report.NOTE_GREY,) * 3
    # a card with no state keeps black ink, and «Речь в цифрах» keeps label, value, note
    plain = _draw_cards([("Длительность ролика", "11:00", "", None)], cols=1, value_first=True)
    assert [t for t, _ in plain] == ["11:00", "Длительность ролика"] and plain[0][1] == (0, 0, 0)
    speech = _draw_cards([("Слов всего", "300", "без повторов")], cols=1)
    assert [t for t, _ in speech] == ["Слов всего", "300", "без повторов"] and speech[1][1] == (0, 0, 0)


def test_pdf_card_height_does_not_depend_on_the_order():
    from bs3 import pdf_report
    pdf = pdf_report.Report(file_label="t", total_pages=1)
    pdf.add_page()
    items = [("Темп речи", "200 слов в минуту", "паузы — 21% времени", ABOVE),
             ("Длительность ролика", "11:00", "разбит на 33 отрезка", None)]
    plain = [(a, b, c) for a, b, c, _ in items]
    assert abs(pdf.cards_height(items, 2) - pdf.cards_height(plain, 2)) < 1e-9
