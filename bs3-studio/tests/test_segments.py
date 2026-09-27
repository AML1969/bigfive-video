"""bs3/segments.py (refactoring plan of 3.1, stage 12): the per-segment helpers left pdf_report, pdf_charts, charts
and narrative and give what the copies gave; each place that shows the representative segment keeps its own rule.

OLD_RULES keeps the five lookups of the representative segment as they were written before the move; every run
compares them with `representative` over a grid of timelines with 0 to 4 segments, each scored or not.
"""
from __future__ import annotations

import inspect
import itertools
import math
import subprocess
import sys
from pathlib import Path

from bs3 import (charts, frame_captions, labels, narrative, pdf_charts, pdf_mbti, pdf_report, scores, segments,
                 textfmt, webapp)
from bs3.norms import TRAIT_KEYS
from bs3.segments import (behavior_by_segment, dominant_emotion, emotion_shares, empty_text, odd_segments,
                          representative, scored, seg_words, segment_rows)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- the lookups before stage 12
def _old_scored(rep):                 # charts._segments = pdf_report._scored = the filters in narrative
    return [t for t in (rep.get("timeline") or []) if t.get("scores")]


def _old_charts(rep):                 # charts.fig_traits_timeline: among the scored, any number of them
    segs = _old_scored(rep)
    rep_i = rep.get("representative_segment")
    return next((t for t in segs if t["segment"] == rep_i), None) if rep_i else None


def _old_pdf_charts(rep):             # pdf_charts._traits_chart: more than one scored
    segs = _old_scored(rep)
    rep_i = rep.get("representative_segment")
    return next((t for t in segs if t["segment"] == rep_i), None) if rep_i and len(segs) > 1 else None


def _old_pdf_report(report):          # pdf_report._rep_segment: at least two scored
    segs = _old_scored(report)
    if len(segs) < 2:
        return None
    return next((t for t in segs if t.get("segment") == report.get("representative_segment")), None)


def _old_whole(rep):                  # webapp._frames_html and frame_captions.moments: the whole timeline
    tl_all = rep.get("timeline") or []
    return next((t for t in tl_all if t.get("segment") == rep.get("representative_segment")), None) if tl_all else None


OLD_RULES = {"charts": (_old_charts, {}), "pdf_charts": (_old_pdf_charts, {"min_scored": 2}),
             "pdf_report": (_old_pdf_report, {"min_scored": 2}), "webapp": (_old_whole, {"among": "all"}),
             "frame_captions": (_old_whole, {"among": "all"})}


def _scores(v: float) -> dict:
    return {k: v for k in TRAIT_KEYS}


def _grid():
    """Timelines of 0…4 segments numbered from 1, every one scored or a gap (scores None or {}), with the
    representative segment missing, None, 0, each number, one past the end and a string."""
    for n in range(5):
        for mask in itertools.product((True, False), repeat=n):
            for rep_i in ("missing", None, 0, *range(1, n + 2), "2"):
                tl = [{"segment": i, "start": 20.0 * (i - 1), "end": 20.0 * i,
                       "scores": _scores(0.4 + 0.05 * i) if ok else (None if i % 2 else {})}
                      for i, ok in enumerate(mask, 1)]
                view = {"timeline": tl}
                if rep_i != "missing":
                    view["representative_segment"] = rep_i
                yield view
    yield {}
    yield {"timeline": None, "representative_segment": 1}


def test_representative_keeps_the_rule_of_every_caller():
    n = 0
    for view in _grid():
        for name, (old, kw) in OLD_RULES.items():
            assert representative(view, **kw) is old(view), (name, view)
            n += 1
    assert n > 1000


def test_representative_by_the_number_of_scored_segments():
    one = {"timeline": [{"segment": 1, "start": 0, "end": 20, "scores": _scores(0.5)},
                        {"segment": 2, "start": 20, "end": 40, "scores": None}], "representative_segment": 1}
    two = {"timeline": [{"segment": 1, "start": 0, "end": 20, "scores": _scores(0.5)},
                        {"segment": 2, "start": 20, "end": 40, "scores": None},
                        {"segment": 3, "start": 40, "end": 60, "scores": _scores(0.6)}], "representative_segment": 3}
    # the web chart marks the one scored segment; the PDF only from two scored segments on
    assert representative(one) is one["timeline"][0]
    assert representative(one, min_scored=2) is None
    assert representative(two, min_scored=2) is two["timeline"][2]
    # a representative segment without a score: the key frames still find it in the whole timeline
    gap = {**two, "representative_segment": 2}
    assert representative(gap) is None and representative(gap, min_scored=2) is None
    assert representative(gap, among="all") is gap["timeline"][1]
    assert representative({"timeline": [], "representative_segment": 1}, among="all") is None
    try:
        representative(two, among="every")
    except ValueError:
        pass
    else:
        raise AssertionError("among='every' was accepted")


def test_a_segment_without_a_number_is_never_the_representative():
    """Malformed timelines (no entry of a real job lacks its number): the old lookups matched an entry without a
    number when the job had no representative segment (None == None), and the two charts raised KeyError on it."""
    for missing in (True, False):
        tl = [{"segment": 1, "start": 0.0, "end": 20.0, "scores": _scores(0.5)},
              {"segment": None, "start": 20.0, "end": 40.0, "scores": _scores(0.6)},
              {"segment": 3, "start": 40.0, "end": 60.0, "scores": _scores(0.4)}]
        if missing:
            del tl[1]["segment"]
        for kw in ({}, {"min_scored": 2}, {"among": "all"}):
            assert representative({"timeline": tl}, **kw) is None
            assert representative({"timeline": tl, "representative_segment": None}, **kw) is None
            assert representative({"timeline": tl, "representative_segment": 2}, **kw) is None
            assert representative({"timeline": tl, "representative_segment": 3}, **kw) is tl[2]


def test_the_callers_pass_their_rules():
    def src(fn) -> str:
        return inspect.getsource(fn)
    assert "representative(rep)" in src(charts.fig_traits_timeline)
    assert "representative(rep, min_scored=2)" in src(pdf_charts._traits_chart)
    assert inspect.getsource(pdf_report).count("representative(report, min_scored=2)") == 4
    assert "representative(rep, among=\"all\")" in src(webapp._frames_html)
    assert "representative(report, among=\"all\")" in src(frame_captions.moments)
    # the same by behaviour: one scored segment is marked on the web chart, not in the PDF appendix
    one = {"duration_sec": 40.0, "representative_segment": 1,
           "timeline": [{"segment": 1, "start": 0.0, "end": 20.0, "scores": _scores(0.5)},
                        {"segment": 2, "start": 20.0, "end": 40.0, "scores": None}]}
    fig = charts.fig_traits_timeline(one, "light")
    assert [s.x0 for s in fig.layout.shapes if s.line.dash == "dash"] == [0.0]
    assert "отрезок для объяснений ★" not in sum(pdf_report._notable(one, True).values(), [])
    two = {**one, "timeline": [one["timeline"][0], {**one["timeline"][1], "scores": _scores(0.6)}]}
    assert pdf_report._notable(two, True) == {0.0: ["отрезок для объяснений ★"]}
    # the key frames count their moment from the start of the representative segment, scored or not
    gap = {**one, "representative_segment": 2, "media": {"fps": 10.0}}
    assert frame_captions.moments(gap, ["key_0_frame50.jpg"])[0] == [25.0]


def test_scored_equals_the_old_filters():
    for view in _grid():
        assert scored(view) == _old_scored(view)
        assert all(a is b for a, b in zip(scored(view), _old_scored(view)))


def test_the_scored_segments_come_from_segments():
    """The web and PDF charts, the PDF and «Как получены оценки» take the scored segments from segments.scored and
    keep no filter of their own: a gap stored as `scores: {}` is no scored segment anywhere."""
    assert "segs = scored(rep)" in inspect.getsource(charts.fig_traits_timeline)
    assert "segs = scored(rep)" in inspect.getsource(pdf_charts._traits_chart)
    assert "tl = scored(view)" in inspect.getsource(narrative.method_notes)
    assert inspect.getsource(pdf_report).count("segments.scored(report)") == 2
    view = {"view_meta": {"main_system": "mm"}, "scores_std_across_segments": dict.fromkeys(TRAIT_KEYS, 0.01),
            "timeline": [{"segment": 1, "start": 0.0, "end": 20.0, "scores": _scores(0.5)},
                         {"segment": 2, "start": 20.0, "end": 40.0, "scores": {}},
                         {"segment": 3, "start": 40.0, "end": 60.0, "scores": _scores(0.52)}]}
    assert "По ходу ролика (2 отрезка с оценкой AMLAI 1.0) оценки устойчивы" in narrative.method_notes(view)


# ---------------------------------------------------------------- one segment
ROWS = {
    "spoken": {"text_en": "we spoke", "emotions_text": {"neutral": 0.2, "joy": 0.7, "fear": 0.1},
               "face": {"expressions": {"happy": 0.3, "sad": 0.6, "neutral": 0.1}}, "speech": {"words": 12}},
    "silent": {"text_en": "  ", "emotions_text": {"neutral": 1.0}, "speech": {"words": 0}},
    "no_text": {"text_en": "", "emotions_text": {"neutral": 1.0}, "speech": {"words": 5}},
    "odd_values": {"emotions_text": {"joy": None, "neutral": "0.5", "anger": "x"}, "speech": {"words": "12"}},
    "unknown": {"emotions_text": {"joy": -0.1, "contempt": 0.9}, "speech": {"words": "x"}},
    "tie": {"face": {"expressions": {"angry": "0.4", "happy": 0.4}}, "speech": None},
    "zero": {"emotions_text": {"joy": 0.0, "neutral": 0.0}, "face": {"expressions": None}, "speech": {"words": 3.7}},
}


def _close(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return a is b
    return list(a) == list(b) and all(math.isclose(a[k], b[k], abs_tol=1e-12) for k in a)


def test_emotion_shares():
    z = dict.fromkeys(labels.HEAT_ROWS, 0.0)
    assert list(z) == ["joy", "surprise", "sadness", "fear", "anger", "disgust", "neutral"]
    assert _close(emotion_shares(ROWS["spoken"], "text"), {**z, "joy": 0.7, "fear": 0.1, "neutral": 0.2})
    assert _close(emotion_shares(ROWS["spoken"], "face"), {**z, "joy": 0.3, "sadness": 0.6, "neutral": 0.1})
    assert emotion_shares(ROWS["silent"], "text") is None and emotion_shares(ROWS["no_text"], "text") is None
    assert _close(emotion_shares(ROWS["odd_values"], "text"), {**z, "neutral": 1.0})
    assert emotion_shares(ROWS["unknown"], "text") is None            # a negative share and a label of no row
    assert _close(emotion_shares(ROWS["tie"], "face"), {**z, "joy": 0.5, "anger": 0.5})
    assert emotion_shares(ROWS["zero"], "text") is None and emotion_shares(ROWS["zero"], "face") is None
    assert emotion_shares({}, "text") is None and emotion_shares({"face": None}, "face") is None


def test_dominant_emotion():
    assert dominant_emotion(ROWS["spoken"], "text") == ("joy", 0.7)
    assert dominant_emotion(ROWS["spoken"], "face") == ("sadness", 0.6)       # sad -> sadness
    assert dominant_emotion(ROWS["silent"], "text") is None
    assert dominant_emotion(ROWS["odd_values"], "text") == ("neutral", 0.5)
    assert dominant_emotion(ROWS["unknown"], "text") == ("contempt", 0.9)     # an unknown label passes through
    assert dominant_emotion(ROWS["tie"], "face") == ("anger", 0.4)            # the first of equal values
    assert dominant_emotion(ROWS["zero"], "text") == ("joy", 0.0)
    assert dominant_emotion(ROWS["zero"], "face") is None
    # the appendix names it as before
    assert [pdf_report._dominant_text(ROWS[k], "text") for k in ("spoken", "silent", "no_text", "unknown")] == [
        "радость 70%", "нет речи", "нет текста", "contempt 90%"]


def test_empty_text_and_words():
    assert [empty_text(ROWS[k]) for k in ("spoken", "silent", "no_text", "odd_values")] == [False, True, True, False]
    assert empty_text({"text_en": None}) is True
    assert [seg_words(ROWS[k]) for k in ROWS] == [12, 0, 5, 12, 0, 0, 3]


def test_segment_rows_and_behaviour():
    t0 = {"segment": 1, "start": 0, "end": 20}
    t1 = {"segment": 2, "start": 20, "end": 40.4}
    t2 = {"segment": 4, "start": 60.4, "end": 80}
    r0, r1, r2 = {"start": 0.2, "end": 20.0}, {"start": 40.4, "end": 50}, {"start": 59.6, "end": 80}
    rows = segment_rows({"timeline": [t0, t1, t2], "analyses": {"per_segment": [r1, r0, r2]}})
    # matched by the start rounded to a second, in time order; the times are the ones of the timeline entry
    assert [(s, e) for s, e, _, _ in rows] == [(0.0, 20.0), (20.0, 40.4), (40.4, 50.0), (60.4, 80.0)]
    assert [(t, r) for _, _, t, r in rows] == [(t0, r0), (t1, None), (None, r1), (t2, r2)]
    assert segment_rows({}) == []
    text = "[0–20 с] Улыбается. [20-40 s] Кивает.\n[1,5–3 с] Смотрит в камеру"
    assert behavior_by_segment({"behavior_description_ru": text}) == [
        (0.0, 20.0, "Улыбается."), (20.0, 40.0, "Кивает."), (1.5, 3.0, "Смотрит в камеру")]
    assert behavior_by_segment({"behavior_description_ru": "без меток"}) == []
    assert behavior_by_segment({}) == []


def _old_odd(rep):                    # narrative.odd_segments before stage 12
    import numpy as np
    tl = [t for t in (rep.get("timeline") or []) if t.get("scores")]
    means = np.array([np.mean([t["scores"][k] for k in TRAIT_KEYS]) for t in tl])
    if len(means) < 4 or means.std() <= 0:
        return []
    z = (means - means.mean()) / means.std()
    return [(t, float(z_)) for t, z_ in zip(tl, z) if abs(z_) > 2.0]


def test_odd_segments():
    views = []
    for n, out in ((6, 5), (6, 0), (4, 1), (3, 1), (5, None)):
        tl = [{"segment": i + 1, "start": 20.0 * i, "end": 20.0 * i + 20,
               "scores": _scores(0.95 if i == out else 0.5 + 0.01 * (i % 3))} for i in range(n)]
        tl.insert(1, {"segment": 99, "start": 5.0, "end": 6.0, "scores": None})
        views.append({"timeline": tl})
    for v in views:
        assert [(t is o, z == oz) for (t, z), (o, oz) in zip(odd_segments(v), _old_odd(v))] == \
            [(True, True)] * len(_old_odd(v))
        assert len(odd_segments(v)) == len(_old_odd(v))
    found = odd_segments(views[0])
    assert len(found) == 1 and found[0][0]["start"] == 100.0 and found[0][1] > 2.0
    assert odd_segments(views[3]) == [] and odd_segments(views[4]) == []
    assert narrative.odd_segments is odd_segments                   # method_notes takes it from segments


def test_the_copies_are_gone():
    gone = {pdf_report: ("_empty_text", "_seg_words", "_scored", "_rep_segment", "_segment_rows"),
            pdf_charts: ("_num", "_segments", "_empty_text", "_seg_words"), charts: ("_segments",)}
    for mod, names in gone.items():
        for name in names:
            assert not hasattr(mod, name), f"{mod.__name__}.{name}"
    for mod in (charts, pdf_charts, pdf_report, narrative, frame_captions, webapp):
        for name in ("scored", "representative", "empty_text", "seg_words", "emotion_shares", "dominant_emotion",
                     "segment_rows", "behavior_by_segment", "odd_segments", "as_float", "HEAT_ROWS"):
            obj = getattr(segments, name)
            if mod is pdf_report and name == "scored":
                obj = scores.scored     # the interview entry with a number; the segments are segments.scored there
            assert getattr(mod, name, obj) is obj, f"{mod.__name__}.{name} is a copy"
    assert "segments.scored(report)" in inspect.getsource(pdf_report._segments_table)
    assert pdf_charts.HEAT_ROWS is labels.HEAT_ROWS and segments.SEC_LABEL is textfmt.SEC_LABEL
    assert math.isnan(segments.as_float(None)) and math.isnan(segments.as_float("x"))
    assert segments.as_float("0.5") == 0.5 and segments.as_float(True) == 1.0


def test_the_data_modules_stay_light():
    """segments loads no numpy until odd_segments runs; the PDF modules load no matplotlib, plotly or torch on
    import."""
    code = ("import sys\n"
            "import bs3.segments\n"
            "print(sorted(m for m in ('numpy', 'matplotlib', 'fpdf') if m in sys.modules))\n"
            "import bs3.pdf_report\n"
            "print(sorted(m for m in ('matplotlib', 'plotly', 'torch', 'cv2', 'gradio') if m in sys.modules))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split("\n")[:2] == ["[]", "[]"], r.stdout
