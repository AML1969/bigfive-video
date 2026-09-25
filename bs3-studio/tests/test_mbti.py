"""MBTI notation of Big Five (design 4, 5, 7; tests 1-14 of design 13.1; change of 2026-09-26: the customer's formula
on the absolute score 0…1 for both systems and both languages, no reference group)."""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import tempfile
import time
from pathlib import Path

from samples import english, rep

from bs3 import mbti, scores
from bs3.norms import TRAIT_KEYS

DOC = {"extraversion": 0.72, "openness": 0.61, "agreeableness": 0.48, "conscientiousness": 0.55,
       "emotional_stability": 0.67}


def _conf(res):
    return [round(res["axes"][ax]["confidence"], 2) for ax in mbti.AXES]


def _one(v, thr=0.5):
    return mbti.bigfive_to_mbti({"extraversion": v}, {"EI": thr})["axes"]["EI"]


# 1
def test_document_example_raw():
    r = mbti.bigfive_to_mbti(DOC)
    assert r["type"] == "EXXX" and r["type_strict"] == "ENTJ"
    assert _conf(r) == [0.44, 0.22, 0.04, 0.10]
    assert [r["axes"][ax]["borderline"] for ax in mbti.AXES] == [False, True, True, True]
    assert r["neuroticism"]["value"] == 0.33
    for system, lang in (("oceanai", "ru"), ("mm", "ru"), ("mean", "en"), ("oceanai", "en")):
        m = mbti.mbti_for(system, DOC, lang)
        assert m["type"] == "EXXX" and m["type_strict"] == "ENTJ" and m["type_name"] == "Руководитель"
        assert [m["axes"][ax]["confidence"] for ax in mbti.AXES] == [0.44, 0.22, 0.04, 0.10]
        assert [m["axes"][ax]["value"] for ax in mbti.AXES] == [0.72, 0.61, 0.48, 0.55]
        assert all(m["axes"][ax]["threshold"] == 0.5 for ax in mbti.AXES)
        assert m["neuroticism"]["value"] == 0.33 and m["neuroticism"]["level"] == "ниже среднего"


# 2
def test_boundaries():
    a = _one(0.5)
    assert a["letter"] == "E" and a["borderline"] and a["confidence"] == 0
    assert not _one(0.65)["borderline"] and _one(0.65)["letter"] == "E"
    assert not _one(0.35)["borderline"] and _one(0.35)["letter"] == "I"
    assert _one(0.6499)["borderline"] and _one(0.3501)["borderline"]
    assert _one(0.0)["confidence"] == 1 and _one(1.0)["confidence"] == 1


# 3
def test_other_threshold():
    assert _one(1.0, 0.6)["confidence"] == 1 and _one(0.0, 0.6)["confidence"] == 1
    assert _one(0.6, 0.6)["letter"] == "E" and _one(0.59, 0.6)["letter"] == "I"
    for i in range(101):
        v = i / 100
        assert math.isclose(_one(v)["confidence"], min(abs(v - 0.5) / 0.5, 1.0), abs_tol=1e-8)


# 4
def test_all_axes_borderline():
    r = mbti.bigfive_to_mbti({k: 0.5 for k in TRAIT_KEYS})
    assert r["type"] == "XXXX" and len(r["type_strict"]) == 4 and "X" not in r["type_strict"]
    assert r["x_count"] == 4
    assert len(r["alternatives"]) == 15 and len(set(r["alternatives"])) == 15
    assert r["type_strict"] not in r["alternatives"]


# 5
def test_degenerate_inputs():
    for bad in (None, float("nan"), "abc", float("inf"), True):
        r = mbti.bigfive_to_mbti({**DOC, "extraversion": bad})
        assert r["type"][0] == "X" and r["type_strict"][0] == "X" and r["axes"]["EI"]["missing"]
        assert r["type_name"] is None
        assert r["alternatives"] and all(t[0] == "X" for t in r["alternatives"])
    r = mbti.bigfive_to_mbti({k: v for k, v in DOC.items() if k != "extraversion"})
    assert r["axes"]["EI"]["missing"] and r["type_strict"][0] == "X"
    r = mbti.bigfive_to_mbti({**DOC, "extraversion": 1.3, "openness": -0.2})
    assert r["axes"]["EI"]["value"] == 1.0 and r["axes"]["EI"]["clipped"]
    assert r["axes"]["SN"]["value"] == 0.0 and r["axes"]["SN"]["clipped"]
    empty = mbti.bigfive_to_mbti({})
    assert empty["type"] == "XXXX" and empty["type_strict"] == "XXXX" and empty["alternatives"] == []
    assert empty["neuroticism"] is None and empty["type_name"] is None
    assert mbti.bigfive_to_mbti({k: v for k, v in DOC.items() if k != "emotional_stability"})["neuroticism"] is None
    m = mbti.mbti_for("oceanai", {"extraversion": None}, "ru")
    assert m["type_strict"] == "XXXX" and m["neuroticism"] is None and m["axes"]["EI"]["word"] == "нет данных"


# 6
def test_neuroticism():
    raw = {**DOC, "emotional_stability": 0.30}
    n = mbti.mbti_for("oceanai", raw, "ru")["neuroticism"]
    assert n["value"] == 0.70 and n["level"] == "выше среднего" and "position" not in n
    b = rep("B")["variant_scores"]["oceanai"]
    m = mbti.mbti_for("oceanai", b, "ru")
    assert m["neuroticism"]["value"] == round(1 - b["emotional_stability"], 3)
    mirror = {"high": "low", "above": "below", "mid": "mid", "below": "above", "low": "high"}
    for i in range(101):
        p = i / 100
        assert scores.level(1 - p) == mirror[scores.level(p)], p
    assert "средний уровень" in m["neuroticism_note"] and "0." not in m["neuroticism_note"]


# 7
def test_absolute_scale_config():
    """No reference group: method "raw", thresholds 0.5, the same letters for any system and language."""
    cfg = mbti.load_config()
    assert cfg["method"] == "raw" and cfg["raw_thresholds"] == {"EI": 0.5, "SN": 0.5, "TF": 0.5, "JP": 0.5}
    for key in ("references", "ru_prov_file", "thresholds"):
        assert key not in cfg, key
    assert cfg["borderline"] == 0.15 and mbti.SCHEMA_VERSION == 2 and mbti.METHOD == "raw"
    b = rep("B")["variant_scores"]
    for system in ("oceanai", "mm"):
        types = {mbti.mbti_for(system, b[system], lang)["type"] for lang in ("ru", "en")}
        assert len(types) == 1, system
        m = mbti.mbti_for(system, b[system], "ru")
        for ax in mbti.AXES:
            a = m["axes"][ax]
            assert a["value"] == scores.shown(b[system][a["trait"]])          # the printed value, two decimals
            assert set(a) <= {"trait", "value", "threshold", "letter", "confidence", "borderline", "word", "missing",
                              "clipped"}
        assert "reference" not in m


# 8
def test_alternatives():
    axes = {"EI": {"borderline": False}, "SN": {"borderline": True}, "TF": {"borderline": False},
            "JP": {"borderline": False}}
    assert mbti.alternatives("ESFJ", axes) == ["ENFJ"]
    axes["JP"] = {"borderline": True}
    assert mbti.alternatives("ESFJ", axes) == ["ESFP", "ENFJ", "ENFP"]
    axes["JP"] = {"borderline": True, "missing": True}
    assert mbti.alternatives("ESFX", axes) == ["ENFX"]


# 9
def test_agreement():
    def m(source, letters, border=()):
        return {"source": source, "axes": {ax: {"letter": letters[i], "borderline": ax in border}
                                           for i, ax in enumerate(mbti.AXES)}}
    a = mbti.agreement(m("ocean_ai", "ESFJ", {"SN"}), m("own_model", "ISTJ", {"TF", "JP"}))
    assert a["axes"] == {"EI": "differ", "SN": "border", "TF": "border", "JP": "border"} and a["n_agree"] == 0
    assert a["pair"] == ["ocean_ai", "own_model"]
    b = mbti.agreement(m("ocean_ai", "ESFJ"), m("own_model", "ESFJ"))
    assert b["n_agree"] == 4 and set(b["axes"].values()) == {"agree"}
    assert mbti.agreement_line(b) == "Обе системы дают один и тот же тип."
    assert mbti.agreement_line(a) == ("Уверенных совпадений нет; расходится E–I; на границе хотя бы у одной из систем: "
                                      "S–N, T–F, J–P.")
    c = mbti.agreement(m("ocean_ai", "ESFJ", {"SN", "TF"}), m("own_model", "ESTJ", {"SN"}))
    assert mbti.agreement_line(c) == "Совпадают 2 из 4 осей; на границе хотя бы у одной из систем: S–N, T–F."
    missing = m("own_model", "ESFJ")
    missing["axes"]["EI"] = {"letter": None, "missing": True, "borderline": True}
    assert mbti.agreement(m("ocean_ai", "ESFJ"), missing)["axes"]["EI"] == "border"


# 10
def test_segments_old_and_new_formats():
    v = scores.clean_view(rep("B"))
    sec = mbti.build_section(v, computed_at="t")
    gaps = [e for e in sec["timeline"] if e["type"] is None]
    assert [e["segment"] for e in gaps] == [10, 11, 12, 14, 15, 26, 33]
    assert all(e["reason"] == "no_primary" for e in gaps)
    assert sec["stability"]["EI"]["of"] == 26
    assert "timeline" not in sec["second"][0]                    # old jobs: no per-segment scores of the own model
    # 3.0 format: variants per segment give lanes of both systems
    r = rep("B")
    for t in r["timeline"]:
        mm = {k: r["variant_scores"]["mm"][k] for k in TRAIT_KEYS}
        oc = t["scores"] if "oceanai" in t["members_used"] else None
        t["variants"] = {"oceanai": oc, "mm": mm}
        t["primary_used"] = "oceanai" if oc else None
    sec3 = mbti.build_section(scores.clean_view(r), computed_at="t")
    assert len([e for e in sec3["timeline"] if e.get("type_strict")]) == 26
    lane2 = sec3["second"][0]["timeline"]
    assert len(lane2) == 33 and all(e["type_strict"] == "ISTP" for e in lane2)
    assert sec3["second"][0]["stability"]["EI"] == {"same": 33, "of": 33}
    # no segments at all
    r = rep("B")
    r["timeline"] = []
    sec0 = mbti.build_section(scores.clean_view(r), computed_at="t")
    assert sec0["timeline"] == [] and sec0["stability"] is None and "timeline" not in sec0["second"][0]
    assert sec0["segments_used"] == 1


# 11
def test_get_mbti_is_pure():
    r = rep("B")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "result.json"
        f.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        before_mtime, before_text = f.stat().st_mtime_ns, f.read_text(encoding="utf-8")
        loaded = json.loads(before_text)
        snapshot = copy.deepcopy(loaded)
        mb = mbti.get_mbti(loaded, scores.clean_view(loaded))
        assert mb["computed_on_render"] is True and mb["schema_version"] == 2
        assert loaded == snapshot and "mbti" not in loaded
        mb2 = mbti.get_mbti(loaded)                                   # the view is built inside
        assert mb2["type"] == mb["type"] and loaded == snapshot
        time.sleep(0.01)
        assert f.stat().st_mtime_ns == before_mtime and f.read_text(encoding="utf-8") == before_text
        assert sorted(os.listdir(d)) == ["result.json"]
    saved = {**r, "mbti": {"schema_version": 2, "type": "SAVED"}}
    got = mbti.get_mbti(saved, scores.clean_view(saved))
    assert got == {"schema_version": 2, "type": "SAVED"} and "computed_on_render" not in got
    # a section of schema 1 (letters by the position in the reference group) is not shown: computed anew
    old = {**r, "mbti": {"schema_version": 1, "type": "EXFJ"}}
    got = mbti.get_mbti(old, scores.clean_view(old))
    assert got["type"] == "ENFJ" and got["computed_on_render"] is True
    no_bf = {"model": {"lang": "ru", "primary": "oceanai"}, "traits": {}, "variant_scores": {}}
    assert mbti.get_mbti(no_bf) is None


# 12
def test_build_section_deterministic():
    for name in ("A", "B"):
        v = scores.clean_view(rep(name))
        a = json.dumps(mbti.build_section(v, computed_at="2026-10-01T12:00:00"), sort_keys=True, ensure_ascii=False)
        b = json.dumps(mbti.build_section(copy.deepcopy(v), computed_at="2026-10-01T12:00:00"), sort_keys=True,
                       ensure_ascii=False)
        assert a == b


# 13
def test_type_names():
    names = mbti.load_config()["type_names_ru"]
    combos = {"".join(t) for t in itertools.product("EI", "SN", "TF", "JP")}
    assert set(names) == combos and len(names) == 16
    assert len(set(names.values())) == 16
    assert names["ENFJ"] == "Наставник" and names["ESFJ"] == "Попечитель"


# 14
def _p(m):
    return [m["axes"][ax]["value"] for ax in mbti.AXES]


def test_golden_sample_a():
    v = scores.clean_view(rep("A"))
    mb = mbti.build_section(v, computed_at="t")
    assert mb["type"] == "XNFJ" and mb["type_strict"] == "ENFJ" and mb["type_name"] == "Наставник" and mb["x_count"] == 1
    assert mb["alternatives"] == ["INFJ"]
    assert _p(mb) == [0.56, 0.65, 0.67, 0.65]
    assert [mb["axes"][ax]["confidence"] for ax in mbti.AXES] == [0.12, 0.3, 0.34, 0.3]
    assert [mb["axes"][ax]["word"] for ax in mbti.AXES] == ["на границе", "умеренно", "умеренно", "умеренно"]
    assert [scores.level(v["traits"][k]["score"]) for k in TRAIT_KEYS] == ["above", "above", "mid", "above", "mid"]
    assert mb["neuroticism"] == {"value": 0.64, "level": "средний уровень",
                                 "note": "шкала не имеет соответствия в MBTI, приводится отдельно"}
    s = mb["second"][0]
    assert s["source"] == "own_model" and s["type"] == "IXXX" and s["type_strict"] == "ISTJ"
    assert _p(s) == [0.31, 0.41, 0.48, 0.51]
    assert mb["agreement"]["axes"] == {"EI": "border", "SN": "border", "TF": "border", "JP": "border"}
    assert mb["agreement"]["n_agree"] == 0
    assert v["view_meta"]["segments_without_primary"] == [16]
    assert mb["modal_types"] == [["ENFJ", 17]]
    assert all(mb["stability"][ax] == {"same": 17, "of": 17} for ax in mbti.AXES)
    # the strict letters never change, but the axes are on the border in part of the segments
    assert mbti.border_counts(mb["timeline"]) == ({"EI": 17, "SN": 7, "TF": 6, "JP": 6}, 17)
    assert mbti.border_text(mb["timeline"]) == "ось E–I на границе во всех 17 отрезках, S–N — в 7, T–F — в 6, J–P — в 6"
    assert round(v["traits"]["extraversion"]["score"], 3) == 0.556
    assert "timeline" not in s


def test_golden_sample_b():
    v = scores.clean_view(rep("B"))
    mb = mbti.build_section(v, computed_at="t")
    assert mb["type"] == "ENFJ" and mb["type_strict"] == "ENFJ" and mb["type_name"] == "Наставник"
    assert mb["alternatives"] == [] and mb["x_count"] == 0
    assert _p(mb) == [0.73, 0.71, 0.87, 0.76]
    assert [mb["axes"][ax]["confidence"] for ax in mbti.AXES] == [0.46, 0.42, 0.74, 0.52]
    assert [mb["axes"][ax]["word"] for ax in mbti.AXES] == ["умеренно", "умеренно", "отчётливо", "умеренно"]
    assert not any(mb["axes"][ax]["borderline"] for ax in mbti.AXES)
    assert [scores.level(v["traits"][k]["score"]) for k in TRAIT_KEYS] == ["above", "above", "above", "high", "mid"]
    assert mb["neuroticism"]["value"] == 0.47 and mb["neuroticism"]["level"] == "средний уровень"
    assert mb["neuroticism_note"] == "Шкала нейротизма (средний уровень) в MBTI не выражается, приводится отдельно"
    s = mb["second"][0]
    assert s["type"] == "ISXX" and s["type_strict"] == "ISTP"
    assert _p(s) == [0.25, 0.34, 0.45, 0.48]
    assert mbti.border_counts(mb["timeline"]) == ({"EI": 0, "SN": 0, "TF": 0, "JP": 0}, 26)
    assert mbti.border_text(mb["timeline"]) == ""
    assert mb["agreement"]["axes"] == {"EI": "differ", "SN": "differ", "TF": "border", "JP": "border"}
    assert mb["agreement"]["n_agree"] == 0
    assert mb["segments_total"] == 33 and mb["segments_used"] == 26
    assert mb["modal_types"] == [["ENFJ", 26]]
    assert all(mb["stability"][ax] == {"same": 26, "of": 26} for ax in mbti.AXES)
    assert "reference" not in mb and all("reference" not in x for x in mb["second"])
    assert mb["reliability"]["EI"] == "высокая (r≈0.74)" and mb["reliability_r"]["TF"] == 0.44
    assert mb["computed_by"] == "BS Profiler 3.0 3.0.0a1" and mb["method"] == "raw" and mb["llm"] is None
    assert mb["schema_version"] == 2
    assert mb["source"] == "ocean_ai" and mb["role"] == "main" and s["role"] == "second_opinion"


def test_fact_card_and_journal_lines():
    mb = mbti.build_section(scores.clean_view(rep("B")), computed_at="t")
    assert mbti.fact_card(mb) == ("Тип MBTI · OCEAN-AI", "ENFJ", "«Наставник» · своя модель: ISXX")
    lines = mbti.journal_lines(mb)
    assert lines[0] == "Тип MBTI (OCEAN-AI): ENFJ «Наставник»; нейротизм — средний уровень"
    assert lines[1] == ("Второе мнение MBTI (своя модель): ISXX, ближайший ISTP; уверенно совпадают 0 осей из 4 "
                        "(E–I ≠, S–N ≠, T–F ≈, J–P ≈)")
    a = mbti.build_section(scores.clean_view(rep("A")), computed_at="t")
    label, value, note = mbti.fact_card(a)
    assert value == "XNFJ" and note.startswith("ближайший ENFJ «Наставник»") and note.endswith("своя модель: IXXX")
    assert mbti.journal_lines(a)[1].startswith("Второе мнение MBTI (своя модель): IXXX, тип не выражен")
    assert mbti.fact_card(None) is None
    for text in [*lines, note]:
        assert "сегмент" not in text and "определяет тип" not in text and "предварительн" not in text


def test_english_section():
    v = scores.clean_view(english("B"))
    mb = mbti.build_section(v, computed_at="t")
    assert mb["source"] == "mean" and "reference" not in mb and mb["method"] == "raw"
    assert mb["type"] == "XXFJ" and mb["type_strict"] == "ENFJ"                   # the mean of the two systems
    assert [s["source"] for s in mb["second"]] == ["ocean_ai", "own_model"]
    assert mb["agreement"]["pair"] == ["ocean_ai", "own_model"]
    label, _, note = mbti.fact_card(mb)
    assert label == "Тип MBTI · среднее двух систем" and "предварительные" not in note
    lines = mbti.journal_lines(mb)
    assert lines[1].startswith("MBTI по системе (OCEAN-AI): ") and lines[2].startswith("MBTI по системе (своя модель): ")
    assert not any("Второе мнение" in ln for ln in lines)
