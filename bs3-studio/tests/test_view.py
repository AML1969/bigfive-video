"""Clean scores (design 6.1, 5.2, 13.1 test_view): scores.clean_view, levels and position phrases."""
from __future__ import annotations

import copy
import json

from samples import english, rep

from bs3 import scores
from bs3.norms import TRAIT_KEYS


def test_clean_view_sample_b_main_scores_are_oceanai_only():
    r = rep("B")
    assert r["traits"]["extraversion"]["score"] == 0.629            # the mixed 2.0 number
    v = scores.clean_view(r)
    assert round(v["traits"]["extraversion"]["score"], 3) == 0.730
    for k in TRAIT_KEYS:
        assert v["traits"][k]["score"] == r["variant_scores"]["oceanai"][k]


def test_clean_view_sample_b_gaps():
    v = scores.clean_view(rep("B"))
    gaps = [t["segment"] for t in v["timeline"] if t["scores"] is None]
    assert gaps == [10, 11, 12, 14, 15, 26, 33]
    assert all(t.get("no_primary") for t in v["timeline"] if t["scores"] is None)
    meta = v["view_meta"]
    assert meta["main_system"] == "oceanai" and meta["main_source"] == "ocean_ai" and meta["lang"] == "ru"
    assert meta["segments_total"] == 33 and meta["segments_used"] == 26
    assert meta["segments_without_primary"] == gaps and meta["primary_missing"] is False
    assert meta["reference"]["id"] == "ru_prov_2026-09-25" and meta["reference"]["n"] == 13


def test_clean_view_spread_recomputed():
    r = rep("B")
    assert r["scores_std_across_segments"]["extraversion"] > 0.2        # ±0.205 with the mixed segments
    v = scores.clean_view(r)
    std = v["scores_std_across_segments"]
    assert abs(std["extraversion"] - 0.023) < 0.0015, std["extraversion"]
    assert max(std[k] for k in TRAIT_KEYS) <= 0.0235
    a = scores.clean_view(rep("A"))["scores_std_across_segments"]
    assert max(a[k] for k in TRAIT_KEYS) <= 0.0375


def test_clean_view_positions_ru():
    v = scores.clean_view(rep("B"))
    e = v["traits"]["extraversion"]
    assert abs(e["position"] - 13 / 14) < 1e-4
    assert e["percentile"] == round(100 * 13 / 14, 1)
    assert e["percentile_ref"] == "ref:ru_prov_2026-09-25"


def test_clean_view_does_not_change_rep():
    r = rep("B")
    before = copy.deepcopy(r)
    scores.clean_view(r)
    assert r == before


def test_clean_view_idempotent():
    for name in ("A", "B"):
        v1 = scores.clean_view(rep(name))
        v2 = scores.clean_view(v1)
        assert json.dumps(v1, sort_keys=True) == json.dumps(v2, sort_keys=True)


def test_clean_view_english_keeps_traits():
    r = english("B")
    v = scores.clean_view(r)
    for k in TRAIT_KEYS:
        assert v["traits"][k]["score"] == r["traits"][k]["score"]
        assert "percentile_ref" not in v["traits"][k] or v["traits"][k]["percentile_ref"] == r["traits"][k].get("percentile_ref")
    assert [t["scores"] for t in v["timeline"]] == [t["scores"] for t in r["timeline"]]
    assert v["scores_std_across_segments"] == r["scores_std_across_segments"]
    assert v["view_meta"]["main_system"] == "mean" and v["view_meta"]["reference"]["id"] == "fiv2"


def test_clean_view_primary_missing_falls_back_to_own_model():
    r = rep("B")
    del r["variant_scores"]["oceanai"]
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    v = scores.clean_view(r)
    assert v["view_meta"]["main_system"] == "mm" and v["view_meta"]["primary_missing"] is True
    assert v["traits"]["extraversion"]["score"] == r["variant_scores"]["mm"]["extraversion"]
    assert all(t["scores"] is not None for t in v["timeline"])


def test_segment_ok_formats():
    r = rep("B")
    assert scores.segment_ok(r, {"members_used": ["oceanai", "mm"]})
    assert not scores.segment_ok(r, {"members_used": ["mm"]})
    assert scores.segment_ok(r, {"members_used": ["mm"], "primary_used": "oceanai"})
    assert not scores.segment_ok(r, {"members_used": ["oceanai", "mm"], "primary_used": None})


def test_levels_and_bands():
    L = scores.level
    assert L(0.85) == "high" and L(0.8499) == "above" and L(0.65) == "above" and L(0.6499) == "mid"
    assert L(0.5) == "mid" and L(0.3501) == "mid" and L(0.35) == "below" and L(0.1501) == "below"
    assert L(0.15) == "low" and L(0.0) == "low" and L(1.0) == "high" and L(None) is None
    assert scores.level_phrase(0.9) == "заметно выше типичного"
    assert scores.level_phrase(0.3) == "ниже типичного"
    assert set(scores.LEVELS_RU) == {"high", "above", "mid", "below", "low"}


def test_position_phrases():
    ref = "ru_prov_2026-09-25"
    assert scores.position_phrase(0.929, ref) == "выше, чем у большинства из 13 русских роликов"
    assert scores.position_phrase(0.2, ref) == "ниже, чем у большинства из 13 русских роликов"
    assert scores.position_phrase(0.6, ref) == "примерно посередине среди 13 русских роликов"
    assert scores.position_phrase(0.72, "fiv2") == "выше, чем у 72% людей в First Impressions V2"
    assert scores.position_phrase(0.30, "fiv2") == "ниже, чем у 70% людей в First Impressions V2"
    assert scores.position_phrase(0.53, "fiv2") == "примерно посередине"
    for p in (0.1, 0.5, 0.9):
        assert "сегмент" not in scores.position_phrase(p, ref)
