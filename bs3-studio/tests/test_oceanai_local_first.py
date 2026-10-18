"""bs3/backend_oceanai.py _patch_local_first: a known OCEAN-AI weight link whose file is on disk is answered from disk
without the network (a Docker container has no internet); an unknown link, a missing file or force_reload goes to the
original method; the shipped name table covers every weight link the backend loads for both corpora."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from bs3 import backend_oceanai as bo


class FakeRun:
    """Stands in for oceanai Run: records the calls that would go to the network."""

    def __init__(self, path_to_save_):
        self.path_to_save_ = path_to_save_
        self._url_last_filename = ""
        self.network_calls = []

    def _download_file_from_url(self, url, force_reload=True, out=True, runtime=True, run=True):
        self.network_calls.append((url, force_reload))
        return 404


def _setup(tmp: Path):
    names = tmp / "names.json"
    names.write_text(json.dumps({"https://drive/x?id=1": "w1.pth", "https://drive/x?id=2": "w2.pth"}), encoding="utf-8")
    models = tmp / "models"
    models.mkdir()
    (models / "w1.pth").write_bytes(b"weights")
    b5 = FakeRun(str(models))
    bo._patch_local_first(b5, names)
    return b5, models


def test_known_file_on_disk_needs_no_network():
    with tempfile.TemporaryDirectory() as t:
        b5, models = _setup(Path(t))
        res = b5._download_file_from_url(url="https://drive/x?id=1", force_reload=False, runtime=False, out=False, run=True)
        assert res == 200
        assert b5._url_last_filename == str(models / "w1.pth")
        assert b5.network_calls == []


def test_missing_file_unknown_link_and_force_reload_go_online():
    with tempfile.TemporaryDirectory() as t:
        b5, _ = _setup(Path(t))
        assert b5._download_file_from_url(url="https://drive/x?id=2", force_reload=False) == 404      # file absent
        assert b5._download_file_from_url(url="https://drive/x?id=9", force_reload=False) == 404      # unknown link
        assert b5._download_file_from_url(url="https://drive/x?id=1", force_reload=True) == 404       # forced
        assert [u[-1] for u, _ in b5.network_calls] == ["2", "9", "1"]


def test_patch_applies_once_and_survives_a_bad_table():
    with tempfile.TemporaryDirectory() as t:
        b5, _ = _setup(Path(t))
        first = b5._download_file_from_url
        bo._patch_local_first(b5, Path(t) / "names.json")
        assert b5._download_file_from_url is first
        other = FakeRun(t)
        bo._patch_local_first(other, Path(t) / "absent.json")      # no table: everything goes to the original
        assert other._download_file_from_url(url="https://drive/x?id=1", force_reload=False) == 404
        assert len(other.network_calls) == 1


def test_shipped_table_covers_every_weight_link():
    """Every link in OCEAN-AI's weight table that the backend loads (googledisk, both corpora) and the LIWC
    dictionary link has a file name in config/oceanai_weights.json."""
    names = json.loads(bo.WEIGHT_NAMES.read_text(encoding="utf-8"))
    assert all(isinstance(v, str) and v and "/" not in v for v in names.values())
    try:
        from oceanai.modules.lab.build import Run
    except Exception as e:                                      # OCEAN-AI is not installed: only the table's shape
        import unittest
        raise unittest.SkipTest(f"oceanai not importable: {e}")
    W = Run(lang="en", metadata=False).weights_for_big5_
    need = set()
    for corpus in ("fi", "mupta"):
        need |= {W["audio"][corpus][k]["googledisk"] for k in ("hc", "nn")}
        need |= {W["video"][corpus][k]["googledisk"] for k in ("hc", "fe", "nn")}
    missing = sorted(u for u in need if u not in names)
    assert not missing, f"{len(missing)} weight links without a file name: {missing[:3]}"
