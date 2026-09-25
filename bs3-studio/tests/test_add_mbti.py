"""scripts/add_mbti.py (design 7.2, task T28) on synthetic jobs in a temporary folder: the section is written only for
a job under the 3.0 jobs root, a job elsewhere is refused and its file stays byte for byte the same."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

from samples import rep

from bs3 import mbti

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "add_mbti.py"
JOB_ID = "20000101_000000"


def _mod():
    spec = importlib.util.spec_from_file_location("bs3_add_mbti", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _job(base: Path, r: dict) -> Path:
    job = base / JOB_ID
    job.mkdir(parents=True)
    (job / "result.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return job


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_writes_section_under_root():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "web_jobs"
        job = _job(root, rep("B"))
        assert m.add_mbti(job, root=root) == "written"
        saved = json.loads((job / "result.json").read_text(encoding="utf-8"))
        sec = saved["mbti"]
        assert sec["schema_version"] == 1 and sec["type"] == "EXFJ" and sec["type_strict"] == "ESFJ"
        assert sec["computed_by"] == "BS Profiler 3.0 3.0.0a1" and "computed_on_render" not in sec
        assert mbti.get_mbti(saved) == sec                 # the page now shows the stored section as it is
        before = _sha(job / "result.json")
        assert m.add_mbti(job, root=root) == "kept"        # stored already: not recomputed without --force
        assert _sha(job / "result.json") == before
        assert m.add_mbti(job, root=root, force=True) == "written"


def test_refuses_outside_root_and_leaves_file_unchanged():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "web_jobs"
        root.mkdir()
        job = _job(Path(d) / "other", rep("A"))
        before = _sha(job / "result.json")
        for target in (job, root / ".." / "other" / JOB_ID, root):
            try:
                m.add_mbti(target, root=root)
            except m.AddRefused:
                pass
            else:
                raise AssertionError(f"not refused: {target}")
        assert _sha(job / "result.json") == before
        assert m.main([str(job)]) == 2                     # the default root is ~/bs3_data/web_jobs: refused too
        assert _sha(job / "result.json") == before


def test_no_big_five_no_section():
    m = _mod()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "web_jobs"
        r = rep("B")
        r["variant_scores"] = {}
        for k in r["traits"]:
            r["traits"][k]["score"] = None
        job = _job(root, r)
        before = _sha(job / "result.json")
        assert m.add_mbti(job, root=root) == "no Big Five"
        assert _sha(job / "result.json") == before
