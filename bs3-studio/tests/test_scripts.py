"""The offline scripts of BS Profiler 3.1 (scripts/*.py): every one of them compiles, and the ones the checks and the
owner run answer --help from the bs3-studio folder. The refactoring moves names these scripts import; a broken import
shows up here, not at the next manual run."""
from __future__ import annotations

import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # bs3-studio/
SCRIPTS = ROOT / "scripts"
HELP = ("import_job", "add_mbti", "rerender_samples", "check_job", "compare_baseline")


def test_every_script_compiles():
    files = sorted(SCRIPTS.glob("*.py"))
    assert {f.stem for f in files} >= set(HELP), files
    with tempfile.TemporaryDirectory() as d:          # the .pyc files go there, not into scripts/__pycache__
        for f in files:
            py_compile.compile(str(f), cfile=str(Path(d) / f"{f.stem}.pyc"), doraise=True)


def test_scripts_answer_help():
    env = {**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"}
    for name in HELP:
        r = subprocess.run([sys.executable, str(SCRIPTS / f"{name}.py"), "--help"], cwd=str(ROOT), env=env,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (name, r.returncode, r.stdout[-500:], r.stderr[-2000:])
        assert r.stdout.strip(), name
