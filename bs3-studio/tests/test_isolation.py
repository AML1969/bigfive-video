"""Isolation of BS Profiler 3.0 from BS 2.0 (design 2.1, 13.1).

- nothing under bs2-studio/ and bs/ differs from the tag v3-base (the last commit before the 3.0 scaffold), neither in
  commits nor in the working tree;
- no file of bs3-studio/ (except README.md) imports bs2 or names its variables; its work dir is named only by the
  offline scripts that read old 2.0 jobs; the 2.0 port and name may appear only in documentation (comments,
  docstrings, the project description), never in code;
- the 3.0 work dirs lie under ~/bs3_data; the product name and version come from bs3/__init__.py.
"""
from __future__ import annotations

import ast
import io
import os
import re
import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # bs3-studio/
REPO = ROOT.parent
SELF = Path(__file__).resolve()

# forbidden everywhere (code, comments, data)
HARD = [re.compile(r"\bimport\s+bs2\b"), re.compile(r"\bfrom\s+bs2\b"), re.compile(r"BS2_")]
# the 2.0 work dir: never in the package, the tests or the unit; only the offline scripts that read old 2.0 jobs
# (design 4.4 and 13.3: freeze the reference group, import and re-render sample jobs, refuse to write there) name it
BS2_DATA = re.compile(r"bs2_data")
BS2_DATA_READERS = {"scripts/freeze_ru_norms.py", "scripts/import_job.py", "scripts/rerender_samples.py",
                    "scripts/add_mbti.py"}
# allowed only in documentation positions (the 3.0 docs say it is independent of BS 2.0 on :7870)
SOFT = [re.compile(r"(?<![\d.])7870(?!\d)"), re.compile(r"BS 2\.0")]
SKIP_DIRS = {"__pycache__", ".git"}
SKIP_SUFFIX = {".pyc", ".jpg", ".jpeg", ".png", ".pdf", ".mp4", ".pt", ".pth", ".onnx"}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def _files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.resolve() == SELF or p.suffix.lower() in SKIP_SUFFIX:
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if rel.as_posix() == "README.md":
            continue
        yield p, rel.as_posix()


def _py_doc_lines(text: str) -> set[int]:
    """Line numbers of comments and docstrings of a Python file."""
    lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            lines.add(tok.start[0])
    tree = ast.parse(text)
    for node in [tree, *ast.walk(tree)]:
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def _soft_ok(rel: str, text: str, line: str, lineno: int, doc_lines: set[int] | None) -> bool:
    if rel.endswith(".py"):
        return lineno in (doc_lines or set())
    if rel.endswith(".sh"):
        s = line.lstrip()
        return s.startswith("#")
    if rel == "pyproject.toml":
        return line.startswith("description = ")
    return False


def test_bs2_and_bs_unchanged_since_v3_base():
    tag = _git("rev-parse", "--verify", "--quiet", "v3-base^{commit}")
    assert tag.returncode == 0, "local tag v3-base is missing: git tag v3-base 1d9fe30"
    diff = _git("diff", "--quiet", "v3-base", "--", "bs2-studio", "bs")
    assert diff.returncode == 0, "bs2-studio/ or bs/ differs from v3-base:\n" + _git(
        "diff", "--stat", "v3-base", "--", "bs2-studio", "bs").stdout
    st = _git("status", "--porcelain", "--", "bs2-studio", "bs")
    assert st.returncode == 0 and st.stdout.strip() == "", "uncommitted changes under bs2-studio/ or bs/:\n" + st.stdout


def test_no_bs2_references_in_bs3_studio():
    bad = []
    for p, rel in _files():
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        doc_lines = None
        for i, line in enumerate(text.splitlines(), 1):
            for rx in HARD:
                if rx.search(line):
                    bad.append(f"{rel}:{i}: {line.strip()[:120]}")
            if BS2_DATA.search(line) and rel not in BS2_DATA_READERS:
                bad.append(f"{rel}:{i}: {line.strip()[:120]}")
            if any(rx.search(line) for rx in SOFT):
                if rel.endswith(".py") and doc_lines is None:
                    doc_lines = _py_doc_lines(text)
                if not _soft_ok(rel, text, line, i, doc_lines):
                    bad.append(f"{rel}:{i}: {line.strip()[:120]}")
    assert not bad, "references to BS 2.0 in bs3-studio:\n" + "\n".join(bad)


def test_work_dirs_under_bs3_data():
    env = {k: v for k, v in os.environ.items() if k not in ("BS3_JOURNAL", "BS3_POOL_DIR")}
    code = ("import sys, bs3.pool, bs3.journal; "
            "print(bs3.pool.POOL_DIR); print(bs3.journal.PATH); "
            "print(sorted(m for m in sys.modules if m == 'bs2' or m.startswith('bs2.')))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    pool, journal, bs2_mods = r.stdout.strip().splitlines()[-3:]
    base = str(Path.home() / "bs3_data")
    assert pool.startswith(base + os.sep), pool
    assert journal.startswith(base + os.sep), journal
    assert bs2_mods == "[]", bs2_mods
    web = (ROOT / "bs3" / "webapp.py").read_text(encoding="utf-8")
    assert re.search(r'expanduser\("~/bs3_data/web_jobs"\)', web), "default work dir of webapp.main is not ~/bs3_data"
    cli = (ROOT / "bs3" / "cli.py").read_text(encoding="utf-8")
    assert re.search(r"default=7880\b", cli), "default port of `bs3 web` is not 7880"


def test_product_name_and_version():
    import bs3
    assert Path(bs3.__file__).resolve().parent == ROOT / "bs3", bs3.__file__
    assert bs3.PRODUCT == "BS Profiler 3.0"
    assert bs3.__version__ == "3.0.0a1"
    assert bs3.PRODUCT_SLUG == "BS_Profiler_3"
