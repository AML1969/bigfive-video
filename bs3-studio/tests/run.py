"""Test runner of BS Profiler 3.0 on the standard library only (pytest is not installed into the shared venv).

Usage:  ~/bs/venv/bin/python bs3-studio/tests/run.py [test_file.py ...] [-k SUBSTRING]
Finds tests/test_*.py, calls every module-level function named test_* in the order of definition, prints one line per
failure and a summary; exit code 1 if anything failed. The files stay pytest-compatible (plain asserts, no fixtures).
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # bs3-studio/: `import bs3` resolves to this working tree
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"bs3_tests_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> int:
    key = None
    files: list[Path] = []
    it = iter(argv)
    for a in it:
        if a == "-k":
            key = next(it, None)
        else:
            p = Path(a)
            files.append(p if p.is_absolute() or p.exists() else HERE / p.name)
    if not files:
        files = sorted(HERE.glob("test_*.py"))
    passed, failed, t0 = 0, [], time.time()
    for f in files:
        try:
            mod = _load(f)
        except Exception:
            failed.append((f.name, "<import>", traceback.format_exc()))
            print(f"FAIL {f.name}: import error")
            continue
        tests = [(n, fn) for n, fn in vars(mod).items()
                 if n.startswith("test_") and inspect.isfunction(fn) and fn.__module__ == mod.__name__]
        tests.sort(key=lambda t: t[1].__code__.co_firstlineno)
        for name, fn in tests:
            if key and key not in name:
                continue
            try:
                fn()
                passed += 1
            except Exception:
                failed.append((f.name, name, traceback.format_exc()))
                print(f"FAIL {f.name}::{name}")
    for fname, name, tb in failed:
        print(f"\n===== {fname}::{name}\n{tb}")
    print(f"\n{passed} passed, {len(failed)} failed in {time.time() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
