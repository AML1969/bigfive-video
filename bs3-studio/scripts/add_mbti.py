"""Write the `mbti` section (design 7.1) into result.json of an old job that lies in the 3.0 work dir (design 7.2;
task T28, optional).

    ~/bs/venv/bin/python bs3-studio/scripts/add_mbti.py ~/bs3_data/web_jobs/<job id> [...] [--force]

Not needed for showing or re-rendering: the page and the PDF compute the section on the fly (mbti.get_mbti, which
never writes). This script is the explicit way to store it. It refuses any job outside ~/bs3_data/web_jobs (in
particular the jobs of 2.0 under ~/bs2_data, which 3.0 never writes); old jobs are brought over with import_job.py
first. A job that already has a section of the current schema (2) is left as it is unless --force is given; an
older section (schema 1, letters by the position in a reference group) is replaced. A job without Big Five
scores gets no section (design 7.1) and its file is not touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

JOBS_ROOT = Path.home() / "bs3_data" / "web_jobs"


class AddRefused(RuntimeError):
    pass


def _inside(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return root in path.parents


def add_mbti(job, *, root: Path = JOBS_ROOT, force: bool = False) -> str:
    """Store the section in <job>/result.json; returns what was done ('written', 'kept', 'no Big Five')."""
    from bs3.mbti import SCHEMA_VERSION, build_section
    from bs3.scores import clean_view

    job = Path(job).expanduser()
    if not _inside(job, Path(root).expanduser()):
        raise AddRefused(f"refused: {job} is not a job under {root}")
    res_path = job.resolve() / "result.json"
    if not res_path.is_file():
        raise AddRefused(f"not a finished job (no result.json): {job}")
    rep = json.loads(res_path.read_text(encoding="utf-8"))
    saved = rep.get("mbti")
    if isinstance(saved, dict) and saved.get("schema_version") == SCHEMA_VERSION and not force:
        return "kept"
    sec = build_section(clean_view(rep))
    if sec is None:
        return "no Big Five"
    rep["mbti"] = sec
    fd, tmp = tempfile.mkstemp(prefix=".result_", suffix=".json", dir=res_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        os.chmod(tmp, 0o644)
        os.replace(tmp, res_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return "written"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs", nargs="+", help=f"job folders under {JOBS_ROOT}")
    ap.add_argument("--force", action="store_true", help="recompute a section that is already stored")
    args = ap.parse_args(argv)
    rc = 0
    for job in args.jobs:
        try:
            what = add_mbti(job, force=args.force)
            print(f"{Path(job).name}: {what}")
        except (AddRefused, OSError, ValueError) as e:
            print(f"{Path(job).name}: {e}", file=sys.stderr)
            rc = 2 if isinstance(e, AddRefused) else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
