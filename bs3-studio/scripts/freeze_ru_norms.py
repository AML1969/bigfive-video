"""Freeze the provisional Russian reference group of BS Profiler 3.0 (design 4.4).

Reads (never writes) <src>/*/result.json of finished jobs, default ~/bs2_data/web_jobs:
  1. jobs with model.lang == "ru", both variant_scores["oceanai"] and ["mm"] (the clean means of each system over its own
     segments) and a job id (folder name) <= the cutoff, so N is reproducible while new jobs keep coming;
  2. repeats of the same file by media.sha256 — the newest job of each file is kept;
  3. values rounded to 4 digits and sorted per system and trait; only numbers are written: no names, paths, job ids or
     fingerprints;
  4. "agreement": Spearman rank correlation of the two systems over the same videos (mean ranks for ties) and how often
     the strict MBTI letters of the two systems coincide with thresholds at the median of each group (method "position").

Usage: ~/bs/venv/bin/python bs3-studio/scripts/freeze_ru_norms.py [--src DIR] [--cutoff 20260925_235959]
       [--frozen-at 2026-09-25] [--out PATH] [--dry-run]
Re-freezing (new data) is the owner's decision only: use a new cutoff and date, the id changes with the date.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bs3.norms import TRAIT_KEYS  # noqa: E402
from bs3.refnorms import date_ru, rank_points, rank_position, ru_group_gen  # noqa: E402

SYSTEMS = ("oceanai", "mm")
AXES = {"EI": "extraversion", "SN": "openness", "TF": "agreeableness", "JP": "conscientiousness"}
JOB_RE = re.compile(r"^\d{8}_\d{6}$")


def _num(x):
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def collect(src: Path, cutoff: str) -> tuple[list[dict], dict]:
    """Clean rows {system: {trait: value}} of the unique Russian videos, plus counters for the log."""
    stats = {"jobs": 0, "ru_both": 0, "no_sha": 0, "unique": 0}
    by_sha: dict[str, tuple[str, dict]] = {}
    for d in sorted(src.iterdir()):
        f = d / "result.json"
        if not (d.is_dir() and JOB_RE.match(d.name) and d.name <= cutoff and f.exists()):
            continue
        stats["jobs"] += 1
        try:
            rep = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (rep.get("model") or {}).get("lang") != "ru":
            continue
        vs = rep.get("variant_scores") or {}
        row = {}
        for s in SYSTEMS:
            vals = {k: _num((vs.get(s) or {}).get(k)) for k in TRAIT_KEYS}
            if any(v is None for v in vals.values()):
                break
            row[s] = vals
        else:
            stats["ru_both"] += 1
            sha = (rep.get("media") or {}).get("sha256")
            if not sha:
                stats["no_sha"] += 1
                continue
            prev = by_sha.get(sha)
            if prev is None or d.name > prev[0]:              # the newest job of a repeated file
                by_sha[sha] = (d.name, row)
    rows = [row for _, row in sorted(by_sha.values(), key=lambda t: t[0])]
    stats["unique"] = len(rows)
    return rows, stats


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def _ranks(values: list[float]) -> list[float]:
    mean_rank = dict(rank_points(values))
    return [mean_rank[v] for v in values]


def spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_ranks(x), _ranks(y))


def freeze(rows: list[dict], ref_id: str, frozen_at: str, cutoff: str) -> dict:
    n = len(rows)
    r4 = [{s: {k: round(row[s][k], 4) for k in TRAIT_KEYS} for s in SYSTEMS} for row in rows]
    sources = {s: {k: sorted(r[s][k] for r in r4) for k in TRAIT_KEYS} for s in SYSTEMS}
    rho = {k: round(spearman([r["oceanai"][k] for r in r4], [r["mm"][k] for r in r4]), 2) for k in TRAIT_KEYS}
    same = {}
    for ax, k in AXES.items():
        same[ax] = sum(
            (rank_position(sources["oceanai"][k], r["oceanai"][k]) >= 0.5) == (rank_position(sources["mm"][k], r["mm"][k]) >= 0.5)
            for r in r4)
    return {"id": ref_id, "kind": "provisional", "frozen_at": frozen_at, "cutoff_job": cutoff,
            "dedup": "media.sha256", "n": n, "sources": sources,
            "agreement": {"n": n, "spearman": rho, "letters_same": same},
            "label_ru": f"{ru_group_gen(n)}, обработанных системой до {date_ru(frozen_at)}"}


def dumps(d: dict) -> str:
    """Indented JSON with every list of numbers on one line (easy to read and to diff)."""
    text = json.dumps(d, ensure_ascii=False, indent=1)
    return re.sub(r"\[\s*([-0-9.,\s]+?)\s*\]", lambda m: "[" + re.sub(r"\s+", "", m.group(1)) + "]", text) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(Path.home() / "bs2_data" / "web_jobs"))
    ap.add_argument("--cutoff", default="20260925_235959")
    ap.add_argument("--frozen-at", default="2026-09-25")
    ap.add_argument("--out", default=str(ROOT / "bs3" / "data" / "norms_ru_provisional.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rows, stats = collect(Path(a.src).expanduser(), a.cutoff)
    if not rows:
        print("no Russian jobs with both systems found", file=sys.stderr)
        return 1
    d = freeze(rows, f"ru_prov_{a.frozen_at}", a.frozen_at, a.cutoff)
    print(f"jobs <= cutoff: {stats['jobs']}, ru with both systems: {stats['ru_both']}, without sha256: "
          f"{stats['no_sha']}, unique videos: {stats['unique']}")
    for s in SYSTEMS:
        med = {k: d["sources"][s][k][(d["n"] - 1) // 2] if d["n"] % 2 else None for k in TRAIT_KEYS}
        print(f"median {s}: " + ", ".join(f"{k[:4]} {v}" for k, v in med.items()))
    print("spearman:", d["agreement"]["spearman"])
    print("letters_same:", d["agreement"]["letters_same"])
    if a.dry_run:
        return 0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(d), encoding="utf-8")
    print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
