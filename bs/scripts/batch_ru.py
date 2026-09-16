"""Batch analysis of a folder of Russian videos with the full pipeline (ensemble OCEAN-AI MuPTA + own model,
segmented long-video path). No labels exist for these videos, so this is inference + descriptive analysis:
agreement between the two systems, stability across segments, position relative to the pool of processed videos.

Main score (variant A, 2026-09-15): OCEAN-AI on MuPTA weights (Russian reference population); the own model
(trained on English FIV2) is kept as a second opinion. Percentiles are ranks within the pool of processed videos.

Usage: batch_ru.py SRC_DIR OUT_DIR [--lang ru] [--explain] [--force] [--summary-only]
Writes OUT_DIR/<safe_name>/result.json, OUT_DIR/summary.csv and OUT_DIR/summary.md (resumable).
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bs_bigfive import pool
from bs_bigfive.norms import RU_TITLES, TRAIT_KEYS
from bs_bigfive.report import build_report

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--lang", default="ru")
ap.add_argument("--explain", action="store_true")
ap.add_argument("--force", action="store_true", help="recompute videos that already have result.json (old one kept as result_prev.json)")
ap.add_argument("--summary-only", action="store_true", help="only rebuild summary.csv/md and pool percentiles from cached results")
a = ap.parse_args()
src, out = Path(a.src), Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "video"


def fingerprint(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
    return f"{p.stat().st_size}:{h.hexdigest()}"


videos = sorted(p for p in src.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"))
seen, todo = {}, []
for p in videos:
    fp = fingerprint(p)
    if fp in seen:
        print(f"skip duplicate: {p.name} == {seen[fp]}")
        continue
    seen[fp] = p.name
    todo.append(p)
print(f"{len(todo)} videos (lang={a.lang})")

be = an = None
if not a.summary_only:
    from bs_bigfive.backend_ensemble import EnsembleBackend, EnsembleConfig
    from bs_bigfive.backend_mm import MMConfig
    from bs_bigfive.backend_oceanai import BackendConfig
    from bs_bigfive.longvideo import LongVideoAnalyzer, video_duration
    for p in todo:
        print(f"  {p.name:45s} {video_duration(p):7.1f} s  {p.stat().st_size / 1e6:7.1f} MB")
    be = EnsembleBackend(EnsembleConfig(members=("oceanai", "mm"), lang=a.lang, oceanai_cfg=BackendConfig(lang=a.lang),
                                        mm_cfg=MMConfig(lang=a.lang))).load()
    an = LongVideoAnalyzer(be, lang=a.lang)
    print(f"main score: {be.cfg.primary or 'mean of members'}")

reports = {}
for i, p in enumerate(todo, 1):
    name = safe(p.stem)
    d = out / name
    d.mkdir(exist_ok=True)
    rj = d / "result.json"
    fail_marker = d / "failed.txt"
    t0 = time.time()
    if a.force and rj.exists() and not a.summary_only:
        rj.replace(d / "result_prev.json")
    if rj.exists():
        rep = json.loads(rj.read_text(encoding="utf-8"))
        print(f"[{i}/{len(todo)}] {p.name}: cached")
    elif a.summary_only:
        continue
    elif fail_marker.exists() and fail_marker.read_text().count("\n") >= 2:
        print(f"[{i}/{len(todo)}] {p.name}: skipped after repeated failures ({fail_marker})")
        continue
    else:
        print(f"[{i}/{len(todo)}] {p.name}: analysing ...", flush=True)
        try:
            res = an.analyze(p, d / "segments")
        except Exception as e:  # noqa: BLE001
            with open(fail_marker, "a", encoding="utf-8") as f:
                f.write(time.strftime("%H:%M:%S ") + str(e).splitlines()[0][:200] + "\n")
            print(f"      FAILED: {str(e).splitlines()[0][:160]}", flush=True)
            if "CUDA" in str(e):
                print("      sticky CUDA error -> exiting so the shell loop restarts a clean process", flush=True)
                raise SystemExit(3)
            continue
        primary = res.get("primary")
        if primary:
            pool.add(p, res["scores"], a.lang, primary, name=p.name)
        rep = build_report(p, res, backend="ensemble", corpus=be.cfg.corpus, lang=a.lang, asr_model=an.asr_model,
                           modalities=("oceanai", "mm"), pool_lang=a.lang if primary else None, primary=primary)
        for key in ("duration_sec", "segments", "timeline", "scores_std", "representative_segment", "variants",
                    "transcript_en"):
            if res.get(key) is not None:
                rep[key] = res[key]
        rep["timings_sec"]["total_wall"] = round(time.time() - t0, 1)
        if a.explain and rep.get("timeline"):
            seg = rep["timeline"][rep["representative_segment"] - 1]
            mm = be.backends["mm"]
            ex = mm.explain_video(seg["file"], d / "explain", asr=False, transcript=seg["transcript"],
                                  behavior=seg.get("behavior_description") or None)
            rep["modality_shares"] = {k: {m: v["share"] for m, v in row.items()}
                                      for k, row in ex["modalities"]["input_x_gradient"].items()}
        rj.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"      done in {rep['timings_sec']['total_wall']} s: " +
              " ".join(f"{k[:5]}={rep['traits'][k]['score']:.2f}" for k in TRAIT_KEYS), flush=True)
    reports[p.name] = (rj, rep)

# ---- percentiles against the final pool (the pool grew during the run), then the summary
rows = []
for vname, (rj, rep) in reports.items():
    tr = rep["traits"]
    primary = rep.get("model", {}).get("primary")
    if primary:
        for k in TRAIT_KEYS:
            pct, n = pool.percentile(k, tr[k]["score"], a.lang)
            tr[k]["percentile"], tr[k]["percentile_ref"] = pct, pool.label(a.lang, n)
        rj.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    var = rep.get("variants") or rep.get("variant_scores") or {}
    row = {"video": vname, "duration_s": rep.get("duration_sec"), "segments": rep.get("segments"),
           "seconds": rep["timings_sec"].get("total_wall"), "words": len((rep.get("transcript") or "").split()),
           "primary": primary or "mean"}
    for k in TRAIT_KEYS:
        row[k] = round(tr[k]["score"], 3)
        row[f"{k}_pct"] = tr[k].get("percentile", tr[k].get("percentile_vs_fiv2"))
        row[f"{k}_std"] = round((rep.get("scores_std") or {}).get(k, 0), 3)
        for m in ("oceanai", "mm"):
            if m in var:
                row[f"{k}_{m}"] = round(var[m][k], 3)
    if "interview" in rep:
        row["interview"] = round(rep["interview"]["score"], 3)
        row["interview_pct"] = rep["interview"].get("percentile", rep["interview"].get("percentile_vs_fiv2"))
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(out / "summary.csv", index=False)
prim = df["primary"].iloc[0] if len(df) else "mean"
ref = ("перцентиль относительно пула обработанных русских роликов (ранг среди N=%d)" % len(df)) if prim != "mean" \
    else "перцентиль относительно train FIV2"
lines = ["# Русские видео: сводка", "",
         f"Видео: {len(df)}, язык {a.lang}. Основная оценка: "
         + ("OCEAN-AI (веса MuPTA); своя модель — второе мнение." if prim == "oceanai" else f"{prim}."), "",
         "| Видео | Длит., с | Сегм. | Откр. | Добр. | Экстр. | Доброж. | Эм. стаб. | Собесед. | Время, с |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for _, r in df.iterrows():
    def cell(k):
        pct = r[f"{k}_pct"]
        return f"{r[k]:.2f} ({pct:.0f}%)" if pct is not None and not (isinstance(pct, float) and np.isnan(pct)) else f"{r[k]:.2f}"
    lines.append(f"| {r['video']} | {r['duration_s']:.0f} | {r['segments']} | " + " | ".join(cell(k) for k in TRAIT_KEYS) +
                 f" | {r.get('interview', float('nan')):.2f} | {r['seconds']:.0f} |")
lines += ["", f"В скобках — {ref}. «Собеседование» — своя модель, шкала FIV2.", ""]
if all(f"{k}_oceanai" in df for k in TRAIT_KEYS):
    lines += ["## Две системы", "", "| Черта | OCEAN-AI (MuPTA) среднее | Своя модель среднее | Корреляция по видео | Средняя |разница| |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for k in TRAIT_KEYS:
        x, y = df[f"{k}_oceanai"], df[f"{k}_mm"]
        corr = float(np.corrcoef(x, y)[0, 1]) if len(df) > 2 and x.std() > 0 and y.std() > 0 else float("nan")
        lines.append(f"| {RU_TITLES[k]} | {x.mean():.3f} | {y.mean():.3f} | {corr:.2f} | {(x - y).abs().mean():.3f} |")
    lines.append("")
lines += ["## Стабильность по сегментам (средний разброс внутри видео)", "",
          ", ".join(f"{RU_TITLES[k].lower()}: {df[f'{k}_std'].mean():.3f}" for k in TRAIT_KEYS), ""]
(out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print(df[["video", "duration_s", "segments", "seconds"] + TRAIT_KEYS + (["interview"] if "interview" in df else [])].to_string(index=False))
print("wrote", out / "summary.csv", "and summary.md")
