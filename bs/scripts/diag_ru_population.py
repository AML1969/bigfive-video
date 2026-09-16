"""Diagnostic for the Russian batch: is the gap between OCEAN-AI (MuPTA) and the own model a reference-population
effect or a Russian-speech effect? Re-scores the already cut segments of every batch video with
  (a) OCEAN-AI on the FIV2 ("fi") weights, transcript translated by the library, and
  (b) the own model with the audio node dropped (face + text + behaviour), stored behaviour descriptions reused.
Usage: diag_ru_population.py OUT_DIR [--only NAME ...]
Writes OUT_DIR/diag_population.json and OUT_DIR/diag_population.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bs_bigfive.backend_mm import MMBackend, MMConfig
from bs_bigfive.backend_oceanai import BackendConfig, OceanAIBackend
from bs_bigfive.norms import RU_SHORT, TRAIT_KEYS

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--only", nargs="*", default=None)
a = ap.parse_args()
out = Path(a.out)
cache_path = out / "diag_population.json"
cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

# OCEAN-AI ties the language flag to the weight set (FI weights need lang="en": the hand-crafted video features and
# the LIWC text features are language-specific), so the FIV2 system is run as English with transcripts translated
# by the same Marian model the own model uses.
systems = {
    "oceanai_fi": lambda: OceanAIBackend(BackendConfig(lang="en", corpus="fi")).load(),
    "mm_noaudio": lambda: MMBackend(MMConfig(lang="ru", drop_modalities=("audio",))).load(),
}
loaded = {}
_translator = MMBackend(MMConfig(lang="ru"))        # only .to_english() is used (lazy Marian ru->en)


def score_video(sys_name, rep):
    be = loaded.setdefault(sys_name, systems[sys_name]())
    rows, w = [], []
    for t in rep["timeline"]:
        if not t.get("scores"):
            continue
        kw = {"asr": False, "transcript": t["transcript"]}
        if sys_name == "oceanai_fi":
            kw["transcript"] = _translator.to_english(t["transcript"])
        if sys_name.startswith("mm"):
            kw["behavior"] = t.get("behavior_description") or None
        try:
            r = be.predict_video(t["file"], **kw)
        except Exception as e:  # noqa: BLE001
            print(f"    seg {t['segment']} skipped ({sys_name}): {str(e).splitlines()[0][:100]}", flush=True)
            continue
        rows.append([r["scores"][k] for k in TRAIT_KEYS])
        w.append(t["end"] - t["start"])
    if not rows:
        return None
    m = np.average(np.array(rows), axis=0, weights=np.array(w))
    return {"scores": dict(zip(TRAIT_KEYS, map(float, m))), "segments": len(rows)}


for d in sorted(out.iterdir()):
    rj = d / "result.json"
    if not rj.exists() or (a.only and d.name not in a.only):
        continue
    rep = json.loads(rj.read_text(encoding="utf-8"))
    entry = cache.setdefault(d.name, {})
    entry["oceanai_mupta"] = {"scores": {k: rep["variants"]["oceanai"][k] for k in TRAIT_KEYS}}
    entry["mm_full"] = {"scores": {k: rep["variants"]["mm"][k] for k in TRAIT_KEYS}}
    for s in systems:
        if s in entry:
            continue
        t0 = time.time()
        print(f"{d.name}: {s} ...", flush=True)
        res = score_video(s, rep)
        if res:
            entry[s] = res
            print(f"    {s}: " + " ".join(f"{k[:5]}={v:.2f}" for k, v in res["scores"].items()) +
                  f"  ({res['segments']} segs, {time.time() - t0:.0f} s)", flush=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

# ---- summary
order = ["oceanai_mupta", "oceanai_fi", "mm_full", "mm_noaudio"]
titles = {"oceanai_mupta": "OCEAN-AI MuPTA", "oceanai_fi": "OCEAN-AI FIV2", "mm_full": "Своя (4 мод.)", "mm_noaudio": "Своя без голоса"}
lines = ["# Диагностика: популяция или русская речь?", "",
         "Те же сегменты, четыре системы. OCEAN-AI FIV2 — те же веса, что для английских роликов (шкала FIV2, транскрипт "
         "переведён). «Своя без голоса» — узел аудио удалён из графа.", "",
         "| Видео | Система | " + " | ".join(RU_SHORT[k] for k in TRAIT_KEYS) + " | среднее |", "| --- | --- | " + " | ".join("---:" for _ in TRAIT_KEYS) + " | ---: |"]
means = {s: [] for s in order}
for name, e in cache.items():
    for s in order:
        if s in e:
            v = [e[s]["scores"][k] for k in TRAIT_KEYS]
            means[s].append(v)
            lines.append(f"| {name[:30]} | {titles[s]} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {np.mean(v):.2f} |")
lines += ["", "## Средние по роликам", "", "| Система | " + " | ".join(RU_SHORT[k] for k in TRAIT_KEYS) + " | среднее |",
          "| --- | " + " | ".join("---:" for _ in TRAIT_KEYS) + " | ---: |"]
for s in order:
    if means[s]:
        m = np.mean(np.array(means[s]), axis=0)
        lines.append(f"| {titles[s]} | " + " | ".join(f"{x:.2f}" for x in m) + f" | {m.mean():.2f} |")
lines += ["", "## Корреляция между системами по роликам (средняя по чертам, n = %d)" % len(cache), ""]
names = [s for s in order if len(means[s]) == len(cache) and len(cache) > 2]
lines += ["| | " + " | ".join(titles[s] for s in names) + " |", "| --- | " + " | ".join("---:" for _ in names) + " |"]
for s1 in names:
    row = []
    for s2 in names:
        cs = []
        for i in range(len(TRAIT_KEYS)):
            x = np.array(means[s1])[:, i]; y = np.array(means[s2])[:, i]
            cs.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan)
        row.append(f"{np.nanmean(cs):.2f}")
    lines.append(f"| {titles[s1]} | " + " | ".join(row) + " |")
(out / "diag_population.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
