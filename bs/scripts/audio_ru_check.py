"""Candidate own-model checkpoints on the Russian batch: re-score the stored segments (transcripts and behaviour
descriptions reused, no Ollama), then compare per-video means with OCEAN-AI FIV2 / MuPTA (from diag_population.json)
and the per-segment stability. Usage: audio_ru_check.py OUT_DIR NAME=CKPT_GLOB [NAME=CKPT_GLOB ...]
Writes OUT_DIR/audio_ru_check.json and .md
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from bs_bigfive.backend_mm import MMBackend, MMConfig
from bs_bigfive.norms import RU_SHORT, TRAIT_KEYS

out = Path(sys.argv[1])
cands = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in sys.argv[2:]]
cache_p = out / "audio_ru_check.json"
cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
diag = json.loads((out / "diag_population.json").read_text(encoding="utf-8")) if (out / "diag_population.json").exists() else {}
videos = sorted(d for d in out.iterdir() if (d / "result.json").exists())

for name, ckpt in cands:
    if name in cache and len(cache[name]) == len(videos):
        print(f"{name}: cached"); continue
    be = MMBackend(MMConfig(lang="ru", checkpoint=ckpt)).load()
    print(f"{name}: modalities {be.modalities}, {len(be.checkpoints)} checkpoints", flush=True)
    cache.setdefault(name, {})
    for d in videos:
        if d.name in cache[name]:
            continue
        rep = json.loads((d / "result.json").read_text(encoding="utf-8"))
        rows, w, t0 = [], [], time.time()
        for t in rep["timeline"]:
            if not t.get("scores") or not Path(t["file"]).exists():
                continue
            try:
                r = be.predict_video(t["file"], asr=False, transcript=t["transcript"], behavior=t.get("behavior_description") or None)
            except Exception as e:  # noqa: BLE001
                print(f"    {d.name} seg {t['segment']} failed: {str(e).splitlines()[0][:100]}", flush=True); continue
            rows.append([r["scores"][k] for k in TRAIT_KEYS]); w.append(t["end"] - t["start"])
        m = np.array(rows); ww = np.array(w) / np.sum(w)
        cache[name][d.name] = {"scores": dict(zip(TRAIT_KEYS, map(float, (m * ww[:, None]).sum(0)))),
                               "std": dict(zip(TRAIT_KEYS, map(float, m.std(0)))), "segments": len(rows)}
        print(f"    {d.name}: " + " ".join(f"{k[:5]}={v:.2f}" for k, v in cache[name][d.name]["scores"].items()) +
              f"  std {np.mean(m.std(0)):.3f} ({time.time() - t0:.0f} s)", flush=True)
        cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    del be
    import torch; torch.cuda.empty_cache()

# ---- summary
names = [n for n, _ in cands if n in cache]
refs = {}
for ref in ("oceanai_fi", "oceanai_mupta"):
    refs[ref] = {v: diag[v][ref]["scores"] for v in diag if ref in diag[v]}
lines = ["# Кандидаты аудио-ветви на русских роликах (9 видео, сегменты и описания из батча)", "",
         "| Голосовой узел | " + " | ".join(RU_SHORT[k] for k in TRAIT_KEYS) + " | Средний разброс по сегментам | Согласие с OCEAN-AI FIV2 | Согласие с OCEAN-AI MuPTA |",
         "| --- | " + " | ".join("---:" for _ in TRAIT_KEYS) + " | ---: | ---: | ---: |"]
NAMES = {"clap_base": "CLAP (базовая модель)", "whisper": "энкодер Whisper", "w2v_emo": "wav2vec2-emotion"}


def corr_with(name, ref):
    vids = sorted(set(cache[name]) & set(refs[ref]))
    if len(vids) < 3:
        return float("nan")
    cs = []
    for k in TRAIT_KEYS:
        x = np.array([cache[name][v]["scores"][k] for v in vids]); y = np.array([refs[ref][v][k] for v in vids])
        if x.std() > 0 and y.std() > 0:
            cs.append(np.corrcoef(x, y)[0, 1])
    return float(np.mean(cs)) if cs else float("nan")


for n in names:
    means = np.mean([[cache[n][v]["scores"][k] for k in TRAIT_KEYS] for v in cache[n]], axis=0)
    std = np.mean([np.mean(list(cache[n][v]["std"].values())) for v in cache[n]])
    lines.append(f"| {NAMES.get(n, n)} | " + " | ".join(f"{x:.2f}" for x in means) + f" | {std:.3f} | {corr_with(n, 'oceanai_fi'):.2f} | {corr_with(n, 'oceanai_mupta'):.2f} |")
lines += ["", "Средние оценки своей модели по девяти роликам (шкала FIV2). Согласие — средняя по чертам корреляция Пирсона "
          "между видео (n = 9) с соответствующей системой OCEAN-AI."]
(out / "audio_ru_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
