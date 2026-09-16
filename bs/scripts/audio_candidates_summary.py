"""Comparison table of the audio-encoder candidates: seed-ensemble test metrics of every run under RUN_ROOT
against the baseline (CLAP) seed ensemble. Usage: audio_candidates_summary.py RUN_ROOT BASELINE_DIR OUT_MD
"""
import json
import sys
from pathlib import Path

run_root, base_dir, out_md = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
TITLES = {"audio": "CLAP (текущий)", "audio_whisper": "энкодер Whisper large-v3", "audio_xlsr": "XLS-R 300m",
          "audio_w2v_emo": "wav2vec2 emotion (audeering)", "audio_egemaps": "eGeMAPS (88 ручных)",
          "audio_e2v": "emotion2vec+ large"}


def row(name, d):
    se = json.loads((d / "seed_ensemble.json").read_text(encoding="utf-8"))
    m = se["mean"]
    per = se.get("per_seed", {})
    macc = [v["mACC"] for v in per.values()]
    iv = m.get("interview", {}).get("ccc", float("nan")) if isinstance(m.get("interview"), dict) else float("nan")
    return (f"| {name} | {len(per)} | {m['mACC']:.4f} | {m['mCCC']:.4f} | {m.get('CCC_flat', float('nan')):.4f} | {iv:.3f} | "
            f"{min(macc):.4f}–{max(macc):.4f} |")


lines = ["# Кандидаты на замену аудио-ветви CLAP (test FIV2, 1997 клипов, среднее 5 seed)", "",
         "| Конфигурация | seed | mACC | mCCC | CCC flat | собесед. CCC | mACC по seed |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
if (base_dir / "seed_ensemble.json").exists():
    lines.append(row("лицо + CLAP + речь + описание (базовая)", base_dir))
for d in sorted(run_root.iterdir()):
    if not (d / "seed_ensemble.json").exists():
        continue
    cand, mode = d.name.rsplit("_", 1)
    title = TITLES.get(cand, cand)
    label = f"лицо + {title} + речь + описание" if mode == "replace" else f"лицо + CLAP + {title} + речь + описание"
    lines.append(row(label, d))
lines += ["", "«replace» — кандидат вместо CLAP; «add» — кандидат как пятый узел графа рядом с CLAP. "
          "mACC = 1 − MAE; CCC flat — по всем оценкам сразу."]
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
