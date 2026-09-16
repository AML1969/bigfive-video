"""Compare rater evaluations (JSONs written by rater_eval.py) side by side.
Usage: rater_ablation_summary.py OUT_MD NAME=EVAL_JSON [NAME=EVAL_JSON ...]
"""
import json
import sys
from pathlib import Path

out_md = Path(sys.argv[1])
runs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in sys.argv[2:]]
variants = ["rater_raw", "rater_calibrated", "oceanai", "own5", "mean(oceanai,own5)",
            "mean(oceanai,own5,rater_calibrated)", "0.4*oceanai+0.4*own5+0.2*rater_calibrated"]
lines = ["# LLM-оценщик: варианты числа кадров (test200 FIV2)", "",
         "| Вариант | " + " | ".join(f"{n} mACC | {n} mCCC" for n, _ in runs) + " |",
         "| --- | " + " | ".join("---: | ---:" for _ in runs) + " |"]
data = {}
for n, p in runs:
    if p.exists():
        data[n] = json.loads(p.read_text(encoding="utf-8"))
for v in variants:
    cells = []
    for n, _ in runs:
        r = data.get(n, {}).get(v)
        cells.append(f"{r['mACC']:.4f} | {r['mCCC']:.3f}" if r else "— | —")
    lines.append(f"| {v} | " + " | ".join(cells) + " |")
lines.append("")
for n, _ in runs:
    d = data.get(n, {})
    if "error_correlation" in d:
        lines.append(f"- {n}: корреляция ошибок {d['error_correlation']}; калибровка (наклон, сдвиг) "
                     + str({k: (round(a, 2), round(b, 2)) for k, (a, b) in d.get('calibration', {}).items()}))
lines += ["", "mACC = 1 − MAE; own5 = своя модель (среднее 5 seed); калибровка — линейная по dev200 (16 кадров)."]
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
