#!/usr/bin/env bash
# Compile the modules touched by variant A and unit-check pool percentiles / report fields / primary resolution.
set -uo pipefail
P="/mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs"
PY="$HOME/bs/venv/bin/python"
cd "$HOME"
"$PY" -m py_compile "$P/bs_bigfive/pool.py" "$P/bs_bigfive/backend_ensemble.py" "$P/bs_bigfive/longvideo.py" \
  "$P/bs_bigfive/report.py" "$P/bs_bigfive/webapp.py" "$P/bs_bigfive/pdf_report.py" "$P/bs_bigfive/cli.py" \
  "$P/scripts/batch_ru.py" && echo "compiled OK"
export BS_POOL_DIR=/tmp/bs_pool_test
rm -rf /tmp/bs_pool_test
"$PY" - <<'EOF'
import json, tempfile, pathlib
from bs_bigfive import pool
from bs_bigfive.backend_ensemble import EnsembleConfig
from bs_bigfive.report import build_report
d = pathlib.Path(tempfile.mkdtemp())
for i in range(5):
    f = d / f"v{i}.bin"; f.write_bytes(bytes([i]) * 2000 + b"x" * i)
    pool.add(f, {"openness": 0.5 + 0.1 * i, "conscientiousness": 0.7, "extraversion": 0.6, "agreeableness": 0.8,
                 "emotional_stability": 0.5}, "ru", "oceanai")
print("pool size", len(pool.entries("ru")), "pct(open 0.7)=", pool.percentile("openness", 0.7, "ru"),
      "pct(cons 0.7 all ties)=", pool.percentile("conscientiousness", 0.7, "ru"), "en pool:", pool.percentile("openness", 0.5, "en"))
rep = build_report(str(d / "v0.bin"), {"scores": {"openness": 0.7, "conscientiousness": 0.7, "extraversion": 0.6,
                   "agreeableness": 0.8, "emotional_stability": 0.5, "interview": 0.4}, "seconds": 1},
                   backend="ensemble", corpus="x", lang="ru", asr_model=None, pool_lang="ru", primary="oceanai")
print(json.dumps(rep["traits"]["openness"], ensure_ascii=False), "|", rep["model"]["scale"], "|", rep["interview"]["percentile_ref"])
rep2 = build_report(str(d / "v0.bin"), {"scores": {k: 0.5 for k in ["openness", "conscientiousness", "extraversion",
                    "agreeableness", "emotional_stability"]}, "seconds": 1}, backend="ensemble", corpus="x", lang="en", asr_model=None)
print("en:", json.dumps(rep2["traits"]["openness"], ensure_ascii=False))
print("auto ru:", EnsembleConfig(lang="ru").primary, "| auto en:", EnsembleConfig(lang="en").primary,
      "| mean:", EnsembleConfig(lang="ru", primary="mean").primary, "|", EnsembleConfig(lang="ru").corpus)
from bs_bigfive.webapp import _bar_html
from bs_bigfive.pdf_report import build_pdf
html = _bar_html(rep["traits"], rep["interview"]); print("bar html ok:", "пул" in html, len(html))
out = build_pdf(rep, str(d / "t.pdf"), explanation=None, media=None, key_frames=[]); print("pdf ok:", pathlib.Path(out).stat().st_size)
EOF
rm -rf /tmp/bs_pool_test
