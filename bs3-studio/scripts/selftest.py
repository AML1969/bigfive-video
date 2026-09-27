"""End-to-end test of BS Profiler 3.0 without Gradio: analyse one video, build the PDF, print a summary.
Usage: selftest.py VIDEO [ru|en]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # bs3-studio/: `import bs3` is this working tree
from bs3.pipeline import Studio, run_analysis  # noqa: E402
from bs3.webapp import export_pdf  # noqa: E402

video, lang = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "ru")
t0 = time.time()
studio = Studio()
rep = run_analysis(studio, Path.home() / "bs3_data" / "web_jobs", video, lang, True,
                   progress=lambda f, d: print(f"  {f * 100:5.1f}% {d}", flush=True))
print(f"analysis done in {time.time() - t0:.0f} s -> {rep['job_dir']}")
an = rep.get("analyses", {})
print("traits:", {k: round(v["score"], 2) for k, v in rep["traits"].items()})
print("emotions_text:", an.get("emotions_text", {}).get("mean"))
print("face:", {k: v for k, v in (an.get("face") or {}).items() if k != "mean"}, (an.get("face") or {}).get("mean"))
print("voice:", an.get("voice", {}).get("mean"))
print("speech:", {k: v for k, v in (an.get("speech") or {}).items() if k not in ("vocabulary", "description")})
print("narrative:", rep.get("narrative", "")[:300])
t1 = time.time()
pdf = export_pdf(rep["job_dir"])
print(f"pdf: {pdf} ({Path(pdf).stat().st_size // 1024} KB, {time.time() - t1:.1f} s)")
