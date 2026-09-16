"""Run the web request path (run_analysis) without Gradio on one clip and print what the UI would show.
Usage: web_selftest.py VIDEO [lang] [explain 0/1]
"""
import os
import sys
from pathlib import Path

from bs_bigfive.webapp import Engine, run_analysis

video = sys.argv[1]
lang = sys.argv[2] if len(sys.argv) > 2 else "en"
explain = (sys.argv[3] != "0") if len(sys.argv) > 3 else True
engine = Engine()
r = run_analysis(engine, Path(os.path.expanduser("~/bs/web_jobs")), video, lang, explain)
rep = r["report"]
print("scores :", {k: v["score"] for k, v in rep["traits"].items()}, "| interview:", rep.get("interview", {}).get("score"))
print("timing :", r["timing"])
print("desc   :", r["description"][:160])
print("transc :", r["transcript"][:120])
print("frames :", [Path(p).name for p, _ in r["frames"]])
print("members:", r["members"].replace("\n", " | "))
print("contrib:", "yes" if r["contrib_html"] else "no", "| words:", r["words"][:120].replace("\n", " / "))
print("saved  :", r["path"])
print("web selftest ok")
