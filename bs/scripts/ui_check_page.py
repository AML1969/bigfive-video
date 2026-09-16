"""Static pages to eyeball the web charts and key frames on dark and light backgrounds (like the Gradio Soft theme).
Usage: ui_check_page.py JOB_DIR OUT_DIR       then: python -m http.server 7872 --directory OUT_DIR
"""
import json
import sys
from pathlib import Path

from bs_bigfive import charts as ch

job, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rep = json.loads((job / "result.json").read_text(encoding="utf-8"))
frames = [(str(p), "") for p in sorted((job / "explain").glob("key_*.jpg"))]
body = (f"<h3>Big Five по ходу ролика</h3><div style='padding:12px;border-radius:8px;background:var(--block)'>{ch.traits_timeline_html(rep)}</div>"
        f"<h3>Ключевые кадры</h3><div style='padding:12px;border-radius:8px;background:var(--block)'>{ch.frames_html(rep, frames)}</div>")
for name, cls, page, block, color in (("dark", "dark", "#0b0f19", "#1f2937", "#e5e7eb"), ("light", "", "#f9fafb", "#ffffff", "#111827")):
    (out / f"{name}.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><style>:root{{--block:{block}}}</style></head>"
        f"<body class='{cls}' style='background:{page};color:{color};font-family:sans-serif;margin:0;padding:16px'>{body}</body></html>",
        encoding="utf-8")
print("written", out / "dark.html", out / "light.html")
