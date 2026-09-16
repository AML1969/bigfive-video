"""Render the web score block (main bars + second opinion + timeline) from a batch result.json to an HTML file.
Usage: check_members_html.py RESULT_JSON OUT_HTML
"""
import json
import sys

from bs_bigfive.webapp import _bar_html, _members_html, _timeline_html

rep = json.loads(open(sys.argv[1], encoding="utf-8").read())
rep.setdefault("variant_scores", rep.get("variants"))
rep.setdefault("scores_std_across_segments", rep.get("scores_std"))
html = _bar_html(rep["traits"], rep.get("interview")) + _members_html(rep) + _timeline_html(rep)
open(sys.argv[2], "w", encoding="utf-8").write("<meta charset='utf-8'><body style='font-family:sans-serif;padding:16px'>" + html)
print("members block present:", "Второе мнение" in html or "Участники ансамбля" in html, "| chars", len(html))
