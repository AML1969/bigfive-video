"""Recompute WCAG contrast for every colour in bs2/palette.py. Exit code 1 if anything fails.
Usage: check_palette.py        (runs with plain python, no package install needed)
"""
import importlib.util
import re
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("palette", Path(__file__).resolve().parents[1] / "bs2" / "palette.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def rgb(c):
    c = c.strip()
    m = re.match(r"rgba?\(([^)]+)\)", c)
    if m:
        parts = [float(x) for x in m.group(1).split(",")]
        return tuple(parts[:3]), (parts[3] if len(parts) > 3 else 1.0)
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)), 1.0


def blend(fg, bg):
    (f, a), (b, _) = rgb(fg), rgb(bg)
    return tuple(a * fi + (1 - a) * bi for fi, bi in zip(f, b))


def lum(t):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = t
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def cr(fg, bg):
    a, b = lum(blend(fg, bg)), lum(rgb(bg)[0])
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


fails = []


def check(name, color, backgrounds, need):
    for bg in backgrounds:
        r = cr(color, bg)
        flag = "OK " if r >= need else "FAIL"
        if r < need:
            fails.append(f"{name} {color} on {bg}: {r:.2f} < {need}")
        print(f"  {flag} {name:34s} {color:24s} on {bg}: {r:5.2f} (need {need})")


for theme, bgs in P.WEB_BACKGROUNDS.items():
    print(f"== web {theme}")
    T = P.THEME[theme]
    for k in ("text", "muted", "hover_text"):
        check(f"text {k}", T[k], bgs if k != "hover_text" else (T["hover_bg"],), 4.5)
    check("axis line", T["axis"], bgs, 3.0)
    check("band border", T["band_border"], bgs, 3.0)
    check("no-data border", T["nodata_border"], bgs, 3.0)
    for group in ("TRAIT_WEB", "VOICE_WEB", "EMO_WEB", "BARS_WEB"):
        for k, c in getattr(P, group)[theme].items():
            check(f"{group}.{k}", c, bgs, 3.0)
    for k in ("main", "second"):
        check(f"RADAR_WEB.{k}", P.RADAR_WEB[theme][k], bgs, 3.0)
    for k, c in P.SPEECH_WEB[theme].items():
        check(f"SPEECH_WEB.{k}", c, bgs, 3.0)

print("== HTML (both themes)")
all_bgs = P.WEB_BACKGROUNDS["dark"] + P.WEB_BACKGROUNDS["light"]
for k in ("main_fill", "interview_fill", "track_outline", "card_border", "status_running", "status_done", "status_stopped"):
    check(f"HTML.{k}", P.HTML[k], all_bgs, 3.0)

print("== PDF (white)")
for group in ("TRAIT_PDF", "VOICE_PDF", "EMO_PDF"):
    for k, c in getattr(P, group).items():
        check(f"{group}.{k}", c, (P.PDF_BACKGROUND,), 3.0)
for k, c in P.SPEECH_PDF.items():
    check(f"SPEECH_PDF.{k}", c, (P.PDF_BACKGROUND,), 3.0)
f = P.SCORE_BAR_PDF["fill"]
check("SCORE_BAR_PDF.fill", "#%02x%02x%02x" % f, ("#%02x%02x%02x" % ((P.SCORE_BAR_PDF["track"],) * 3), "#ffffff"), 3.0)

print("\nFAILURES:" if fails else "\nall checks passed")
for x in fails:
    print("  " + x)
sys.exit(1 if fails else 0)
