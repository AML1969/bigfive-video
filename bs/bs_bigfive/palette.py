"""Single source of colours for the BS 1.0 web page (Gradio 5.8 Default theme, zinc neutrals).

Backgrounds of the Default theme:
  dark:  page #0f0f11, block #27272a, text #f4f4f5
  light: page #ffffff, block #ffffff, text #27272a
The HTML infographics (score bars, tables, status bar) use inline styles, so one fixed colour has to work on BOTH
themes: every value below reaches the WCAG 2.1 minimum on the dark and on the light background at the same time
(text >= 4.5:1, graphical marks such as bars, lines, markers and outlines >= 3:1). Where no single colour can do that
(red warning text), a CSS class with a `.dark` override is used (SKIP_TEXT). Secondary notes are not grey: they take
the theme text colour at MUTED_OPACITY, which stays readable in both themes.
The Plotly chart sits in a transparent iframe and switches CHART[theme] from the Gradio `dark` class.
check() recomputes every ratio; run it after changing a value.
"""
from __future__ import annotations

import re

BACKGROUNDS = {"dark": {"page": "#0f0f11", "block": "#27272a", "text": "#f4f4f5"},
               "light": {"page": "#ffffff", "block": "#ffffff", "text": "#27272a"}}

# Big Five series: the same colour for a trait's score bar and its chart line; >= 3:1 on both blocks.
# The colours are close in lightness, so the chart also gives every trait its own marker symbol.
TRAIT_COLORS = {"openness": "#3b82f6", "conscientiousness": "#16a34a", "extraversion": "#ea580c",
                "agreeableness": "#a855f7", "emotional_stability": "#e11d48", "interview": "#b7791f"}
TRAIT_SYMBOLS = {"openness": "circle", "conscientiousness": "square", "extraversion": "diamond",
                 "agreeableness": "triangle-up", "emotional_stability": "x", "interview": "star"}

OUTLINE = "#808080"            # hollow bar tracks, card frame, chart frame of the explanation segment, hatch
SECOND_OPINION = "#6d89c7"     # second-opinion bars: muted blue, thinner bar
STATUS = {"running": "#ea580c", "done": "#2e8b57", "stopped": "#dc2626", "error": "#dc2626"}
SKIP_TEXT = {"light": "#b91c1c", "dark": "#fca5a5"}     # «пропущен: нет лица или речи» row (CSS class + .dark)
TABLE_RULE = "rgba(128,128,128,.45)"                     # decorative table rules (Gradio draws them in text colour)
MUTED_OPACITY = 0.75                                     # notes and footnotes: theme text colour, dimmed

# Plotly chart chrome per theme (text, tick labels, grid, axis lines, hover label)
CHART = {
    "dark": dict(text="#e5e7eb", sub="#cbd5e1", grid="rgba(229,231,235,0.16)", axis="rgba(229,231,235,0.55)",
                 hover_bg="#111827"),
    "light": dict(text="#111827", sub="#374151", grid="rgba(17,24,39,0.12)", axis="rgba(17,24,39,0.55)",
                  hover_bg="#ffffff"),
}
FONT_FAMILY = "'Source Sans Pro', ui-sans-serif, system-ui, sans-serif"   # the Gradio Default theme font

# Gradio theme overrides (webapp._theme): white button labels need a darker fill than the Default orange / red
BUTTON_PRIMARY, BUTTON_PRIMARY_HOVER = "#c2410c", "#9a3412"     # orange-700 / 800 (Default: 500, 2.8:1 with white)
BUTTON_STOP, BUTTON_STOP_HOVER = "#b91c1c", "#991b1b"           # red-700 / 800 (Default light: 500, 3.8:1)
SUBDUED_TEXT_LIGHT = "#52525b"  # zinc-600: Radio info line and placeholders on white (zinc-400 is 1.9:1)


# ---------------------------------------------------------------- contrast arithmetic (WCAG 2.1)
def _rgba(c: str):
    c = c.strip()
    m = re.fullmatch(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)", c)
    if m:
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4) or 1))
    h = c.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 1.0)


def blend(fg: str, bg: str, opacity: float = 1.0):
    """Colour actually seen: fg (with its own alpha and an extra opacity) over an opaque bg."""
    r, g, b, a = _rgba(fg)
    br, bgc, bb, _ = _rgba(bg)
    a *= opacity
    return (r * a + br * (1 - a), g * a + bgc * (1 - a), b * a + bb * (1 - a))


def luminance(rgb) -> float:
    def ch(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = rgb[:3]
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(fg: str, bg: str, opacity: float = 1.0) -> float:
    l1, l2 = luminance(blend(fg, bg, opacity)), luminance(_rgba(bg))
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def check():
    """Every colour pair the page relies on: [(what, fg, bg, ratio, minimum, ok)]."""
    rows = []

    def add(what, fg, bg, minimum, opacity=1.0):
        r = contrast(fg, bg, opacity)
        rows.append((what, fg if opacity == 1 else f"{fg}@{opacity}", bg, round(r, 2), minimum, r >= minimum))

    for t in ("dark", "light"):
        bg = BACKGROUNDS[t]
        for k, c in TRAIT_COLORS.items():
            add(f"{t}: trait bar/line {k}", c, bg["block"], 3.0)
            add(f"{t}: hover swatch {k}", c, CHART[t]["hover_bg"], 3.0)
        add(f"{t}: outline on block", OUTLINE, bg["block"], 3.0)
        add(f"{t}: outline on page (status bar)", OUTLINE, bg["page"], 3.0)
        add(f"{t}: second-opinion bar", SECOND_OPINION, bg["block"], 3.0)
        for k, c in STATUS.items():
            add(f"{t}: status bar {k}", c, bg["page"], 3.0)
        add(f"{t}: skipped-segment text", SKIP_TEXT[t], bg["block"], 4.5)
        add(f"{t}: muted note text", bg["text"], bg["block"], 4.5, MUTED_OPACITY)
        add(f"{t}: chart caption text (.85)", bg["text"], bg["block"], 4.5, 0.85)
        add(f"{t}: chart text", CHART[t]["text"], bg["block"], 4.5)
        add(f"{t}: chart tick labels", CHART[t]["sub"], bg["block"], 4.5)
        add(f"{t}: chart axis line", CHART[t]["axis"], bg["block"], 3.0)
        add(f"{t}: hover text", CHART[t]["text"], CHART[t]["hover_bg"], 4.5)
    for name, fill in (("primary button", BUTTON_PRIMARY), ("primary button hover", BUTTON_PRIMARY_HOVER),
                       ("stop button", BUTTON_STOP), ("stop button hover", BUTTON_STOP_HOVER)):
        add(f"both: {name} label", "#ffffff", fill, 4.5)
    add("light: subdued text (Radio info)", SUBDUED_TEXT_LIGHT, BACKGROUNDS["light"]["block"], 4.5)
    return rows


if __name__ == "__main__":
    bad = 0
    for what, fg, bg, r, mn, ok in check():
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {r:5.2f} >= {mn:<3} {what:45s} {fg} on {bg}")
    raise SystemExit(1 if bad else 0)
