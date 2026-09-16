"""Single source of colours for BS 2.0 (web charts, HTML infographics, PDF charts).

Why two web palettes: plotly charts are transparent iframes, so they sit on the Gradio background, which is dark
(#1f2937 block / #0b0f19 page) or light (#ffffff block / #f9fafb page). No single text colour reaches 4.5:1 on both,
and several series colours cannot reach 3:1 on both, so every web figure is built twice (theme='dark' / 'light') and
the iframe picks one from the Gradio `dark` class on the page body.
PDF charts are on white paper and use their own, darker palette (PDF_*). Do not "unify" the palettes: a colour that
works on white fails on the dark block and vice versa. scripts/check_palette.py recomputes every contrast ratio.

Rules checked by scripts/check_palette.py (WCAG 2.1):
  text >= 4.5:1, graphical marks (lines, bars, markers, swatches, bar tracks) >= 3:1 on every background of the theme.
"""
from __future__ import annotations

WEB_BACKGROUNDS = {"dark": ("#1f2937", "#0b0f19"), "light": ("#ffffff", "#f9fafb")}
PDF_BACKGROUND = "#ffffff"

# ---------------------------------------------------------------- web chart chrome (text, axes, grid, hover)
THEME = {
    "dark": dict(text="#e5e7eb", muted="#d1d5db", grid="#374151", axis="#6b7280", hover_bg="#111827",
                 hover_border="#6b7280", hover_text="#f3f4f6", sep="#111827", band="rgba(229,231,235,0.10)",
                 band_border="#9ca3af", nodata="rgba(248,113,113,0.14)", nodata_border="#f87171"),
    "light": dict(text="#1f2937", muted="#374151", grid="#e5e7eb", axis="#6b7280", hover_bg="#ffffff",
                  hover_border="#6b7280", hover_text="#111827", sep="#ffffff", band="rgba(31,41,55,0.08)",
                  band_border="#4b5563", nodata="rgba(185,28,28,0.10)", nodata_border="#b91c1c"),
}
FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"

# ---------------------------------------------------------------- web series colours (>= 3:1 on both backgrounds of a theme)
TRAIT_WEB = {
    "dark": {"openness": "#60a5fa", "conscientiousness": "#34d399", "extraversion": "#fb923c",
             "agreeableness": "#c084fc", "emotional_stability": "#f87171", "interview": "#fbbf24"},
    "light": {"openness": "#1d4ed8", "conscientiousness": "#047857", "extraversion": "#c2410c",
              "agreeableness": "#7e22ce", "emotional_stability": "#be123c", "interview": "#a16207"},
}
TRAIT_SYMBOL = {"openness": "circle", "conscientiousness": "square", "extraversion": "triangle-up",
                "agreeableness": "diamond", "emotional_stability": "triangle-down", "interview": "star"}

# voice dimensions use hues that are NOT used by the traits (same «Таймлайн» tab)
VOICE_WEB = {
    "dark": {"arousal": "#f472b6", "dominance": "#2dd4bf", "valence": "#a3e635"},
    "light": {"arousal": "#be185d", "dominance": "#0f766e", "valence": "#4d7c0f"},
}
VOICE_SYMBOL = {"arousal": "circle", "dominance": "square", "valence": "diamond"}

# emotions: stacked bands are separated by THEME[t]["sep"] lines, so each band only needs >= 3:1 to the background
EMO_WEB = {
    "dark": {"joy": "#facc15", "surprise": "#fb923c", "neutral": "#94a3b8", "sadness": "#60a5fa", "fear": "#a78bfa",
             "anger": "#f87171", "disgust": "#4ade80"},
    "light": {"joy": "#a16207", "surprise": "#c2410c", "neutral": "#64748b", "sadness": "#1d4ed8", "fear": "#6d28d9",
              "anger": "#b91c1c", "disgust": "#15803d"},
}
EMO_ALIAS = {"happy": "joy", "sad": "sadness", "angry": "anger"}     # face-expression labels -> text-emotion keys

# radar and score bars: the main score is blue, the second opinion is a neutral grey (another model, another scale)
RADAR_WEB = {
    "dark": dict(main="#60a5fa", main_fill="rgba(96,165,250,0.22)", second="#cbd5e1"),
    "light": dict(main="#1d4ed8", main_fill="rgba(29,78,216,0.16)", second="#475569"),
}
SPEECH_WEB = {
    "dark": dict(bars="#6b7280", pauses="#f3f4f6"),
    "light": dict(bars="#6b7280", pauses="#111827"),
}
BARS_WEB = {"dark": dict(speech="#60a5fa", face="#fb923c"), "light": dict(speech="#1d4ed8", face="#c2410c")}

# ---------------------------------------------------------------- HTML infographics (inline styles, must work on BOTH themes)
HTML = dict(
    main_fill="#3b82f6",                 # score bars: 3.99 on #1f2937, 3.68 on white
    interview_fill="#b7791f",            # 4.03 / 3.64
    second_fill="currentColor",          # second opinion: theme text colour at opacity .55
    track="transparent", track_outline="#6b7280",   # outline 3.04 on #1f2937, 4.83 on white
    card_border="#6b7280", table_rule="#6b7280",
    status_running="#ea580c", status_done="#16a34a", status_stopped="#dc2626",
)

# ---------------------------------------------------------------- PDF (white paper)
TRAIT_PDF = {"openness": "#1d4ed8", "conscientiousness": "#0f766e", "extraversion": "#c2410c",
             "agreeableness": "#7e22ce", "emotional_stability": "#b91c1c", "interview": "#8a6d3b"}
TRAIT_MARKER_PDF = {"openness": "o", "conscientiousness": "s", "extraversion": "^", "agreeableness": "D",
                    "emotional_stability": "v", "interview": "o"}
VOICE_PDF = {"arousal": "#be185d", "dominance": "#0e7490", "valence": "#4d7c0f"}
VOICE_MARKER_PDF = {"arousal": "o", "dominance": "s", "valence": "^"}
EMO_PDF = {"joy": "#a16207", "surprise": "#c2410c", "neutral": "#64748b", "sadness": "#1f5a99", "fear": "#6d28d9",
           "anger": "#b91c1c", "disgust": "#15803d"}
SPEECH_PDF = dict(bars="#64748b", pauses="#111827")
SCORE_BAR_PDF = dict(track=242, outline=130, fill=(29, 78, 216), interview=(138, 109, 59), mid_tick=85)


def emo(theme: str, key: str) -> str:
    return EMO_WEB[theme].get(EMO_ALIAS.get(key, key), "#888888")


def emo_pdf(key: str) -> str:
    return EMO_PDF.get(EMO_ALIAS.get(key, key), "#888888")
