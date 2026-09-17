"""PDF charts for BS 2.0: matplotlib PNGs placed across the full text width of the A4 pages by pdf_report.py.

Colours come from palette.py (the *_PDF dictionaries, contrast-checked on white by scripts/check_palette.py), never
from the web palettes: a colour that reads on the dark Gradio block fails on paper and vice versa.

Sizes: every figure is drawn at its printed size, TEXT_W_MM wide (the page width minus the margins of pdf_report.Report),
and pdf_report places it at exactly that width, so 1 pt in matplotlib is 1 pt on paper: tick labels and legends 8 pt,
axis titles 9 pt, chart titles 9.5-10 pt. PNGs are rendered at 300 dpi. Every legend sits OUTSIDE the axes, under
them, so the plot area keeps the full width and the text never covers data.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .charts import EMO_RU, VOICE_RU, _segments
from .norms import RU_TITLES, TRAIT_KEYS
from .palette import SPEECH_PDF, THEME, TRAIT_MARKER_PDF, TRAIT_PDF, VOICE_MARKER_PDF, VOICE_PDF, emo_pdf
from .pdf_report import TEXT_W_MM

NAN = float("nan")
_L = THEME["light"]                 # white paper = the light web theme's chrome colours
INK = _L["text"]                    # titles, axis labels (14.7:1 on white)
MUTED = _L["muted"]                 # tick labels (10.3:1)
AXIS = _L["axis"]                   # spines (4.83:1)
GRID = "#d0d0d0"                    # supplementary grid lines, deliberately quiet (1.54:1)
OUTLINE = "#333333"                 # outline of legend swatches, so light bands still show a clear edge (12.6:1)
NODATA_HATCH = AXIS                 # grey hatching = "no data" in every PDF chart
BARS_TEXT = "#475569"               # tick labels / axis title of the tempo scale: slate like the bars, 7.58:1 as text

RC = {
    "font.size": 8.5, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelsize": 9, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "legend.fontsize": 8, "legend.title_fontsize": 8, "legend.frameon": False,
    "figure.titlesize": 10, "figure.titleweight": "bold", "figure.dpi": 300, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "hatch.linewidth": 0.6,
}
# printed heights, mm (the width is always TEXT_W_MM)
HEIGHT_MM = {"traits": 70, "emotions": 115, "voice": 62, "speech": 62}
TIME_TICKS = 13                     # at most this many labels on a full-width time axis (6 min -> every 30 s)
SERIES_LW, SERIES_MS = 1.6, 3.5     # line width / marker size of the series, pt on paper
# horizontal extent of the plot area of every time chart, mm from the left and the right edge of the image: the left
# margin fits the widest scale («100%» and «доля» of the emotions chart), the right one the pause scale of the speech
# chart. Charts stacked on a page then share one time axis (0:30 at the same x on all of them); a chart whose labels
# need more room (a four-digit tempo scale) takes it and is the only one off by that much
PLOT_LEFT_MM, PLOT_RIGHT_MM = 16.5, 11.5


def _figure(plt, name: str, nrows: int = 1, **kw):
    """Figure at its printed size: TEXT_W_MM × HEIGHT_MM[name]."""
    return plt.subplots(nrows, 1, figsize=(TEXT_W_MM / 25.4, HEIGHT_MM[name] / 25.4), layout="constrained", **kw)


def _align_plot(fig) -> None:
    """Replace the horizontal extent of every axes (twin axes too) by the common PLOT_LEFT_MM / PLOT_RIGHT_MM. Constrained
    layout still places titles, the legend and the vertical margins; it is switched off afterwards, so the common x
    extent is not recomputed per figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_mm = fig.bbox.width / TEXT_W_MM
    axes = fig.axes
    x0 = min(ax.get_position().x0 for ax in axes) * fig.bbox.width
    x1 = max(ax.get_position().x1 for ax in axes) * fig.bbox.width
    boxes = [ax.get_tightbbox(renderer) for ax in axes]
    need_l = (x0 - min(b.x0 for b in boxes)) / px_mm + 0.8
    need_r = (max(b.x1 for b in boxes) - x1) / px_mm + 0.8
    left, right = max(PLOT_LEFT_MM, need_l), max(PLOT_RIGHT_MM, need_r)
    fig.set_layout_engine("none")
    for ax in axes:
        p = ax.get_position()
        ax.set_position([left / TEXT_W_MM, p.y0, 1 - (left + right) / TEXT_W_MM, p.height])


def _rgba(css: str) -> Tuple[float, float, float, float]:
    """'rgba(31,41,55,0.08)' or '#1f2937' -> matplotlib RGBA tuple."""
    s = css.strip()
    if s.startswith("rgba"):
        r, g, b, a = [float(v) for v in s[s.index("(") + 1:s.index(")")].split(",")]
        return r / 255, g / 255, b / 255, a
    s = s.lstrip("#")
    return int(s[0:2], 16) / 255, int(s[2:4], 16) / 255, int(s[4:6], 16) / 255, 1.0


def _num(v) -> float:
    try:
        return NAN if v is None else float(v)
    except (TypeError, ValueError):
        return NAN


def _duration(rep: dict, rows: List[dict]) -> float:
    ends = [float(r.get("end") or 0) for r in rows]
    return max([float(rep.get("duration_sec") or 0)] + ends + [1.0])


def _time_axis(ax, dur: float, max_ticks: int = TIME_TICKS, label: bool = True) -> None:
    """x axis from 0 to the end of the video with round steps; m:ss labels for any length («0:05» in a 15-second
    clip), as in the tables and on the web charts; h:mm:ss from an hour on."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    ax.set_xlim(0, dur)
    step = next((s for s in (5, 10, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600) if dur / s <= max_ticks), 7200)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    if dur >= 3600:
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f"{int(round(v)) // 3600}:{int(round(v)) % 3600 // 60:02d}:{int(round(v)) % 60:02d}"))
        text = "время ролика, ч:мин:с"
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v)) // 60}:{int(round(v)) % 60:02d}"))
        text = "время ролика, мин:с"
    if label:
        ax.set_xlabel(text)


def _fig_legend(fig, handles, labels, **kw):
    """Legend outside the axes, under the whole chart: the fewest rows that fit the figure width, then the fewest
    columns for that many rows (7 entries on two rows are 4 + 3, not 6 + 1), read row by row like text. A legend
    above the axes would collide with a suptitle: constrained layout reserves the top margin for only one of them."""
    renderer = fig.canvas.get_renderer()
    n = len(handles)
    if not n:
        return None
    for rows in range(1, n + 1):
        ncols = math.ceil(n / rows)
        # matplotlib fills a legend column by column, the first columns getting the extra entries; listing the
        # entries column by column out of a row-major grid makes the rows read left to right
        order = [r * ncols + c for c in range(ncols) for r in range(math.ceil(n / ncols)) if r * ncols + c < n]
        args = ([handles[i] for i in order], [labels[i] for i in order])
        if ncols == 1:
            break
        probe = fig.legend(*args, ncols=ncols, **kw)        # measured and removed again
        fits = probe.get_window_extent(renderer).width <= 0.99 * fig.bbox.width
        probe.remove()
        if fits:
            break
    return fig.legend(*args, loc="outside lower center", ncols=ncols, **kw)


def _nodata_patch(label: str):
    from matplotlib.patches import Patch
    return Patch(facecolor="white", edgecolor=NODATA_HATCH, hatch="///", lw=0.6, label=label)


def _nodata_span(ax, start: float, end: float) -> None:
    ax.axvspan(start, end, facecolor="white", edgecolor=NODATA_HATCH, hatch="///", lw=0, zorder=0.6)


# ---------------------------------------------------------------- Big Five timeline
def _traits_chart(plt, rep: dict, out_dir: Path) -> Optional[str]:
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory
    segs = _segments(rep)
    if not segs:
        return None
    timeline = rep.get("timeline") or segs        # full timeline: skipped segments break the lines (NaN) instead of
    dur = _duration(rep, timeline)                # being silently interpolated across
    fig, ax = _figure(plt, "traits")
    keys = list(TRAIT_KEYS) + (["interview"] if all("interview" in t["scores"] for t in segs) else [])
    x = [(t["start"] + t["end"]) / 2 for t in timeline]
    for k in keys:
        ax.plot(x, [_num((t.get("scores") or {}).get(k)) if t.get("scores") else NAN for t in timeline],
                color=TRAIT_PDF[k], marker=TRAIT_MARKER_PDF[k], ms=SERIES_MS, lw=SERIES_LW, ls=":" if k == "interview" else "-",
                label=RU_TITLES[k], clip_on=False, zorder=3)
    handles, labels = ax.get_legend_handles_labels()
    rep_i = rep.get("representative_segment")
    t_rep = next((t for t in segs if t["segment"] == rep_i), None) if rep_i and len(segs) > 1 else None
    if t_rep:
        band, border = _rgba(_L["band"]), _L["band_border"]
        ax.axvspan(t_rep["start"], t_rep["end"], facecolor=band, edgecolor=border, lw=0.8, ls="--", zorder=1)
        ax.text((t_rep["start"] + t_rep["end"]) / 2, 0.02, "★", transform=blended_transform_factory(ax.transData, ax.transAxes),
                ha="center", va="bottom", fontsize=10, color=border, zorder=4)
        handles.append(Patch(facecolor=band, edgecolor=border, lw=0.8, ls="--"))
        labels.append("★ отрезок для объяснений")
    skipped = [t for t in (rep.get("timeline") or []) if not t.get("scores")]
    for t in skipped:
        _nodata_span(ax, t["start"], t["end"])
    if skipped:
        handles.append(_nodata_patch("")); labels.append("отрезок пропущен (нет оценки)")
    ax.set_ylim(0, 1); ax.set_ylabel("оценка 0…1"); ax.set_title("Big Five по ходу ролика")
    _time_axis(ax, dur)
    ax.set_axisbelow(True); ax.grid(color=GRID, lw=0.5)
    _fig_legend(fig, handles, labels, handlelength=2.2, columnspacing=1.4)
    _align_plot(fig)
    p = out_dir / "chart_traits.png"
    fig.savefig(p); plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- emotions (text + face), stacked shares
def _emotions_chart(plt, rep: dict, per: List[dict], out_dir: Path) -> str:
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter
    from .analyses.emotions_text import EMOTION_ORDER
    from .analyses.face_expr import EXPR_ORDER
    dur = _duration(rep, per)
    fig, axes = _figure(plt, "emotions", nrows=2, sharex=True)
    fig.suptitle("Эмоции по ходу ролика (доли, сумма = 100%)", x=0.01, ha="left")
    any_gap = False
    for ax, key, order, title in ((axes[0], "emotions_text", EMOTION_ORDER, "По речи (что говорит)"),
                                  (axes[1], "face", EXPR_ORDER, "По выражению лица (как выглядит)")):
        # each segment is one step from its start to its end: the bands cover exactly the analysed time, and a
        # segment without data is a hatched gap, not a band collapsing to zero between two midpoints
        xs: List[float] = []
        cols: Dict[str, List[float]] = {k: [] for k in order}
        for r in per:
            if key == "face":
                d = (r.get("face") or {}).get("expressions") or {}
            else:
                no_speech = "text_en" in r and not str(r.get("text_en") or "").strip()   # empty text is scored "neutral"
                d = {} if no_speech else (r.get(key) or {})
            vals = {k: _num(d.get(k)) for k in order}
            vals = {k: (0.0 if math.isnan(v) else max(0.0, v)) for k, v in vals.items()}
            total = sum(vals.values())
            ok = bool(d) and total > 1e-6
            if not ok:
                _nodata_span(ax, r["start"], r["end"]); any_gap = True
            xs += [r["start"], r["end"]]
            for k in order:
                # shares are rescaled to sum to 1: rounded model outputs (0.99…) would leave a white sliver at the
                # top of the stack that reads as "missing data"
                v = vals[k] / total if ok else NAN
                cols[k] += [v, v]
        ax.stackplot(xs, [cols[k] for k in order], colors=[emo_pdf(k) for k in order], edgecolor="white", linewidth=0.5)
        ax.set_ylim(0, 1); ax.set_yticks([0, .25, .5, .75, 1]); ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("доля"); ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", color="white", alpha=1.0, lw=0.5)          # drawn over the bands on purpose
    _time_axis(axes[1], dur)
    # one legend for both panels, under the time axis, left to right in the stacking order (bottom band first), the
    # same order as the columns of the table of averages
    handles = [Patch(facecolor=emo_pdf(k), edgecolor=OUTLINE, lw=0.5, label=EMO_RU.get(k, k)) for k in EMOTION_ORDER]
    if any_gap:
        handles.append(_nodata_patch("нет данных"))       # what "no data" means is in the note under the table
    # compact swatches and spacing: all seven emotions and «нет данных» fit on one line
    _fig_legend(fig, handles, [h.get_label() for h in handles], title="Цвет эмоции одинаков на обоих графиках:",
                alignment="left", handlelength=1.3, handletextpad=0.5, columnspacing=1.1)
    _align_plot(fig)
    p = out_dir / "chart_emotions.png"
    fig.savefig(p); plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- voice dimensions
def _voice_chart(plt, rep: dict, per: List[dict], out_dir: Path) -> str:
    dur = _duration(rep, per)
    x = [(r["start"] + r["end"]) / 2 for r in per]
    fig, ax = _figure(plt, "voice")
    ax.set_title("Голос: модель эмоций в речи")
    for d, name in VOICE_RU.items():
        ax.plot(x, [_num((r.get("voice") or {}).get(d)) for r in per], color=VOICE_PDF[d], marker=VOICE_MARKER_PDF[d],
                ms=SERIES_MS, lw=SERIES_LW, label=name, clip_on=False, zorder=3)
    ax.set_ylim(0, 1); ax.set_ylabel("0 = низко, 1 = высоко")
    ax.set_axisbelow(True); ax.grid(color=GRID, lw=0.5)
    h, lab = ax.get_legend_handles_labels()
    no_voice = [r for r in per if not r.get("voice")]          # the lines break there: say why
    for r in no_voice:
        _nodata_span(ax, r["start"], r["end"])
    if no_voice:
        h.append(_nodata_patch("")); lab.append("нет данных")
    _fig_legend(fig, h, lab, handlelength=1.8, columnspacing=1.4)
    _time_axis(ax, dur)
    _align_plot(fig)
    p = out_dir / "chart_voice.png"
    fig.savefig(p); plt.close(fig)
    return str(p)


# ---------------------------------------------------------------- speech tempo and pauses
def _speech_chart(plt, rep: dict, per: List[dict], out_dir: Path) -> str:
    import matplotlib.patheffects as pe
    from matplotlib.ticker import PercentFormatter
    dur = _duration(rep, per)
    x = [(r["start"] + r["end"]) / 2 for r in per]
    fig, ax = _figure(plt, "speech")
    ax.set_title("Речь: темп и паузы")
    bars_c, pause_c = SPEECH_PDF["bars"], SPEECH_PDF["pauses"]
    with_wpm = [r for r in per if (r.get("speech") or {}).get("words_per_min_speech") is not None]
    no_wpm = [r for r in per if r.get("speech") and r["speech"].get("words_per_min_speech") is None]
    no_speech = [r for r in per if not r.get("speech")]         # no speech analytics at all for the segment
    values = [float(r["speech"]["words_per_min_speech"]) for r in with_wpm]
    ax.bar([(r["start"] + r["end"]) / 2 for r in with_wpm], values,
           width=[max(4.0, 0.8 * (r["end"] - r["start"])) for r in with_wpm], color=bars_c, alpha=1.0, zorder=2)
    for r in no_wpm + no_speech:
        _nodata_span(ax, r["start"], r["end"])
    # five equal steps, like the 0-100% pause scale on the right, so both grids line up; the step is a round
    # number (120 → 0/120/…/600, 300 → 0/300/…/1500) instead of 20·ceil(max/100), which gave 260 or 340
    top = max(values) if values else 0
    step = next((s for s in (20, 40, 60, 80, 100, 120, 160, 200, 240, 300, 400, 500, 600, 800, 1000) if 5 * s >= top),
                200 * math.ceil(top / 1000))
    ax.set_ylim(0, 5 * step); ax.set_yticks(range(0, 5 * step + 1, step))
    # left scale text: a darker slate of the bar colour (7.58:1 on white; the bar colour itself is only 4.76:1)
    ax.set_ylabel("слов в минуту", color=BARS_TEXT); ax.tick_params(axis="y", colors=BARS_TEXT)
    ax.set_axisbelow(True); ax.grid(color=GRID, lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(x, [_num((r.get("speech") or {}).get("pause_share")) for r in per], color=pause_c, lw=SERIES_LW, marker="s",
             ms=SERIES_MS, mfc="white", mec=pause_c, clip_on=False, zorder=3,
             path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
    ax2.set_ylim(0, 1); ax2.set_yticks([0, .2, .4, .6, .8, 1]); ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    # the right scale has no axis title: the legend names it («доля пауз, % (правая шкала)») and its ticks are in the
    # line colour; a title there would widen the common right margin of every chart (PLOT_RIGHT_MM) by 5 mm
    ax2.tick_params(axis="y", colors=pause_c)
    ax2.spines["left"].set_visible(False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=bars_c, edgecolor=bars_c),
               Line2D([], [], color=pause_c, lw=SERIES_LW, marker="s", ms=SERIES_MS, mfc="white", mec=pause_c)]
    labels = ["слов в минуту (левая шкала)", "доля пауз, % (правая шкала)"]
    if no_wpm or no_speech:          # one hatch style = one legend entry
        handles.append(_nodata_patch(""))
        labels.append("темп не посчитан (речи меньше 1 с)" if not no_speech else
                      "нет данных о речи" if not no_wpm else "нет данных о темпе")
    _fig_legend(fig, handles, labels, handlelength=1.8, columnspacing=1.4)
    _time_axis(ax, dur)
    _align_plot(fig)
    p = out_dir / "chart_speech.png"
    fig.savefig(p); plt.close(fig)
    return str(p)


def save_pdf_charts(rep: dict, out_dir: str | Path) -> Dict[str, str]:
    """PNG files for the PDF: traits timeline, emotions (text + face), voice, speech. Returns {name: path}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chart_voice_speech.png").unlink(missing_ok=True)     # older exports drew voice and speech side by side
    files: Dict[str, str] = {}
    per = (rep.get("analyses") or {}).get("per_segment") or []
    with plt.rc_context(RC):            # wraps figure creation AND savefig, so sizes and dpi apply to every PNG
        p = _traits_chart(plt, rep, out_dir)
        if p:
            files["traits"] = p
        if per:
            files["emotions"] = _emotions_chart(plt, rep, per, out_dir)
            # a chart without any data is not drawn: pdf_report says so in one line instead of a blank full-width frame
            if any(r.get("voice") for r in per):
                files["voice"] = _voice_chart(plt, rep, per, out_dir)
            if any(r.get("speech") for r in per):
                files["speech"] = _speech_chart(plt, rep, per, out_dir)
    return files
