"""Charts for the BS 1.0 web page: the Big Five timeline (same chart as BS 2.0) and inline key frames.

The Plotly figure is embedded as a responsive <iframe srcdoc> (gr.Plot draws plotly at a fixed ~700 px). The iframe is
transparent, so its text sits on the Gradio block background. A small script inside the iframe
  * draws only once the iframe has a width (ResizeObserver): a chart rendered while its block is hidden or not laid
    out yet is drawn when it appears, instead of being stuck at Plotly's default 700 px;
  * reads the page theme (Gradio's `dark` class on an ancestor of the iframe, else prefers-color-scheme) and colours
    text, legend, axes, grid and hover label from palette.CHART before the first draw and again on every theme toggle;
  * switches to a taller figure with a smaller legend when the block is narrow and resizes the iframe to match.
Colours: palette.py (checked by palette.check()).
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import List, Tuple

from .norms import TRAIT_KEYS
from .palette import CHART, FONT_FAMILY, OUTLINE, TRAIT_COLORS, TRAIT_SYMBOLS  # noqa: F401  (TRAIT_COLORS re-exported)
from .report import seg_label

RU_TITLES = {"openness": "Открытость опыту", "conscientiousness": "Добросовестность", "extraversion": "Экстраверсия",
             "agreeableness": "Доброжелательность", "emotional_stability": "Эмоциональная стабильность",
             "interview": "Впечатление «собеседование»"}
# names in the hover label of a narrow chart (phone): the label has to fit on one side of the pointer
RU_SHORT = {"openness": "Открытость опыту", "conscientiousness": "Добросовестность", "extraversion": "Экстраверсия",
            "agreeableness": "Доброжелательность", "emotional_stability": "Эмоц. стабильность",
            "interview": "«Собеседование»"}

CHART_HEIGHT = 420          # wide block
NARROW_EXTRA = 180          # extra height when the block is narrower than NARROW_PX (the legend wraps into more rows)
NARROW_PX = 560
TICK_STEPS = (10, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1800, 3600)   # seconds between time ticks
TICK_PX = 48                # at least this many pixels per time tick (the script thins the ticks on narrow blocks)
FONT_CSS = "https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap"

_PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="__FONT_CSS__">
<script src="__PLOTLY_JS__" charset="utf-8"></script>
<style>html,body{margin:0;padding:0;background:transparent;overflow:hidden}
#bs-chart{width:100%;height:__H__px}
.bs-fail{font:14px/1.4 __FONT__;padding:8px 0}</style>
</head><body><div id="bs-chart"></div>
<script>
(function () {
  var FIG = __FIG__;
  var THEME = __THEME__;
  var H = __H__, H_NARROW = __HN__, NARROW = __NARROW__, EXTRA = __EXTRA__;
  var gd = document.getElementById('bs-chart');
  var st = {drawn: false, dark: null, narrow: null, w: 0};

  function isDark() {
    // Gradio puts the `dark` class on the page <body> (or on the app's parent when embedded): that is the truth.
    // The OS preference is only a fallback when the parent page cannot be read.
    try {
      var el = window.frameElement;
      while (el) { if (el.classList && el.classList.contains('dark')) return true; el = el.parentElement; }
      var d = window.parent.document;
      return d.body.classList.contains('dark') || d.documentElement.classList.contains('dark');
    } catch (e) {}
    return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  }
  function colours(dark) {
    var c = dark ? THEME.dark : THEME.light;
    var u = {'font.color': c.text, 'legend.font.color': c.text, 'legend.bgcolor': 'rgba(0,0,0,0)',
             'hoverlabel.bgcolor': c.hover_bg, 'hoverlabel.font.color': c.text, 'hoverlabel.bordercolor': c.axis};
    Object.keys(FIG.layout).forEach(function (k) {
      if (/^[xy]axis\d*$/.test(k)) {
        u[k + '.gridcolor'] = c.grid; u[k + '.linecolor'] = c.axis; u[k + '.zerolinecolor'] = c.grid;
        u[k + '.tickfont.color'] = c.sub; u[k + '.title.font.color'] = c.text;
      }
    });
    (FIG.layout.annotations || []).forEach(function (a, i) { u['annotations[' + i + '].font.color'] = c.text; });
    return u;
  }
  function setPath(obj, path, val) {
    var parts = path.replace(/\[(\d+)\]/g, '.$1').split('.'), o = obj;
    for (var i = 0; i < parts.length - 1; i++) {
      if (o[parts[i]] == null || typeof o[parts[i]] !== 'object') o[parts[i]] = /^\d+$/.test(parts[i + 1]) ? [] : {};
      o = o[parts[i]];
    }
    o[parts[parts.length - 1]] = val;
  }
  function fitFrame(h) {
    gd.style.height = h + 'px';
    try { if (window.frameElement) window.frameElement.style.height = (h + EXTRA) + 'px'; } catch (e) {}
  }
  function timeTicks(w) {
    // m:ss ticks thinned to the plot width, so a narrow chart keeps horizontal, non-overlapping labels
    var ta = FIG.layout.meta && FIG.layout.meta.time_axis;
    if (!ta) return null;
    var m = FIG.layout.margin || {}, pw = w - (m.l || 0) - (m.r || 0);
    var most = Math.max(3, Math.min(10, Math.floor(pw / ta.tick_px)));
    var step = ta.steps[ta.steps.length - 1];
    for (var i = 0; i < ta.steps.length; i++) { if (ta.dur / ta.steps[i] <= most) { step = ta.steps[i]; break; } }
    var vals = [], text = [];
    for (var v = 0; v <= ta.dur; v += step) { vals.push(v); text.push(Math.floor(v / 60) + ':' + ('0' + (v % 60)).slice(-2)); }
    return {step: step, vals: vals, text: text};
  }
  function hoverFor(narrow) {
    // wide: one label for the whole segment (x unified); narrow: that label is wider than the chart and gets cut,
    // so each point gets its own short two-line label instead
    FIG.layout.hovermode = narrow ? 'closest' : 'x unified';
    FIG.data.forEach(function (tr) {
      var m = tr.meta;
      if (!m || typeof m !== 'object' || !m.hover_wide) return;
      if (narrow && !m.hover_narrow) {        // the segment row of the wide label: no hover of its own
        delete tr.hovertemplate; tr.hoverinfo = 'skip';   // (plotly ignores hoverinfo while a hovertemplate is set)
      } else {
        tr.hovertemplate = narrow ? m.hover_narrow : m.hover_wide; delete tr.hoverinfo;
      }
    });
  }
  function draw(dark, narrow, w) {
    // a full redraw, not relayout: relaying out `height` makes plotly.js pin the current width (autosize off)
    var u = colours(dark), h = narrow ? H_NARROW : H, t = timeTicks(w);
    u['legend.font.size'] = narrow ? 12 : 13;
    u['hoverlabel.font.size'] = narrow ? 12 : 13;
    u['height'] = h;
    if (t) { u['xaxis.tickvals'] = t.vals; u['xaxis.ticktext'] = t.text; st.step = t.step; }
    Object.keys(u).forEach(function (p) { setPath(FIG.layout, p, u[p]); });
    delete FIG.layout.width;
    FIG.layout.autosize = true;
    hoverFor(narrow);
    st.drawn = true; st.dark = dark; st.narrow = narrow; st.w = w;
    fitFrame(h);
    window.Plotly.newPlot(gd, FIG.data, FIG.layout, FIG.config);
  }
  function render() {
    var w = document.documentElement.clientWidth || 0;
    if (w < 40) return;                       // hidden or not laid out yet: the ResizeObserver calls again
    // 20 px of hysteresis: the taller narrow figure may add a page scrollbar that narrows the block again
    var dark = isDark(), narrow = st.narrow ? w < NARROW + 20 : w < NARROW;
    if (!window.Plotly) {                     // plotly.js did not load (no connection to the CDN)
      gd.className = 'bs-fail'; gd.style.color = (dark ? THEME.dark : THEME.light).text;
      gd.textContent = 'График не загрузился: нет связи с cdn.plot.ly. Оценки по отрезкам есть в таблице выше.';
      fitFrame(48); gd.style.height = 'auto';
      return;
    }
    if (!st.drawn || narrow !== st.narrow) {  // first draw, or the block crossed the narrow threshold
      draw(dark, narrow, w);
      return;
    }
    if (dark !== st.dark) {                   // theme toggled: colours only
      st.dark = dark;
      window.Plotly.relayout(gd, colours(dark));
    }
    if (Math.abs(w - st.w) >= 1) {
      st.w = w;
      var t = timeTicks(w);
      if (t && t.step !== st.step) {         // the block got wider or narrower: more or fewer time ticks
        st.step = t.step;
        var q = window.Plotly.relayout(gd, {'xaxis.tickvals': t.vals, 'xaxis.ticktext': t.text});
        if (q && q.catch) q.catch(function () {});
      }
      var r = window.Plotly.Plots && window.Plotly.Plots.resize ? window.Plotly.Plots.resize(gd)
                                                                : window.Plotly.relayout(gd, {autosize: true});
      if (r && r.catch) r.catch(function () {});
    }
  }
  function start() {
    render();
    try { new ResizeObserver(function () { render(); }).observe(document.documentElement); } catch (e) {}
    window.addEventListener('resize', render);
    // theme toggles: the `dark` class may change on any ancestor of the iframe (<body>, the app's parent, <html>)
    try {
      var mo = new MutationObserver(function () { render(); }), el = window.frameElement;
      while (el) { mo.observe(el, {attributes: true, attributeFilter: ['class']}); el = el.parentElement; }
    } catch (e) {}
    try { window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render); } catch (e) {}
  }
  // the page font (Source Sans Pro) is measured by Plotly when it lays out the legend: load it first, but never wait long
  var ready = Promise.resolve();
  try {
    if (document.fonts && document.fonts.load) {
      ready = Promise.race([document.fonts.load('13px "Source Sans Pro"'),
                            new Promise(function (res) { setTimeout(res, 1500); })]);
    }
  } catch (e) {}
  ready.then(start, start);
})();
</script>
</body></html>
"""


def plot_html(fig, extra_height: int = 8, title: str = "График") -> str:
    """Plotly figure -> <iframe srcdoc> that draws once it has a width and follows the Gradio theme (module doc)."""
    import plotly.io as pio
    from plotly.offline import get_plotlyjs_version
    fig.update_layout(title=None, autosize=True)
    height = int(fig.layout.height or CHART_HEIGHT)
    spec = json.loads(pio.to_json(fig, validate=False))
    # no modebar: over a transparent background its icons are 1.6-2.7:1 and a PNG export loses the page background
    spec["config"] = {"responsive": True, "displaylogo": False, "displayModeBar": False}
    fig_js = json.dumps(spec, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = (_PAGE.replace("__PLOTLY_JS__", f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js")
            .replace("__FONT_CSS__", FONT_CSS).replace("__FONT__", FONT_FAMILY)
            .replace("__THEME__", json.dumps(CHART)).replace("__HN__", str(height + NARROW_EXTRA))
            .replace("__H__", str(height)).replace("__NARROW__", str(NARROW_PX)).replace("__EXTRA__", str(extra_height))
            .replace("__FIG__", fig_js))
    srcdoc = page.replace("&", "&amp;").replace('"', "&quot;")
    return (f"<iframe title='{title}' style='width:100%;height:{height + extra_height}px;border:0;display:block' "
            f"scrolling='no' srcdoc=\"{srcdoc}\"></iframe>")


def _time_ticks(dur: float, most: int = 10):
    """Ticks in m:ss like the table and the hover («4:40–5:00»); at most `most` steps for any length. The iframe
    script recomputes them with the same steps for the real plot width (timeTicks)."""
    step = next((s for s in TICK_STEPS if dur / s <= most), TICK_STEPS[-1])
    vals = list(range(0, int(dur) + 1, step))
    return vals, [f"{v // 60}:{v % 60:02d}" for v in vals]


def _interview_drawn(segs: List[dict]) -> bool:
    return bool(segs) and all("interview" in t["scores"] for t in segs)


def fig_traits_timeline(rep: dict):
    import plotly.graph_objects as go
    tl = list(rep.get("timeline") or [])
    segs = [t for t in tl if t.get("scores")]
    skipped = [t for t in tl if not t.get("scores")]
    fig = go.Figure()
    # every segment keeps its place on the x axis; a skipped one has no score, so the lines break there
    x = [(t["start"] + t["end"]) / 2 for t in tl]
    labels = [seg_label(t["start"], t["end"]) for t in tl]
    dur = max(float(rep.get("duration_sec") or 0), float(tl[-1]["end"]) if tl else 0.0) or 1.0
    keys = list(TRAIT_KEYS) + (["interview"] if _interview_drawn(segs) else [])
    if skipped:     # grey hatch under the lines, one legend entry (listed after the traits)
        xs, ys = [], []
        for s in skipped:
            xs += [s["start"], s["end"], s["end"], s["start"], s["start"], None]
            ys += [0, 0, 1, 1, 0, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=0), fill="toself", fillcolor="rgba(0,0,0,0)",
                                 fillpattern=dict(shape="/", fgcolor=OUTLINE, size=8, solidity=0.3),
                                 name="пропущено: нет лица или речи", hoverinfo="skip", legendrank=2000))
    # invisible helper: the first row of the hover label names the segment, so the trait rows carry only the value
    # (the iframe script switches the hover to per-point labels on a narrow block; `meta` carries both templates)
    seg_rows = [f"отрезок {lab}" + ("" if t.get("scores") else " пропущен: нет лица или речи") for t, lab in zip(tl, labels)]
    fig.add_trace(go.Scatter(x=x, y=[0.5] * len(x), customdata=seg_rows, mode="markers", marker=dict(opacity=0),
                             showlegend=False, hovertemplate="%{customdata}<extra></extra>",
                             meta=dict(hover_wide="%{customdata}<extra></extra>")))
    for k in keys:
        wide = "%{y:.2f}<extra>" + RU_TITLES[k] + "</extra>"
        fig.add_trace(go.Scatter(x=x, y=[(t.get("scores") or {}).get(k) for t in tl], mode="lines+markers",
                                 name=RU_TITLES[k], connectgaps=False, customdata=labels,
                                 line=dict(color=TRAIT_COLORS[k], width=2.5, dash="dot" if k == "interview" else "solid"),
                                 marker=dict(size=8, symbol=TRAIT_SYMBOLS[k], line=dict(width=0)),
                                 hovertemplate=wide,
                                 meta=dict(hover_wide=wide,
                                           hover_narrow=f"<b>{RU_SHORT[k]}</b><br>%{{customdata}}: %{{y:.2f}}<extra></extra>")))
    rep_i = rep.get("representative_segment")
    t = next((t for t in segs if t["segment"] == rep_i), None) if rep_i else None
    if t:           # dashed frame (does not tint the lines) and a label above the plot pointing at it
        fig.add_vrect(x0=t["start"], x1=t["end"], fillcolor="rgba(0,0,0,0)", layer="below",
                      line=dict(color=OUTLINE, width=1.5, dash="dash"))
        mid = (t["start"] + t["end"]) / 2
        # the ▼ stays centred over the frame; the words run towards the wider side of the plot
        right_side = mid > dur / 2
        fig.add_annotation(x=mid, xref="x", y=1, yref="paper", yanchor="bottom", showarrow=False,
                           text="отрезок объяснений ▼" if right_side else "▼ отрезок объяснений",
                           xanchor="right" if right_side else "left", xshift=7 if right_side else -7,
                           font=dict(size=12))
    ticks, ticktext = _time_ticks(dur)
    fig.update_layout(height=CHART_HEIGHT, autosize=True, margin=dict(l=60, r=20, t=40, b=70), hovermode="x unified",
                      legend=dict(orientation="h", yref="container", y=0, yanchor="bottom", x=0, font=dict(size=13)),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=13, color=CHART["light"]["text"], family=FONT_FAMILY),
                      hoverlabel=dict(font=dict(family=FONT_FAMILY, size=13)),
                      meta=dict(time_axis=dict(dur=dur, steps=list(TICK_STEPS), tick_px=TICK_PX)))
    fig.update_xaxes(title_text="время ролика, мин:с", range=[0, dur], tickvals=ticks, ticktext=ticktext, showgrid=True,
                     unifiedhovertitle=dict(text="Big Five на отрезке"))
    fig.update_yaxes(title_text="оценка черты, 0…1", range=[0, 1], tickformat=".1f", showgrid=True)
    return fig


def traits_timeline_html(rep: dict) -> str:
    tl = rep.get("timeline") or []
    segs = [t for t in tl if t.get("scores")]
    if not tl:
        return ("<p style='opacity:.8'>Ролик до 30 секунд анализируется целиком, одним отрезком, поэтому графика по ходу "
                "ролика нет: смотрите оценки выше.</p>")
    if len(segs) < 2:
        return ("<p style='opacity:.8'>Для графика нужны оценки хотя бы двух отрезков, а лицо и речь нашлись только в одном. "
                "Смотрите оценки выше.</p>")
    rep_i = rep.get("representative_segment")
    parts = ["Каждая линия — одна черта, у каждой свой значок; значок на линии — оценка отрезка ролика (обычно 20 секунд)."]
    if _interview_drawn(segs):
        parts.append("Точечная линия — впечатление «собеседование».")
    marks = []
    if rep_i and any(t["segment"] == rep_i for t in segs):
        marks.append("серая пунктирная рамка — отрезок, по которому построены объяснения")
    if len(segs) < len(tl):
        marks.append("серая штриховка — пропущенные отрезки (нет лица или речи)")
    if marks:
        parts.append("; ".join(marks)[0].upper() + "; ".join(marks)[1:] + ".")
    parts.append("Наведите курсор, чтобы увидеть значения.")
    return (plot_html(fig_traits_timeline(rep), title="График: Big Five по ходу ролика")
            + f"<p style='font-size:13px;opacity:.85;margin:4px 0 0'>{' '.join(parts)}</p>")


def frames_html(rep: dict, frames: List[Tuple[str, str]], max_side: int = 1280) -> str:
    """Key frames as inline JPEGs with the moment of the video as caption; a click (or Enter) enlarges a frame.
    Frames are embedded at up to 1280 px so the enlarged view stays sharp; the grid shows them downscaled."""
    from PIL import Image
    paths = [p for p, _ in frames if Path(p).exists()]
    if not paths:
        return "<p style='opacity:.8'>Ключевые кадры не построены (объяснения отключены или лицо не найдено).</p>"
    seg = next((t for t in (rep.get("timeline") or []) if t.get("segment") == rep.get("representative_segment")), None)
    fps = float((rep.get("media") or {}).get("fps") or 0) or None
    # frame numbers count from the start of the explained segment, or of the whole video when it has no timeline
    start = float(seg["start"]) if seg else 0.0
    items = []
    for p in paths:
        try:
            with Image.open(p) as im0:
                im = im0.convert("RGB")
            im.thumbnail((max_side, max_side))
            w, h = im.size
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:  # noqa: BLE001
            continue
        m = re.search(r"_frame(\d+)", Path(p).stem)
        items.append((b64, w, h, start + int(m.group(1)) / fps if (m and fps) else None, m.group(1) if m else None))
    if not items:
        return "<p style='opacity:.8'>Ключевые кадры не удалось прочитать.</p>"
    timed = all(t is not None for _, _, _, t, _ in items)
    secs = [int(t) for _, _, _, t, _ in items] if timed else []
    tenths = len(set(secs)) < len(secs)          # two frames within one second: «0:42,1» and «0:42,8»
    cells = []
    for b64, w, h, t, n in items:
        if timed:
            caption = f"{int(t) // 60}:{int(t) % 60:02d}" + (f",{int(t * 10) % 10}" if tenths else "")
        else:
            caption = (f"кадр №{n}" + (" отрезка" if seg else "")) if n else "кадр"
        cells.append(
            f"<figure style='margin:0'><img class='bs-kf' src='data:image/jpeg;base64,{b64}' alt='Ключевой кадр, {caption}' "
            "title='Щёлкните, чтобы увеличить' tabindex='0' role='button' onclick=\"this.classList.toggle('bs-big')\" "
            "onkeydown=\"if(event.key==='Enter'||event.key===' '){this.classList.toggle('bs-big');event.preventDefault()}"
            "else if(event.key==='Escape'){this.classList.remove('bs-big')}\" "
            f"style='display:block;width:100%;height:auto;aspect-ratio:{w}/{h};max-height:360px;object-fit:contain;"
            "cursor:zoom-in;border-radius:8px;background:rgba(128,128,128,.12)'>"
            f"<figcaption style='text-align:center;font-size:14px;margin-top:4px'>{caption}</figcaption></figure>")
    where = f" (отрезок {seg_label(seg['start'], seg['end'])})" if seg else ""
    what = "подпись — момент ролика (мин:с)" if timed else "подпись — номер кадра"
    return ("<style>.bs-big{position:fixed!important;inset:4vh 4vw;width:92vw!important;height:92vh!important;"
            "max-height:none!important;aspect-ratio:auto!important;z-index:9999;background:rgba(0,0,0,.9)!important;box-shadow:0 0 0 100vmax rgba(0,0,0,.85);"
            "cursor:zoom-out!important;object-fit:contain;border-radius:0!important}"
            f".bs-kf:focus-visible{{outline:2px solid {OUTLINE};outline-offset:2px}}</style>"
            f"<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px'>{''.join(cells)}</div>"
            f"<p style='font-size:13px;opacity:.85;margin-top:8px'>Кадры, сильнее всего повлиявшие на оценку своей модели{where}; "
            f"рамкой отмечено найденное лицо, {what}. Щелчок или Enter увеличивает кадр, повторный щелчок или Esc закрывает.</p>")
