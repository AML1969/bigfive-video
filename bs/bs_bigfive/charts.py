"""Charts for the BS 1.0 web page: the Big Five timeline (same chart as BS 2.0) and inline key frames.

Plotly figures are embedded as a responsive <iframe srcdoc> (gr.Plot draws plotly at a fixed ~700 px). The iframe is
transparent, so its text sits on the Gradio block background: a small script inside the iframe reads the page theme
(Gradio's `dark` class, else prefers-color-scheme) and recolours text, legend, axes and grid for contrast.
"""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import List, Tuple

from .norms import TRAIT_KEYS
from .report import seg_label

RU_TITLES = {"openness": "Открытость опыту", "conscientiousness": "Добросовестность", "extraversion": "Экстраверсия",
             "agreeableness": "Доброжелательность", "emotional_stability": "Эмоциональная стабильность",
             "interview": "Впечатление «собеседование»"}
# series colours chosen to stay >= 3:1 against both the dark block (#1f2937) and white
TRAIT_COLORS = {"openness": "#3b82f6", "conscientiousness": "#16a34a", "extraversion": "#ea580c",
                "agreeableness": "#a855f7", "emotional_stability": "#e11d48", "interview": "#b7791f"}

THEME_SCRIPT = """
<script>
(function () {
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
  function apply() {
    if (!window.Plotly) return;
    var dark = isDark();
    var txt = dark ? '#e5e7eb' : '#111827', sub = dark ? '#cbd5e1' : '#374151';
    var grid = dark ? 'rgba(229,231,235,0.16)' : 'rgba(17,24,39,0.12)', axis = dark ? 'rgba(229,231,235,0.55)' : 'rgba(17,24,39,0.55)';
    document.querySelectorAll('.plotly-graph-div').forEach(function (gd) {
      if (!gd.layout) return;
      var u = {'font.color': txt, 'legend.font.color': txt, 'legend.bgcolor': 'rgba(0,0,0,0)',
               'hoverlabel.bgcolor': dark ? '#111827' : '#ffffff', 'hoverlabel.font.color': txt,
               'hoverlabel.bordercolor': axis};
      Object.keys(gd.layout).forEach(function (k) {
        if (/^[xy]axis\\d*$/.test(k)) {
          u[k + '.gridcolor'] = grid; u[k + '.linecolor'] = axis; u[k + '.zerolinecolor'] = grid;
          u[k + '.tickfont.color'] = sub; u[k + '.title.font.color'] = txt;
        }
        if (/^polar\\d*$/.test(k)) {
          ['radialaxis', 'angularaxis'].forEach(function (a) {
            u[k + '.' + a + '.gridcolor'] = grid; u[k + '.' + a + '.linecolor'] = axis; u[k + '.' + a + '.tickfont.color'] = sub;
          });
        }
      });
      (gd.layout.annotations || []).forEach(function (a, i) { u['annotations[' + i + '].font.color'] = txt; });
      window.Plotly.relayout(gd, u);
    });
  }
  window.addEventListener('load', function () { setTimeout(apply, 50); });
  try { new MutationObserver(apply).observe(window.parent.document.body, {attributes: true, attributeFilter: ['class']}); } catch (e) {}
  try { window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', apply); } catch (e) {}
})();
</script>
"""


def plot_html(fig, extra_height: int = 24) -> str:
    fig.update_layout(title=None, autosize=True)
    height = int((fig.layout.height or 360) + extra_height)
    html = fig.to_html(include_plotlyjs="cdn", full_html=True, config={"responsive": True, "displaylogo": False})
    html = html.replace("<body>", "<body style='margin:0;background:transparent'>", 1).replace("</body>", THEME_SCRIPT + "</body>", 1)
    srcdoc = html.replace("&", "&amp;").replace('"', "&quot;")
    return f"<iframe style='width:100%;height:{height}px;border:0;display:block' scrolling='no' srcdoc=\"{srcdoc}\"></iframe>"


def fig_traits_timeline(rep: dict):
    import plotly.graph_objects as go
    segs = [t for t in (rep.get("timeline") or []) if t.get("scores")]
    fig = go.Figure()
    x = [(t["start"] + t["end"]) / 2 for t in segs]
    labels = [seg_label(t["start"], t["end"]) for t in segs]
    keys = list(TRAIT_KEYS) + (["interview"] if segs and all("interview" in t["scores"] for t in segs) else [])
    for k in keys:
        fig.add_trace(go.Scatter(x=x, y=[t["scores"].get(k) for t in segs], mode="lines+markers", name=RU_TITLES[k],
                                 line=dict(color=TRAIT_COLORS[k], width=2.5, dash="dot" if k == "interview" else "solid"),
                                 marker=dict(size=6), customdata=labels,
                                 hovertemplate="%{customdata}: %{y:.2f}<extra>" + RU_TITLES[k] + "</extra>"))
    rep_i = rep.get("representative_segment")
    t = next((t for t in segs if t["segment"] == rep_i), None) if rep_i else None
    if t:
        fig.add_vrect(x0=t["start"], x1=t["end"], fillcolor="rgba(128,128,128,0.18)", line_width=0,
                      annotation_text="отрезок объяснений", annotation_position="top left")
    for s in rep.get("timeline") or []:
        if not s.get("scores"):
            fig.add_vrect(x0=s["start"], x1=s["end"], fillcolor="rgba(225,29,72,0.12)", line_width=0)
    fig.update_layout(height=380, autosize=True, margin=dict(l=60, r=20, t=24, b=100), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, font=dict(size=13)),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13, color="#374151"))
    fig.update_xaxes(title_text="время ролика, с", showgrid=True)
    fig.update_yaxes(title_text="оценка черты, 0…1", range=[0, 1], showgrid=True)
    return fig


def traits_timeline_html(rep: dict) -> str:
    segs = [t for t in (rep.get("timeline") or []) if t.get("scores")]
    if len(segs) < 2:
        return ("<p style='opacity:.8'>Ролик короче 30 секунд анализируется одним отрезком, поэтому графика по ходу ролика "
                "нет: смотрите оценки выше.</p>")
    return plot_html(fig_traits_timeline(rep)) + (
        "<p style='font-size:13px;opacity:.85;margin:4px 0 0'>Каждая линия — одна черта; точка — оценка отрезка ролика "
        "(по 20 секунд). Пунктир — впечатление «собеседование». Серая полоса — отрезок, по которому построены объяснения. "
        "Наведите курсор, чтобы увидеть значения.</p>")


def frames_html(rep: dict, frames: List[Tuple[str, str]], max_side: int = 640) -> str:
    """Key frames as inline JPEGs with the moment of the video as caption; click enlarges."""
    from PIL import Image
    paths = [p for p, _ in frames if Path(p).exists()]
    if not paths:
        return "<p style='opacity:.8'>Ключевые кадры не построены (объяснения отключены или лицо не найдено).</p>"
    seg = next((t for t in (rep.get("timeline") or []) if t.get("segment") == rep.get("representative_segment")), None)
    fps = float((rep.get("media") or {}).get("fps") or 0) or None
    cells = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:  # noqa: BLE001
            continue
        m = re.search(r"_frame(\d+)", Path(p).stem)
        if m and seg and fps:
            t = seg["start"] + int(m.group(1)) / fps
            caption = f"{int(t) // 60}:{int(t) % 60:02d}"
        else:
            caption = f"кадр {m.group(1)}" if m else "кадр"
        cells.append(
            f"<figure style='margin:0'><img src='data:image/jpeg;base64,{b64}' alt='{caption}' title='Щёлкните, чтобы увеличить' "
            "onclick=\"this.classList.toggle('bs-big')\" style='width:100%;height:220px;object-fit:contain;cursor:zoom-in;"
            "border-radius:8px;background:rgba(128,128,128,.12)'>"
            f"<figcaption style='text-align:center;font-size:13px;margin-top:4px'>{caption}</figcaption></figure>")
    where = f" (отрезок {seg_label(seg['start'], seg['end'])})" if seg else ""
    return ("<style>.bs-big{position:fixed!important;inset:4vh 4vw;width:92vw!important;height:92vh!important;"
            "z-index:9999;background:rgba(0,0,0,.9)!important;cursor:zoom-out!important;object-fit:contain}</style>"
            f"<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px'>{''.join(cells)}</div>"
            f"<p style='font-size:13px;opacity:.85;margin-top:8px'>Кадры, сильнее всего повлиявшие на оценку своей модели{where}; "
            "рамкой отмечено найденное лицо, подпись — момент ролика. Щелчок увеличивает кадр, повторный щелчок закрывает.</p>")
