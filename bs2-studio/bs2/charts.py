"""Charts for the BS 2.0 interface (plotly, interactive) and for the PDF (matplotlib PNG). All functions take the
result.json dict; they never compute anything new."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .norms import RU_TITLES, TRAIT_KEYS
from .report import seg_label

TRAIT_COLORS = {"openness": "#4c8bf5", "conscientiousness": "#2e8b57", "extraversion": "#e8731a",
                "agreeableness": "#9b59b6", "emotional_stability": "#c0392b", "interview": "#8a6d3b"}
EMO_RU = {"joy": "радость", "surprise": "удивление", "neutral": "нейтрально", "sadness": "грусть", "fear": "страх",
          "anger": "злость", "disgust": "отвращение",
          "happy": "радость", "sad": "грусть", "angry": "злость"}
EMO_COLORS = {"joy": "#f1c40f", "happy": "#f1c40f", "surprise": "#e67e22", "neutral": "#95a5a6", "sadness": "#3498db",
              "sad": "#3498db", "fear": "#8e44ad", "anger": "#c0392b", "angry": "#c0392b", "disgust": "#27ae60"}
VOICE_RU = {"arousal": "возбуждение", "dominance": "уверенность", "valence": "позитивность"}
VOICE_COLORS = {"arousal": "#e8731a", "dominance": "#2e8b57", "valence": "#4c8bf5"}
FONT_COLOR = "#a9adb3"          # readable on both the light and the dark Gradio theme


def _segments(rep: dict) -> List[dict]:
    return [t for t in (rep.get("timeline") or []) if t.get("scores")]


def _x(segs: List[dict]):
    return [(t["start"] + t["end"]) / 2 for t in segs], [seg_label(t["start"], t["end"]) for t in segs]


def _layout(fig, title: str, y_title: str = "", height: int = 340, y_range=(0, 1)):
    fig.update_layout(title=dict(text=title, x=0.01, y=0.98, font=dict(size=15)), height=height, autosize=True,
                      margin=dict(l=50, r=20, t=48, b=80), legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
                      hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=12, color=FONT_COLOR))
    fig.update_xaxes(title_text="время, с", showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    if y_range:
        fig.update_yaxes(range=list(y_range))
    return fig


# ---------------------------------------------------------------- plotly (web)
def fig_traits_timeline(rep: dict):
    import plotly.graph_objects as go
    segs = _segments(rep)
    fig = go.Figure()
    if not segs:
        return _layout(fig, "Big Five по сегментам (короткий ролик — один отрезок)")
    x, labels = _x(segs)
    keys = list(TRAIT_KEYS) + (["interview"] if all("interview" in t["scores"] for t in segs) else [])
    for k in keys:
        fig.add_trace(go.Scatter(x=x, y=[t["scores"].get(k) for t in segs], mode="lines+markers", name=RU_TITLES[k],
                                 line=dict(color=TRAIT_COLORS[k], width=2, dash="dot" if k == "interview" else "solid"),
                                 customdata=labels, hovertemplate="%{customdata}: %{y:.2f}<extra>" + RU_TITLES[k] + "</extra>"))
    rep_i = rep.get("representative_segment")
    if rep_i:
        t = next((t for t in segs if t["segment"] == rep_i), None)
        if t:
            fig.add_vrect(x0=t["start"], x1=t["end"], fillcolor="rgba(120,120,120,0.12)", line_width=0,
                          annotation_text="сегмент объяснений", annotation_position="top left")
    for t in rep.get("timeline") or []:
        if not t.get("scores"):
            fig.add_vrect(x0=t["start"], x1=t["end"], fillcolor="rgba(200,60,60,0.10)", line_width=0)
    return _layout(fig, "Big Five по ходу ролика", "оценка 0…1")


def fig_radar(rep: dict):
    import plotly.graph_objects as go
    names = [RU_TITLES[k] for k in TRAIT_KEYS]
    main = [rep["traits"][k]["score"] for k in TRAIT_KEYS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=main + main[:1], theta=names + names[:1], fill="toself", name="Основная оценка",
                                  line=dict(color="#4c8bf5")))
    var = rep.get("variant_scores") or {}
    primary = (rep.get("model") or {}).get("primary")
    for m, v in var.items():
        if m != primary and primary:
            vals = [v[k] for k in TRAIT_KEYS]
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=names + names[:1], name="Второе мнение (своя модель, шкала FIV2)",
                                          line=dict(color="#9bb7e8", dash="dot")))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=True, tickfont=dict(size=10))),
                      height=380, autosize=True, margin=dict(l=30, r=30, t=40, b=60), legend=dict(orientation="h", y=-0.12),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_COLOR),
                      title=dict(text="Профиль Big Five", x=0.01, y=0.98, font=dict(size=15)))
    return fig


def _stacked(fig, x, labels, per_rows: List[dict], order: List[str], row=None, col=None, showlegend=True, group="e"):
    import plotly.graph_objects as go
    for k in order:
        ys = [(r or {}).get(k, 0.0) for r in per_rows]
        kw = dict(row=row, col=col) if row else {}
        fig.add_trace(go.Scatter(x=x, y=ys, mode="lines", stackgroup=group, name=EMO_RU.get(k, k),
                                 line=dict(width=0.5, color=EMO_COLORS.get(k, "#888")), showlegend=showlegend,
                                 legendgroup=k, customdata=labels,
                                 hovertemplate="%{customdata}: %{y:.0%}<extra>" + EMO_RU.get(k, k) + "</extra>"), **kw)


def fig_emotions_timeline(rep: dict):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from .analyses.emotions_text import EMOTION_ORDER
    from .analyses.face_expr import EXPR_ORDER
    per = (rep.get("analyses") or {}).get("per_segment") or []
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Эмоции по речи (что говорит)", "Выражение лица (как выглядит)"))
    if not per:
        return _layout(fig, "Эмоции по ходу ролика", height=520)
    x = [(r["start"] + r["end"]) / 2 for r in per]
    labels = [seg_label(r["start"], r["end"]) for r in per]
    _stacked(fig, x, labels, [r.get("emotions_text") for r in per], EMOTION_ORDER, row=1, col=1, group="t")
    _stacked(fig, x, labels, [(r.get("face") or {}).get("expressions") for r in per], EXPR_ORDER, row=2, col=1,
             showlegend=False, group="f")
    fig.update_layout(height=600, autosize=True, margin=dict(l=50, r=20, t=70, b=80), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0), font=dict(color=FONT_COLOR),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      title=dict(text="Эмоции по ходу ролика (доли, сумма = 100%)", x=0.01, y=0.99, font=dict(size=15)))
    fig.update_yaxes(range=[0, 1], tickformat=".0%", gridcolor="rgba(128,128,128,0.2)")
    fig.update_xaxes(title_text="время, с", row=2, col=1)
    return fig


def fig_voice_timeline(rep: dict):
    import plotly.graph_objects as go
    per = (rep.get("analyses") or {}).get("per_segment") or []
    fig = go.Figure()
    rows = [r for r in per if r.get("voice")]
    if not rows:
        return _layout(fig, "Голос: возбуждение, уверенность, позитивность")
    x = [(r["start"] + r["end"]) / 2 for r in rows]
    labels = [seg_label(r["start"], r["end"]) for r in rows]
    for d, name in VOICE_RU.items():
        fig.add_trace(go.Scatter(x=x, y=[r["voice"].get(d) for r in rows], mode="lines+markers", name=name,
                                 line=dict(color=VOICE_COLORS[d], width=2), customdata=labels,
                                 hovertemplate="%{customdata}: %{y:.2f}<extra>" + name + "</extra>"))
    return _layout(fig, "Голос по ходу ролика (модель эмоций в речи, 0…1)", "0…1")


def fig_speech_timeline(rep: dict):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    per = (rep.get("analyses") or {}).get("per_segment") or []
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    rows = [r for r in per if r.get("speech")]
    if not rows:
        return _layout(fig, "Речь: темп и паузы", y_range=None)
    x = [(r["start"] + r["end"]) / 2 for r in rows]
    labels = [seg_label(r["start"], r["end"]) for r in rows]
    fig.add_trace(go.Bar(x=x, y=[r["speech"].get("words_per_min_speech") or 0 for r in rows], name="слов в минуту",
                         marker_color="#4c8bf5", opacity=0.75, customdata=labels,
                         hovertemplate="%{customdata}: %{y:.0f} слов/мин<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=[r["speech"].get("pause_share", 0) for r in rows], name="доля пауз", mode="lines+markers",
                             line=dict(color="#e8731a", width=2), customdata=labels,
                             hovertemplate="%{customdata}: паузы %{y:.0%}<extra></extra>"), secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=[r["speech"].get("fillers_per_100", 0) / 100 for r in rows], name="заполнители (на слово)",
                             mode="lines", line=dict(color="#95a5a6", width=1.5, dash="dot"), customdata=labels,
                             hovertemplate="%{customdata}: %{y:.1%} слов-заполнителей<extra></extra>"), secondary_y=True)
    fig.update_layout(title=dict(text="Речь по ходу ролика", x=0.01, y=0.98, font=dict(size=15)), height=340, autosize=True,
                      margin=dict(l=50, r=50, t=48, b=80), legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
                      hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0.35,
                      font=dict(color=FONT_COLOR))
    fig.update_yaxes(title_text="слов в минуту", secondary_y=False, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(title_text="доля", range=[0, 1], tickformat=".0%", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="время, с")
    return fig


def fig_emotion_bars(rep: dict):
    import plotly.graph_objects as go
    an = rep.get("analyses") or {}
    fig = go.Figure()
    text_mean = (an.get("emotions_text") or {}).get("mean") or {}
    face_mean = (an.get("face") or {}).get("mean") or {}
    order = ["joy", "surprise", "neutral", "sadness", "fear", "anger", "disgust"]
    face_map = {"joy": "happy", "sadness": "sad", "anger": "angry"}
    names = [EMO_RU[k] for k in order]
    if text_mean:
        fig.add_trace(go.Bar(x=names, y=[text_mean.get(k, 0) for k in order], name="по речи", marker_color="#4c8bf5"))
    if face_mean:
        fig.add_trace(go.Bar(x=names, y=[face_mean.get(face_map.get(k, k), 0) for k in order], name="по лицу", marker_color="#e8731a"))
    fig.update_layout(barmode="group", height=340, autosize=True, margin=dict(l=50, r=20, t=48, b=70),
                      title=dict(text="Средний профиль эмоций за ролик", x=0.01, y=0.98, font=dict(size=15)),
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_COLOR))
    fig.update_yaxes(range=[0, 1], tickformat=".0%", gridcolor="rgba(128,128,128,0.2)")
    return fig


# ---------------------------------------------------------------- matplotlib (PDF)
def save_pdf_charts(rep: dict, out_dir: str | Path) -> Dict[str, str]:
    """PNG files for the PDF: traits timeline, emotions (text + face), voice + speech. Returns {name: path}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .analyses.emotions_text import EMOTION_ORDER
    from .analyses.face_expr import EXPR_ORDER

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    segs = _segments(rep)
    per = (rep.get("analyses") or {}).get("per_segment") or []
    if segs:
        x = [(t["start"] + t["end"]) / 2 for t in segs]
        fig, ax = plt.subplots(figsize=(9, 3.2), dpi=150)
        keys = list(TRAIT_KEYS) + (["interview"] if all("interview" in t["scores"] for t in segs) else [])
        for k in keys:
            ax.plot(x, [t["scores"].get(k) for t in segs], marker="o", ms=3, lw=1.6, color=TRAIT_COLORS[k],
                    ls=":" if k == "interview" else "-", label=RU_TITLES[k])
        ax.set_ylim(0, 1); ax.set_xlabel("время, с"); ax.set_ylabel("оценка 0…1"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3, loc="lower left"); ax.set_title("Big Five по ходу ролика", fontsize=10, loc="left")
        fig.tight_layout(); p = out_dir / "chart_traits.png"; fig.savefig(p); plt.close(fig); files["traits"] = str(p)
    if per:
        x = [(r["start"] + r["end"]) / 2 for r in per]
        fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), dpi=150, sharex=True)
        for ax, key, order, title in ((axes[0], "emotions_text", EMOTION_ORDER, "Эмоции по речи"),
                                      (axes[1], "face", EXPR_ORDER, "Выражение лица")):
            rows = [(r.get(key) or {}).get("expressions") if key == "face" else r.get(key) for r in per]
            ys = [[(rr or {}).get(k, 0.0) for rr in rows] for k in order]
            ax.stackplot(x, ys, labels=[EMO_RU.get(k, k) for k in order], colors=[EMO_COLORS.get(k, "#888") for k in order], alpha=0.9)
            ax.set_ylim(0, 1); ax.set_title(title, fontsize=10, loc="left"); ax.grid(alpha=0.2)
        axes[0].legend(fontsize=7, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.35)); axes[1].set_xlabel("время, с")
        fig.tight_layout(); p = out_dir / "chart_emotions.png"; fig.savefig(p); plt.close(fig); files["emotions"] = str(p)
        rows_v = [r for r in per if r.get("voice")]
        rows_s = [r for r in per if r.get("speech")]
        if rows_v or rows_s:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), dpi=150)
            if rows_v:
                xv = [(r["start"] + r["end"]) / 2 for r in rows_v]
                for d, name in VOICE_RU.items():
                    axes[0].plot(xv, [r["voice"].get(d) for r in rows_v], marker="o", ms=3, lw=1.6, color=VOICE_COLORS[d], label=name)
                axes[0].set_ylim(0, 1); axes[0].set_title("Голос (0…1)", fontsize=10, loc="left"); axes[0].grid(alpha=0.3)
                axes[0].legend(fontsize=7); axes[0].set_xlabel("время, с")
            if rows_s:
                xs = [(r["start"] + r["end"]) / 2 for r in rows_s]
                w = [max(4.0, 0.8 * (r["end"] - r["start"])) for r in rows_s]
                axes[1].bar(xs, [r["speech"].get("words_per_min_speech") or 0 for r in rows_s], width=w, color="#4c8bf5", alpha=0.75, label="слов в минуту")
                ax2 = axes[1].twinx()
                ax2.plot(xs, [r["speech"].get("pause_share", 0) for r in rows_s], color="#e8731a", marker="o", ms=3, lw=1.6, label="доля пауз")
                ax2.set_ylim(0, 1); axes[1].set_title("Речь: темп и паузы", fontsize=10, loc="left"); axes[1].set_xlabel("время, с")
                axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
            fig.tight_layout(); p = out_dir / "chart_voice_speech.png"; fig.savefig(p); plt.close(fig); files["voice_speech"] = str(p)
    return files
