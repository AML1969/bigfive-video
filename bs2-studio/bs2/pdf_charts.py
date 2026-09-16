"""PDF charts for BS 2.0 (matplotlib PNGs placed on white A4 pages by pdf_report.py). Colours come from
palette.py (PDF_* dictionaries), never from the web palettes."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .charts import EMO_COLORS, EMO_RU, TRAIT_COLORS, VOICE_COLORS, VOICE_RU, _segments
from .norms import RU_TITLES, TRAIT_KEYS


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

