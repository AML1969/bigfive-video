"""BS 2.0 web UI (Gradio): Big Five + emotions, voice, face and speech analytics with interactive charts.

Run:  bs2 web [--port 7870]      (inside WSL; open http://localhost:7870 on Windows). Independent of bs 1.0 (:7860).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

from .charts import (EMO_RU, VOICE_RU, fig_emotion_bars, fig_emotions_timeline, fig_radar, fig_speech_timeline,
                     fig_traits_timeline, fig_voice_timeline)
from .narrative2 import analyses_sentences, key_facts
from .norms import TRAIT_KEYS
from .pipeline import Studio, run_analysis
from .report import DISCLAIMER_RU, INTERVIEW_DISCLAIMER_RU, fmt_secs, seg_label
from .webparts import MEMBER_TITLES, TRAIT_TITLES, _bar_html, _contrib_html, _members_html, _words_text

log = logging.getLogger("bs2.web")
CARD = ("display:inline-block;min-width:150px;margin:4px;padding:10px 14px;border:1px solid rgba(128,128,128,0.35);"
        "border-radius:10px;vertical-align:top")


def _facts_html(rep: dict) -> str:
    cards = "".join(f"<div style='{CARD}'><div style='font-size:12px;color:#888'>{lab}</div>"
                    f"<div style='font-size:20px;font-weight:600;margin:2px 0'>{val}</div>"
                    f"<div style='font-size:12px;color:#888'>{note}</div></div>" for lab, val, note in key_facts(rep))
    return f"<div>{cards}</div>" if cards else ""


def _segments_table(rep: dict) -> str:
    per = (rep.get("analyses") or {}).get("per_segment") or []
    if not per:
        return ""
    th = "padding:4px 8px;font-size:12px;text-align:center;vertical-align:bottom;line-height:1.15"
    head = "".join(f"<th style='{th}'>{h}</th>" for h in ("Отрезок", "Эмоция речи", "Выражение лица", "Возбуждение", "Уверенность",
                                                          "Позитивность", "Слов/мин", "Паузы"))
    rows = ""
    for r in per:
        te = r.get("emotions_text") or {}
        fa = (r.get("face") or {}).get("expressions") or {}
        vo = r.get("voice") or {}
        sp = r.get("speech") or {}
        e_t = max(te.items(), key=lambda kv: kv[1])[0] if te else "—"
        e_f = max(fa.items(), key=lambda kv: kv[1])[0] if fa else "—"
        cells = [seg_label(r["start"], r["end"]), EMO_RU.get(e_t, e_t), EMO_RU.get(e_f, e_f),
                 f"{vo.get('arousal', float('nan')):.2f}" if vo else "—", f"{vo.get('dominance', float('nan')):.2f}" if vo else "—",
                 f"{vo.get('valence', float('nan')):.2f}" if vo else "—",
                 f"{sp.get('words_per_min_speech') or 0:.0f}" if sp else "—", f"{sp.get('pause_share', 0):.0%}" if sp else "—"]
        rows += "<tr>" + "".join(f"<td style='padding:3px 8px;text-align:center;white-space:nowrap'>{c}</td>" for c in cells) + "</tr>"
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:13px'><tr>{head}</tr>{rows}</table></div>"
            "<div style='font-size:12px;color:#666;margin-top:6px'>Доминирующая эмоция по речи и по лицу, три измерения голоса "
            "(0…1), темп и доля пауз — по каждому отрезку.</div>")


def _speech_html(rep: dict) -> str:
    sp = (rep.get("analyses") or {}).get("speech") or {}
    if not sp:
        return ""
    items = [("Слов всего", sp.get("words")), ("Уникальных слов", sp.get("unique_words")),
             ("Темп (слов/мин речи)", sp.get("words_per_min_speech")), ("Темп по времени ролика", sp.get("words_per_min_wall")),
             ("Доля пауз", f"{sp.get('pause_share', 0):.0%}"), ("Длинных пауз (>2 с)", sp.get("long_pauses")),
             ("Слов-заполнителей", f"{sp.get('fillers', 0)} ({sp.get('fillers_per_100', 0):.1f} на 100 слов)"),
             ("Средняя фраза, слов", sp.get("mean_sentence")), ("Разнообразие словаря", sp.get("ttr"))]
    cards = "".join(f"<div style='{CARD}'><div style='font-size:12px;color:#888'>{k}</div>"
                    f"<div style='font-size:18px;font-weight:600'>{v if v is not None else '—'}</div></div>" for k, v in items)
    vocab = ", ".join(f"{w} ({n})" for w, n in sp.get("vocabulary", [])[:15])
    return (f"<div>{cards}</div><p style='font-size:14px'>{sp.get('description', '')}</p>"
            f"<p style='font-size:13px;color:#666'>Частые слова: {vocab}</p>")


def _face_html(rep: dict) -> str:
    fa = (rep.get("analyses") or {}).get("face") or {}
    if not fa:
        return ""
    m = fa.get("mean") or {}
    rows = "".join(f"<tr><td style='padding:2px 8px'>{EMO_RU.get(k, k)}</td><td style='padding:2px 8px;text-align:right'>{v:.0%}</td></tr>"
                   for k, v in sorted(m.items(), key=lambda kv: -kv[1]))
    extra = []
    if fa.get("head_motion") is not None:
        extra.append(f"движение головы {fa['head_motion']:.2f} (в долях ширины лица между кадрами)")
    if fa.get("face_share") is not None:
        extra.append(f"лицо найдено в {fa['face_share']:.0%} проанализированных кадров")
    return (f"<table style='border-collapse:collapse;font-size:13px'>{rows}</table>"
            f"<p style='font-size:13px;color:#666'>{'; '.join(extra)}. Модель выражений обучена на фотографиях FER-2013 и "
            "склонна видеть «грусть» и «страх» в спокойном лице; смотрите на изменения по ходу ролика, а не на абсолютные доли.</p>")


def export_pdf(job_dir: str | Path) -> str:
    from .charts import save_pdf_charts
    from .media import probe_media
    from .pdf_report import build_pdf
    job = Path(job_dir)
    rep = json.loads((job / "result.json").read_text(encoding="utf-8"))
    expl_path = job / "explain" / "explanation.json"
    expl = json.loads(expl_path.read_text(encoding="utf-8")) if expl_path.exists() else None
    rep["chart_files"] = save_pdf_charts(rep, job / "charts")
    frames = sorted(str(p) for p in (job / "explain").glob("key_*.jpg")) if (job / "explain").exists() else []
    media = rep.get("media")
    if not media or "error" in media:
        inp = next(job.glob("input.*"), None)
        media = probe_media(inp) if inp else None
    stem = re.sub(r"[^A-Za-z0-9А-Яа-яЁё._-]+", "_", Path(rep.get("original_file_name") or "video").stem)[:60]
    return build_pdf(rep, job / f"BS2_report_{stem}.pdf", explanation=expl, media=media, key_frames=frames)


def _status_html(frac: float, desc: str, done: bool = False, label: str | None = None) -> str:
    pct = max(2, min(100, int(frac * 100)))
    color = "#2e8b57" if done else "#e8731a"
    head = label or ("Готово" if done else "Идёт обработка")
    txt = desc if done else f"{pct}% · {desc}"
    return (f"<div style='margin:4px 0 8px'><div style='display:flex;justify-content:space-between;font-size:14px;margin-bottom:4px'>"
            f"<b>{head}</b><span>{txt}</span></div><div style='background:#e8e8e8;border-radius:6px;height:10px'>"
            f"<div style='width:{pct}%;background:{color};height:10px;border-radius:6px;transition:width .5s'></div></div></div>")


def build_app(studio: Studio, work_dir: Path):
    import gradio as gr
    from .longvideo import AnalysisCancelled

    N_REST = 22

    def render(rep: dict) -> tuple:
        job = Path(rep["job_dir"])
        expl_path = job / "explain" / "explanation.json"
        expl = json.loads(expl_path.read_text(encoding="utf-8")) if expl_path.exists() else None
        lang = (rep.get("model") or {}).get("lang", "ru")
        narrative = (rep.get("narrative") or "") + " " + analyses_sentences(rep)
        frames = [(p, f"кадр {Path(p).stem.split('_frame')[-1]}") for p in rep.get("key_frames", [])]
        members = rep.get("variant_scores") or {}
        primary = (rep.get("model") or {}).get("primary")
        member_txt = "\n".join(f"{MEMBER_TITLES.get(m, m)}{' — основная оценка' if m == primary else ''}: "
                               + ", ".join(f"{TRAIT_TITLES[k][:12]} {v[k]:.2f}" for k in TRAIT_KEYS) for m, v in members.items())
        member_txt += f"\nВремя обработки: {fmt_secs(rep['timings_sec'].get('total_wall', 0))}; модели: {rep.get('model', {}).get('corpus')}"
        return (fig_radar(rep), _bar_html(rep["traits"], rep.get("interview")) + _members_html(rep), _facts_html(rep),
                narrative.strip(), fig_traits_timeline(rep), fig_emotions_timeline(rep), fig_voice_timeline(rep),
                fig_speech_timeline(rep), fig_emotion_bars(rep), _segments_table(rep), _speech_html(rep),
                rep.get("transcript", ""), _face_html(rep), frames, _contrib_html(expl),
                _words_text(expl, rep, lang, expl_path) if expl else "",
                rep.get("behavior_description_ru") or rep.get("behavior_description", ""), member_txt,
                json.dumps(rep, ensure_ascii=False, indent=2), str(job / "result.json"), str(job), gr.update(interactive=True))

    def analyze(video, lang, explain):
        if not video:
            raise gr.Error("Загрузите видео")
        state = {"frac": 0.0, "desc": "запуск", "t0": time.time()}

        def cb(frac, desc=None, **kw):
            state["frac"] = float(frac)
            state["desc"] = str(desc if desc is not None else kw.get("desc", "")) or state["desc"]

        result: dict = {}

        def work():
            try:
                result["r"] = run_analysis(studio, work_dir, video, lang, explain, progress=cb)
            except BaseException as e:  # noqa: BLE001
                result["e"] = e

        th = threading.Thread(target=work, daemon=True)
        th.start()
        while th.is_alive():
            th.join(1.0)
            yield (_status_html(state["frac"], state["desc"]),) + (gr.update(),) * N_REST
        if "e" in result:
            e = result["e"]
            if isinstance(e, AnalysisCancelled):
                raise gr.Error("Обработка остановлена. Проверьте язык речи и запустите заново.")
            raise e
        rep = result["r"]
        yield (_status_html(1.0, f"обработано за {fmt_secs(time.time() - state['t0'])}", done=True),) + render(rep)

    def stop():
        studio.stop_event.set()
        return _status_html(0.0, "текущий сегмент дорабатывается, затем обработка прерывается (до ~20 с)", done=True,
                            label="Остановлено")

    def make_pdf(job_dir):
        if not job_dir:
            raise gr.Error("Сначала проанализируйте видео")
        try:
            return export_pdf(job_dir)
        except Exception as e:  # noqa: BLE001
            raise gr.Error(f"Не удалось собрать PDF: {e}")

    with gr.Blocks(title="BS 2.0 — Big Five, эмоции, голос, речь", theme=gr.themes.Soft()) as demo:
        job_state = gr.State("")
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("# BS 2.0 — анализ человека по видео\nBig Five (первое впечатление), эмоции по речи и по лицу, "
                            "характеристики голоса, манера речи, объяснения. Всё считается локально.")
            with gr.Column(scale=1, min_width=220):
                pdf_btn = gr.DownloadButton("Экспорт в PDF", variant="primary", interactive=False)
        status = gr.HTML(value="")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                video = gr.Video(label="Видео", sources=["upload"], height=300)
                lang = gr.Radio(choices=["ru", "en"], value="ru", label="Язык речи",
                                info="ru: основная оценка OCEAN-AI (веса MuPTA), своя модель — второе мнение; en: среднее двух систем")
                explain = gr.Checkbox(value=True, label="Объяснения (ключевые кадры, вклад модальностей, слова)")
                with gr.Row():
                    btn = gr.Button("Анализировать", variant="primary")
                    stop_btn = gr.Button("Остановить обработку", variant="stop")
            with gr.Column(scale=2, min_width=480):
                facts = gr.HTML(label="Ключевые факты")
                narrative = gr.Textbox(label="Пояснение простыми словами", lines=9, max_lines=12)
        with gr.Tabs():
            with gr.Tab("Обзор"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=360):
                        radar = gr.Plot(label="Профиль Big Five")
                    with gr.Column(scale=1, min_width=360):
                        bars = gr.HTML(label="Оценки")
            with gr.Tab("Таймлайн"):
                traits_plot = gr.Plot(label="Big Five по ходу ролика")
                emo_plot = gr.Plot(label="Эмоции по ходу ролика")
                with gr.Row():
                    with gr.Column(scale=1, min_width=360):
                        voice_plot = gr.Plot(label="Голос")
                    with gr.Column(scale=1, min_width=360):
                        speech_plot = gr.Plot(label="Речь")
            with gr.Tab("Эмоции и голос"):
                emo_bars = gr.Plot(label="Средний профиль эмоций")
                seg_table = gr.HTML(label="По отрезкам")
            with gr.Tab("Речь"):
                speech_html = gr.HTML(label="Речевая аналитика")
                transcript = gr.Textbox(label="Транскрипт речи", lines=10, max_lines=14)
            with gr.Tab("Мимика и кадры"):
                face_html = gr.HTML(label="Выражение лица")
                gallery = gr.Gallery(label="Ключевые кадры (по сегменту объяснений)", columns=5, height=200)
            with gr.Tab("Объяснения"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=360):
                        contrib = gr.HTML(label="Вклад модальностей")
                    with gr.Column(scale=1, min_width=360):
                        words_detail = gr.Textbox(label="Слова, на которые откликнулась модель", lines=8, max_lines=12)
                desc = gr.Textbox(label="Описание поведения по сегментам", lines=8, max_lines=12)
            with gr.Tab("Данные"):
                members = gr.Textbox(label="Участники ансамбля и время обработки", lines=4, max_lines=8)
                raw = gr.Code(label="result.json", language="json", lines=24)
                path = gr.Textbox(label="Сохранено в", interactive=False)
        gr.Markdown(f"<small>{DISCLAIMER_RU}<br>{INTERVIEW_DISCLAIMER_RU}<br>Эмоции, голос и мимика — сигналы моделей, обученных "
                    "на англоязычных корпусах и фотографиях; это наблюдения о поведении на видео, а не диагноз.</small>")
        outputs = [status, radar, bars, facts, narrative, traits_plot, emo_plot, voice_plot, speech_plot, emo_bars, seg_table,
                   speech_html, transcript, face_html, gallery, contrib, words_detail, desc, members, raw, path, job_state, pdf_btn]
        assert len(outputs) == N_REST + 1
        run_ev = btn.click(analyze, inputs=[video, lang, explain], outputs=outputs, show_progress="hidden", api_name=False)
        stop_btn.click(stop, inputs=None, outputs=[status], cancels=[run_ev], show_progress="hidden", api_name=False)
        pdf_btn.click(make_pdf, inputs=[job_state], outputs=[pdf_btn], api_name=False)
    return demo


def main(port: int = 7870, members: str = "oceanai,mm", work_dir: str | None = None, share: bool = False,
         asr_model: str = "openai/whisper-large-v3-turbo", ollama_model: str = "qwen2.5vl:7b", mm_ckpt: str | None = None,
         host: str = "0.0.0.0"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")
    wd = Path(work_dir or os.path.expanduser("~/bs2_data/web_jobs"))
    wd.mkdir(parents=True, exist_ok=True)
    studio = Studio(members=tuple(m.strip() for m in members.split(",") if m.strip()), asr_model=asr_model,
                    ollama_model=ollama_model, mm_ckpt=mm_ckpt)
    demo = build_app(studio, wd)
    demo.queue(default_concurrency_limit=1).launch(server_name=host, server_port=port, share=share, show_api=False,
                                                    show_error=True, quiet=False)
