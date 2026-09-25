"""Re-render finished sample jobs with BS Profiler 3.0 without a new analysis and check the page and the PDF (design
13.3, items 1-5; tasks T20, T22).

    ~/bs/venv/bin/python bs3-studio/scripts/rerender_samples.py A=~/bs2_data/web_jobs/<id> B=~/bs2_data/web_jobs/<id>
                                                                [--html-dir DIR] [--pdf-dir DIR]

Each argument is a finished job of the old work dir, optionally with the tag of a design sample (A or B, design 13.2):
tagged jobs are also checked against the golden values of that sample; untagged ones get the general checks only.
No job ids or names are kept in this file: they are given on the command line.

For every job:
1. sha256 of every file of the source job (except segments/ and seg*.mp4) is taken;
2. import_job.py copies result.json and explain/ into ~/bs3_data/web_jobs/<id>/ (an existing copy is replaced);
3. webapp.page_outputs runs on the copy and the page is checked: 27 values; the characterization starts with its
   header and «Коротко»; the first key fact is the MBTI card; the MBTI panels carry the agreement line; the main system
   has a letter strip; the own model of an old job gets C18; segments without the main system are gaps on the
   timeline chart and are named by C13 in «Как получены оценки»; «Краткие выводы» appears nowhere; result.json of
   the copy gets no `mbti` section; nothing on the page mentions a group of processed videos (reference group,
   position, «типичный», percentiles of Russian speech), and the characterization does not compare the two systems
   (change of 2026-09-26);
4. webapp.export_pdf runs on the copy: the PDF is built; its text (pdftotext) contains «Характеристика личности»,
   «Тип MBTI (перевод шкал Big Five)», «Как получены оценки», «BS Profiler 3.0 · стр.», the type of the main system
   and, in the appendix «Значения по отрезкам», the type of every typed segment; it does not contain «Краткие
   выводы», «сегмент» (outside the transcript, which is the person's own speech), the name of the old version or
   anything about a group of processed videos; it
   has at most 2 pages more than the PDF of the old version in the source job (when there is one);
5. sha256 of the source job is taken again and must not have changed.

--html-dir DIR: also save the characterization and the «Тип MBTI» tab of every job as page_<tag>_char.html and
page_<tag>_mbti.html (for reading them in a browser). --pdf-dir DIR: also copy the PDF of every job there as
report_<tag>.pdf. Exit code 0 when every check passed.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_job import import_job, tree_sha256  # noqa: E402

# golden values of the design samples (design 13.2, recomputed for the absolute scale of 2026-09-26): types,
# agreement, the clean extraversion, segments without OCEAN-AI and the summary of the letter strip
EXPECT = {
    "A": {"type": "XNFJ", "type_strict": "ENFJ", "alternatives": ["INFJ"], "second": ("IXXX", "ISTJ"),
          "agreement": {"EI": "border", "SN": "border", "TF": "border", "JP": "border"},
          "extraversion": "0.56", "dropped": [16], "header": ["ENFJ", "«Наставник»", "ось E–I на границе"],
          "strip": "ENFJ во всех 17 отрезках с оценкой; все четыре оси совпадают с итогом во всех отрезках"},
    "B": {"type": "ENFJ", "type_strict": "ENFJ", "alternatives": [], "second": ("ISXX", "ISTP"),
          "agreement": {"EI": "differ", "SN": "differ", "TF": "border", "JP": "border"},
          "extraversion": "0.73", "dropped": [10, 11, 12, 14, 15, 26, 33], "header": ["ENFJ", "«Наставник»"],
          "strip": "ENFJ во всех 26 отрезках с оценкой; все четыре оси совпадают с итогом во всех отрезках"},
}
# nothing about a group of processed videos anywhere (change of 2026-09-26)
# («на русских роликах её значения ниже» about the scale of the own model is not a group: it stays)
RELATIVE = re.compile(r"опорн|положени[ея] (?:в|среди|оценки)|типичн|\d+ русск\w* ролик|среди (?:тех же )?русских|"
                      r"обработанных (?:русских|системой)|предварительн|большинства из|сейчас их \d|проверка на \d",
                      re.I)
# the characterization does not compare the two systems (change of 2026-09-26)
NO_SYSTEMS = re.compile(r"Согласие двух систем|Вторая система|Второе мнение|Системы по отдельности|противоположно")
PAGE_CSS = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1f2937;"
            "background:#fff;--block-background-fill:#fff}h2{font-size:17px;margin:28px 0 10px}"
            "section{border:1px solid #808080;border-radius:8px;padding:12px 14px}")


class Checks:
    def __init__(self, name: str):
        self.name, self.failed, self.n = name, [], 0

    def ok(self, cond, what: str) -> None:
        self.n += 1
        if not cond:
            self.failed.append(what)
            print(f"  FAIL {what}")


def _page(title: str, blocks: list[tuple[str, str]]) -> str:
    body = "".join(f"<h2>{t}</h2><section>{h}</section>" for t, h in blocks)
    return (f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{PAGE_CSS}</style></head><body>{body}</body></html>")


PDF_MUST = ("Характеристика личности", "Тип MBTI (перевод шкал Big Five)", "Как получены оценки", "BS Profiler 3.0 · стр.")
PDF_MUST_NOT = (("Краткие выводы", re.compile(r"Краткие выводы")), ("сегмент", re.compile(r"сегмент", re.I)),
                ("the name of the old version", re.compile(r"BS\s+2\.0")),
                ("a group of processed videos", RELATIVE), ("«Согласие двух систем»", re.compile(r"Согласие двух систем")))
MAX_EXTRA_PAGES = 2


def pdf_text(path: Path) -> str:
    """Text of a PDF (pdftotext of poppler-utils), whitespace collapsed to single spaces."""
    r = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, text=True, check=True)
    return re.sub(r"\s+", " ", r.stdout)


def pdf_pages(path: Path) -> int:
    r = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True)
    m = re.search(r"^Pages:\s+(\d+)", r.stdout, re.M)
    return int(m.group(1)) if m else 0


def check_pdf(c: Checks, src: Path, dest: Path, mb: dict | None, exp: dict | None, pdf_dir: Path | None,
              label: str) -> None:
    """Item 4 of design 13.3: build the PDF of the copy and check its text and length."""
    from bs3.webapp import export_pdf
    pdf = Path(export_pdf(dest))
    c.ok(pdf.exists() and pdf.parent == dest, "PDF built inside the copy")
    if not pdf.exists():
        return
    try:
        text, pages = pdf_text(pdf), pdf_pages(pdf)
    except (OSError, subprocess.CalledProcessError) as e:
        c.ok(False, f"pdftotext / pdfinfo (poppler-utils) available: {type(e).__name__}")
        return
    for s in PDF_MUST:
        c.ok(s in text, f"PDF contains «{s}»")
    # the transcript is the person's own speech: a word there is not a wording of the report
    body = re.split(r"Приложение [А-Я]\. Транскрипт речи", text)[0]
    for what, rx in PDF_MUST_NOT:
        c.ok(not rx.search(body), f"PDF does not contain «{what}»")
    if mb:
        c.ok(mb["type_strict"] in text and mb["type"] in text, f"PDF shows the type {mb['type']} / {mb['type_strict']}")
        # appendix «Значения по отрезкам»: the MBTI column carries the type of every typed segment
        appx = text.split("Значения по отрезкам")[-1]
        seg_types = {e["type"] for e in mb.get("timeline") or [] if e.get("type")}
        c.ok(" MBTI " in appx and all(t in appx for t in seg_types),
             f"MBTI column in the appendix with {len(seg_types)} segment type(s)")
    if exp:
        c.ok(exp["strip"] in text, "PDF strip summary as in design 13.2")
    old = sorted(p for p in src.glob("*_report_*.pdf"))
    if old:
        n_old = pdf_pages(old[0])
        c.ok(pages <= n_old + MAX_EXTRA_PAGES, f"PDF has {pages} pages, the old one {n_old} (at most +{MAX_EXTRA_PAGES})")
    else:
        print(f"  note: no PDF of the old version in the source job, page growth not checked ({pages} pages)")
    c.ok("mbti" not in json.loads((dest / "result.json").read_text(encoding="utf-8")),
         "result.json of the copy still has no mbti section after the PDF")
    if pdf_dir:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, pdf_dir / f"report_{label}.pdf")


def check_job(src: Path, tag: str | None, html_dir: Path | None, pdf_dir: Path | None = None) -> Checks:
    from bs3 import caveats, mbti
    from bs3.charts import fig_traits_timeline
    from bs3.scores import clean_view
    from bs3.webapp import N_PAGE, page_outputs

    label = tag or src.name
    c = Checks(label)
    print(f"== {label}")
    before = tree_sha256(src)                                                  # 1
    dest = import_job(src, force=True)                                         # 2
    res = dest / "result.json"
    rep = json.loads(res.read_text(encoding="utf-8"))
    c.ok(rep.get("job_dir") == str(dest), "job_dir of the copy points into the copy")
    outs = page_outputs(rep)                                                   # 3
    c.ok(len(outs) == N_PAGE == 27, f"page_outputs gives 27 values (N_REST − 1), got {len(outs)}")
    (radar, bars, facts, char, traits_plot, _emo, _voice, _speech, _bars2, _segs, _sp, _tr, _face, _fplot, _frames,
     _contrib, _words, _desc, members, raw, _path, _job, method, emo_intro, types, strip, read) = outs

    view = clean_view(json.loads(res.read_text(encoding="utf-8")))
    mb = mbti.get_mbti(rep, view)
    meta = view["view_meta"]

    # characterization: header first, then «Коротко»
    c.ok(char.startswith("<div class='bs3-char-text'"), "characterization is the HTML of characterization.build")
    i_head, i_short = char.find("border-bottom:1px solid"), char.find("<b>Коротко.</b>")
    c.ok(0 <= i_head < i_short, "characterization starts with the header, then «Коротко»")
    c.ok("определяет тип" not in char and "сегмент" not in char, "no «определяет тип», no «сегмент» in the text")
    c.ok(not NO_SYSTEMS.search(char), "the characterization does not compare the two systems")
    for name, h in (("characterization", char), ("key facts", facts), ("bars", bars), ("method", method),
                    ("types", types), ("strip", strip), ("read", read)):
        m_rel = RELATIVE.search(re.sub(r"<[^>]+>", " ", h))
        c.ok(m_rel is None, f"nothing about a group of processed videos in {name}" + (f": «{m_rel.group(0)}»" if m_rel else ""))
    stem = Path(rep.get("original_file_name") or "").stem
    c.ok(not stem or stem not in char, "no file name in the characterization")
    # key facts: the type card first
    title = mbti.type_title(mb) if mb else None
    first_label = re.search(r"<div style='[^']*'><div style='[^']*'>([^<]*)</div>", facts)
    c.ok(bool(first_label) and first_label.group(1) == title, f"first key fact is «{title}»")
    card = mbti.fact_card(mb)
    c.ok(card is not None and f">{card[1]}</div>" in facts, "the type card shows the type")
    # MBTI tab: panels with the agreement line, the strip of the main system, the reading guide
    if mb and mb.get("agreement"):
        c.ok(mbti.agreement_line(mb["agreement"]) in types, "agreement line under the panels")
    c.ok("OCEAN-AI — основная оценка" in strip if meta["lang"] == "ru" and mb and mb.get("source") == "ocean_ai"
         else bool(strip), "letter strip of the main system")
    old_second = [s for s in (mb or {}).get("second") or [] if not s.get("timeline")]
    c.ok((caveats.text("C18") in strip) == bool(old_second), "C18 exactly when the second system has no strip")
    c.ok(caveats.text("C8") in strip, "C8 under the strip")
    for code in ("C3", "C4", "C5", "C9", "C16"):
        c.ok(caveats.text(code) in read, f"{code} in «Как читать тип MBTI»")
    c.ok(caveats.c6(meta["lang"]) in read and caveats.c7(meta["lang"]) in read, "C6 and C7 in «Как читать тип MBTI»")
    # segments without the main system: gaps on the chart, C13 in «Как получены оценки»
    dropped = meta["segments_without_primary"]
    fig = fig_traits_timeline(view, "light")
    gaps = sum(1 for y in fig.data[0].y if y is None)
    c.ok(gaps == len(dropped), f"{len(dropped)} gaps on the timeline chart (got {gaps})")
    if dropped:
        c.ok("нет оценки" in traits_plot, "gap band «нет оценки» on the timeline chart")
        c.ok(caveats.c13(len(dropped), meta["segments_total"]) in method, "C13 in «Как получены оценки»")
        n_dash = strip.count(">—</div>")
        c.ok(n_dash >= 4 * len(dropped), f"«—» on the {len(dropped)} segments without the main system in the strip")
    c.ok(bool(method), "«Как получены оценки» filled")
    c.ok(bool(emo_intro) == bool((view.get("analyses") or {}).get("voice") or (view.get("analyses") or {})
                                 .get("emotions_text")), "«Эмоции и голос: коротко» filled when there are analyses")
    # data tab
    if mb and mb.get("computed_on_render"):
        c.ok(caveats.text("C22") in members, "C22 in «Участники ансамбля и время обработки»")
    # nothing of the 2.0 summary, nothing written
    c.ok(not any(isinstance(o, str) and "Краткие выводы" in o for o in outs), "«Краткие выводы» nowhere on the page")
    for name, h in (("method", method), ("emo_intro", emo_intro), ("types", types), ("strip", strip), ("read", read)):
        c.ok("сегмент" not in h, f"no «сегмент» in {name}")
    c.ok("mbti" not in json.loads(res.read_text(encoding="utf-8")), "result.json of the copy has no mbti section")

    # the design samples: golden values (design 13.2)
    exp = EXPECT.get(tag or "")
    if exp:
        c.ok(mb["type"] == exp["type"] and mb["type_strict"] == exp["type_strict"],
             f"type {mb['type']}/{mb['type_strict']} == {exp['type']}/{exp['type_strict']}")
        c.ok(mb["alternatives"] == exp["alternatives"], f"alternatives {mb['alternatives']}")
        s = mb["second"][0]
        c.ok((s["type"], s["type_strict"]) == exp["second"], f"own model {s['type']}/{s['type_strict']}")
        c.ok(mb["agreement"]["axes"] == exp["agreement"] and mb["agreement"]["n_agree"] == 0,
             f"agreement {mb['agreement']['axes']}")
        c.ok(dropped == exp["dropped"], f"segments without OCEAN-AI {dropped}")
        m = re.search(r"Экстраверсия</b> <span[^>]*>([^<]*)<", bars)
        c.ok(bool(m) and m.group(1) == exp["extraversion"], f"extraversion bar shows the score only: {m and m.group(1)}")
        for h in exp["header"]:
            c.ok(h in char[:i_short], f"header shows {h}")
        c.ok(exp["strip"] in strip, "strip summary as in design 13.2")

    check_pdf(c, src, dest, mb, exp, pdf_dir, label)                               # 4

    after = tree_sha256(src)                                                   # 5
    c.ok(after == before, f"source job unchanged (sha256 of {len(before)} files)")

    if html_dir:
        html_dir.mkdir(parents=True, exist_ok=True)
        (html_dir / f"page_{label}_char.html").write_text(
            _page(f"Характеристика личности · {label}", [("Ключевые факты", facts), ("Характеристика личности", char)]),
            encoding="utf-8")
        (html_dir / f"page_{label}_mbti.html").write_text(
            _page(f"Тип MBTI · {label}", [("Тип MBTI по двум системам", types), ("Тип по ходу ролика", strip),
                                         ("Как читать тип MBTI", read)]), encoding="utf-8")
    print(f"  {c.n - len(c.failed)} of {c.n} checks passed; {mb['type'] if mb else '—'}/"
          f"{mb['type_strict'] if mb else '—'}, words {len(re.sub('<[^>]+>', ' ', char).split())}")
    return c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs", nargs="+", help="[A=|B=]<job folder of the old work dir>")
    ap.add_argument("--html-dir", help="save the characterization and the MBTI tab of every job here")
    ap.add_argument("--pdf-dir", help="copy the PDF of every job here as report_<tag>.pdf")
    args = ap.parse_args(argv)
    html_dir = Path(args.html_dir).expanduser() if args.html_dir else None
    pdf_dir = Path(args.pdf_dir).expanduser() if args.pdf_dir else None
    failed = 0
    for arg in args.jobs:
        tag, _, path = arg.rpartition("=")
        try:
            c = check_job(Path(path).expanduser().resolve(), tag or None, html_dir, pdf_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {arg}: {type(e).__name__}: {e}")
            failed += 1
            continue
        failed += bool(c.failed)
    print("OK" if not failed else f"{failed} job(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
