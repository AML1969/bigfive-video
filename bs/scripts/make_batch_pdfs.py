"""Full PDF reports for every finished video of a batch run (same content as the web «Экспорт в PDF»).
For each OUT_DIR/<name>/result.json: media metadata of the source file, explanations (modality contribution,
key frames, words) on the representative segment if not computed yet, full transcript translation, then the PDF.
Usage: make_batch_pdfs.py OUT_DIR SRC_DIR [--lang ru] [--only NAME ...] [--force]
"""
import argparse
import json
import re
import time
from pathlib import Path

from bs_bigfive.media import probe_media
from bs_bigfive.pdf_report import build_pdf

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("src")
ap.add_argument("--lang", default="ru")
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--force", action="store_true", help="recompute explanations even if explain/explanation.json exists")
ap.add_argument("--retranslate", action="store_true", help="translate the attributed words again (e.g. after a prompt change)")
a = ap.parse_args()
out, src = Path(a.out), Path(a.src)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "video"


sources = {}
for p in sorted(src.iterdir()):
    if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"):
        sources.setdefault(safe(p.stem), p)

_mm = None


def mm_backend():
    global _mm
    if _mm is None:
        from bs_bigfive.backend_mm import MMBackend, MMConfig
        _mm = MMBackend(MMConfig(lang=a.lang)).load()
    return _mm


done = []
for d in sorted(out.iterdir()):
    rj = d / "result.json"
    if not d.is_dir() or not rj.exists() or (a.only and d.name not in a.only):
        continue
    t0 = time.time()
    rep = json.loads(rj.read_text(encoding="utf-8"))
    # keys as the web report names them
    rep.setdefault("variant_scores", rep.get("variants"))
    rep.setdefault("scores_std_across_segments", rep.get("scores_std"))
    srcfile = sources.get(d.name)
    if srcfile is not None and (a.force or not rep.get("media") or "error" in rep["media"]):
        rep["media"] = probe_media(srcfile)
        rep["media"]["file_name"] = srcfile.name
        rep["original_file_name"] = srcfile.name
    ex_dir = d / "explain"
    ej = ex_dir / "explanation.json"
    expl = None
    if ej.exists() and not a.force:
        expl = json.loads(ej.read_text(encoding="utf-8"))
    elif rep.get("timeline"):
        seg = rep["timeline"][rep["representative_segment"] - 1]
        for old in ex_dir.glob("key_*.jpg"):      # frames of a previous representative segment must not linger
            old.unlink()
        try:
            expl = mm_backend().explain_video(seg["file"], ex_dir, asr=False, transcript=seg["transcript"],
                                              behavior=seg.get("behavior_description") or None)
        except Exception as e:  # noqa: BLE001
            print(f"  {d.name}: explanation failed: {str(e).splitlines()[0][:160]}")
    if expl:
        rep["modality_shares"] = {k: {m: v["share"] for m, v in row.items()}
                                  for k, row in expl["modalities"]["input_x_gradient"].items()}
    if a.lang != "en" and rep.get("transcript") and not rep.get("transcript_en"):
        try:
            rep["transcript_en"] = mm_backend().to_english(rep["transcript"])
        except Exception as e:  # noqa: BLE001
            print(f"  {d.name}: translation failed: {str(e).splitlines()[0][:160]}")
    # interface language: the description (and the transcript of English speech) is shown in Russian, the
    # originals stay in result.json; the same translations as the web page (webapp._display_translations)
    if a.force:
        rep.pop("behavior_description_ru", None)
        rep.pop("transcript_ru", None)
        rep.pop("translated_by", None)
    from bs_bigfive.webapp import _display_translations
    _display_translations(rep)
    # the same word lists as the web page: words looked up in the Russian texts the PDF shows (words.word_lists)
    from bs_bigfive.words import lists_outdated, word_lists
    if expl and (a.retranslate or lists_outdated(expl, rep)):
        expl["readable_words"] = word_lists(expl, rep)
        ej.write_text(json.dumps(expl, ensure_ascii=False, indent=2), encoding="utf-8")
    rj.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    # only the frames of the current explanation (stale files from an earlier run are ignored and removed)
    wanted = set(Path(p).name for p in (expl or {}).get("frames", {}).get("key_frame_files", []))
    frames = []
    for p in sorted(ex_dir.glob("key_*.jpg")) if ex_dir.exists() else []:
        if not wanted or p.name in wanted:
            frames.append(str(p))
        else:
            p.unlink()
    stem = re.sub(r"[^A-Za-z0-9А-Яа-яЁё._-]+", "_", Path(rep.get("original_file_name") or d.name).stem)[:60]
    pdf = build_pdf(rep, d / f"BigFive_report_{stem}.pdf", explanation=expl, media=rep.get("media"), key_frames=frames)
    print(f"{d.name}: {Path(pdf).name} ({Path(pdf).stat().st_size // 1024} KB, frames {len(frames)}, "
          f"{round(time.time() - t0, 1)} s)", flush=True)
    done.append(pdf)
print(f"{len(done)} PDF reports written")
