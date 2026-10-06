"""Pass over the videos of the own Russian set (docs/dataset/4_Как_обучить_AMLAI_2.0.md, section 4): the features of every
20-second segment exactly as the product computes them, plus the scores of AMLAI 1.0 on the way.

  python -m training.ru_extract --videos /mnt/d/ru_set_v1/work --labels /mnt/d/ru_set_v1/labels.csv --out ~/data/ru_v1

Why not training/mm_extract.py: it reads the ready texts of First Impressions V2 (Rev transcripts, Qwen3-VL descriptions of
the whole clip). The product transcribes with Whisper, translates Russian to English with opus-mt and describes 16 frames
of each segment with the Ollama vision model; AMLAI 2.0 must learn on these inputs. So the segments are cut and scored by
the product's own LongVideoAnalyzer + MMBackend, and the model input of every scored segment is captured in `_score`.

One file per video, OUT/features/<video_name>.pt:
  {"video", "duration_sec", "segments": [{"segment", "start", "end", "transcript", "behavior_description",
   "scores_amlai1"}], "segments_planned", "segments_failed", "segment_errors", "x": {modality: FloatTensor[S, D]},
   "video_scores_amlai1", "modalities", "checkpoints", "bs3_version", "created"}
A finished video is skipped on a rerun. A video that fails is logged and the pass goes on, except for the failures the
product itself treats as fatal (Ollama down, a GPU fault) and after five failures in a row: then the pass stops.
OUT/extract_failed.txt always lists the videos of labels.csv that have no feature file, with the last reason;
OUT/amlai1_scores.csv (one row per video, trait names as in labels.csv) is rebuilt from the feature files at the end."""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from bs3 import __version__
from bs3.errors import is_fatal
from bs3.norms import TRAIT_KEYS

from .ru_labels import TRAITS

log = logging.getLogger("bs.ru_extract")
# the product names the traits by bs3.norms.TRAIT_KEYS, labels.csv by the FIV2 columns of ru_labels.TRAITS
PRODUCT_TO_LABEL = dict(zip(TRAIT_KEYS, TRAITS)) | {"interview": "interview"}
MAX_FAILURES_IN_A_ROW = 5


def feature_file(out: Path, video_name: str) -> Path:
    return out / "features" / f"{video_name}.pt"


def make_capture_backend(cfg):
    """AMLAI 1.0 (bs3.backend_mm.MMBackend) that keeps the model input of every scored segment, appended only after a
    successful forward pass so a failed segment leaves no orphan input. Built here so that the test suite can run this
    module without torch models: the import of the heavy backend happens on call."""
    from bs3.backend_mm import MMBackend

    class CaptureMM(MMBackend):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.inputs: list[dict] = []

        def _score(self, batch):
            out = super()._score(batch)
            self.inputs.append({m: v[0].detach().cpu().float() for m, v in batch.items()})
            return out

    return CaptureMM(cfg)


def extract_video(video: Path, analyzer, backend, out: Path, video_name: str | None = None) -> dict:
    """Score `video` with the product's analyzer over `backend` (a capture backend with `.inputs`), align the captured
    inputs with the scored segments of the timeline and save the feature file. Returns the saved record."""
    video_name = video_name or video.stem
    backend.inputs.clear()
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"ru_extract_{video_name}_", dir=str(out / "tmp")))
    try:
        res = analyzer.analyze(video, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)       # a fresh work dir per video: timeline.json of another video must never be reused
    timeline = res.get("timeline") or []
    if not timeline:                                 # a clip not longer than single_max is scored whole, as one segment
        timeline = [{"segment": 1, "start": 0.0, "end": float(res.get("duration_sec") or 0.0), "scores": res["scores"],
                     "transcript": res.get("transcript", ""), "behavior_description": res.get("behavior_description", "")}]
    ok = [t for t in timeline if t.get("scores")]
    if len(ok) != len(backend.inputs):
        raise RuntimeError(f"{video_name}: {len(ok)} scored segments but {len(backend.inputs)} captured inputs")
    if not ok:
        raise RuntimeError(f"{video_name}: no segment could be scored")
    mods = list(backend.inputs[0].keys())
    x = {m: torch.stack([inp[m] for inp in backend.inputs]) for m in mods}
    record = {
        "video": video_name, "duration_sec": res.get("duration_sec"),
        "segments": [{"segment": t.get("segment", i + 1), "start": t.get("start"), "end": t.get("end"),
                      "transcript": t.get("transcript", ""), "behavior_description": t.get("behavior_description", ""),
                      "scores_amlai1": t["scores"]} for i, t in enumerate(ok)],
        "segments_planned": int(res.get("segments") or len(timeline)), "segments_failed": len(timeline) - len(ok),
        "segment_errors": [str(t.get("error", ""))[:200] for t in timeline if not t.get("scores")],
        "x": x, "video_scores_amlai1": res["scores"], "modalities": mods,
        "checkpoints": list(getattr(backend, "checkpoints", [])), "bs3_version": __version__,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    p = feature_file(out, video_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = p.with_suffix(".pt.tmp")
    torch.save(record, tmp_p)
    tmp_p.replace(p)
    return record


def scores_table(out: Path) -> pd.DataFrame:
    """One row per feature file: the AMLAI 1.0 scores of the video (trait names as in labels.csv), segments, duration."""
    rows = []
    for p in sorted((out / "features").glob("*.pt")):
        d = torch.load(p, map_location="cpu")
        row = {"video_name": d["video"], "duration_sec": d.get("duration_sec"), "segments": len(d["segments"]),
               "segments_failed": d.get("segments_failed", 0)}
        row.update({PRODUCT_TO_LABEL[k]: d["video_scores_amlai1"].get(k) for k in TRAIT_KEYS + ["interview"]})
        rows.append(row)
    return pd.DataFrame(rows)


def _read_failures(path: Path) -> dict:
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                name, reason = line.split("\t", 1)
                out[name] = reason
    return out


def run(videos_dir: Path, labels: pd.DataFrame, out: Path, analyzer, backend, limit: int = 0, log_fn=log.info) -> dict:
    """The pass itself; `analyzer` and `backend` are built by main() (or by a test). Returns counts."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "tmp").mkdir(exist_ok=True)
    for stale in (out / "features").glob("*.pt.tmp"):
        stale.unlink()                                 # a half-written file of an interrupted run
    all_names = labels["video_name"].astype(str).tolist()
    names = all_names[:limit] if limit else all_names
    failed_log = out / "extract_failed.txt"
    failures = _read_failures(failed_log)
    done = skipped = failed = in_a_row = 0
    t0 = time.time()
    stop = None
    try:
        for i, name in enumerate(names, 1):
            if feature_file(out, name).exists():
                skipped += 1
                continue
            video = videos_dir / f"{name}.mp4"
            if not video.is_file():
                failed += 1
                failures[name] = f"нет файла {video}"
                log_fn(f"{name}: нет файла {video}")
                continue
            t1 = time.time()
            try:
                rec = extract_video(video, analyzer, backend, out, name)
            except Exception as e:  # noqa: BLE001 — one broken video must not stop a pass of several days
                failed += 1
                in_a_row += 1
                reason = str(e).splitlines()[0][:200] if str(e).strip() else type(e).__name__
                failures[name] = reason
                log_fn(f"{name}: сбой — {reason[:160]}")
                if is_fatal(e):                        # Ollama down or a GPU fault: the next videos would fail the same way
                    stop = f"{name}: фатальный сбой (Ollama или видеокарта), проход остановлен"
                    break
                if in_a_row >= MAX_FAILURES_IN_A_ROW:
                    stop = f"{MAX_FAILURES_IN_A_ROW} сбоев подряд, проход остановлен; последняя причина: {reason[:160]}"
                    break
                continue
            in_a_row = 0
            done += 1
            failures.pop(name, None)
            log_fn(f"[{i}/{len(names)}] {name}: {len(rec['segments'])} отрезков за {time.time() - t1:.0f} с "
                   f"(ролик {rec['duration_sec'] or 0:.0f} с)")
    finally:
        # the failures file tells the truth about the whole labels list: every video without a feature file, last reason
        missing = [n for n in all_names if not feature_file(out, n).exists()]
        failed_log.write_text("".join(f"{n}\t{failures.get(n, 'не обработан')}\n" for n in missing), encoding="utf-8")
        table = scores_table(out)
        if len(table):
            table.to_csv(out / "amlai1_scores.csv", index=False, encoding="utf-8")
        log_fn(f"итог: {done} новых, {skipped} уже были, {failed} сбоев; без признаков {len(missing)} из {len(all_names)}; "
               f"{time.time() - t0:.0f} с; amlai1_scores.csv — {len(table)} роликов")
    if stop:
        log_fn(stop)
        raise SystemExit(stop)
    return {"done": done, "skipped": skipped, "failed": failed, "n_table": len(table), "missing": len(missing)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m training.ru_extract", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="папка work/ набора: <код>.mp4")
    ap.add_argument("--labels", required=True, help="labels.csv набора: проходятся только ролики из него")
    ap.add_argument("--out", required=True, help="куда писать features/, amlai1_scores.csv, extract_failed.txt")
    ap.add_argument("--limit", type=int, default=0, help="только первые N роликов (проба)")
    ap.add_argument("--mm-ckpt", default=None, help="веса AMLAI 1.0 (по умолчанию settings.MM_CHECKPOINTS)")
    ap.add_argument("--ollama-model", default=None)
    ap.add_argument("--asr-model", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("numba", "urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from bs3 import LANG
    from bs3.backend_mm import MMConfig
    from bs3.longvideo import LongVideoAnalyzer

    kw = {"lang": LANG}
    if a.mm_ckpt:
        kw["checkpoint"] = a.mm_ckpt
    if a.ollama_model:
        kw["ollama_model"] = a.ollama_model
    if a.asr_model:
        kw["asr_model"] = a.asr_model
    backend = make_capture_backend(MMConfig(**kw)).load()
    analyzer = LongVideoAnalyzer(backend, lang=LANG, asr_model=backend.cfg.asr_model)
    labels = pd.read_csv(a.labels, encoding="utf-8")
    run(Path(a.videos).expanduser(), labels, Path(a.out).expanduser(), analyzer, backend, limit=a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
