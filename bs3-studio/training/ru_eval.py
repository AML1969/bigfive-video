"""Accuracy of AMLAI 2.0 on the own Russian set and the comparison with its rivals (docs/dataset/4_Как_обучить_AMLAI_2.0.md,
sections 6-8): per-video scores from the segment features of training/ru_extract.py, CCC / Pearson / ACC per trait with
bootstrap intervals, and the rivals constant answer, AMLAI 1.0 and OCEAN-AI, raw and calibrated.

  # the test, once: five fold models averaged, calibration of the rivals fitted on all non-test videos
  python -m training.ru_eval --data ~/data/ru_v1 --labels /mnt/d/ru_set_v1/labels.csv \
      --ckpt "~/bs/amlai2/v1/fold*/best.pt" --mode test --oceanai ~/data/ru_v1/oceanai --out ~/data/ru_v1/eval_test.json
  # out-of-fold: every non-test video scored by the model of its own fold, pooled; rivals calibrated on the other folds
  python -m training.ru_eval --data ~/data/ru_v1 --labels ... --ckpt-pattern "~/bs/amlai2_runs/A_lr1e-5/fold{fold}/best.pt" \
      --mode oof --out ~/data/ru_v1/eval_oof_A.json

Calibration = a straight line per trait that gives the rival the mean and the spread of the label on the fitting videos
(a = sd(label) / sd(score), b = mean(label) - a * mean(score)); with it the rival's CCC equals its Pearson correlation,
the best any linear transform can do. Every rival is compared with AMLAI 2.0 on the videos BOTH have a score for (nothing
is imputed), with the paired bootstrap of the difference on the same resamples. mACC is printed next to the constant
answer because it flatters. A video of labels.csv without a feature file stops a test run unless --allow-missing."""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bs3.mm.model import ModelConfig, PersonalityFusionModel
from bs3.norms import TRAIT_KEYS

from .ru_labels import RU, TRAITS
from .ru_train import ccc, pearson, segment_durations, video_metrics

log = logging.getLogger("bs.ru_eval")
# the product names the traits by bs3.norms.TRAIT_KEYS, labels.csv by the FIV2 columns
PRODUCT_TO_LABEL = dict(zip(TRAIT_KEYS, TRAITS)) | {"interview": "interview"}
OUTPUTS = TRAITS + ["interview"]
MAIN = "AMLAI 2.0"


# ---------------------------------------------------------------- scores per video
def resolve_checkpoints(spec: str) -> list[str]:
    """Like MMBackend.load: one path, a comma-separated list, or a glob (several files are averaged)."""
    paths = []
    for part in str(spec).split(","):
        part = os.path.expanduser(part.strip())
        if not part:
            continue
        hits = sorted(glob.glob(part)) if any(ch in part for ch in "*?[") else [part]
        if not hits or not all(os.path.isfile(h) for h in hits):
            raise SystemExit(f"--ckpt: ничего нет по пути {part}")
        paths += hits
    if not paths:
        raise SystemExit(f"--ckpt: пустой список {spec!r}")
    return paths


def load_models(paths: list[str], device: str) -> list:
    models = []
    for p in paths:
        ckpt = torch.load(p, map_location=device)
        m = PersonalityFusionModel(ModelConfig(**ckpt["config"])).to(device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        m.targets = list(ckpt["targets"])
        m.modalities = list(ckpt["modalities"])
        models.append(m)
    if not models:
        raise SystemExit("no checkpoint found")
    if len({tuple(m.targets) for m in models}) != 1 or len({tuple(m.modalities) for m in models}) != 1:
        raise SystemExit("the checkpoints have different outputs or modalities and cannot be averaged")
    return models


@torch.no_grad()
def model_scores(models: list, record: dict, device: str) -> dict:
    """Duration-weighted mean over the segments of the averaged outputs of all models, as the product does."""
    missing = [m for m in models[0].modalities if m not in record["x"]]
    if missing:
        raise SystemExit(f"{record['video']}: в признаках нет {missing}, которые нужны модели (другой проход ru_extract?)")
    x = {m: record["x"][m].to(device) for m in models[0].modalities}
    out = np.mean([mdl(x).cpu().numpy() for mdl in models], axis=0)            # [S, T]
    dur = segment_durations(record["segments"])
    vec = (out * (dur / dur.sum())[:, None]).sum(axis=0)
    return dict(zip(models[0].targets, map(float, vec)))


def amlai1_scores(record: dict) -> dict:
    return {PRODUCT_TO_LABEL[k]: float(v) for k, v in record["video_scores_amlai1"].items() if k in PRODUCT_TO_LABEL}


def oceanai_scores(folder: Path, name: str) -> dict | None:
    """The scores of a `bs3 infer --backend oceanai` report; a file holding several reports is matched by the input name."""
    p = folder / f"{name}.json"
    if not p.is_file():
        return None
    rep = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(rep, list):
        hits = [r for r in rep if Path(str(r.get("input", ""))).stem == name]
        if len(hits) == 1:
            rep = hits[0]
        elif len(rep) == 1:
            rep = rep[0]
        else:
            raise SystemExit(f"{p}: {len(rep)} отчётов, по имени {name} найдено {len(hits)}")
    return {PRODUCT_TO_LABEL[k]: float(v["score"]) for k, v in rep["traits"].items() if k in PRODUCT_TO_LABEL}


def calibrate(fit_scores: np.ndarray, fit_labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Per column: a = sd(label) / sd(score), b = mean(label) - a * mean(score), fitted on the rows of `fit_*` where both
    are known."""
    out = np.full_like(scores, np.nan, dtype=float)
    for j in range(scores.shape[1]):
        ok = ~np.isnan(fit_scores[:, j]) & ~np.isnan(fit_labels[:, j])
        if ok.sum() < 3:
            continue
        sx = fit_scores[ok, j].std()
        a = fit_labels[ok, j].std() / sx if sx > 0 else 0.0
        b = fit_labels[ok, j].mean() - a * fit_scores[ok, j].mean()
        out[:, j] = a * scores[:, j] + b
    return out


def bootstrap_pair(labels: np.ndarray, main: np.ndarray, rival: np.ndarray | None, B: int, seed: int) -> dict:
    """95% intervals over resamples of the videos: mean CCC and per-trait CCC of `main` (and of `rival`), and the paired
    difference rival - main on the same resamples. Arrays hold the five traits of the common videos."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    draws = {"main": [], "rival": [], "main_t": [], "rival_t": []}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        cm = [ccc(labels[idx, i], main[idx, i]) for i in range(5)]
        draws["main"].append(np.mean(cm))
        draws["main_t"].append(cm)
        if rival is not None:
            cr = [ccc(labels[idx, i], rival[idx, i]) for i in range(5)]
            draws["rival"].append(np.mean(cr))
            draws["rival_t"].append(cr)

    def ci(v):
        v = np.asarray(v, dtype=float)
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    out = {"n": int(n), "main_mCCC_ci": ci(draws["main"]),
           "main_ccc_ci": {t: ci(np.asarray(draws["main_t"])[:, i]) for i, t in enumerate(TRAITS)}}
    if rival is not None:
        d = np.asarray(draws["rival"]) - np.asarray(draws["main"])
        out["rival_mCCC_ci"] = ci(draws["rival"])
        out["rival_ccc_ci"] = {t: ci(np.asarray(draws["rival_t"])[:, i]) for i, t in enumerate(TRAITS)}
        out["difference_ci"] = ci(d)                   # rival - main: negative = AMLAI 2.0 ahead
    return out


# ---------------------------------------------------------------- the evaluation
def collect(df: pd.DataFrame, data: Path, models: list, device: str, oce_dir: Path | None) -> dict:
    """Per video of `df`: the label, the AMLAI 2.0 score, the AMLAI 1.0 score, the OCEAN-AI score (NaN without a report).
    Videos without a feature file are listed in 'missing' and left out."""
    out = {"label": [], MAIN: [], "AMLAI 1.0": [], "OCEAN-AI": [], "names": [], "missing": [], "no_oceanai": []}
    for row in df.to_dict("records"):
        name = str(row["video_name"])
        p = data / "features" / f"{name}.pt"
        if not p.is_file():
            out["missing"].append(name)
            continue
        rec = torch.load(p, map_location="cpu")
        out["names"].append(name)
        out["label"].append([float(row[t]) for t in OUTPUTS])
        m2 = model_scores(models, rec, device)
        out[MAIN].append([m2.get(t, np.nan) for t in OUTPUTS])
        a1 = amlai1_scores(rec)
        out["AMLAI 1.0"].append([a1.get(t, np.nan) for t in OUTPUTS])
        oc = oceanai_scores(oce_dir, name) if oce_dir else None
        if oce_dir and oc is None:
            out["no_oceanai"].append(name)
        out["OCEAN-AI"].append([oc.get(t, np.nan) if oc else np.nan for t in OUTPUTS])
    for k in ("label", MAIN, "AMLAI 1.0", "OCEAN-AI"):
        out[k] = np.array(out[k], dtype=float).reshape(-1, len(OUTPUTS))
    return out


def evaluate(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data).expanduser()
    labels = pd.read_csv(args.labels, encoding="utf-8")
    labels["fold"] = labels["fold"].astype("Int64")
    for t in OUTPUTS:
        labels[t] = pd.to_numeric(labels[t], errors="coerce")
    labels = labels.dropna(subset=TRAITS)
    train = labels[labels["split"] == "train"]
    if args.mode == "test":
        groups = [(labels[labels["split"] == "test"], train, load_models(resolve_checkpoints(args.ckpt), device))]
    else:
        if not args.ckpt_pattern or "{fold}" not in args.ckpt_pattern:
            raise SystemExit("--mode oof needs --ckpt-pattern with {fold}")
        groups = []
        for k in sorted(train["fold"].dropna().unique()):
            path = os.path.expanduser(args.ckpt_pattern.format(fold=int(k)))
            groups.append((train[train["fold"] == k], train[train["fold"] != k], load_models([path], device)))
    oce_dir = Path(args.oceanai).expanduser() if args.oceanai else None

    rows, missing, no_oceanai = [], [], []
    for eval_rows, fit_rows, models in groups:
        ev, fit = collect(eval_rows, data, models, device, oce_dir), collect(fit_rows, data, models, device, oce_dir)
        missing += ev["missing"]
        no_oceanai += ev["no_oceanai"]
        if not ev["names"]:
            raise SystemExit("no evaluation video has a feature file")
        systems = {MAIN: ev[MAIN],
                   "постоянный ответ": np.repeat(np.nanmean(fit["label"], axis=0, keepdims=True), len(ev["names"]), axis=0),
                   "AMLAI 1.0": ev["AMLAI 1.0"], "AMLAI 1.0 с калибровкой": calibrate(fit["AMLAI 1.0"], fit["label"], ev["AMLAI 1.0"])}
        if oce_dir:
            systems["OCEAN-AI"] = ev["OCEAN-AI"]
            systems["OCEAN-AI с калибровкой"] = calibrate(fit["OCEAN-AI"], fit["label"], ev["OCEAN-AI"])
        for i, name in enumerate(ev["names"]):
            r = {"video_name": name, **{f"label_{t}": ev["label"][i, j] for j, t in enumerate(OUTPUTS)}}
            for sname, arr in systems.items():
                for j, t in enumerate(OUTPUTS):
                    r[f"{sname}:{t}"] = arr[i, j]
            rows.append(r)
    if missing:
        log.warning("%d ролик(ов) без файла признаков не вошли: %s", len(missing), missing[:10])
        if args.mode == "test" and not args.allow_missing:
            raise SystemExit(f"в тесте {len(missing)} ролик(ов) без признаков ({missing[:5]}…): разберитесь или передайте --allow-missing")
    if no_oceanai:
        log.warning("%d ролик(ов) без отчёта OCEAN-AI (документ 4, раздел 5): %s", len(no_oceanai), no_oceanai[:10])
    pred = pd.DataFrame(rows)
    names = list(dict.fromkeys(c.split(":")[0] for c in pred.columns if ":" in c))
    systems = {s: pred[[f"{s}:{t}" for t in OUTPUTS]].to_numpy(dtype=float) for s in names}
    lab = pred[[f"label_{t}" for t in OUTPUTS]].to_numpy(dtype=float)
    ok_rows = {s: ~np.isnan(a[:, :5]).any(axis=1) for s, a in systems.items()}

    metrics, boot = {}, {}
    main_arr = systems[MAIN]
    for s, arr in systems.items():
        k = ok_rows[MAIN] & ok_rows[s]                      # the videos both the system and AMLAI 2.0 have a score for
        if k.sum() < 3:
            log.warning("%s: только %d общих роликов, пропущено", s, int(k.sum()))
            continue
        cols = [j for j in range(len(OUTPUTS)) if not np.isnan(arr[k][:, j]).all() and not np.isnan(lab[k][:, j]).all()]
        metrics[s] = video_metrics(lab[k][:, cols], arr[k][:, cols], [OUTPUTS[j] for j in cols])
        b = bootstrap_pair(lab[k][:, :5], main_arr[k][:, :5], None if s == MAIN else arr[k][:, :5], args.bootstrap, args.seed)
        if s == MAIN:
            boot[s] = {"n": b["n"], "mCCC_ci": b["main_mCCC_ci"], "ccc_ci": b["main_ccc_ci"]}
        else:
            main_here = video_metrics(lab[k][:, :5], main_arr[k][:, :5], TRAITS)["mCCC"]
            boot[s] = {"n": b["n"], "mCCC_ci": b["rival_mCCC_ci"], "ccc_ci": b["rival_ccc_ci"],
                       "difference": {"point": metrics[s]["mCCC"] - main_here, "ci": b["difference_ci"],
                                      "amlai2_ahead": bool(b["difference_ci"][1] < 0),
                                      "rival_ahead": bool(b["difference_ci"][0] > 0)}}
    self_corr = {}
    if all(f"self_{t}" in labels.columns for t in TRAITS):
        sl = labels.set_index("video_name")
        for j, t in enumerate(TRAITS):
            s_self = pd.to_numeric(sl.reindex(pred["video_name"].astype(str))[f"self_{t}"], errors="coerce").to_numpy(dtype=float)
            k = ~np.isnan(s_self) & ok_rows[MAIN]
            self_corr[t] = {"n": int(k.sum()), "pearson": pearson(s_self[k], main_arr[k, j]) if k.sum() >= 3 else float("nan")}
    result = {"mode": args.mode, "n_videos": int(len(pred)), "n_labels": int(sum(len(g[0]) for g in groups)),
              "videos_without_features": missing, "videos_without_oceanai": no_oceanai, "systems": metrics,
              "bootstrap": boot, "self_corr": self_corr, "args": vars(args)}
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pred.to_csv(out.with_suffix(".pred.csv"), index=False, encoding="utf-8")
    print(table(result))
    print(f"[ok] {out} и {out.with_suffix('.pred.csv')}")
    return result


def _f(x, nd=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def table(result: dict) -> str:
    """The comparison as text: CCC per trait, mean CCC with its interval, mACC, n, and the difference to AMLAI 2.0
    (rival minus AMLAI 2.0: negative means AMLAI 2.0 is ahead) with its interval."""
    head = ["Система"] + [RU[t][:6] for t in TRAITS] + ["mCCC", "95%", "mACC", "n", "Δ к AMLAI 2.0"]
    lines = [f"Роликов: {result['n_videos']} из {result['n_labels']} ({result['mode']})", " | ".join(head)]
    for s, m in result["systems"].items():
        b = result["bootstrap"].get(s, {})
        ci = b.get("mCCC_ci", [np.nan, np.nan])
        d = b.get("difference")
        delta = "—" if d is None else f"{d['point']:+.2f} [{d['ci'][0]:+.2f}; {d['ci'][1]:+.2f}]"
        cells = ([s] + [_f(m["ccc"].get(t, np.nan)) for t in TRAITS]
                 + [_f(m["mCCC"]), f"[{_f(ci[0])}; {_f(ci[1])}]", _f(m["mACC"], 3), str(b.get("n", m["n"])), delta])
        lines.append(" | ".join(cells))
    iv = {s: m["ccc"].get("interview") for s, m in result["systems"].items() if "interview" in m["ccc"]}
    if iv:
        lines.append("собеседование, CCC: " + ", ".join(f"{s} {_f(v)}" for s, v in iv.items()))
    sc = result.get("self_corr") or {}
    if sc:
        lines.append("связь AMLAI 2.0 с самооценкой (Пирсон, для сведения): "
                     + ", ".join(f"{RU[t][:6]} {_f(v['pearson'])} (n={v['n']})" for t, v in sc.items()))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m training.ru_eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="папка с features/ от ru_extract.py")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--mode", default="test", choices=["test", "oof"])
    ap.add_argument("--ckpt", default=None, help="test: путь, список через запятую или маска best.pt (усредняются)")
    ap.add_argument("--ckpt-pattern", default=None, help="oof: путь с {fold}, например runs/A/fold{fold}/best.pt")
    ap.add_argument("--oceanai", default=None, help="папка с <код>.json от bs3 infer --backend oceanai")
    ap.add_argument("--out", required=True, help="eval.json; рядом пишется eval.pred.csv")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-missing", action="store_true", help="test: считать и без роликов, у которых нет признаков")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "test" and not args.ckpt:
        raise SystemExit("--mode test needs --ckpt")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
