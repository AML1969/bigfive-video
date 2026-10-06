"""Train AMLAI 2.0 on the own Russian set (docs/dataset/4_Как_обучить_AMLAI_2.0.md, sections 4 and 6): the fusion model of
AMLAI 1.0 on the segment features of training/ru_extract.py with the video labels of labels.csv, one fold at a time.

  python -m training.ru_train --data ~/data/ru_v1 --labels /mnt/d/ru_set_v1/labels.csv --fold 1 --seed 1 \
      --lr 1e-5 --init ~/bs/mm_runs_seeds/seed1/best.pt --out ~/bs/amlai2_runs/A_lr1e-5/fold1      # variant A
  python -m training.ru_train --data ~/data/ru_v1 --labels ... --fold 1 --seed 1 --lr 1e-4 --out .../B/fold1   # variant B

Samples are segments, the label is the video's; a segment weighs 1 / (segments of its video) so long videos do not
dominate. The validation fold is scored per video (duration-weighted mean of its segments, as the product does) and the
best epoch / early stopping follow the mean CCC of the five traits. Test videos (split == test) are never read.
With --init the architecture comes from that checkpoint (only dropout and the dimensions of the features are taken from
the run), so the weights always fit. best.pt has the keys of training/mm_train.py (state_dict, config, modalities,
targets, epoch, dev), so the product loads it with --mm-ckpt. val_pred.csv next to it holds the per-video predictions
of the validation fold."""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bs3.mm.model import ModelConfig, PersonalityFusionModel

from .ru_labels import TRAITS

log = logging.getLogger("bs.ru_train")
TARGET_SETS = {"big5": TRAITS, "big5+interview": TRAITS + ["interview"]}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- metrics (per video)
def ccc(t: np.ndarray, p: np.ndarray) -> float:
    mt, mp = t.mean(), p.mean()
    return float(2 * ((t - mt) * (p - mp)).mean() / (t.var() + p.var() + (mt - mp) ** 2 + 1e-12))


def pearson(t: np.ndarray, p: np.ndarray) -> float:
    if len(t) < 2 or t.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(t, p)[0, 1])


def video_metrics(t: np.ndarray, p: np.ndarray, targets: list[str]) -> dict:
    """CCC, Pearson and ACC = 1 - MAE per output; mCCC / mACC over the five traits only (interview is reported apart)."""
    out = {"n": int(len(t)), "ccc": {}, "pearson": {}, "acc": {}}
    for i, k in enumerate(targets):
        out["ccc"][k] = ccc(t[:, i], p[:, i])
        out["pearson"][k] = pearson(t[:, i], p[:, i])
        out["acc"][k] = float(1 - np.abs(t[:, i] - p[:, i]).mean())
    five = [k for k in targets if k in TRAITS]
    out["mCCC"] = float(np.mean([out["ccc"][k] for k in five])) if five else float("nan")
    out["mACC"] = float(np.mean([out["acc"][k] for k in five])) if five else float("nan")
    return out


def segment_durations(segments: list[dict]) -> np.ndarray:
    """Duration of every segment; a segment without times weighs like one of 20 seconds."""
    out = []
    for s in segments:
        a, b = s.get("start"), s.get("end")
        out.append(max(1e-3, float(b) - float(a)) if a is not None and b is not None else 20.0)
    return np.array(out, dtype=float)


# ---------------------------------------------------------------- data
class SegmentTable:
    """The segments of a set of videos: features per modality [N, D], the label of the video of each segment [N, T],
    the segment weight 1 / segments-of-its-video, the duration of each segment and the video index. Every feature file
    must carry the same modalities (one pass of ru_extract); a file that differs stops the run with its name."""

    def __init__(self, data: Path, labels: pd.DataFrame, modalities: list[str] | None, targets: list[str]):
        self.targets = targets
        feats: dict[str, list] = {}
        ys, w, dur, vid, names, missing = [], [], [], [], [], []
        mods = modalities
        for row in labels.to_dict("records"):
            name = str(row["video_name"])
            p = data / "features" / f"{name}.pt"
            if not p.is_file():
                missing.append(name)
                continue
            d = torch.load(p, map_location="cpu")
            if mods is None:
                mods = list(d["modalities"])
            absent = [m for m in mods if m not in d["x"]]
            extra = [m for m in d["modalities"] if m not in mods] if modalities is None else []
            if absent or extra:
                raise SystemExit(f"{p.name}: модальности {d['modalities']} не совпадают с {mods} (признаки из другого прохода?)")
            n = len(d["segments"])
            if n == 0 or any(int(d["x"][m].shape[0]) != n for m in mods):
                raise SystemExit(f"{p.name}: {n} отрезков, а признаков {[int(d['x'][m].shape[0]) for m in mods]}")
            for m in mods:
                feats.setdefault(m, []).append(d["x"][m])
            ys += [[float(row[t]) for t in targets]] * n
            w += [1.0 / n] * n
            dur += segment_durations(d["segments"]).tolist()
            vid += [len(names)] * n
            names.append(name)
        if missing:
            log.warning("%d video(s) of labels.csv have no feature file, left out: %s", len(missing), missing[:5])
        if not names:
            raise SystemExit("no feature files for these videos: run training/ru_extract.py first")
        self.modalities = mods
        self.x = {m: torch.cat(v).float() for m, v in feats.items()}
        self.y = torch.tensor(ys, dtype=torch.float32)
        assert all(int(v.shape[0]) == len(self.y) for v in self.x.values())
        self.w = torch.tensor(w, dtype=torch.float32)
        self.dur = np.array(dur, dtype=float)
        self.vid = np.array(vid)
        self.names = names
        self.missing = missing

    def __len__(self):
        return len(self.y)

    def batch(self, idx):
        return {m: v[idx] for m, v in self.x.items()}, self.y[idx], self.w[idx]

    def per_video(self, seg_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Duration-weighted mean of the segment predictions and the label, per video (in the order of self.names)."""
        n_videos = len(self.names)
        preds = np.zeros((n_videos, seg_pred.shape[1]))
        labels = np.zeros((n_videos, seg_pred.shape[1]))
        y = self.y.numpy()
        for v in range(n_videos):
            m = self.vid == v
            wgt = self.dur[m] / self.dur[m].sum()
            preds[v] = (seg_pred[m] * wgt[:, None]).sum(axis=0)
            labels[v] = y[m][0]
        return labels, preds


def label_frame(labels: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """labels.csv with the target columns as floats; rows without a label are dropped."""
    df = labels.copy()
    for t in targets:
        df[t] = pd.to_numeric(df[t], errors="coerce")
    return df.dropna(subset=targets)


@torch.no_grad()
def predict(model, table: SegmentTable, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(table), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(table)))
        feats, _, _ = table.batch(idx)
        out.append(model({m: v.to(device) for m, v in feats.items()}).cpu())
    return torch.cat(out).numpy()


def pick_device(allow_cpu: bool) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        return "cpu"
    raise SystemExit("no CUDA device; pass --allow-cpu to train on the CPU (slow)")


def build_config(args, dims: dict, n_traits: int) -> tuple[ModelConfig, dict | None]:
    """The model configuration of the run: from --init when given (its architecture, these features' dimensions, this
    run's dropout), else from the command line. Returns (config, checkpoint or None)."""
    if not args.init:
        return ModelConfig(modality_dims=dims, hidden_dim=args.hidden_dim, num_heads=args.heads, out_dim=args.out_dim,
                           dropout=args.dropout, n_traits=n_traits), None
    ckpt = torch.load(Path(args.init).expanduser(), map_location="cpu")
    if list(ckpt["modalities"]) != list(dims):
        raise SystemExit(f"--init {args.init}: modalities {ckpt['modalities']} differ from the features {list(dims)}")
    c = dict(ckpt["config"])
    if c.get("n_traits") != n_traits:
        raise SystemExit(f"--init {args.init}: {c.get('n_traits')} outputs, the run needs {n_traits} (--targets)")
    if c.get("modality_dims") != dims:
        raise SystemExit(f"--init {args.init}: feature dimensions {c.get('modality_dims')} differ from these features {dims}")
    for key, flag in (("hidden_dim", "--hidden-dim"), ("num_heads", "--heads"), ("out_dim", "--out-dim")):
        given = getattr(args, {"hidden_dim": "hidden_dim", "num_heads": "heads", "out_dim": "out_dim"}[key])
        if given != c.get(key):
            log.warning("%s %s ignored: the architecture comes from --init (%s = %s)", flag, given, key, c.get(key))
    c.update(modality_dims=dims, n_traits=n_traits, dropout=args.dropout)
    return ModelConfig(**c), ckpt


def train(args) -> dict:
    seed_everything(args.seed)
    device = pick_device(args.allow_cpu)
    targets = TARGET_SETS[args.targets]
    labels = label_frame(pd.read_csv(args.labels, encoding="utf-8"), targets)
    folds = labels["fold"].astype("Int64")
    train_rows = labels[(labels["split"] == "train") & (folds != args.fold).fillna(False)]
    val_rows = labels[(labels["split"] == "train") & (folds == args.fold).fillna(False)]
    if val_rows.empty or train_rows.empty:
        raise SystemExit(f"fold {args.fold}: {len(train_rows)} training and {len(val_rows)} validation videos")
    mods = [m for m in args.modalities.split(",") if m] if args.modalities else None
    data = Path(args.data).expanduser()
    tr = SegmentTable(data, train_rows, mods, targets)
    va = SegmentTable(data, val_rows, tr.modalities, targets)
    test_names = set(labels.loc[labels["split"] == "test", "video_name"].astype(str))
    assert not (set(tr.names) | set(va.names)) & test_names, "a test video reached training"
    dims = {m: int(tr.x[m].shape[1]) for m in tr.modalities}
    cfg, ckpt = build_config(args, dims, len(targets))
    model = PersonalityFusionModel(cfg).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["state_dict"])
        log.info("initialised from %s (epoch %s)", args.init, ckpt.get("epoch"))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("fold %d: train %d videos / %d segments, val %d videos / %d segments, outputs %s, device %s",
             args.fold, len(tr.names), len(tr), len(va.names), len(va), targets, device)

    best, best_epoch, patience, history = -9.0, -1, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr))
        total, nb = 0.0, 0
        for i in range(0, len(tr), args.batch_size):
            feats, y, w = tr.batch(perm[i:i + args.batch_size])
            pred = model({m: v.to(device) for m, v in feats.items()})
            err = torch.abs(pred - y.to(device)).mean(dim=1)            # MAE per segment
            w = w.to(device)
            loss = (err * w).sum() / w.sum()                            # a video weighs 1 whatever its length
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        y_true, y_pred = va.per_video(predict(model, va, device))
        m = video_metrics(y_true, y_pred, targets)
        score = m["mCCC"]
        sched.step(score)
        history.append({"epoch": epoch, "train_loss": total / max(1, nb), "val_mCCC": score, "val_mACC": m["mACC"],
                        "lr": opt.param_groups[0]["lr"]})
        improved = score > best
        log.info("epoch %3d loss %.4f | val mCCC %.4f mACC %.4f %s", epoch, total / max(1, nb), score, m["mACC"], "*" if improved else "")
        if improved:
            best, best_epoch, patience = score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__, "modalities": list(cfg.modality_dims),
                        "targets": targets, "epoch": epoch, "dev": m}, out_dir / "best.pt")
        else:
            patience += 1
            if patience >= args.patience:
                log.info("early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    saved = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(saved["state_dict"])
    y_true, y_pred = va.per_video(predict(model, va, device))
    m = video_metrics(y_true, y_pred, targets)
    pred_df = pd.DataFrame(y_pred, columns=[f"pred_{t}" for t in targets])
    pred_df.insert(0, "video_name", va.names)
    for i, t in enumerate(targets):
        pred_df[f"label_{t}"] = y_true[:, i]
    pred_df["fold"] = args.fold
    pred_df.to_csv(out_dir / "val_pred.csv", index=False, encoding="utf-8")
    result = {"fold": args.fold, "targets": targets, "modalities": list(cfg.modality_dims), "init": args.init,
              "train_videos": len(tr.names), "train_segments": len(tr), "val_videos": len(va.names), "val_segments": len(va),
              "videos_without_features": tr.missing + va.missing, "best_epoch": best_epoch, "epochs_run": len(history),
              "seconds": round(time.time() - t0, 1), "val": m, "config": cfg.__dict__, "args": vars(args), "history": history}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("DONE fold %d | best epoch %d | val mCCC %.4f mACC %.4f | %s", args.fold, best_epoch, m["mCCC"], m["mACC"], out_dir)
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m training.ru_train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="папка с features/ от ru_extract.py")
    ap.add_argument("--labels", required=True, help="labels.csv набора")
    ap.add_argument("--fold", type=int, required=True, help="проверочная часть 1..5; обучение — остальные")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="начальные веса (best.pt AMLAI 1.0) — вариант А; без него — с нуля")
    ap.add_argument("--targets", default="big5+interview", choices=list(TARGET_SETS))
    ap.add_argument("--modalities", default=None, help="например face,audio,text,behavior (по умолчанию все из признаков)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--out-dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-cpu", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(args)
    return 0


if __name__ == "__main__":
    main()
