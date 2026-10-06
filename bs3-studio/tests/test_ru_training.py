"""training/ru_extract.py, ru_train.py and ru_eval.py on fakes and synthetic features (docs/dataset/4_Как_обучить_AMLAI_2.0.md):
no model of the product, no GPU, no video. The extractor is driven by a fake analyzer that behaves like LongVideoAnalyzer
(a timeline, some segments without scores, the capture backend filled in `_score` order, fatal and ordinary failures); the
trainer and the evaluator run on a tiny fusion model over random features with a planted linear signal."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bs3.errors import OllamaUnavailable
from bs3.norms import TRAIT_KEYS
from training import ru_eval, ru_extract, ru_train
from training.ru_labels import LABEL_COLUMNS, TRAITS

OUTPUTS = TRAITS + ["interview"]
DIMS = {"face": 8, "audio": 6}
QUIET = lambda *a: None  # noqa: E731


# ---------------------------------------------------------------- fakes for the extractor
class FakeBackend:
    def __init__(self):
        self.inputs = []
        self.checkpoints = ["fake/best.pt"]
        self.modalities = list(DIMS)


class FakeAnalyzer:
    """Three segments, the second fails (no scores, no captured input), like the product on a broken segment."""

    def __init__(self, backend, fail_video=None, fatal_video=None):
        self.backend, self.fail_video, self.fatal_video = backend, fail_video, fatal_video
        self.calls = []

    def analyze(self, video, work_dir):
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        self.calls.append(Path(video).stem)
        if Path(video).stem == self.fail_video:
            raise RuntimeError("no segment could be analysed")
        if Path(video).stem == self.fatal_video:
            raise OllamaUnavailable("Ollama request failed: <urlopen error [Errno 111] Connection refused>")
        scores = {k: 0.4 for k in TRAIT_KEYS} | {"interview": 0.5}
        timeline = []
        for i, (s, e) in enumerate([(0, 20), (20, 40), (40, 55)], 1):
            if i == 2:
                timeline.append({"segment": i, "start": s, "end": e, "scores": None, "error": "boom"})
                continue
            self.backend.inputs.append({m: torch.ones(d) * i for m, d in DIMS.items()})       # what _score would capture
            timeline.append({"segment": i, "start": s, "end": e, "scores": scores, "transcript": f"речь {i}",
                             "behavior_description": f"behaviour {i}"})
        return {"scores": scores, "timeline": timeline, "duration_sec": 55.0, "segments": 3, "transcript": "речь"}


def _videos(folder: Path, codes):
    vids = folder / "work"
    vids.mkdir(exist_ok=True)
    for code in codes:
        (vids / f"{code}.mp4").write_bytes(b"")
    return vids


def test_extract_aligns_inputs_with_scored_segments_and_skips_finished_videos():
    be = FakeBackend()
    an = FakeAnalyzer(be, fail_video="R0003")
    with tempfile.TemporaryDirectory(prefix="bs3_ru_extract_") as d:
        vids = _videos(Path(d), ("R0001", "R0002", "R0003"))
        labels = pd.DataFrame({"video_name": ["R0001", "R0002", "R0003", "R0004"]})     # R0004 has no file
        out = Path(d) / "feat"
        counts = ru_extract.run(vids, labels, out, an, be, log_fn=QUIET)
        assert counts == {"done": 2, "skipped": 0, "failed": 2, "n_table": 2, "missing": 2}
        rec = torch.load(out / "features" / "R0001.pt")
        assert rec["modalities"] == ["face", "audio"] and tuple(rec["x"]["face"].shape) == (2, 8)
        assert [s["segment"] for s in rec["segments"]] == [1, 3] and rec["segments"][1]["transcript"] == "речь 3"
        assert float(rec["x"]["audio"][1, 0]) == 3.0                   # the input of segment 3, not of the failed one
        assert rec["segments_planned"] == 3 and rec["segments_failed"] == 1 and rec["segment_errors"] == ["boom"]
        assert rec["video_scores_amlai1"]["interview"] == 0.5 and rec["bs3_version"]
        table = pd.read_csv(out / "amlai1_scores.csv")
        assert list(table["video_name"]) == ["R0001", "R0002"] and list(table["segments"]) == [2, 2]
        assert "non-neuroticism" in table.columns and "emotional_stability" not in table.columns   # names as in labels.csv
        assert list(table["segments_failed"]) == [1, 1]
        failed = (out / "extract_failed.txt").read_text(encoding="utf-8")
        assert "R0003\tno segment" in failed and "R0004\tнет файла" in failed and "R0001" not in failed
        assert not list((out / "tmp").iterdir())                        # work dirs removed
        # a rerun skips what is done; a video that now succeeds leaves the failures file
        an.fail_video = None
        counts2 = ru_extract.run(vids, labels, out, an, be, log_fn=QUIET)
        assert counts2["skipped"] == 2 and counts2["done"] == 1 and counts2["missing"] == 1
        assert (out / "extract_failed.txt").read_text(encoding="utf-8").startswith("R0004\t")


def test_extract_stops_on_a_fatal_failure_and_after_five_in_a_row():
    with tempfile.TemporaryDirectory(prefix="bs3_ru_extract_") as d:
        codes = [f"R{i:04d}" for i in range(1, 9)]
        vids = _videos(Path(d), codes)
        labels = pd.DataFrame({"video_name": codes})
        be = FakeBackend()
        an = FakeAnalyzer(be, fatal_video="R0002")                      # Ollama down: the pass must not go on
        try:
            ru_extract.run(vids, labels, Path(d) / "a", an, be, log_fn=QUIET)
        except SystemExit as e:
            assert "фатальный" in str(e.code)
        else:
            raise AssertionError("a fatal failure must stop the pass")
        assert an.calls == ["R0001", "R0002"]
        failed = (Path(d) / "a" / "extract_failed.txt").read_text(encoding="utf-8")
        assert failed.count("\n") == 7 and "R0002\tOllama" in failed and "R0008\tне обработан" in failed
        assert (Path(d) / "a" / "amlai1_scores.csv").is_file()        # the table is still rebuilt

        class AlwaysBroken(FakeAnalyzer):
            def analyze(self, video, work_dir):
                self.calls.append(Path(video).stem)
                raise RuntimeError("ffmpeg failed")

        an2 = AlwaysBroken(FakeBackend())
        try:
            ru_extract.run(vids, labels, Path(d) / "b", an2, an2.backend, log_fn=QUIET)
        except SystemExit as e:
            assert "подряд" in str(e.code)
        else:
            raise AssertionError("five failures in a row must stop the pass")
        assert len(an2.calls) == 5


def test_extract_refuses_misaligned_inputs():
    be = FakeBackend()

    class Misaligned(FakeAnalyzer):
        def analyze(self, video, work_dir):
            res = super().analyze(video, work_dir)
            self.backend.inputs.append({m: torch.zeros(d) for m, d in DIMS.items()})     # one input too many
            return res

    with tempfile.TemporaryDirectory(prefix="bs3_ru_extract_") as d:
        v = Path(d) / "R0009.mp4"
        v.write_bytes(b"")
        try:
            ru_extract.extract_video(v, Misaligned(be), be, Path(d))
        except RuntimeError as e:
            assert "captured inputs" in str(e)
        else:
            raise AssertionError("misaligned inputs must raise")


# ---------------------------------------------------------------- synthetic features for trainer and evaluator
def synthetic_store(folder: Path, n_videos=40, seed=0, drop_audio_every=0):
    rng = np.random.default_rng(seed)
    (folder / "features").mkdir(parents=True)
    rows = []
    for i in range(n_videos):
        name = f"R{1000 + i:04d}"
        n_seg = int(rng.integers(2, 5))
        x = {m: torch.tensor(rng.normal(0, 1, (n_seg, d)), dtype=torch.float32) for m, d in DIMS.items()}
        signal = float(x["face"].mean()) * 0.15 + float(x["audio"].mean()) * 0.1
        label = {t: float(np.clip(0.5 + signal + rng.normal(0, 0.03), 0.05, 0.95)) for t in TRAITS}
        label["interview"] = float(np.clip(0.5 + signal, 0.05, 0.95))
        amlai1 = {k: float(np.clip(0.35 + 0.6 * (label[t] - 0.5) + rng.normal(0, 0.05), 0, 1)) for k, t in zip(TRAIT_KEYS, TRAITS)}
        amlai1["interview"] = 0.4
        mods = list(DIMS)
        if drop_audio_every and i % drop_audio_every == 0:
            x.pop("audio")
            mods = ["face"]
        segs = [{"segment": j + 1, "start": 20.0 * j, "end": 20.0 * (j + 1), "transcript": "", "behavior_description": "",
                 "scores_amlai1": amlai1} for j in range(n_seg)]
        torch.save({"video": name, "duration_sec": 20.0 * n_seg, "segments": segs, "x": x, "video_scores_amlai1": amlai1,
                    "modalities": mods, "checkpoints": [], "bs3_version": "test", "created": "now"},
                   folder / "features" / f"{name}.pt")
        rows.append({"video_name": name, **label, "n_ratings": 3,
                     **{f"self_{t}": float(np.clip(label[t] + rng.normal(0, 0.2), 0, 1)) for t in TRAITS},
                     **{f"sd_{t}": 0.1 for t in TRAITS},
                     "cluster": f"c{i % 4}", "split": "test" if i < 8 else "train", "fold": pd.NA if i < 8 else (i % 5) + 1})
    labels = pd.DataFrame(rows)[LABEL_COLUMNS]
    labels.to_csv(folder / "labels.csv", index=False)
    return labels


def _train(folder: Path, fold: int, out: Path, init: str | None = None, extra=()):
    argv = ["--data", str(folder), "--labels", str(folder / "labels.csv"), "--fold", str(fold), "--out", str(out),
            "--epochs", "4", "--patience", "3", "--batch-size", "16", "--lr", "3e-3", "--hidden-dim", "16", "--out-dim", "16",
            "--heads", "2", "--seed", str(fold), "--allow-cpu", *extra]
    if init:
        argv += ["--init", init]
    args = ru_train.build_parser().parse_args(argv)
    return ru_train.train(args)


def test_train_fold_writes_product_compatible_checkpoint_and_never_sees_the_test():
    with tempfile.TemporaryDirectory(prefix="bs3_ru_train_") as d:
        folder = Path(d)
        labels = synthetic_store(folder)
        out = folder / "runs" / "fold1"
        res = _train(folder, 1, out)
        ckpt = torch.load(out / "best.pt")
        assert set(ckpt) >= {"state_dict", "config", "modalities", "targets", "epoch", "dev"}
        assert ckpt["modalities"] == ["face", "audio"] and ckpt["targets"] == OUTPUTS and ckpt["config"]["n_traits"] == 6
        val = pd.read_csv(out / "val_pred.csv")
        fold1 = set(labels[(labels["split"] == "train") & (labels["fold"] == 1)]["video_name"])
        assert set(val["video_name"]) == fold1 and (val["fold"] == 1).all()
        assert not (set(val["video_name"]) & set(labels[labels["split"] == "test"]["video_name"]))
        assert res["train_videos"] + res["val_videos"] == int((labels["split"] == "train").sum())
        assert res["best_epoch"] >= 1 and len(res["history"]) <= 4
        assert json.loads((out / "result.json").read_text(encoding="utf-8"))["fold"] == 1
        # variant A: --init takes the architecture from the checkpoint (hidden 16) even when the run asks for 32
        res2 = _train(folder, 2, folder / "runs" / "fold2", init=str(out / "best.pt"), extra=("--hidden-dim", "32"))
        assert res2["config"]["hidden_dim"] == 16 and res2["init"] == str(out / "best.pt")
        bad = dict(ckpt)
        bad["config"] = dict(ckpt["config"], n_traits=5)
        torch.save(bad, folder / "bad.pt")
        try:
            _train(folder, 2, folder / "runs" / "fold2b", init=str(folder / "bad.pt"))
        except SystemExit as e:
            assert "outputs" in str(e.code)
        else:
            raise AssertionError("an --init with other outputs must be refused")


def test_train_refuses_a_store_with_mixed_modalities():
    with tempfile.TemporaryDirectory(prefix="bs3_ru_train_") as d:
        folder = Path(d)
        synthetic_store(folder, drop_audio_every=3)
        try:
            _train(folder, 1, folder / "runs" / "fold1")
        except SystemExit as e:
            assert "модальности" in str(e.code) and ".pt" in str(e.code)
        else:
            raise AssertionError("mixed modalities must be refused")


def test_eval_test_and_oof_with_rivals_calibration_and_missing_reports():
    with tempfile.TemporaryDirectory(prefix="bs3_ru_eval_") as d:
        folder = Path(d)
        labels = synthetic_store(folder)
        for k in range(1, 6):
            _train(folder, k, folder / "runs" / f"fold{k}")
        # OCEAN-AI reports in the shape of bs3 infer: for 6 of 8 test videos and for some train videos; one file is a list
        oce = folder / "oceanai"
        oce.mkdir()
        lab_ix = labels.set_index("video_name")
        for name in list(labels["video_name"][:6]) + list(labels["video_name"][8:30]):
            lab = lab_ix.loc[name]
            rep = {"input": f"/x/{name}.mp4", "traits": {k: {"score": float(np.clip(0.7 + 0.3 * (lab[t] - 0.5), 0, 1))}
                                                         for k, t in zip(TRAIT_KEYS, TRAITS)}}
            (oce / f"{name}.json").write_text(json.dumps(rep), encoding="utf-8")
        first = labels["video_name"][0]
        other = {"input": "/x/R9999.mp4", "traits": {k: {"score": 0.11} for k in TRAIT_KEYS}}
        mine = json.loads((oce / f"{first}.json").read_text(encoding="utf-8"))
        (oce / f"{first}.json").write_text(json.dumps([other, mine]), encoding="utf-8")       # a two-report file
        out = folder / "eval_test.json"
        ckpts = ",".join(str(folder / "runs" / f"fold{k}" / "best.pt") for k in (1, 2)) + "," + str(folder / "runs" / "fold[345]" / "best.pt")
        argv = ["--data", str(folder), "--labels", str(folder / "labels.csv"), "--mode", "test", "--ckpt", ckpts,
                "--oceanai", str(oce), "--out", str(out), "--bootstrap", "200"]
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            res = ru_eval.evaluate(ru_eval.build_parser().parse_args(argv))
        assert res["n_videos"] == 8 and res["n_labels"] == 8 and res["videos_without_features"] == []
        assert set(res["systems"]) == {"AMLAI 2.0", "постоянный ответ", "AMLAI 1.0", "AMLAI 1.0 с калибровкой",
                                       "OCEAN-AI", "OCEAN-AI с калибровкой"}
        text = printed.getvalue()
        assert "Δ к AMLAI 2.0" in text and "собеседование" in text and "самооценкой" in text
        pred = pd.read_csv(out.with_suffix(".pred.csv"))
        assert len(pred) == 8 and "AMLAI 2.0:openness" in pred.columns
        assert abs(pred.loc[pred["video_name"] == first, "OCEAN-AI:openness"].iloc[0]
                   - float(np.clip(0.7 + 0.3 * (lab_ix.loc[first, "openness"] - 0.5), 0, 1))) < 1e-9     # the right report of the list
        # no imputation: OCEAN-AI is compared on its 6 videos, AMLAI 2.0 on all 8; the constant answer has CCC 0
        assert res["bootstrap"]["OCEAN-AI"]["n"] == 6 and res["bootstrap"]["AMLAI 2.0"]["n"] == 8
        assert res["systems"]["OCEAN-AI"]["n"] == 6 and sorted(res["videos_without_oceanai"]) == sorted(labels["video_name"][6:8])
        assert abs(res["systems"]["постоянный ответ"]["mCCC"]) < 1e-9
        for s in res["systems"]:
            b = res["bootstrap"][s]
            assert len(b["mCCC_ci"]) == 2 and set(b["ccc_ci"]) == set(TRAITS)
            if s != "AMLAI 2.0":
                assert set(b["difference"]) == {"point", "ci", "amlai2_ahead", "rival_ahead"}
        assert set(res["self_corr"]) == set(TRAITS) and res["self_corr"]["openness"]["n"] == 8
        # calibration on the fit set: CCC equals Pearson by construction
        fit = labels[labels["split"] == "train"]
        a1 = np.array([[torch.load(folder / "features" / f"{n}.pt")["video_scores_amlai1"][k] for k in TRAIT_KEYS] for n in fit["video_name"]])
        y = fit[TRAITS].to_numpy(dtype=float)
        cal = ru_eval.calibrate(a1, y, a1)
        for j in range(5):
            assert abs(ru_train.ccc(y[:, j], cal[:, j]) - ru_train.pearson(y[:, j], cal[:, j])) < 1e-9
        # out-of-fold: every train video once, scored by its own fold's model
        out2 = folder / "eval_oof.json"
        argv = ["--data", str(folder), "--labels", str(folder / "labels.csv"), "--mode", "oof",
                "--ckpt-pattern", str(folder / "runs" / "fold{fold}" / "best.pt"), "--out", str(out2), "--bootstrap", "100"]
        with contextlib.redirect_stdout(io.StringIO()):
            res2 = ru_eval.evaluate(ru_eval.build_parser().parse_args(argv))
        pred2 = pd.read_csv(out2.with_suffix(".pred.csv"))
        assert res2["n_videos"] == 32 and pred2["video_name"].is_unique
        assert set(pred2["video_name"]) == set(labels[labels["split"] == "train"]["video_name"])
        assert "OCEAN-AI" not in res2["systems"]                     # no --oceanai given
        # a test video without a feature file stops the test unless --allow-missing
        (folder / "features" / f"{labels['video_name'][1]}.pt").unlink()
        argv = ["--data", str(folder), "--labels", str(folder / "labels.csv"), "--mode", "test",
                "--ckpt", str(folder / "runs" / "fold*" / "best.pt"), "--out", str(folder / "e3.json"), "--bootstrap", "50"]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ru_eval.evaluate(ru_eval.build_parser().parse_args(argv))
        except SystemExit as e:
            assert "без признаков" in str(e.code)
        else:
            raise AssertionError("a missing test video must stop the run")
        with contextlib.redirect_stdout(io.StringIO()):
            res3 = ru_eval.evaluate(ru_eval.build_parser().parse_args(argv + ["--allow-missing"]))
        assert res3["n_videos"] == 7 and res3["videos_without_features"] == [labels["video_name"][1]]
        try:
            ru_eval.resolve_checkpoints(str(folder / "runs" / "nowhere" / "best.pt"))
        except SystemExit as e:
            assert "ничего нет" in str(e.code)
        else:
            raise AssertionError("a wrong --ckpt path must be named")
