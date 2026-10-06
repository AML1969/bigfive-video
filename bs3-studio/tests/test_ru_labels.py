"""training/ru_labels.py on a synthetic set (docs/dataset/5_Как_разметить_набор.md): the scoring keys, the validity and rater
rules, the one-way ICC against its theoretical value, the split by people or by whole clusters (5 / 6 / 8 clusters, unequal
clusters, videos without a cluster), a second version of the set that keeps the old split (in memory and through the
command line), and the files the script writes. No model, no video: tables only."""
from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from training import ru_labels as rl
from training.ru_labels import CARD, IPIP, TRAITS, Labelling, icc_oneway, rater_correlations, score_card, score_ipip

QUIET = {"log": lambda *a: None}


def synthetic(n_videos=320, raters_per_video=4, noise_sd=1.3, seed=1, n_clusters=8, cluster_sizes=None):
    """Participants rate each other's videos across clusters; the truth is one 1..7 profile per video.
    `cluster_sizes`: optional list of shares (sum 1) for unequal clusters."""
    rng = np.random.default_rng(seed)
    codes = [f"R{1000 + i:04d}" for i in range(n_videos)]
    if cluster_sizes:
        bounds = np.cumsum([0] + list(cluster_sizes)) * n_videos
        clusters = np.array([f"c{np.searchsorted(bounds, i, side='right') - 1}" for i in range(n_videos)])
    else:
        clusters = np.array([f"c{i % n_clusters}" for i in range(n_videos)])
    true = rng.normal(4, 1.0, size=(n_videos, 5)).clip(1, 7)
    videos = pd.DataFrame({"code": codes, "cluster": clusters, "status": "принят",
                           "glasses": rng.choice(["да", "нет"], n_videos), "beard": rng.choice(["да", "нет"], n_videos)})
    videos.loc[:5, "status"] = "пересъёмка"
    videos.loc[3, "code"] = "р1003"                                # Cyrillic letter, to be normalised
    videos.loc[8, "glasses"] = "Да"                                # spelling the labeller must tolerate
    q = {}
    for t_i, t in enumerate(TRAITS):
        d, rv = IPIP[t]
        base = 3 + (true[:, t_i] - 4) * 0.3 + rng.normal(0, 0.9, n_videos)
        for i in d:
            q[f"q{i:02d}"] = np.rint(base + rng.normal(0, 0.8, n_videos)).clip(1, 5)
        for i in rv:
            q[f"q{i:02d}"] = np.rint(6 - base + rng.normal(0, 0.8, n_videos)).clip(1, 5)
    s = {}
    for t, (d, rv) in CARD.items():
        t_i = TRAITS.index(t)
        s[f"s{d:02d}"] = np.rint(true[:, t_i] + rng.normal(0, 1.2, n_videos)).clip(1, 7)
        s[f"s{rv:02d}"] = np.rint(8 - true[:, t_i] + rng.normal(0, 1.2, n_videos)).clip(1, 7)
    self_ = pd.DataFrame({"code": codes, "date": "2026-11-01", **q, **s})
    self_ = pd.concat([self_, self_.iloc[[7]].assign(date="2026-11-03")])    # one duplicate answer
    rows = []
    for j, rater in enumerate(codes):
        pool = [i for i in range(n_videos) if clusters[i] != clusters[j] and i != j]
        for i in rng.choice(pool, size=raters_per_video, replace=False):
            row = {"rater": rater, "video": codes[i], "date": "2026-11-10", "known": "нет", "interview": 0,
                   "glasses": videos.loc[i, "glasses"], "beard": videos.loc[i, "beard"], "confidence": 2,
                   "noise": "нет", "comment": ""}
            for t, (d, rv) in CARD.items():
                t_i = TRAITS.index(t)
                row[f"s{d:02d}"] = int(np.rint(true[i, t_i] + rng.normal(0, noise_sd)).clip(1, 7))
                row[f"s{rv:02d}"] = int(np.rint(8 - true[i, t_i] + rng.normal(0, noise_sd)).clip(1, 7))
            row["interview"] = int(np.rint(true[i, 2] * 0.5 + true[i, 3] * 0.5 + rng.normal(0, noise_sd)).clip(1, 7))
            rows.append(row)
    ratings = pd.DataFrame(rows)
    for rater in codes[10:13]:                                      # three flat raters: 4 everywhere
        ratings.loc[ratings["rater"] == rater, [f"s{i:02d}" for i in range(1, 11)]] = 4
    ratings.loc[ratings["rater"] == codes[20], "known"] = "знаком(а) лично"
    ratings.loc[ratings.index[:3], "noise"] = "плохой звук"
    own = ratings.iloc[[0]].copy()                                  # one rater scoring their own video
    own["rater"] = own["video"]
    ratings = pd.concat([ratings, own], ignore_index=True)
    return videos, self_, ratings, true


def test_keys_and_code_normalisation():
    probe = {f"s{CARD[t][0]:02d}": 6 for t in TRAITS} | {f"s{CARD[t][1]:02d}": 2 for t in TRAITS}
    assert all(abs(v - 5 / 6) < 1e-9 for v in score_card(pd.Series(probe)).values())     # 6 direct, 2 reverse -> 5 of 6
    probe50 = {f"q{i:02d}": 5 for i in range(1, 51)}
    for _, rv in IPIP.values():
        for i in rv:
            probe50[f"q{i:02d}"] = 1
    assert all(abs(v - 1) < 1e-9 for v in score_ipip(pd.Series(probe50)).values())
    three_missing = dict(probe50)
    for i in (1, 11, 21):
        three_missing[f"q{i:02d}"] = np.nan
    out = score_ipip(pd.Series(three_missing))
    assert np.isnan(out["extraversion"]) and all(not np.isnan(out[t]) for t in TRAITS if t != "extraversion")
    assert rl.norm_code(" р0147 ") == "R0147"
    assert sorted(sum([d + r for d, r in IPIP.values()], [])) == list(range(1, 51))


def test_rater_correlation_is_centred():
    """A rater who gives every video the profile of the trait means: high raw correlation, none after centring."""
    truth = pd.DataFrame({t: m + np.linspace(-0.1, 0.1, 8) for t, m in zip(TRAITS, (0.4, 0.6, 0.3, 0.8, 0.5))},
                         index=[f"V{i}" for i in range(8)])
    rows = [{"rater": f"O{j}", "video": vid, **truth.loc[vid].to_dict()} for j in range(3) for vid in truth.index]
    profile = truth.mean() + pd.Series([0.05, -0.05, 0.05, -0.05, 0.0], index=TRAITS)
    rows += [{"rater": "X", "video": vid, **profile.to_dict()} for vid in truth.index]
    built = pd.DataFrame(rows)
    raw = float(rater_correlations(built, centre=False)["X"])
    centred = float(rater_correlations(built)["X"])
    assert raw > 0.5 and centred <= 0, (raw, centred)
    exact = pd.DataFrame(rows[:24] + [{"rater": "X2", "video": vid, **truth.mean().to_dict()} for vid in truth.index])
    assert rater_correlations(exact)["X2"] == 0.0                   # no variation after centring = no link, the sign fires


def test_icc_degenerate_cases_give_nan_not_a_crash():
    long = pd.DataFrame({"video": ["a", "a", "b", "b", "c", "c"], "flat": [0.5] * 6, "one": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    flat = icc_oneway(long, "video", "flat")
    assert all(np.isnan(flat[k]) for k in ("icc1", "icc1k", "ceiling"))
    single = icc_oneway(long.drop_duplicates("video"), "video", "one")      # one rating per video: nothing repeated
    assert np.isnan(single["icc1"])
    assert rl.ratings_for(0.3) == 10 and rl.ratings_for(np.nan) is None and rl.ratings_for(0.0) is None


def test_labelling_on_a_synthetic_set():
    videos, self_, ratings, true = synthetic()
    job = Labelling(videos, self_, ratings, **QUIET)
    lab = job.labels
    assert list(lab.columns) == rl.LABEL_COLUMNS
    assert lab["video_name"].is_unique and lab["n_ratings"].min() >= 2
    assert set(lab["split"]) == {"train", "test"} and job.split_mode == "clusters"
    assert lab.loc[lab["split"] == "test", "fold"].isna().all() and lab.loc[lab["split"] == "train", "fold"].notna().all()
    share = (lab["split"] == "test").mean()
    assert 0.10 <= share <= 0.22, share               # one of eight equal clusters: the share closest to 15-20%
    folds = lab.loc[lab["split"] == "train", "fold"].value_counts()
    assert len(folds) == 5 and folds.min() > 0
    assert lab.groupby("cluster")["split"].nunique().max() == 1                    # whole clusters (8 of them)
    assert lab[lab["split"] == "train"].groupby("cluster")["fold"].nunique().max() == 1
    tr = pd.DataFrame(true, columns=TRAITS, index=videos["code"].map(rl.norm_code))
    m = lab.set_index("video_name").join(tr.add_prefix("true_"))
    assert all(np.corrcoef(m[t], m[f"true_{t}"])[0, 1] > 0.7 for t in TRAITS)
    assert all(0.05 < m[[f"self_{t}", f"true_{t}"]].dropna().corr().iloc[0, 1] < 0.6 for t in TRAITS)
    var_true = float(np.var(true[:, 2]))
    expected = var_true / (var_true + 1.3 ** 2 / 2)
    for t in TRAITS:
        x = job.reliability[t]
        assert abs(x["icc1"] - expected) < 0.12, (t, x["icc1"], expected)
        assert abs(rl.spearman_brown(x["icc1"], x["k0"]) - x["icc1k"]) < 0.03
    per = job.raters.set_index("rater")
    assert per.loc[[f"R{1010 + i:04d}" for i in range(3)], "excluded"].all()
    assert set(per.index) == set(job.ratings_scored["rater"])                      # every rater, not only the valid ones
    scored = job.ratings_scored
    known = scored[scored["rater"] == "R1020"]
    assert len(known) and (~known["valid"]).all() and (known["reason"] == "знакомство").all()
    ex = job.exclusions
    assert (ex["reason"] == "оценка собственного ролика").sum() == 1
    assert (ex["reason"] == "самооценка заполнена несколько раз").sum() == 1
    assert "Да" not in scored["glasses"].values                     # spelling normalised before the attention check
    assert 0 <= job.alpha["extraversion"] <= 1
    checks = job.checks_md()
    assert "Оценок для 0,8" in checks and "ровно с двумя оценками" in checks and "Распределение меток" in checks


def test_missing_answers_are_handled_as_the_doc_says():
    videos, self_, ratings, _ = synthetic(n_videos=160, n_clusters=4)
    v0 = videos.loc[videos["status"] == "принят", "code"].iloc[0]
    videos.loc[videos["code"] == v0, "glasses"] = np.nan                        # passport without the answer
    ratings.loc[ratings.index[10], "known"] = np.nan                             # service answer missing
    ratings.loc[ratings.index[11], "s03"] = 9                                    # out of the 1-7 scale
    ratings.loc[ratings.index[12], "s05"] = np.nan                               # gap in the card
    # the later self-assessment of R1007 is empty: the earlier complete one must survive
    dup = self_.iloc[[7]].assign(date="2026-11-05")
    dup[[f"q{i:02d}" for i in range(1, 51)]] = np.nan
    self_ = pd.concat([self_, dup])
    job = Labelling(videos, self_, ratings, **QUIET)
    r = job.ratings_scored
    assert not r.loc[r["video"] == v0, "flag_attn"].any()                        # nothing to compare against
    assert (job.exclusions["reason"] == "в паспорте нет ответа про очки или бороду").sum() == 1
    assert r.loc[r.index[10], "reason"] == "нет ответа про знакомство или помехи" and not r.loc[r.index[10], "valid"]
    assert r.loc[r.index[11], "reason"] == "неполная" and np.isnan(r.loc[r.index[11], "s03"])
    assert job.exclusions["reason"].str.startswith("ответ вне пределов шкалы 1–7").sum() == 1
    assert r.loc[r.index[12], "reason"] == "неполная"
    assert np.isfinite(job.labels.set_index("video_name").loc["R1007", "self_openness"])
    assert (job.exclusions["action"] == "оставлена последняя полная").sum() >= 1


def test_split_by_people_with_five_clusters_and_whole_clusters_with_six():
    v5, s5, r5, _ = synthetic(n_clusters=5)
    job5 = Labelling(v5, s5, r5, **QUIET)
    f5 = job5.labels.loc[job5.labels["split"] == "train", "fold"].value_counts()
    assert job5.split_mode == "people" and len(f5) == 5 and f5.min() > 0
    v6, s6, r6, _ = synthetic(n_clusters=6)
    job6 = Labelling(v6, s6, r6, **QUIET)
    lab6 = job6.labels
    f6 = lab6.loc[lab6["split"] == "train", "fold"].value_counts()
    share6 = (lab6["split"] == "test").mean()
    assert job6.split_mode == "clusters" and len(f6) == 5 and f6.min() > 0
    assert lab6.groupby("cluster")["split"].nunique().max() == 1 and 0.13 <= share6 <= 0.22, share6


def test_unequal_clusters_fall_back_to_people_and_a_clusterless_video_is_reported():
    vu, su, ru, _ = synthetic(n_clusters=6, cluster_sizes=[0.6, 0.1, 0.08, 0.08, 0.07, 0.07])
    job = Labelling(vu, su, ru, **QUIET)
    assert job.split_mode == "people" and any("кластеры неравные" in n for n in job.notes)
    f = job.labels.loc[job.labels["split"] == "train", "fold"].value_counts()
    assert len(f) == 5 and f.max() <= 2 * f.mean() and 0.15 <= (job.labels["split"] == "test").mean() <= 0.25
    v8, s8, r8, _ = synthetic()
    loose = Labelling(v8, s8, r8, **QUIET).labels["video_name"].iloc[5]     # a video that does get a label
    v8.loc[v8["code"] == loose, "cluster"] = np.nan
    job8 = Labelling(v8, s8, r8, **QUIET)
    assert job8.split_mode == "clusters"                                          # the design does not change
    assert (job8.exclusions["reason"] == "нет кластера").sum() == 1
    lab8 = job8.labels
    assert loose in set(lab8["video_name"]) and lab8[lab8["cluster"].notna()].groupby("cluster")["split"].nunique().max() == 1
    assert any("без кластера" in n for n in job8.notes) and "без кластера" in job8.checks_md()


def test_second_version_keeps_the_old_split():
    videos, self_, ratings, _ = synthetic()
    lab1 = Labelling(videos, self_, ratings, **QUIET).labels
    v2, s2, r2, _ = synthetic(n_videos=360, raters_per_video=5, seed=1)
    lab2 = Labelling(v2, s2, r2, prev=lab1, **QUIET).labels
    both = lab1.merge(lab2, on="video_name", suffixes=("_1", "_2"))
    assert len(both) > 200
    assert (both["split_1"] == both["split_2"]).all()
    assert both["fold_1"].astype(float).equals(both["fold_2"].astype(float))
    new = lab2[~lab2["video_name"].isin(lab1["video_name"])]
    test_clusters = set(lab1.loc[lab1["split"] == "test", "cluster"])
    assert len(new) > 0 and (new["cluster"].isin(test_clusters) == (new["split"] == "test")).all()
    assert lab2[lab2["split"] == "train"].groupby("cluster")["fold"].nunique().max() == 1
    # a new cluster in the second version (the 40 new videos form one) goes whole to the test while the test is short,
    # else to a fold; the old videos keep their clusters and their places
    v3, s3, r3, _ = synthetic(n_videos=360, raters_per_video=5, seed=1)
    v3.loc[v3.index >= 320, "cluster"] = "c8"
    lab3 = Labelling(v3, s3, r3, prev=lab1, **QUIET).labels
    assert lab3.groupby("cluster")["split"].nunique().max() == 1 and 0.12 <= (lab3["split"] == "test").mean() <= 0.3
    # version 1 split by people with an EMPTY cluster column: the new videos of version 2 must not all land in the test
    ve, se, re_, _ = synthetic(n_videos=200, n_clusters=4)
    ve["cluster"] = np.nan
    lab_e1 = Labelling(ve, se, re_, **QUIET).labels
    ve2, se2, re2, _ = synthetic(n_videos=260, raters_per_video=5, n_clusters=4)
    ve2["cluster"] = np.nan
    lab_e2 = Labelling(ve2, se2, re2, prev=lab_e1, **QUIET).labels
    new_e = lab_e2[~lab_e2["video_name"].isin(lab_e1["video_name"])]
    assert len(new_e) > 20 and 0.0 < (new_e["split"] == "test").mean() < 0.6
    assert 0.15 <= (lab_e2["split"] == "test").mean() <= 0.25


def test_files_written_labels_not_overwritten_and_prev_through_the_command_line():
    videos, self_, ratings, _ = synthetic(n_videos=120, n_clusters=4)
    with tempfile.TemporaryDirectory(prefix="bs3_ru_labels_") as d:
        folder = Path(d) / "v1"
        folder.mkdir()
        videos.to_csv(folder / "videos.csv", index=False)
        self_.to_csv(folder / "self.csv", index=False)
        ratings.to_csv(folder / "ratings.csv", index=False)
        (folder / "work").mkdir()
        for code in videos["code"].map(rl.norm_code)[:50]:
            (folder / "work" / f"{code}.mp4").write_bytes(b"")
        with contextlib.redirect_stdout(io.StringIO()):
            assert rl.main(["--set", str(folder)]) == 0
        for name in ("labels.csv", "ratings_scored.csv", "self_scored.csv", "raters.csv", "exclusions.csv", "checks.md"):
            assert (folder / name).is_file(), name
        lab = pd.read_csv(folder / "labels.csv")
        assert list(lab.columns) == rl.LABEL_COLUMNS and len(lab) > 50
        checks = (folder / "checks.md").read_text(encoding="utf-8")
        assert "## Надёжность меток" in checks and "начальное число 20261006" in checks and "Потолок" in checks
        ex = pd.read_csv(folder / "exclusions.csv")
        assert (ex["reason"] == "нет копии ролика work/<код>.mp4").sum() > 0       # 50 copies for more videos
        raised = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rl.main(["--set", str(folder)])
        except SystemExit as e:
            raised = e
        assert raised is not None and "уже есть" in str(raised.code)
        with contextlib.redirect_stdout(io.StringIO()):
            assert rl.main(["--set", str(folder), "--force"]) == 0
        # version 2 through the command line: the old split survives the CSV round trip (fold read back as float)
        v2, s2, r2, _ = synthetic(n_videos=150, raters_per_video=5, n_clusters=4)
        folder2 = Path(d) / "v2"
        folder2.mkdir()
        v2.to_csv(folder2 / "videos.csv", index=False)
        s2.to_csv(folder2 / "self.csv", index=False)
        r2.to_csv(folder2 / "ratings.csv", index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            assert rl.main(["--set", str(folder2), "--prev", str(folder / "labels.csv")]) == 0
        lab2 = pd.read_csv(folder2 / "labels.csv")
        both = lab.merge(lab2, on="video_name", suffixes=("_1", "_2"))
        assert len(both) > 50 and (both["split_1"] == both["split_2"]).all()
        assert both["fold_1"].astype(float).equals(both["fold_2"].astype(float))
        assert "взято из прежней версии" in (folder2 / "checks.md").read_text(encoding="utf-8") or "по людям" in (folder2 / "checks.md").read_text(encoding="utf-8")
