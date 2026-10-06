"""Labelling of the own Russian video set (docs/dataset/5_Как_разметить_набор.md): the three tables of the organizer memo
(videos.csv, self.csv, ratings.csv) -> labels.csv of 4_Как_обучить_AMLAI_2.0.md plus the check tables.

  python -m training.ru_labels --set /mnt/d/ru_set_v1                 # writes into the set folder
  python -m training.ru_labels --set /mnt/d/ru_set_v2 --prev /mnt/d/ru_set_v1/labels.csv   # new version: old split kept

Rules (memo sections 7-10, doc 5 sections 3-9): codes normalised to R + four digits; answers outside their scale are
cleared and listed; duplicates and self-ratings removed and listed in exclusions.csv (a repeated self-assessment keeps
the last complete one); card scores 0..1 (reverse row = 8 - x), 50 statements 0..1 (reverse = 6 - x, up to two empty
answers per trait); a rating is invalid when incomplete, when the service answers are missing, when the rater knows the
person or had trouble, or when the rater is excluded (two of three signs); a video needs two valid ratings; the label is
the mean of the valid ratings; one-way ICC for the unbalanced design; the split is by person, whole clusters from six
roughly equal clusters on, and a previous labels.csv keeps the split of the videos it already has. An existing labels.csv
is never overwritten without --force. Only pandas and numpy; nothing here imports bs3."""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "non-neuroticism"]
RU = {"openness": "открытость", "conscientiousness": "добросовестность", "extraversion": "экстраверсия",
      "agreeableness": "доброжелательность", "non-neuroticism": "эмоциональная устойчивость", "interview": "собеседование"}
# ten-scale card: trait -> (direct row, reverse row); answers 1-7, reverse = 8 - x
CARD = {"extraversion": (1, 6), "agreeableness": (7, 2), "conscientiousness": (3, 8),
        "non-neuroticism": (9, 4), "openness": (5, 10)}
# fifty statements: trait -> (direct items, reverse items); answers 1-5, reverse = 6 - x
IPIP = {"extraversion": ([1, 11, 21, 31, 41], [6, 16, 26, 36, 46]),
        "agreeableness": ([7, 17, 27, 37, 42, 47], [2, 12, 22, 32]),
        "conscientiousness": ([3, 13, 23, 33, 43, 48], [8, 18, 28, 38]),
        "non-neuroticism": ([9, 19], [4, 14, 24, 29, 34, 39, 44, 49]),
        "openness": ([5, 15, 25, 35, 40, 45, 50], [10, 20, 30])}
CARD_COLS = [f"s{i:02d}" for i in range(1, 11)]
IPIP_COLS = [f"q{i:02d}" for i in range(1, 51)]
CODE = re.compile(r"^R\d{4}$")
MIN_CLUSTERS = 6             # whole-cluster split: one cluster for the test and five for the folds (doc 5, section 9)
ACCEPTED = "принят"
NO = "нет"
YES_NO = ("да", "нет")
EMPTY = ("", "nan", "none")  # what an empty cell looks like after astype(str).lower()
LABEL_COLUMNS = (["video_name"] + TRAITS + ["interview", "n_ratings"] + [f"self_{t}" for t in TRAITS]
                 + [f"sd_{t}" for t in TRAITS] + ["cluster", "split", "fold"])


# ---------------------------------------------------------------- scoring (doc 5, section 5 and 11)
def norm_code(x) -> str:
    """'r0147', ' R0147 ', Cyrillic 'Р0147' -> 'R0147'."""
    return str(x).strip().upper().replace("Р", "R").replace(" ", "")


def score_card(row) -> dict:
    """Five trait scores 0..1 from s01..s10 of one row (a rating or a self-assessment)."""
    out = {}
    for t, (d, r) in CARD.items():
        a, b = row[f"s{d:02d}"], 8 - row[f"s{r:02d}"]
        out[t] = ((a + b) / 2 - 1) / 6
    return out


def score_ipip(row, max_missing: int = 2) -> dict:
    """Five trait scores 0..1 from q01..q50; a trait with more than `max_missing` empty answers is NaN."""
    out = {}
    for t, (d, r) in IPIP.items():
        vals = [row[f"q{i:02d}"] for i in d] + [6 - row[f"q{i:02d}"] for i in r]
        vals = pd.Series(vals, dtype="float")
        out[t] = np.nan if vals.isna().sum() > max_missing else (vals.mean() - 1) / 4
    return out


def icc_oneway(long: pd.DataFrame, target: str, value: str) -> dict:
    """One-way random-effects ICC for an unbalanced design (every video has its own raters):
    ICC(1) of a single rating and ICC(1,k) of the mean rating, k0 = the adjusted mean number of ratings.
    NaN when there is nothing to estimate (fewer than two videos, no repeated ratings, no variation between videos)."""
    nothing = {"icc1": np.nan, "icc1k": np.nan, "k0": np.nan, "ceiling": np.nan}
    g = long.groupby(target)[value]
    k = g.size()
    n, N = len(k), int(k.sum())
    if n < 2 or N <= n:
        return nothing | {"n_videos": n, "n_ratings": N}
    means, grand = g.mean(), long[value].mean()
    ssb = float((k * (means - grand) ** 2).sum())
    dev = long[value] - long[target].map(means)
    ssw = float((dev ** 2).sum())
    msb, msw = ssb / (n - 1), ssw / (N - n)
    k0 = (N - float((k ** 2).sum()) / N) / (n - 1)
    den1 = msb + (k0 - 1) * msw
    icc1 = (msb - msw) / den1 if den1 > 0 else np.nan
    icc1k = (msb - msw) / msb if msb > 0 else np.nan
    ceiling = float(np.sqrt(max(icc1k, 0))) if np.isfinite(icc1k) else np.nan
    return {"icc1": icc1, "icc1k": icc1k, "k0": k0, "n_videos": n, "n_ratings": N, "ceiling": ceiling}


def spearman_brown(icc1: float, k: float) -> float:
    return k * icc1 / (1 + (k - 1) * icc1)


def ratings_for(icc1: float, target: float = 0.8, k_max: int = 50) -> int | None:
    """The smallest number of ratings per video whose mean reaches reliability `target`; None when not within k_max."""
    if not np.isfinite(icc1) or icc1 <= 0:
        return None
    for k in range(1, k_max + 1):
        if spearman_brown(icc1, k) >= target:
            return k
    return None


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Alpha of the columns of `items` (already recoded), rows with an empty answer dropped."""
    x = items.dropna()
    if len(x) < 3 or x.shape[1] < 2:
        return np.nan
    k = x.shape[1]
    total_var = x.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - x.var(ddof=1).sum() / total_var)) if total_var > 0 else np.nan


def rater_correlations(v: pd.DataFrame, centre: bool = True, min_videos: int = 4) -> pd.Series:
    """Third sign of a careless rater (doc 5, section 6): one correlation per rater over all pairs video x trait between
    their trait scores and the mean scores of the other raters of the same videos. With `centre` the mean of every trait
    over all valid ratings is subtracted first: otherwise the different levels of the traits alone give a positive
    correlation even to a rater who gives every video the same profile. NaN with fewer than `min_videos` videos that have
    other ratings; 0 when the rater's centred scores do not vary (no link at all)."""
    mid = v[TRAITS].mean() if centre else 0.0
    out = {}
    for rater, grp in v.groupby("rater"):
        others = v[(v["rater"] != rater) & (v["video"].isin(grp["video"]))]
        om = others.groupby("video")[TRAITS].mean() - mid
        if len(om) < min_videos:
            out[rater] = np.nan
            continue
        mine = grp.set_index("video")[TRAITS].loc[om.index] - mid
        a, b = mine.to_numpy().ravel(), om.to_numpy().ravel()
        out[rater] = 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
    return pd.Series(out, dtype="float")


# ---------------------------------------------------------------- split (doc 5, sections 9-10)
def _balanced(test: set, fold_of: dict, n_total: int, n_folds: int) -> str | None:
    """None when the whole-cluster placement is acceptable, else the reason it is not (doc 5, section 9: clusters must
    be roughly equal; here: test share within 10-25%, no fold larger than twice the mean fold, every fold used)."""
    share = len(test) / max(1, n_total)
    sizes = [sum(1 for f in fold_of.values() if f == k) for k in range(1, n_folds + 1)]
    if share < 0.10 or share > 0.25:
        return f"доля теста {share:.0%}"
    if min(sizes) == 0:
        return "пустая часть"
    if max(sizes) > 2 * (sum(sizes) / n_folds):
        return f"части неравные ({', '.join(map(str, sizes))})"
    return None


def split_videos(lab: pd.DataFrame, rng, test_share: float, n_folds: int, prev: pd.DataFrame | None, notes: list | None = None):
    """(test: set of codes, fold_of: code -> fold, mode: 'clusters' | 'people'). Whole clusters when there are MIN_CLUSTERS or
    more roughly equal clusters; with `prev` (labels.csv of the previous version) old videos keep split and fold and only the
    new ones are placed: by their cluster when the first version was split by clusters, else at random in the same shares.
    A video without a cluster is reported and placed on its own; it never changes the design."""
    notes = notes if notes is not None else []
    codes = lab["video_name"].tolist()
    cluster_of = {c: ("" if pd.isna(cl) else str(cl).strip()) for c, cl in zip(lab["video_name"], lab["cluster"])}
    by_cluster: dict[str, list] = {}
    for c in codes:
        by_cluster.setdefault(cluster_of[c], []).append(c)
    loose = by_cluster.pop("", [])                      # videos without a cluster
    if loose:
        notes.append(f"{len(loose)} ролик(ов) без кластера размещены по одному")
    test, fold_of = set(), {}
    if prev is not None and len(prev):
        old = prev.drop_duplicates("video_name").set_index("video_name")
        for c in codes:
            if c in old.index:
                if old.loc[c, "split"] == "test":
                    test.add(c)
                elif pd.notna(old.loc[c, "fold"]):
                    fold_of[c] = int(old.loc[c, "fold"])
    new = [c for c in codes if c not in test and c not in fold_of]
    if not new:
        return test, fold_of, "prev"
    n_total = len(codes)
    sizes = {f: sum(1 for c in fold_of.values() if c == f) for f in range(1, n_folds + 1)}

    def smallest_fold():
        return min(sizes, key=lambda f: (sizes[f], f))

    def place_people(items):
        rng.shuffle(items)
        want_test = int(round(test_share * n_total)) - len(test)
        for c in items[:max(0, want_test)]:
            test.add(c)
        for c in items[max(0, want_test):]:
            f = smallest_fold()
            fold_of[c] = f
            sizes[f] += 1

    if prev is not None and len(prev):
        p = prev.assign(cluster=prev["cluster"].fillna("").astype(str).str.strip())
        p = p[p["cluster"] != ""]
        spans = (p.groupby("cluster")["split"].nunique() > 1).any() if len(p) else True
        several = (p[p["split"] == "train"].groupby("cluster")["fold"].nunique() > 1).any() if len(p) else True
        by_people = bool(spans or several)
    else:
        by_people = len(by_cluster) < MIN_CLUSTERS
    if by_people:
        place_people(new)
        return test, fold_of, "people"

    # ---- whole clusters
    placed_test = {cluster_of[c] for c in test if cluster_of[c]}
    placed_fold = {cluster_of[c]: f for c, f in fold_of.items() if cluster_of[c]}
    new_clusters = [cl for cl in by_cluster if cl not in placed_test and cl not in placed_fold]
    for c in new:                                        # a new video of a known cluster follows it
        cl = cluster_of[c]
        if cl in placed_test:
            test.add(c)
        elif cl in placed_fold:
            fold_of[c] = placed_fold[cl]
            sizes[placed_fold[cl]] += 1
    rng.shuffle(new_clusters)
    if prev is None or not len(prev):
        # how many whole clusters go to the test: the share closest to the middle of 15-20%, at least five left
        best, best_gap = 1, None
        for m in range(1, len(new_clusters) - n_folds + 1):
            share = sum(len(by_cluster[cl]) for cl in new_clusters[:m]) / n_total
            gap = abs(share - (test_share - 0.025))
            if best_gap is None or gap < best_gap:
                best, best_gap = m, gap
        for cl in new_clusters[:best]:
            test.update(by_cluster[cl])
        new_clusters = new_clusters[best:]
        for cl in sorted(new_clusters, key=lambda x: -len(by_cluster[x])):   # largest first, into the smallest fold
            f = smallest_fold()
            for c in by_cluster[cl]:
                fold_of[c] = f
            sizes[f] += len(by_cluster[cl])
        reason = _balanced(test, fold_of, n_total - len(loose), n_folds)
        if reason:
            notes.append(f"кластеры неравные ({reason}): разбиение по людям")
            test.clear()
            fold_of.clear()
            for f in sizes:
                sizes[f] = 0
            place_people(list(codes))
            return test, fold_of, "people"
    else:
        for cl in new_clusters:                          # a cluster seen for the first time: whole, test while it is short
            if len(test) / n_total < test_share - 0.025:
                test.update(by_cluster[cl])
            else:
                f = smallest_fold()
                for c in by_cluster[cl]:
                    fold_of[c] = f
                sizes[f] += len(by_cluster[cl])
    loose_new = [c for c in loose if c in new]
    if loose_new:
        place_people(loose_new)
    return test, fold_of, "clusters"


# ---------------------------------------------------------------- the pipeline
def fmt(x, nd: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


class Labelling:
    """Runs the rules on three DataFrames and keeps every intermediate table; `write` puts them into a folder."""

    def __init__(self, videos: pd.DataFrame, self_: pd.DataFrame, ratings: pd.DataFrame, *, seed: int = 20261006,
                 test_share: float = 0.2, n_folds: int = 5, prev: pd.DataFrame | None = None, work: Path | None = None,
                 log=print):
        self.seed, self.test_share, self.n_folds, self.prev, self.work, self.log = seed, test_share, n_folds, prev, work, log
        self.excl: list[dict] = []
        self.notes: list[str] = []
        self.run(videos.copy(), self_.copy(), ratings.copy())

    def exclude(self, kind: str, code: str, reason: str, action: str = "", **extra):
        self.excl.append({"what": kind, "code": code, "reason": reason, "action": action, **extra})

    # ---- section 3: codes, ranges, duplicates
    def _codes(self, df: pd.DataFrame, cols: list[str], table: str):
        for c in cols:
            df[c] = df[c].map(norm_code)
            bad = df[~df[c].map(lambda s: bool(CODE.match(s)))]
            for v in bad[c].unique():
                self.exclude(table, v, f"код не подходит под образец R0000 (столбец {c})", "строки не взяты")
            df.drop(bad.index, inplace=True)
        return df

    def _ranges(self, df: pd.DataFrame, cols: list[str], lo: int, hi: int, table: str, key_cols: list[str]):
        """Answers outside lo..hi become empty and are listed; the completeness rules then do the rest."""
        present = [c for c in cols if c in df.columns]
        for c in present:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            bad = df[c].notna() & ~df[c].between(lo, hi)
            for _, row in df[bad].iterrows():
                self.exclude(table, row[key_cols[0]], f"ответ вне пределов шкалы {lo}–{hi} (столбец {c}: {row[c]:g})",
                             "ответ очищен", **({"video": row[key_cols[1]]} if len(key_cols) > 1 else {}))
            df.loc[bad, c] = np.nan

    def run(self, videos, self_, ratings):
        need = {"videos.csv": (videos, ["code", "cluster", "status", "glasses", "beard"]),
                "self.csv": (self_, ["code", "date"] + IPIP_COLS),
                "ratings.csv": (ratings, ["rater", "video", "known", "noise", "glasses", "beard", "interview"] + CARD_COLS)}
        for name, (df, cols) in need.items():
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise SystemExit(f"{name}: нет столбцов {missing} (памятка, раздел 10)")
        for df, cols in ((videos, ["code"]), (self_, ["code"]), (ratings, ["rater", "video"])):
            for c in cols:
                df[c] = df[c].astype(str)
        videos = self._codes(videos, ["code"], "videos.csv")
        self_ = self._codes(self_, ["code"], "self.csv")
        ratings = self._codes(ratings, ["rater", "video"], "ratings.csv")
        for col in ("glasses", "beard", "status"):
            videos[col] = videos[col].astype(str).str.strip().str.lower().str.rstrip(".")
        for col in ("known", "noise", "glasses", "beard"):
            ratings[col] = ratings[col].astype(str).str.strip().str.lower().str.rstrip(".")
        self._ranges(ratings, CARD_COLS + ["interview"], 1, 7, "ratings.csv", ["rater", "video"])
        self._ranges(ratings, ["confidence"], 1, 3, "ratings.csv", ["rater", "video"])
        self._ranges(self_, IPIP_COLS, 1, 5, "self.csv", ["code"])
        self._ranges(self_, CARD_COLS, 1, 7, "self.csv", ["code"])
        dup_v = videos[videos.duplicated("code", keep=False)]
        for code in dup_v["code"].unique():
            self.exclude("videos.csv", code, "код встречается в videos.csv несколько раз", "оставлена последняя строка")
        videos = videos.drop_duplicates("code", keep="last")
        accepted = videos[videos["status"] == ACCEPTED].copy()
        self.log(f"ролики: получено {len(videos)}, принято {len(accepted)}")

        # a repeated self-assessment: the last COMPLETE one wins (complete = every trait scorable, doc 5 section 3)
        if len(self_):
            full = self_.apply(lambda row: not any(np.isnan(v) for v in score_ipip(row).values()), axis=1)
            self_ = self_.assign(_full=full.astype(int))
            if "date" in self_:
                self_ = self_.sort_values(["_full", "date"], kind="stable")
            else:
                self_ = self_.sort_values("_full", kind="stable")
            dup = self_[self_.duplicated("code", keep=False)]
            for code in dup["code"].unique():
                self.exclude("self.csv", code, "самооценка заполнена несколько раз", "оставлена последняя полная")
            self_ = self_.drop_duplicates("code", keep="last").drop(columns="_full")

        r = ratings
        if "date" in r:
            r = r.sort_values("date", kind="stable")
        own = r[r["rater"] == r["video"]]
        for _, row in own.iterrows():
            self.exclude("ratings.csv", row["rater"], "оценка собственного ролика", "строка не взята", video=row["video"])
        r = r[r["rater"] != r["video"]]
        dup_r = r[r.duplicated(["rater", "video"], keep=False)]
        for (rater, video), _ in dup_r.groupby(["rater", "video"]):
            self.exclude("ratings.csv", rater, "ролик оценён этим наблюдателем дважды", "оставлена последняя", video=video)
        r = r.drop_duplicates(["rater", "video"], keep="last")
        not_acc = r[~r["video"].isin(accepted["code"])]
        for video in not_acc["video"].unique():
            self.exclude("ratings.csv", video, "оценка ролика, которого нет среди принятых", "в набор не идёт, строка сохранена",
                         n=int((not_acc["video"] == video).sum()))
        r = r[r["video"].isin(accepted["code"])].copy()

        # ---- section 4: the set folder and the passports
        if self.work is not None and self.work.is_dir():
            files = {p.stem for p in self.work.glob("*.mp4")}
            for code in sorted(set(accepted["code"]) - files):
                self.exclude("work/", code, "нет копии ролика work/<код>.mp4", "сделать копию (памятка, раздел 4)")
            for code in sorted(files - set(accepted["code"])):
                self.exclude("work/", code, "файл в work/ без принятого ролика в videos.csv", "проверить статус ролика")
        for code in sorted(set(accepted["code"]) - set(self_["code"])):
            self.exclude("self.csv", code, "у принятого ролика нет самооценки", "в обучение идёт, в анализ связи с самооценкой — нет")
        pas = accepted.set_index("code")
        no_passport = pas[~pas["glasses"].isin(YES_NO) | ~pas["beard"].isin(YES_NO)]
        for code in no_passport.index:
            self.exclude("videos.csv", code, "в паспорте нет ответа про очки или бороду", "проверка внимательности для ролика не делается")
        cl = accepted["cluster"].astype(str).str.strip().str.lower()
        for code in accepted.loc[cl.isin(EMPTY), "code"]:
            self.exclude("videos.csv", code, "нет кластера", "размещён в разбиении по одному")

        # ---- section 5: scores; section 6: validity
        complete = r[CARD_COLS].notna().all(axis=1) & r["interview"].notna()
        no_service = r["known"].isin(EMPTY) | r["noise"].isin(EMPTY)
        for _, row in r[~complete].iterrows():
            self.exclude("ratings.csv", row["rater"], "неполная оценка: пропуск в карте или в вопросе о собеседовании",
                         "недействительна", video=row["video"])
        r = pd.concat([r, r.apply(score_card, axis=1, result_type="expand")], axis=1)
        r["interview01"] = (r["interview"] - 1) / 6
        r["complete"] = complete
        r["flag_same"] = r[CARD_COLS].nunique(axis=1) == 1
        gap = pd.DataFrame({t: (r[f"s{d:02d}"] - (8 - r[f"s{rv:02d}"])).abs() for t, (d, rv) in CARD.items()})
        r["flag_contra"] = (gap >= 4).sum(axis=1) >= 2
        # attention check (memo section 8): the rater's answer against the passport, which was verified at acceptance;
        # compared only where the passport has a real answer. A rater's blank answer against a real passport counts as a miss.
        pg, pb = r["video"].map(pas["glasses"]), r["video"].map(pas["beard"])
        r["flag_attn"] = (pg.isin(YES_NO) & (r["glasses"] != pg)) | (pb.isin(YES_NO) & (r["beard"] != pb))
        r["flagged"] = r[["flag_same", "flag_contra", "flag_attn"]].any(axis=1)
        r["valid"] = complete & ~no_service & (r["known"] == NO) & (r["noise"] == NO)
        r["reason"] = ""
        r.loc[~complete, "reason"] = "неполная"
        r.loc[complete & no_service, "reason"] = "нет ответа про знакомство или помехи"
        r.loc[complete & ~no_service & (r["known"] != NO), "reason"] = "знакомство"
        r.loc[complete & ~no_service & (r["known"] == NO) & (r["noise"] != NO), "reason"] = "помехи"
        self.log(f"оценки: {len(ratings)} строк, {len(r)} по принятым роликам, {int((~complete).sum())} неполных, "
                 f"{int(r['flagged'].sum())} помечено, {int((~r['valid']).sum())} недействительных")

        v = r[r["valid"]]
        all_raters = sorted(r["rater"].unique())
        per = v.groupby("rater").agg(n_valid=("video", "size"), flagged_share=("flagged", "mean")).reindex(all_raters)
        per["n_valid"] = per["n_valid"].fillna(0).astype(int)
        per["sd_answers"] = v.groupby("rater")[CARD_COLS].apply(lambda x: float(np.std(x.to_numpy().ravel())))
        per["corr_others"] = rater_correlations(v) if len(v) else np.nan
        per["sign_flagged"] = per["flagged_share"] > 0.30
        per["sign_flat"] = per["sd_answers"] < 0.5
        per["sign_unrelated"] = per["corr_others"].fillna(1) <= 0
        per["signs"] = per[["sign_flagged", "sign_flat", "sign_unrelated"]].sum(axis=1)
        per["excluded"] = per["signs"] >= 2
        per["n_all"] = r.groupby("rater").size().reindex(all_raters).fillna(0).astype(int)
        for rater, row in per[per["excluded"]].iterrows():
            self.exclude("наблюдатель", rater, "два признака небрежности из трёх (раздел 6)", "все оценки недействительны",
                         flagged_share=round(float(row["flagged_share"]), 2), sd_answers=round(float(row["sd_answers"]), 2),
                         corr_others=None if pd.isna(row["corr_others"]) else round(float(row["corr_others"]), 2))
        bad_raters = set(per.index[per["excluded"]])
        r.loc[r["rater"].isin(bad_raters) & r["valid"], "reason"] = "наблюдатель исключён"
        r.loc[r["rater"].isin(bad_raters), "valid"] = False
        self.log(f"наблюдатели: {len(per)}, исключено {len(bad_raters)}")
        v = r[r["valid"]].copy()

        # ---- section 7: the label of a video
        agg = v.groupby("video")
        lab = agg[TRAITS + ["interview01"]].mean().rename(columns={"interview01": "interview"})
        lab["n_ratings"] = agg.size()
        lab = lab.join(agg[TRAITS].std(ddof=0).add_prefix("sd_"))
        few = lab[lab["n_ratings"] < 2]
        for code in few.index:
            self.exclude("ролик", code, "меньше двух действительных оценок", "в набор не идёт; при доборе вернётся",
                         n_valid=int(few.loc[code, "n_ratings"]))
        for code in sorted(set(accepted["code"]) - set(lab.index)):
            self.exclude("ролик", code, "нет ни одной действительной оценки", "в набор не идёт; при доборе вернётся", n_valid=0)
        lab = lab[lab["n_ratings"] >= 2]
        s50 = self_.set_index("code").apply(score_ipip, axis=1, result_type="expand") if len(self_) else pd.DataFrame(columns=TRAITS)
        self_scored = s50.add_prefix("self50_")
        if len(self_) and all(c in self_.columns for c in CARD_COLS):
            self_scored = self_scored.join(self_.set_index("code").apply(score_card, axis=1, result_type="expand").add_prefix("selfcard_"))
        lab = lab.join(s50.add_prefix("self_"), how="left").join(pas[["cluster"]], how="left")
        lab["cluster"] = lab["cluster"].astype(str).str.strip().where(~lab["cluster"].astype(str).str.strip().str.lower().isin(EMPTY), np.nan)
        lab.index.name = "video_name"
        lab = lab.reset_index()
        self.log(f"размечено роликов: {len(lab)}; без двух оценок: {len(set(accepted['code']) - set(lab['video_name']))}")

        # ---- section 8: reliability and sanity
        vv = v[v["video"].isin(lab["video_name"])]
        rel = {t: icc_oneway(vv, "video", t) for t in TRAITS}
        rel["interview"] = icc_oneway(vv, "video", "interview01")
        self_corr = {}
        for t in TRAITS:
            pair = lab[[t, f"self_{t}"]].dropna()
            self_corr[t] = float(pair.corr().iloc[0, 1]) if len(pair) >= 3 and pair[t].std() > 0 and pair[f"self_{t}"].std() > 0 else np.nan
        alpha = {}
        for t, (d, rv) in IPIP.items():
            if len(self_):
                items = pd.concat([self_[[f"q{i:02d}" for i in d]], 6 - self_[[f"q{i:02d}" for i in rv]]], axis=1)
                alpha[t] = cronbach_alpha(items)

        # ---- section 9: the split
        rng = np.random.default_rng(self.seed)
        test, fold_of, mode = split_videos(lab, rng, self.test_share, self.n_folds, self.prev, self.notes)
        lab["split"] = np.where(lab["video_name"].isin(test), "test", "train")
        lab["fold"] = lab["video_name"].map(fold_of).astype("Int64")
        self.split_mode = mode
        self.labels = lab[LABEL_COLUMNS]
        self.ratings_scored = r
        self.self_scored = self_scored.reset_index().rename(columns={"index": "code"}) if len(self_scored) else pd.DataFrame()
        self.raters = per.reset_index()
        self.exclusions = pd.DataFrame(self.excl)
        self.reliability, self.self_corr, self.alpha = rel, self_corr, alpha
        self.n_videos_received, self.n_accepted, self.n_ratings_rows = len(videos), len(accepted), len(ratings)

    # ---- the files (doc 5, sections 7, 8, 10)
    def checks_md(self) -> str:
        lab = self.labels
        n_test = int((lab["split"] == "test").sum())
        folds = lab.loc[lab["split"] == "train", "fold"].value_counts().sort_index()
        mode = {"clusters": "целыми кластерами", "people": "по людям", "prev": "взято из прежней версии"}[self.split_mode]
        lines = [f"# Проверки разметки ({date.today().isoformat()})\n",
                 f"Роликов получено {self.n_videos_received}, принято {self.n_accepted}, размечено {len(lab)}; строк оценок "
                 f"{self.n_ratings_rows}, действительных {int(self.ratings_scored['valid'].sum())}; наблюдателей {len(self.raters)}, "
                 f"исключено {int(self.raters['excluded'].sum())}.\n",
                 f"Оценок на ролик: минимум {int(lab['n_ratings'].min()) if len(lab) else 0}, среднее "
                 f"{lab['n_ratings'].mean():.1f}; роликов ровно с двумя оценками: {int((lab['n_ratings'] == 2).sum())} из {len(lab)}.\n",
                 f"Разбиение: начальное число {self.seed}, тест {n_test} роликов ({n_test / max(1, len(lab)):.0%}), {mode}; части: "
                 + ", ".join(f"{int(f)} — {int(n)}" for f, n in folds.items()) + "."
                 + (" Замечания: " + "; ".join(self.notes) + "." if self.notes else "") + "\n",
                 "## Надёжность меток\n", "| Черта | ICC(1) | ICC(1,k) | Потолок | k0 | Оценок для 0,8 |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for t in TRAITS + ["interview"]:
            x = self.reliability[t]
            k80 = ratings_for(x["icc1"])
            lines.append(f"| {RU[t]} | {fmt(x['icc1'])} | {fmt(x['icc1k'])} | {fmt(x['ceiling'])} | {fmt(x['k0'], 1)} | "
                         f"{k80 if k80 is not None else '—'} |")
        lines += ["\nICC(1) — надёжность одной оценки, ICC(1,k) — средней оценки ролика (самой метки), потолок — корень из ICC(1,k): "
                  "сильнее этого модель с меткой коррелировать не может. «Оценок для 0,8» — сколько оценок на ролик довело бы "
                  "надёжность метки до 0,8 (формула Спирмена — Брауна).\n", "## Метки\n",
                  "| Черта | Среднее | Разброс | Доля выше 0,5 | Связь с самооценкой | Альфа 50 утверждений |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for t in TRAITS:
            lines.append(f"| {RU[t]} | {fmt(lab[t].mean())} | {fmt(lab[t].std(ddof=0))} | {(lab[t] > 0.5).mean():.0%} | "
                         f"{fmt(self.self_corr[t])} | {fmt(self.alpha.get(t, np.nan))} |")
        iv = lab["interview"]
        lines.append(f"| {RU['interview']} | {fmt(iv.mean())} | {fmt(iv.std(ddof=0))} | {(iv > 0.5).mean():.0%} | — | — |")
        lines.append("\nРаспределение меток по десяти отрезкам шкалы 0…1 (число роликов от 0,0–0,1 до 0,9–1,0):\n")
        for t in TRAITS + ["interview"]:
            hist = np.histogram(lab[t].dropna(), bins=np.linspace(0, 1, 11))[0]
            lines.append(f"- {RU[t]}: {' '.join(str(int(h)) for h in hist)}")
        warn = []
        for t in TRAITS:
            if lab[t].std(ddof=0) < 0.08:
                warn.append(f"разброс метки «{RU[t]}» меньше 0,08")
            side = max((lab[t] > 0.5).mean(), (lab[t] < 0.5).mean()) if len(lab) else 0
            if side > 0.8:
                warn.append(f"{side:.0%} роликов по одну сторону от 0,5 по черте «{RU[t]}»")
            if np.isfinite(self.self_corr[t]) and self.self_corr[t] < -0.1:
                warn.append(f"связь с самооценкой по черте «{RU[t]}» отрицательная ({self.self_corr[t]:.2f}): проверьте ключи")
            if not np.isfinite(self.reliability[t]["icc1k"]):
                warn.append(f"метка «{RU[t]}» не меняется между роликами: надёжность не определена")
        lines.append("\n**Предупреждения:** " + ("; ".join(warn) + "." if warn else "нет."))
        lines.append("\n## Что исключено\n")
        if len(self.exclusions):
            for (what, reason), n in self.exclusions.groupby(["what", "reason"]).size().items():
                lines.append(f"- {what}: {reason} — {n}")
        else:
            lines.append("- ничего")
        return "\n".join(lines) + "\n"

    def write(self, folder: Path, force: bool = False):
        out = folder / "labels.csv"
        if out.exists() and not force:
            raise SystemExit(f"{out} уже есть: готовый labels.csv не пересчитывается (документ 4, раздел 5). "
                             "Для новой версии набора делайте новую папку и передайте старый файл через --prev; "
                             "перезаписать намеренно — --force.")
        self.labels.to_csv(out, index=False, encoding="utf-8")
        self.ratings_scored.to_csv(folder / "ratings_scored.csv", index=False, encoding="utf-8")
        self.self_scored.to_csv(folder / "self_scored.csv", index=False, encoding="utf-8")
        self.raters.to_csv(folder / "raters.csv", index=False, encoding="utf-8")
        self.exclusions.to_csv(folder / "exclusions.csv", index=False, encoding="utf-8")
        (folder / "checks.md").write_text(self.checks_md(), encoding="utf-8")
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        self.log(f"записано в {folder}: labels.csv ({len(self.labels)} роликов, sha256 {digest[:12]}…), ratings_scored.csv, "
                 "self_scored.csv, raters.csv, exclusions.csv, checks.md")


def read_set(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = []
    for name in ("videos.csv", "self.csv", "ratings.csv"):
        p = folder / name
        if not p.is_file():
            raise SystemExit(f"нет файла {p} (памятка, раздел 10)")
        tables.append(pd.read_csv(p, encoding="utf-8", dtype=str))
    return tuple(tables)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m training.ru_labels", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="папка набора с videos.csv, self.csv, ratings.csv (и work/)")
    ap.add_argument("--prev", default=None, help="labels.csv прежней версии: разбиение старых роликов берётся из него")
    ap.add_argument("--seed", type=int, default=20261006, help="начальное число генератора для разбиения")
    ap.add_argument("--test-share", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="перезаписать существующий labels.csv")
    a = ap.parse_args(argv)
    folder = Path(a.set).expanduser()
    videos, self_, ratings = read_set(folder)
    prev = pd.read_csv(a.prev, encoding="utf-8") if a.prev else None
    work = folder / "work"
    job = Labelling(videos, self_, ratings, seed=a.seed, test_share=a.test_share, n_folds=a.folds, prev=prev,
                    work=work if work.is_dir() else None)
    job.write(folder, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
