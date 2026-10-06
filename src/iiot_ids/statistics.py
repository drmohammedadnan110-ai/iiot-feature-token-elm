"""Seed-paired inference and dependency-aware case resampling are distinct analyses."""

import numpy as np
from scipy import stats

from .metrics import macro_f1


def holm(pvalues):
    p = np.asarray(pvalues, float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid p-values")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def seed_summary(scores):
    values = np.asarray(scores, float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite seed scores required")
    mean, sd = values.mean(), values.std(ddof=1)
    delta = stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sample_sd": sd,
        "ci95": [mean - delta, mean + delta],
        "unit": "training seed on a fixed test population, not independent traffic populations",
    }


def paired_seed_tests(frame, reference):
    """Require shared TEST checksum and class order in addition to matched seeds."""
    for column in ["test_sha256", "classes"]:
        if column not in frame or frame[column].isna().any() or frame[column].nunique() != 1:
            raise ValueError(f"Paired comparisons require one shared {column}")
    if not {"seed", "model", "macro_f1"} <= set(frame) or frame.duplicated(["seed", "model"]).any():
        raise ValueError("Require one score per model/seed")
    pivot = frame.pivot(index="seed", columns="model", values="macro_f1")
    if pivot.isna().any().any() or reference not in pivot or len(pivot) < 2:
        raise ValueError("Seed sets must match, with >=2 paired runs and the reference model")
    summaries = {name: seed_summary(pivot[name]) for name in pivot}
    comparisons = {}
    for name in pivot:
        if name == reference:
            continue
        difference = (pivot[reference] - pivot[name]).to_numpy()
        summary = seed_summary(difference)
        sd = difference.std(ddof=1)
        if sd == 0:
            statistic, pvalue = (0.0, 1.0) if np.all(difference == 0) else (None, 0.0)
            effect = None
        else:
            test = stats.ttest_1samp(difference, 0)
            statistic, pvalue = float(test.statistic), float(test.pvalue)
            effect = float(difference.mean() / sd)
        comparisons[name] = {
            **summary,
            "t": statistic,
            "p_raw": pvalue,
            "cohens_dz": effect,
            "zero_variance_difference": bool(sd == 0),
        }
    for entry, adjusted in zip(comparisons.values(), holm([v["p_raw"] for v in comparisons.values()])):
        entry["p_holm"] = adjusted
    return {
        "reference": reference,
        "seeds": pivot.index.tolist(),
        "summaries": summaries,
        "comparisons": comparisons,
        "test_sha256": frame.test_sha256.iloc[0],
        "classes": frame.classes.iloc[0],
    }


def align(a, b):
    required = ["row_id", "seed", "group", "y_true", "prediction"]
    for frame in [a, b]:
        if (
            not len(frame)
            or not set(required) <= set(frame)
            or frame[required].isna().any().any()
            or frame.duplicated(["seed", "row_id"]).any()
        ):
            raise ValueError("Incomplete, missing or repeated prediction metadata")
    a, b = [f.sort_values(["seed", "row_id"]).reset_index(drop=True) for f in [a, b]]
    metadata = required[:-1]
    if not a[metadata].equals(b[metadata]):
        raise ValueError("Paired predictions disagree on IDs, seeds, labels or groups")
    return a, b


def paired_group_bootstrap(a, b, classes, draws=2000, seed=42):
    a, b = align(a, b)
    if draws < 2 or a.seed.nunique() != 1:
        raise ValueError("Bootstrap one matched seed at a time, using >=2 draws")
    for frame in [a, b]:
        if (
            ((frame[["y_true", "prediction"]] < 0) | (frame[["y_true", "prediction"]] >= len(classes)))
            .any()
            .any()
        ):
            raise ValueError("Label outside declared class space")
    groups = [g.index.to_numpy() for _, g in a.groupby("group", sort=True)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        aa, bb = a.iloc[indices], b.iloc[indices]
        values.append(
            [
                float((aa.y_true == aa.prediction).mean() - (bb.y_true == bb.prediction).mean()),
                macro_f1(aa.y_true, aa.prediction, len(classes))
                - macro_f1(bb.y_true, bb.prediction, len(classes)),
            ]
        )
    values = np.asarray(values)
    return {
        name: {
            "ci95_difference": np.quantile(values[:, i], [0.025, 0.975]).tolist(),
            "draws": draws,
            "resampling_unit": "group",
            "groups": len(groups),
        }
        for i, name in enumerate(["accuracy", "macro_f1"])
    }
