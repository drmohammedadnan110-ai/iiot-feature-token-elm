"""Explicit feature schemas; TRAIN-only median/scale/vocabulary fitting."""

import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .utils import read_json, write_json


@dataclass
class Schema:
    numeric: list[str]
    categorical: list[str]
    label: str
    classes: list[str]
    group_columns: list[str] = None
    forbidden: list[str] = None
    category_aliases: dict = None

    def __post_init__(self):
        self.group_columns = self.group_columns or []
        self.forbidden = self.forbidden or []
        self.category_aliases = self.category_aliases or {}
        features = self.numeric + self.categorical
        if not features or len(set(features)) != len(features):
            raise ValueError("Require nonempty, distinct feature columns")
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("Require >=2 distinct ordered class names")
        denied = set(
            [self.label, "row_id", "split", "group", "fingerprint"] + self.forbidden + self.group_columns
        )
        if set(features) & denied:
            raise ValueError(f"Metadata/label/group feature leakage: {set(features) & denied}")
        if set(self.category_aliases) - set(self.categorical):
            raise ValueError("Category aliases may reference only categorical predictors")

    @classmethod
    def load(cls, path):
        return cls(**read_json(path))

    def check(self, frame, labeled=False):
        missing = set(self.numeric + self.categorical + ([self.label] if labeled else [])) - set(frame)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if labeled:
            unknown = set(frame[self.label].astype(str)) - set(self.classes)
            if unknown:
                raise ValueError(f"Unknown labels; provide an explicit label mapping: {sorted(unknown)}")

    def labels(self, frame):
        self.check(frame, labeled=True)
        return frame[self.label].map({c: i for i, c in enumerate(self.classes)}).to_numpy(np.int64)

    def category(self, series, name):
        series = series.astype("string").fillna("").str.strip()
        return series.replace(self.category_aliases.get(name, {}))


class Preprocessor:
    def __init__(self, schema):
        self.schema = schema
        self.state = None

    def fit(self, frame):
        self.schema.check(frame)
        if self.state is not None:
            raise RuntimeError("Preprocessor already fitted; create a new instance for a new experiment")
        if "split" not in frame or set(frame.split) != {"train"}:
            raise ValueError("Preprocessing may fit TRAIN only")
        numeric = (
            frame[self.schema.numeric]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        medians = numeric.median().fillna(0)
        imputed = numeric.fillna(medians)
        means, stds = imputed.mean(), imputed.std(ddof=0)
        stds = stds.mask(stds < 1e-12, 1.0).fillna(1.0)
        vocab = {}
        for name in self.schema.categorical:
            values = self.schema.category(frame[name], name)
            vocab[name] = {v: i + 1 for i, v in enumerate(sorted(set(values) - {""}))}
        self.state = {
            "medians": medians.tolist(),
            "means": means.tolist(),
            "stds": stds.tolist(),
            "vocab": vocab,
            "fit_rows": len(frame),
            "fit_partition": "train",
        }
        return self

    def transform(self, frame):
        if self.state is None:
            raise RuntimeError("Preprocessor is not fitted")
        self.schema.check(frame)
        x = frame[self.schema.numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        x = np.where(np.isfinite(x), x, np.asarray(self.state["medians"]))
        x = (x - self.state["means"]) / self.state["stds"]
        c = np.empty((len(frame), len(self.schema.categorical)), dtype=np.int64)
        for j, name in enumerate(self.schema.categorical):
            c[:, j] = (
                self.schema.category(frame[name], name)
                .map(self.state["vocab"][name])
                .fillna(0)
                .to_numpy(np.int64)
            )
        with np.errstate(over="ignore"):
            x = x.astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite transformed inputs")
        return x, c

    @property
    def cardinalities(self):
        return [len(self.state["vocab"][name]) + 1 for name in self.schema.categorical]

    def save(self, path):
        write_json(path, {"schema": asdict(self.schema), "state": self.state})

    @classmethod
    def load(cls, path):
        payload = read_json(path)
        obj = cls(Schema(**payload["schema"]))
        obj.state = payload["state"]
        return obj


def suggest_schema(csv, label, out):
    """Draft only; excludes likely identifiers and all label-like columns for review."""
    frame = pd.read_csv(csv, nrows=10000, keep_default_na=False)
    if label not in frame:
        raise ValueError(f"Label column absent: {label}")
    forbidden = [
        c
        for c in frame
        if c == label
        or re.search(
            r"label|attack|class|(^|[._ ])id$|ip[._ ]|[._ ]ip|src|dst|time|host|mac|file|capture|session",
            c,
            re.IGNORECASE,
        )
    ]
    numeric, categorical = [], []
    for name in frame.columns.difference(forbidden, sort=False):
        converted = pd.to_numeric(frame[name], errors="coerce")
        (numeric if converted.notna().mean() > 0.95 else categorical).append(name)
    write_json(
        out,
        {
            "numeric": numeric,
            "categorical": categorical,
            "label": label,
            "classes": sorted(frame[label].astype(str).unique()),
            "group_columns": [],
            "forbidden": forbidden,
        },
    )
    return {
        "status": "DRAFT: review features, units, label classes and provenance; sampled first 10000 rows only",
        "potential_identifier_or_label_columns_excluded": forbidden,
    }
