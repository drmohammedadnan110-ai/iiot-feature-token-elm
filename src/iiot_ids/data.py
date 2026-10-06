"""Source manifests and reproducible group-preserving 64/16/20 reference partitions."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .preprocessing import Schema
from .utils import fresh_dir, sha256, write_json


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def fingerprints(frame, schema):
    normalized = frame[schema.numeric + schema.categorical].copy()
    for name in schema.numeric:
        normalized[name] = pd.to_numeric(normalized[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        normalized[name] = normalized[name].mask(normalized[name] == 0, 0.0)
    for name in schema.categorical:
        normalized[name] = schema.category(normalized[name], name)
    # Fixed-key pandas fingerprints are grouping aids; file provenance uses SHA-256.
    return pd.util.hash_pandas_object(normalized, index=False).map(lambda x: f"{x:016x}")


def connected_groups(frame, schema):
    for name in schema.group_columns:
        if name not in frame:
            raise ValueError(f"Missing configured group column: {name}")
        missing = frame[name].isna() | frame[name].astype(str).str.strip().str.lower().isin(
            ["", "nan", "none"]
        )
        if missing.any():
            raise ValueError(
                f"Missing values in configured grouping column {name}; supply verified identifiers"
            )
    parent = np.arange(len(frame))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for column in ["fingerprint"] + schema.group_columns:
        seen = {}
        for index, value in enumerate(frame[column].astype(str)):
            if not value or value.lower() in ["nan", "none"]:
                continue
            if value in seen:
                union(index, seen[value])
            else:
                seen[value] = index
    row_ids = frame.row_id.to_numpy()
    return [row_ids[find(i)] for i in range(len(frame))]


def assert_partitions(frame):
    required = {"row_id", "split", "group", "fingerprint"}
    if not required <= set(frame):
        raise ValueError(f"Missing split metadata: {required - set(frame)}")
    if not len(frame) or frame.row_id.duplicated().any():
        raise ValueError("Empty data or repeated row ID")
    if not set(frame.split) <= {"train", "val", "test", "excluded"}:
        raise ValueError("Unrecognized split")
    active = frame[frame.split != "excluded"]
    for key in ["group", "fingerprint"]:
        if active[key].isna().any() or (active[key] == "").any():
            raise ValueError(f"Missing {key}")
        if (active.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Cross-partition leakage through {key}")


def allocate(frame, labels, seed):
    """Exact nominal fractions for independent rows; approximate fractions for groups."""
    indices = np.arange(len(frame))
    if frame.group.nunique() == len(frame):
        dev, _test = train_test_split(indices, test_size=0.2, stratify=labels, random_state=seed)
        train, val = train_test_split(dev, test_size=0.2, stratify=labels[dev], random_state=seed)
    else:
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        dev, _test = next(splitter.split(indices, labels, frame.group))
        inner = StratifiedGroupKFold(5, shuffle=True, random_state=seed + 1)
        tr, va = next(inner.split(dev, labels[dev], frame.group.iloc[dev]))
        train, val = dev[tr], dev[va]
    result = np.full(len(frame), "test", dtype=object)
    result[train], result[val] = "train", "val"
    return result


def prepare(paths, schema_path, out, seed=42, split_manifest=None):
    schema = Schema.load(schema_path)
    parts, sources = [], []
    for path in paths:
        frame = read_csv(path)
        schema.check(frame, labeled=True)
        if {"row_id", "split", "group", "fingerprint"} & set(frame):
            raise ValueError("Raw input has reserved pipeline metadata columns")
        digest = sha256(path)
        frame["row_id"] = [f"{digest}:{i}" for i in range(len(frame))]
        parts.append(frame)
        sources.append({"filename": str(path), "sha256": digest, "rows": len(frame)})
    frame = pd.concat(parts, ignore_index=True)
    frame["fingerprint"] = fingerprints(frame, schema)
    frame["group"] = connected_groups(frame, schema)
    if split_manifest:
        fixed = read_csv(split_manifest)
        if fixed.row_id.duplicated().any() or set(fixed.row_id) != set(frame.row_id):
            raise ValueError("Supplied split manifest must contain every row ID exactly once")
        frame["split"] = frame.row_id.map(fixed.set_index("row_id").split)
    else:
        frame["split"] = allocate(frame, schema.labels(frame), seed)
    assert_partitions(frame)
    for split in ["train", "val", "test"]:
        if set(frame.loc[frame.split == split, schema.label]) != set(schema.classes):
            raise ValueError(
                f"Missing class in {split}; supply a reviewed split manifest or a suitable group scheme"
            )
    out = fresh_dir(out)
    for split in ["train", "val", "test"]:
        frame.loc[frame.split == split].to_csv(out / f"{split}.csv", index=False, lineterminator="\n")
    frame[["row_id", "group", "fingerprint", "split", schema.label]].to_csv(
        out / "manifest.csv", index=False, lineterminator="\n"
    )
    from dataclasses import asdict

    write_json(out / "schema.json", asdict(schema))
    write_json(
        out / "audit.json",
        {
            "sources": sources,
            "source_schema_sha256": sha256(schema_path),
            "schema_sha256": sha256(out / "schema.json"),
            "manifest_sha256": sha256(out / "manifest.csv"),
            "seed": seed,
            "split_files": {s: sha256(out / f"{s}.csv") for s in ["train", "val", "test"]},
            "counts": frame.groupby(["split", schema.label]).size().to_dict(),
            "duplicate_predictor_rows": int(frame.fingerprint.duplicated().sum()),
            "conflicting_label_fingerprints": int(
                (frame.groupby("fingerprint")[schema.label].nunique() > 1).sum()
            ),
            "group_columns": schema.group_columns,
            "original_paper_membership_reproduced": False,
            "note": "Reference partition, not authenticated Protocol B membership. Group preservation can change nominal ratios.",
        },
    )
    return out


def development_data(directory):
    """Load only TRAIN and VALIDATION values; inspect TEST metadata via saved manifest."""
    from pathlib import Path

    root = Path(directory)
    audit = __import__("json").loads((root / "audit.json").read_text())
    if (
        sha256(root / "schema.json") != audit["schema_sha256"]
        or sha256(root / "manifest.csv") != audit["manifest_sha256"]
    ):
        raise ValueError("Changed prepared schema or manifest since preparation")
    schema = Schema.load(root / "schema.json")
    manifest = read_csv(root / "manifest.csv")
    assert_partitions(manifest)
    result = {}
    for split in ["train", "val"]:
        if sha256(root / f"{split}.csv") != audit["split_files"][split]:
            raise ValueError(f"Changed {split} data since preparation")
        result[split] = read_csv(root / f"{split}.csv")
        schema.check(result[split], labeled=True)
        if set(result[split].split) != {split}:
            raise ValueError("Wrong partition contents")
    return schema, result, audit
