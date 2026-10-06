"""Reproducible synthetic plumbing checks, never a source of paper performance claims."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import fresh_dir, read_json, sha256, write_json

EDGE_CLASSES = [
    "Normal",
    "DDoS_UDP",
    "DDoS_ICMP",
    "SQL_Injection",
    "DDoS_TCP",
    "DDoS_HTTP",
    "Password",
    "Vulnerability_Scanner",
    "Port_Scanning",
    "XSS",
    "Uploading",
    "Backdoor",
    "OS_Fingerprinting",
    "Ransomware",
    "MITM",
]


def synthetic_data(out, classes=None, minority_rows=60, seed=2026):
    """Gaussian toy features with overlap, missing values, and categorical covariates."""
    classes = classes or EDGE_CLASSES
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2, (len(classes), 8))
    pieces = []
    for k, label in enumerate(classes):
        n = minority_rows * (4 if k == 0 else 1)
        features = rng.normal(centers[k], 1.4, (n, 8))
        features[rng.random(features.shape) < 0.025] = np.nan
        part = pd.DataFrame(features, columns=[f"toy_numeric_{i}" for i in range(8)])
        part["toy_protocol"] = rng.choice(["tcp", "udp", "icmp"], n)
        part["toy_device"] = [f"device_type_{i}" for i in rng.integers(0, 4, n)]
        part["label"] = label
        pieces.append(part)
    pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed).to_csv(
        out / "synthetic.csv", index=False, lineterminator="\n"
    )
    schema = {
        "numeric": [f"toy_numeric_{i}" for i in range(8)],
        "categorical": ["toy_protocol", "toy_device"],
        "label": "label",
        "classes": classes,
        "group_columns": [],
        "forbidden": [],
    }
    write_json(out / "schema.json", schema)
    return out / "synthetic.csv", out / "schema.json"


def run_demo(out, seeds=(7,)):
    from .baselines import benchmark
    from .data import prepare
    from .engine import fit, test_bundle
    from .plots import plot_history, plot_predictions
    from .statistics import paired_group_bootstrap, paired_seed_tests

    out = fresh_dir(out)
    csv, schema = synthetic_data(out / "raw")
    prepared = prepare([csv], schema, out / "prepared")
    reports = out / "reports"
    reports.mkdir()
    audit = read_json(prepared / "audit.json")
    config = {
        "data_dir": str(prepared.resolve()),
        "device": "cpu",
        "threads": 2,
        "model": {"width": 32, "layers": 1, "heads": 4, "expansion": 2, "dropout": 0.1},
        "training": {
            "epochs": 12,
            "patience": 4,
            "batch_size": 64,
            "lr": 0.003,
            "weight_decay": 0.0001,
            "gamma": 2.0,
            "class_weighted": True,
        },
        "elm": {"hidden": 64, "activation": "gelu", "ridge": 0.1, "initialization": "orthogonal"},
    }
    rows = []
    for seed in seeds:
        config["seed"] = seed
        model = out / f"proposed_{seed}"
        fit(config, model)
        tested = out / f"test_{seed}"
        metrics = test_bundle(model, prepared, tested)
        baseline = out / f"logistic_{seed}"
        benchmark(prepared, "logistic", baseline, seed, evaluate_test=True)
        base_metrics = read_json(baseline / "test_metrics.json")
        for name, result in [("proposed", metrics), ("logistic", base_metrics)]:
            rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "macro_f1": result["macro_f1"],
                    "accuracy": result["accuracy"],
                    "test_sha256": audit["split_files"]["test"],
                    "classes": json.dumps(EDGE_CLASSES),
                }
            )
        run = reports / f"seed_{seed}"
        run.mkdir()
        for source, name in [
            (tested / "predictions.csv", "proposed_predictions.csv"),
            (tested / "metrics.json", "proposed_metrics.json"),
            (baseline / "test_predictions.csv", "logistic_predictions.csv"),
            (baseline / "test_metrics.json", "logistic_metrics.json"),
            (model / "history.csv", "history.csv"),
        ]:
            shutil.copyfile(source, run / name)
        bundle = read_json(model / "bundle.json")
        bundle["config"]["data_dir"] = "<demo>/prepared"
        for source in bundle["data_audit"]["sources"]:
            source["filename"] = "<demo>/raw/synthetic.csv"
        write_json(run / "training_provenance.json", bundle)
        shutil.copyfile(tested / "provenance.json", run / "test_provenance.json")
    scores = pd.DataFrame(rows)
    scores.to_csv(reports / "seed_scores.csv", index=False, lineterminator="\n")
    if len(seeds) >= 2:
        write_json(reports / "paired_seed_statistics.json", paired_seed_tests(scores, "proposed"))
    first = reports / f"seed_{seeds[0]}"
    a, b = [pd.read_csv(first / f"{name}_predictions.csv") for name in ["proposed", "logistic"]]
    write_json(reports / "group_bootstrap.json", paired_group_bootstrap(a, b, EDGE_CLASSES, draws=300))
    plot_predictions(a, EDGE_CLASSES, reports / "figures", title="Synthetic demonstration — not paper data")
    plot_history(first / "history.csv", reports / "figures" / "training_history.png")
    write_json(
        reports / "scope.json",
        {
            "data": "SYNTHETIC Gaussian toy data, not network captures",
            "purpose": "Executed integration evidence only; does not validate paper accuracy or real-world detection",
            "rows": len(pd.read_csv(csv)),
            "classes": EDGE_CLASSES,
            "seeds": list(seeds),
            "source_sha256": sha256(csv),
            "test_sha256": audit["split_files"]["test"],
            "protocol": "Fixed split seed 42; training seeds vary; no HPO; both methods predeclared before TEST evaluation",
            "bundle_provenance": "Reports redact local input paths; numeric model bundles remain in the local demo directory",
        },
    )
    write_json(
        reports / "checksums.json",
        {p.relative_to(reports).as_posix(): sha256(p) for p in sorted(reports.rglob("*")) if p.is_file()},
    )
    return {"reports": str(reports), "scope": "SYNTHETIC ONLY", "runs": rows}
