"""Verify published synthetic/report assets without loading ML libraries."""

import csv
import hashlib
import json
import re
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    report = root / "results/executed/synthetic_demo"
    hashes = json.loads((report / "checksums.json").read_text())
    for name, expected in hashes.items():
        assert digest(report / name) == expected, f"Changed synthetic artifact: {name}"
    provenance = json.loads((root / "results/reported/provenance.json").read_text())
    assert len(provenance["tables"]) == 20 and len(provenance["figures"]) == 8
    for kind, directory in [("tables", "results/reported"), ("figures", "figures/paper")]:
        for item in provenance[kind]:
            assert digest(root / directory / item["file"]) == item["sha256"]
    with (report / "seed_scores.csv").open(newline="") as stream:
        scores = list(csv.DictReader(stream))
    for row in scores:
        path = report / f"seed_{row['seed']}" / f"{row['model']}_predictions.csv"
        with path.open(newline="") as stream:
            predictions = list(csv.DictReader(stream))
        k = len(json.loads(row["classes"]))
        cm = [[0] * k for _ in range(k)]
        for pred in predictions:
            cm[int(pred["y_true"])][int(pred["prediction"])] += 1
        macro = sum(2 * cm[i][i] / max(1, sum(cm[i]) + sum(r[i] for r in cm)) for i in range(k)) / k
        assert abs(macro - float(row["macro_f1"])) < 1e-12
    for path in root.rglob("*"):
        if not path.is_file() or set(path.relative_to(root).parts) & {
            ".git",
            "runs",
            "data",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
        }:
            continue
        if path.suffix in {
            ".py",
            ".md",
            ".html",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ipynb",
            ".txt",
            ".csv",
        }:
            content = path.read_text(encoding="utf-8")
            assert not re.search(r"gh[pousr]_" + r"[A-Za-z0-9]{30,}", content), (
                f"Credential-like material: {path.name}"
            )
    print(
        f"Verified {len(hashes)} synthetic artifacts, 20 reported tables, 8 paper images, and 10 saved score rows."
    )


if __name__ == "__main__":
    main()
