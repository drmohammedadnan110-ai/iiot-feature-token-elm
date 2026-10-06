"""Explicit source-only transfer and controlled leave-one-attack-out evaluation."""

import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .data import development_data, read_csv
from .engine import Pipeline, fit, prediction_frame
from .metrics import evaluate, probabilities
from .utils import fresh_dir, read_json, sha256, write_json


def unknown_metrics(known_validation_scores, test_scores, test_is_unknown, known_fpr=0.05):
    if not 0 < known_fpr < 1:
        raise ValueError("known_fpr must be between zero and one")
    validation = np.asarray(known_validation_scores)
    scores, unknown = np.asarray(test_scores), np.asarray(test_is_unknown, bool)
    if not len(validation) or len(scores) != len(unknown) or np.unique(unknown).size != 2:
        raise ValueError("Need known-only validation scores and both known/unknown TEST cases")
    if not np.isfinite(validation).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite novelty scores")
    threshold = float(np.quantile(validation, 1 - known_fpr, method="higher"))
    flagged = scores > threshold
    fpr, tpr, _ = roc_curve(unknown, scores)
    return {
        "threshold": threshold,
        "threshold_fit": "known VALIDATION only",
        "requested_known_fpr": known_fpr,
        "validation_false_unknown_rate": float((validation > threshold).mean()),
        "unknown_recall": float(flagged[unknown].mean()),
        "known_false_unknown_rate": float(flagged[~unknown].mean()),
        "auroc": float(roc_auc_score(unknown, scores)),
        "aupr": float(average_precision_score(unknown, scores)),
        "fpr_at_95tpr": float(fpr[tpr >= 0.95].min()),
        "note": "FPR@95TPR is a retrospective TEST ROC metric, not the validation-chosen operating threshold.",
    }


def leave_one_out(config, heldout, out, known_fpr=0.05):
    schema, data, audit = development_data(config["data_dir"])
    if heldout not in schema.classes or heldout == "Normal":
        raise ValueError("Hold out an attack class, not Normal")
    if len(schema.classes) < 3:
        raise ValueError("Need at least two known classes after holdout")
    out = fresh_dir(out)
    root = Path(config["data_dir"])
    if sha256(root / "test.csv") != audit["split_files"]["test"]:
        raise ValueError("Changed original TEST file")
    native_test = read_csv(root / "test.csv")
    known_schema = deepcopy(schema)
    known_schema.classes.remove(heldout)
    prepared = out / "prepared"
    prepared.mkdir()
    exclusions = {}
    for split in ["train", "val"]:
        part = data[split]
        keep = part[schema.label] != heldout
        exclusions[split] = int((~keep).sum())
        part.loc[keep].to_csv(prepared / f"{split}.csv", index=False, lineterminator="\n")
    native_test.to_csv(prepared / "test.csv", index=False, lineterminator="\n")
    manifest = read_csv(root / "manifest.csv")
    excluded = manifest.split.isin(["train", "val"]) & (manifest[schema.label] == heldout)
    manifest.loc[excluded, "split"] = "excluded"
    manifest.to_csv(prepared / "manifest.csv", index=False, lineterminator="\n")
    write_json(prepared / "schema.json", asdict(known_schema))
    audit["split_files"] = {s: sha256(prepared / f"{s}.csv") for s in ["train", "val", "test"]}
    audit["schema_sha256"] = sha256(prepared / "schema.json")
    audit["manifest_sha256"] = sha256(prepared / "manifest.csv")
    audit.update(heldout=heldout, excluded_from_fitting_and_calibration=exclusions)
    write_json(prepared / "audit.json", audit)
    revised = deepcopy(config)
    revised["data_dir"] = str(prepared)
    fit(revised, out / "model")
    pipeline = Pipeline(out / "model", config.get("device", "cpu"))
    validation = read_csv(prepared / "val.csv")
    vp = probabilities(pipeline.scores(validation), pipeline.temperature)
    score = pipeline.scores(native_test)
    p = probabilities(score, pipeline.temperature)
    unknown = (native_test[schema.label] == heldout).to_numpy()
    result = unknown_metrics(1 - vp.max(1), 1 - p.max(1), unknown, known_fpr)
    known = native_test.loc[~unknown]
    result["known_class_metrics"] = evaluate(known_schema.labels(known), p[~unknown], known_schema.classes)
    result["heldout_class"] = heldout
    predictions = prediction_frame(
        native_test, score, known_schema.classes, pipeline.temperature, revised.get("seed", 7)
    )
    predictions["is_unknown"] = unknown
    predictions["novelty"] = 1 - p.max(1)
    predictions["flagged_unknown"] = predictions.novelty > result["threshold"]
    predictions.to_csv(out / "predictions.csv", index=False, lineterminator="\n")
    write_json(out / "metrics.json", result)
    return result


def transfer(bundle, target_csv, mapping_path, out, device="cpu"):
    """No target fitting; explicit feature units and label ontology supplied by caller."""
    model = Pipeline(bundle, device)
    mapping = read_json(mapping_path)
    target = read_csv(target_csv)
    expected = model.schema.numeric + model.schema.categorical
    if set(mapping["features"]) != set(expected):
        raise ValueError("Mapping must define every source-model predictor exactly once")
    source_style = pd.DataFrame(index=target.index)
    for name in expected:
        spec = mapping["features"][name]
        if spec["column"] not in target or not spec.get("meaning"):
            raise ValueError(f"Missing column or semantic justification for {name}")
        values = target[spec["column"]]
        if name in model.schema.numeric:
            if "unit" not in spec:
                raise ValueError(f"State unit conversion explicitly for {name}")
            source_style[name] = pd.to_numeric(values, errors="coerce") * spec.get("scale", 1) + spec.get(
                "offset", 0
            )
        else:
            source_style[name] = values.replace(spec.get("values", {}))
    label_column = mapping["label_column"]
    labels = target[label_column].map(mapping["labels"])
    if labels.isna().any() or not set(labels) <= set(model.schema.classes):
        raise ValueError("Every target label must map explicitly into the source model class space")
    source_style[model.schema.label] = labels
    digest = sha256(target_csv)
    source_style["row_id"] = [f"{digest}:{i}" for i in range(len(target))]
    output = fresh_dir(out)
    scores = model.scores(source_style)
    predicted = prediction_frame(
        source_style, scores, model.schema.classes, model.temperature, model.bundle["seed"]
    )
    predicted["y_true"] = model.schema.labels(source_style)
    predicted.to_csv(output / "predictions.csv", index=False, lineterminator="\n")
    result = evaluate(predicted.y_true, probabilities(scores, model.temperature), model.schema.classes)
    write_json(output / "metrics.json", result)
    write_json(
        output / "provenance.json",
        {
            "mode": "direct source-only transfer",
            "target_fitting": False,
            "temperature_source": "source validation",
            "target_sha256": sha256(target_csv),
            "mapping_sha256": sha256(mapping_path),
            "bundle_sha256": sha256(Path(bundle) / "bundle.json"),
            "mapping": mapping,
        },
    )
    return result


def profile(bundle, csv, out, batch_size=1, warmup=10, repetitions=50, device="cpu"):
    if min(batch_size, repetitions) < 1 or warmup < 0:
        raise ValueError("Invalid profiling settings")
    model = Pipeline(bundle, device)
    frame = read_csv(csv).iloc[:batch_size]
    if len(frame) != batch_size:
        raise ValueError("Not enough input observations")
    times = []
    for i in range(warmup + repetitions):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        model.predict(frame, batch_size)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if i >= warmup:
            times.append(elapsed)
    result = {
        "batch_size": batch_size,
        "repetitions": repetitions,
        "warmup": warmup,
        "device": device,
        "median_batch_ms": np.median(times) * 1000,
        "p95_batch_ms": np.quantile(times, 0.95) * 1000,
        "measured_samples_per_second": repetitions * batch_size / sum(times),
        "scope": "preprocessing + encoder + CPU ELM + probability/frame construction, excluding file I/O and model load",
    }
    write_json(out, result)
    return result
