"""Two-stage training and a separate, frozen TEST evaluation entry point."""

import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import assert_partitions, development_data, read_csv
from .elm import RidgeELM
from .metrics import evaluate, fit_temperature, macro_f1, probabilities
from .model import FeatureEncoder, TrainingModel, focal_loss, inverse_frequency_weights
from .preprocessing import Preprocessor
from .utils import environment, fresh_dir, load_state, read_json, save_state, seed_all, sha256, write_json


def loader(x, c, y, batch, shuffle=False, seed=7):
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(c), torch.from_numpy(y)),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def neural_scores(model, batches, device):
    model.eval()
    return np.concatenate([model(x.to(device), c.to(device)).cpu().numpy() for x, c, _ in batches])


@torch.no_grad()
def representations(encoder, batches, device):
    encoder.eval()
    for x, c, y in batches:
        yield encoder(x.to(device), c.to(device)).cpu().numpy(), y.numpy()


def prediction_frame(frame, scores, classes, temperature, seed):
    p = probabilities(scores, temperature)
    result = frame[[col for col in ["row_id", "group", "fingerprint", "split"] if col in frame]].copy()
    if "row_id" not in result:
        result["row_id"] = [f"prediction_{i}" for i in range(len(frame))]
    result["seed"] = seed
    result["prediction"] = p.argmax(1)
    result["predicted_class"] = [classes[i] for i in p.argmax(1)]
    for i in range(len(classes)):
        result[f"score_{i}"] = scores[:, i]
        result[f"prob_{i}"] = p[:, i]
    return result


def fit(config, out):
    config = deepcopy(config)
    out = fresh_dir(out)
    seed = int(config.get("seed", 7))
    seed_all(seed, config.get("threads", 4))
    schema, data, audit = development_data(config["data_dir"])
    train_frame, val_frame = data["train"], data["val"]
    preprocessor = Preprocessor(schema).fit(train_frame)
    preprocessor.save(out / "preprocessor.json")
    x, c = preprocessor.transform(train_frame)
    vx, vc = preprocessor.transform(val_frame)
    y, vy = schema.labels(train_frame), schema.labels(val_frame)
    cfg = config.get("training", {})
    epochs, patience, batch = cfg.get("epochs", 100), cfg.get("patience", 15), cfg.get("batch_size", 512)
    if min(epochs, patience, batch) < 1:
        raise ValueError("epochs, patience, batch_size must be positive")
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = config.get("model", {})
    encoder = FeatureEncoder(len(schema.numeric), preprocessor.cardinalities, **model_cfg).to(device)
    model = TrainingModel(encoder, len(schema.classes)).to(device)
    weights = inverse_frequency_weights(y, len(schema.classes)).to(device)
    if not cfg.get("class_weighted", True):
        weights = torch.ones_like(weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("weight_decay", 1e-4)
    )
    train_loader = loader(x, c, y, batch, True, seed)
    val_loader = loader(vx, vc, vy, batch, False, seed)
    start, history, best, stale = time.perf_counter(), [], -np.inf, 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xn, xc, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xn.to(device), xc.to(device))
            loss = focal_loss(logits, target.to(device), weights, cfg.get("gamma", 2.0))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if cfg.get("clip_grad_norm"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["clip_grad_norm"], error_if_nonfinite=True
                )
            optimizer.step()
            loss_sum += loss.item() * len(target)
        scores = neural_scores(model, val_loader, device)
        score = macro_f1(vy, scores.argmax(1), len(schema.classes))
        history.append({"epoch": epoch, "train_loss": loss_sum / len(y), "validation_macro_f1": score})
        if score > best:
            best, stale = score, 0
            save_state(model, out / "encoder_training.npz")
            selected_epoch = epoch
        else:
            stale += 1
        print(f"epoch={epoch} loss={loss_sum / len(y):.6f} validation_macro_f1={score:.6f}", flush=True)
        if stale >= patience:
            break
    pd.DataFrame(history).to_csv(out / "history.csv", index=False, lineterminator="\n")
    load_state(model, out / "encoder_training.npz", device)
    encoder = model.encoder.eval().requires_grad_(False)
    save_state(encoder, out / "encoder.npz")
    encoder_digest = sha256(out / "encoder.npz")
    elm = RidgeELM(encoder.width, len(schema.classes), seed=seed, **config.get("elm", {}))
    elm.fit_batches(representations(encoder, loader(x, c, y, batch), device))
    elm.save(out)
    val_scores = np.concatenate([elm.predict(z) for z, _ in representations(encoder, val_loader, device)])
    temperature = fit_temperature(val_scores, vy)
    preds = prediction_frame(val_frame, val_scores, schema.classes, temperature, seed)
    preds["y_true"] = vy
    preds.to_csv(out / "validation_predictions.csv", index=False, lineterminator="\n")
    write_json(
        out / "validation_metrics.json", evaluate(vy, probabilities(val_scores, temperature), schema.classes)
    )
    write_json(
        out / "validation_uncalibrated_metrics.json",
        evaluate(vy, probabilities(val_scores, 1), schema.classes),
    )
    bundle = {
        "format_version": 1,
        "classes": schema.classes,
        "seed": seed,
        "temperature": temperature,
        "selected_epoch": selected_epoch,
        "encoder_validation_macro_f1": best,
        "validation_macro_f1": macro_f1(vy, val_scores.argmax(1), len(schema.classes)),
        "data_audit": audit,
        "config": config,
        "environment": environment(),
        "fit_rows": len(y),
        "validation_rows": len(vy),
        "class_weights": weights.cpu().numpy(),
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "elm_parameters": elm.weight.size + elm.bias.size + elm.beta.size,
        "elapsed_seconds": time.perf_counter() - start,
        "status": "fitted and calibrated; TEST not evaluated by fit",
        "artifacts": {
            name: sha256(out / name) for name in ["encoder.npz", "elm.npz", "elm.json", "preprocessor.json"]
        },
    }
    assert encoder_digest == sha256(out / "encoder.npz")
    write_json(out / "bundle.json", bundle)
    return bundle


class Pipeline:
    def __init__(self, directory, device="cpu"):
        root = Path(directory)
        self.bundle = read_json(root / "bundle.json")
        if self.bundle["format_version"] != 1:
            raise ValueError("Unsupported bundle version")
        if set(self.bundle["artifacts"]) != {"encoder.npz", "elm.npz", "elm.json", "preprocessor.json"}:
            raise ValueError("Incomplete bundle integrity manifest")
        for name, digest in self.bundle["artifacts"].items():
            if (
                name not in {"encoder.npz", "elm.npz", "elm.json", "preprocessor.json"}
                or sha256(root / name) != digest
            ):
                raise ValueError(f"Bundle integrity mismatch: {name}")
        self.preprocessor = Preprocessor.load(root / "preprocessor.json")
        self.schema = self.preprocessor.schema
        if self.schema.classes != self.bundle["classes"]:
            raise ValueError("Class order differs inside bundle")
        self.encoder = FeatureEncoder(
            len(self.schema.numeric),
            self.preprocessor.cardinalities,
            **self.bundle["config"].get("model", {}),
        ).to(device)
        load_state(self.encoder, root / "encoder.npz", device)
        self.encoder.eval().requires_grad_(False)
        self.elm = RidgeELM.load(root)
        self.device = device
        self.temperature = self.bundle["temperature"]

    def scores(self, frame, batch_size=1024):
        if not len(frame) or batch_size < 1:
            raise ValueError("Require nonempty inputs and a positive batch size")
        x, c = self.preprocessor.transform(frame)
        batches = loader(x, c, np.zeros(len(x), dtype=np.int64), batch_size)
        return np.concatenate(
            [self.elm.predict(z) for z, _ in representations(self.encoder, batches, self.device)]
        )

    def predict(self, frame, batch_size=1024):
        scores = self.scores(frame, batch_size)
        return prediction_frame(frame, scores, self.schema.classes, self.temperature, self.bundle["seed"])


def test_bundle(bundle, data_dir, out, device="cpu"):
    pipeline = Pipeline(bundle, device)
    root = Path(data_dir)
    for name, key in [("schema.json", "schema_sha256"), ("manifest.csv", "manifest_sha256")]:
        if sha256(root / name) != pipeline.bundle["data_audit"][key]:
            raise ValueError(f"Changed locked {name}")
    assert_partitions(read_csv(root / "manifest.csv"))
    expected = pipeline.bundle["data_audit"]["split_files"]["test"]
    if sha256(root / "test.csv") != expected:
        raise ValueError("TEST data differs from the locked preparation audit")
    frame = read_csv(root / "test.csv")
    if set(frame.split) != {"test"}:
        raise ValueError("Expected TEST partition")
    labels = pipeline.schema.labels(frame)
    output = fresh_dir(out)
    scores = pipeline.scores(frame)
    predictions = prediction_frame(
        frame, scores, pipeline.schema.classes, pipeline.temperature, pipeline.bundle["seed"]
    )
    predictions["y_true"] = labels
    predictions.to_csv(output / "predictions.csv", index=False, lineterminator="\n")
    result = evaluate(labels, probabilities(scores, pipeline.temperature), pipeline.schema.classes)
    write_json(output / "metrics.json", result)
    write_json(
        output / "provenance.json",
        {
            "bundle_sha256": sha256(Path(bundle) / "bundle.json"),
            "test_sha256": expected,
            "prediction_sha256": sha256(output / "predictions.csv"),
            "test_used_for_selection": False,
        },
    )
    return result
