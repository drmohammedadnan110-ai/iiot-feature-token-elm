"""Frozen random features and a float64, streamed ridge normal-equation solve."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erf

from .utils import read_json, write_json


class RidgeELM:
    def __init__(
        self, inputs, classes, hidden=512, activation="gelu", ridge=0.01, initialization="orthogonal", seed=7
    ):
        if min(inputs, classes, hidden) < 1 or ridge <= 0 or not np.isfinite(ridge):
            raise ValueError("Require positive dimensions and finite positive ridge")
        if activation not in ["gelu", "relu", "tanh"] or initialization not in [
            "gaussian",
            "orthogonal",
            "trigonometric",
        ]:
            raise ValueError("Unknown ELM activation or initialization")
        self.config = {
            "inputs": inputs,
            "classes": classes,
            "hidden": hidden,
            "activation": activation,
            "ridge": ridge,
            "initialization": initialization,
            "seed": seed,
        }
        rng = np.random.default_rng(seed)
        if initialization == "gaussian":
            self.weight = rng.normal(0, 1 / np.sqrt(inputs), (inputs, hidden))
        else:
            blocks = []
            for _ in range((hidden + inputs - 1) // inputs):
                if initialization == "orthogonal":
                    q, r = np.linalg.qr(rng.normal(size=(inputs, inputs)))
                    q *= np.where(np.diag(r) >= 0, 1, -1)
                else:
                    # DCT-II orthonormal basis, independently signed and permuted per block.
                    i, j = np.arange(inputs)[:, None], np.arange(inputs)[None, :]
                    q = np.sqrt(2 / inputs) * np.cos(np.pi * (i + 0.5) * j / inputs)
                    q[:, 0] /= np.sqrt(2)
                    q = q[:, rng.permutation(inputs)] * rng.choice([-1, 1], inputs)
                blocks.append(q)
            self.weight = np.concatenate(blocks, axis=1)[:, :hidden]
        self.bias = rng.uniform(-1, 1, hidden)
        self.beta = None

    def features(self, x):
        z = np.asarray(x, dtype=np.float64) @ self.weight + self.bias
        activation = self.config["activation"]
        if activation == "gelu":
            return z * 0.5 * (1 + erf(z / np.sqrt(2)))
        return np.maximum(z, 0) if activation == "relu" else np.tanh(z)

    def fit_batches(self, batches):
        width, classes = self.config["hidden"], self.config["classes"]
        gram, target = np.zeros((width, width)), np.zeros((width, classes))
        count = 0
        for x, labels in batches:
            labels = np.asarray(labels, dtype=np.int64)
            if len(x) != len(labels) or (labels < 0).any() or (labels >= classes).any():
                raise ValueError("Invalid ELM labels")
            h = self.features(x)
            if not np.isfinite(h).all():
                raise ValueError("Nonfinite ELM features")
            gram += h.T @ h
            # Avoid materializing N x K one-hot targets for the full corpus.
            for cls in range(classes):
                target[:, cls] += h[labels == cls].sum(axis=0)
            count += len(labels)
        if not count:
            raise ValueError("No TRAIN observations for ELM")
        regularized = gram + self.config["ridge"] * np.eye(width)
        self.beta = cho_solve(cho_factor(regularized, lower=True), target)
        self.fit_rows = count
        self.solve_residual = float(
            np.linalg.norm(regularized @ self.beta - target) / max(np.linalg.norm(target), 1e-15)
        )
        if not np.isfinite(self.beta).all():
            raise FloatingPointError("Nonfinite ELM coefficients; review conditioning and ridge")
        return self

    def predict(self, x):
        if self.beta is None:
            raise RuntimeError("ELM is not fitted")
        return self.features(x) @ self.beta

    def save(self, root):
        from pathlib import Path

        root = Path(root)
        np.savez_compressed(root / "elm.npz", weight=self.weight, bias=self.bias, beta=self.beta)
        write_json(
            root / "elm.json",
            {"config": self.config, "fit_rows": self.fit_rows, "solve_residual": self.solve_residual},
        )

    @classmethod
    def load(cls, root):
        from pathlib import Path

        root = Path(root)
        meta = read_json(root / "elm.json")
        obj = cls(**meta["config"])
        with np.load(root / "elm.npz", allow_pickle=False) as arrays:
            obj.weight, obj.bias, obj.beta = [arrays[k] for k in ["weight", "bias", "beta"]]
        obj.fit_rows, obj.solve_residual = meta["fit_rows"], meta["solve_residual"]
        return obj
