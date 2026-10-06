"""Classical reference baselines on the same TRAIN-only preprocessing and fixed splits."""

from pathlib import Path

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from .data import development_data, read_csv
from .engine import prediction_frame
from .metrics import evaluate, fit_temperature, probabilities
from .preprocessing import Preprocessor
from .utils import environment, fresh_dir, seed_all, sha256, write_json

NAMES = [
    "logistic",
    "naive_bayes",
    "knn",
    "svm",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "adaboost",
    "hist_gradient_boosting",
    "mlp",
    "xgboost",
    "lightgbm",
    "catboost",
]


def make_estimator(name, seed, parameters):
    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    constructors = {
        "logistic": (LogisticRegression, {"max_iter": 1000}),
        "naive_bayes": (GaussianNB, {}),
        "knn": (KNeighborsClassifier, {}),
        "svm": (SVC, {"probability": False}),
        "decision_tree": (DecisionTreeClassifier, {}),
        "random_forest": (RandomForestClassifier, {"n_jobs": 1}),
        "extra_trees": (ExtraTreesClassifier, {"n_jobs": 1}),
        "adaboost": (AdaBoostClassifier, {}),
        "hist_gradient_boosting": (HistGradientBoostingClassifier, {"early_stopping": False}),
        "mlp": (MLPClassifier, {"early_stopping": False, "max_iter": 200}),
    }
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**{"n_jobs": 1, **parameters, "random_state": seed})
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**{"n_jobs": 1, "verbosity": -1, **parameters, "random_state": seed})
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            **{
                "thread_count": 1,
                "verbose": False,
                "allow_writing_files": False,
                **parameters,
                "random_seed": seed,
            }
        )
    if name not in constructors:
        raise ValueError(f"Unknown baseline; choose {NAMES}")
    constructor, defaults = constructors[name]
    options = {**defaults, **parameters}
    if "random_state" in constructor().get_params():
        options["random_state"] = seed
    return constructor(**options)


def scores(estimator, x, classes):
    if hasattr(estimator, "predict_proba"):
        p = np.clip(estimator.predict_proba(x), 1e-12, 1)
        if list(estimator.classes_) != list(range(classes)):
            raise ValueError("Estimator class order mismatch")
        return np.log(p)
    result = estimator.decision_function(x)
    return np.column_stack([-result, result]) if result.ndim == 1 else result


def benchmark(data_dir, name, out, seed=7, parameters=None, evaluate_test=False):
    seed_all(seed)
    schema, data, audit = development_data(data_dir)
    preprocessor = Preprocessor(schema).fit(data["train"])
    transformed = {key: preprocessor.transform(frame) for key, frame in data.items()}
    cardinalities = preprocessor.cardinalities
    encoding = None
    if cardinalities:
        encoding = OneHotEncoder(
            categories=[np.arange(n) for n in cardinalities], sparse_output=False, handle_unknown="error"
        )
        encoding.fit(transformed["train"][1])

    def matrix(values):
        x, c = values
        return np.column_stack([x, encoding.transform(c)]) if encoding is not None else x

    estimator = make_estimator(name, seed, parameters or {})
    estimator.fit(matrix(transformed["train"]), schema.labels(data["train"]))
    validation_scores = scores(estimator, matrix(transformed["val"]), len(schema.classes))
    temperature = fit_temperature(validation_scores, schema.labels(data["val"]))
    out = fresh_dir(out)
    if evaluate_test:
        root = Path(data_dir)
        if sha256(root / "test.csv") != audit["split_files"]["test"]:
            raise ValueError("Changed TEST partition")
        data["test"] = read_csv(root / "test.csv")
        transformed["test"] = preprocessor.transform(data["test"])
    for split in ["val", "test"] if evaluate_test else ["val"]:
        score = (
            validation_scores
            if split == "val"
            else scores(estimator, matrix(transformed[split]), len(schema.classes))
        )
        frame = prediction_frame(data[split], score, schema.classes, temperature, seed)
        frame["y_true"] = schema.labels(data[split])
        frame.to_csv(out / f"{split}_predictions.csv", index=False, lineterminator="\n")
        write_json(
            out / f"{split}_metrics.json",
            evaluate(frame.y_true, probabilities(score, temperature), schema.classes),
        )
    write_json(
        out / "provenance.json",
        {
            "model": name,
            "seed": seed,
            "parameters": parameters or {},
            "temperature": temperature,
            "classes": schema.classes,
            "data_audit": audit,
            "environment": environment(),
            "test_evaluated": evaluate_test,
            "note": "Reference defaults, not undocumented manuscript baseline settings; estimator objects are not pickled.",
        },
    )
    return out
