"""Table 5 search domain; every trial trains on TRAIN and scores VALIDATION only."""

from copy import deepcopy

import optuna

from .engine import fit
from .utils import fresh_dir, write_json


def suggest(trial, base):
    config = deepcopy(base)
    config["model"] = {
        "width": trial.suggest_categorical("width", [64, 96, 128]),
        "layers": trial.suggest_int("layers", 1, 4),
        "heads": trial.suggest_categorical("heads", [4, 8]),
        "expansion": trial.suggest_int("expansion", 2, 4),
        "dropout": trial.suggest_float("dropout", 0, 0.3),
    }
    config.setdefault("training", {}).update(
        {
            "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
            "gamma": trial.suggest_float("gamma", 1, 3),
        }
    )
    config["elm"] = {
        "hidden": trial.suggest_categorical("hidden", [128, 256, 512, 1024, 2048]),
        "activation": trial.suggest_categorical("activation", ["gelu", "relu", "tanh"]),
        "ridge": trial.suggest_float("ridge", 1e-6, 10, log=True),
        "initialization": trial.suggest_categorical(
            "initialization", ["gaussian", "orthogonal", "trigonometric"]
        ),
    }
    return config


def search(base, out, trials=100, sampler="tpe"):
    if trials < 1 or sampler not in ["tpe", "random", "sobol"]:
        raise ValueError("Invalid trial count or sampler")
    out = fresh_dir(out)
    seed = base.get("seed", 7)
    method = {
        "tpe": lambda: optuna.samplers.TPESampler(seed=seed, multivariate=False),
        "random": lambda: optuna.samplers.RandomSampler(seed=seed),
        "sobol": lambda: optuna.samplers.QMCSampler(seed=seed, scramble=True),
    }[sampler]()
    study = optuna.create_study(
        direction="maximize", sampler=method, storage=f"sqlite:///{(out / 'study.db').resolve().as_posix()}"
    )

    def objective(trial):
        config = suggest(trial, base)
        write_json(out / f"trial_{trial.number:03d}_config.json", config)
        bundle = fit(config, out / f"trial_{trial.number:03d}")
        return bundle["validation_macro_f1"]

    study.optimize(objective, n_trials=trials, n_jobs=1)
    study.trials_dataframe().to_csv(out / "trials.csv", index=False, lineterminator="\n")
    best_path = out / f"trial_{study.best_trial.number:03d}"
    from .utils import read_json

    write_json(
        out / "selected.json",
        {
            "bundle": str(best_path),
            "trial": study.best_trial.number,
            "validation_macro_f1": study.best_value,
            "trials": trials,
            "sampler": sampler,
            "test_evaluated": False,
        },
    )
    write_json(
        out / "selected_config.json", read_json(out / f"trial_{study.best_trial.number:03d}_config.json")
    )
    return best_path
