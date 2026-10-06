"""Explicit preparation, development, and held-out evaluation commands."""

import argparse
import json
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="Feature-token attention + ridge ELM research pipeline")
    sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("schema", help="Draft a schema; manually review before preparing data")
    p.add_argument("--csv", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-manifest")
    for command in ["train", "search", "leave-one-out"]:
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int)
        if command == "search":
            p.add_argument("--trials", type=int, default=100)
            p.add_argument("--sampler", choices=["tpe", "random", "sobol"], default="tpe")
        if command == "leave-one-out":
            p.add_argument("--heldout", required=True)
            p.add_argument("--known-fpr", type=float, default=0.05)
    for command in ["test", "predict", "transfer", "profile"]:
        p = sub.add_parser(command)
        p.add_argument("--bundle", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--data" if command == "test" else "--csv", required=True)
        if command == "transfer":
            p.add_argument("--mapping", required=True)
        if command in ["predict", "profile"]:
            p.add_argument("--batch-size", type=int, default=1 if command == "profile" else 1024)
        if command == "profile":
            p.add_argument("--warmup", type=int, default=10)
            p.add_argument("--repetitions", type=int, default=50)
    p = sub.add_parser("baseline")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--parameters", help="JSON file of estimator options")
    p.add_argument("--test", action="store_true", help="Explicit final held-out evaluation")
    p = sub.add_parser("statistics")
    p.add_argument("--csv", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("bootstrap")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("figures")
    p.add_argument("--predictions", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--history")
    p = sub.add_parser("demo", help="Synthetic integration demonstration; not manuscript reproduction")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[7])
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    from .utils import clean, read_config, read_json, write_json

    command = args.command
    result = None
    if command == "schema":
        from .preprocessing import suggest_schema

        result = suggest_schema(args.csv, args.label, args.out)
    elif command == "prepare":
        from .data import prepare

        result = prepare(args.csv, args.schema, args.out, args.seed, args.split_manifest)
    elif command in ["train", "search", "leave-one-out"]:
        config = read_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        if command == "train":
            from .engine import fit

            fitted = fit(config, args.out)
            result = {"bundle": args.out, "validation_macro_f1": fitted["validation_macro_f1"]}
        elif command == "search":
            from .search import search

            result = search(config, args.out, args.trials, args.sampler)
        else:
            from .generalization import leave_one_out

            result = leave_one_out(config, args.heldout, args.out, args.known_fpr)
    elif command == "test":
        from .engine import test_bundle

        result = test_bundle(args.bundle, args.data, args.out, args.device)
    elif command == "predict":
        from .data import read_csv
        from .engine import Pipeline

        target = Path(args.out)
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        Pipeline(args.bundle, args.device).predict(read_csv(args.csv), args.batch_size).to_csv(
            target, index=False, lineterminator="\n"
        )
        result = target
    elif command in ["transfer", "profile"]:
        from .generalization import profile, transfer

        if command == "transfer":
            result = transfer(args.bundle, args.csv, args.mapping, args.out, args.device)
        else:
            result = profile(
                args.bundle, args.csv, args.out, args.batch_size, args.warmup, args.repetitions, args.device
            )
    elif command == "baseline":
        from .baselines import benchmark

        result = benchmark(
            args.data,
            args.model,
            args.out,
            args.seed,
            read_json(args.parameters) if args.parameters else None,
            args.test,
        )
    elif command in ["statistics", "bootstrap"]:
        import pandas as pd

        from .statistics import paired_group_bootstrap, paired_seed_tests

        if command == "statistics":
            result = paired_seed_tests(pd.read_csv(args.csv), args.reference)
        else:
            from .preprocessing import Schema

            result = paired_group_bootstrap(
                pd.read_csv(args.a),
                pd.read_csv(args.b),
                Schema.load(args.schema).classes,
                args.draws,
                args.seed,
            )
        write_json(args.out, result)
    elif command == "figures":
        import pandas as pd

        from .plots import plot_history, plot_predictions
        from .preprocessing import Schema

        plot_predictions(pd.read_csv(args.predictions), Schema.load(args.schema).classes, args.out)
        if args.history:
            plot_history(args.history, Path(args.out) / "training_history.png")
        result = args.out
    elif command == "demo":
        from .demo import run_demo

        result = run_demo(args.out, args.seeds)
    if result is not None:
        print(json.dumps(clean(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
