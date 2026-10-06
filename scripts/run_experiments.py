"""Fixed-split seed/ablation runner. TEST requires an explicit flag."""

import argparse
import json
from copy import deepcopy

import pandas as pd

from iiot_ids.engine import fit, test_bundle
from iiot_ids.utils import fresh_dir, read_config, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21, 42, 99])
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    base, out = read_config(args.config), fresh_dir(args.out)
    variants = {"proposed": {}}
    if args.ablations:
        variants.update(
            {
                "no_attention": {"model": {"attention": False}},
                "gamma_zero": {"training": {"gamma": 0}},
                "no_class_weight": {"training": {"class_weighted": False}},
                "gaussian_elm": {"elm": {"initialization": "gaussian"}},
            }
        )
    write_json(out / "plan.json", {"seeds": args.seeds, "variants": variants, "test_requested": args.test})
    scores = []
    for name, changes in variants.items():
        for seed in args.seeds:
            config = deepcopy(base)
            config["seed"] = seed
            for section, values in changes.items():
                config.setdefault(section, {}).update(values)
            directory = out / f"{name}_{seed}"
            bundle = fit(config, directory)
            if args.test:
                result = test_bundle(
                    directory, config["data_dir"], out / f"{name}_{seed}_test", config.get("device", "cpu")
                )
                scores.append(
                    {
                        "seed": seed,
                        "model": name,
                        "macro_f1": result["macro_f1"],
                        "test_sha256": bundle["data_audit"]["split_files"]["test"],
                        "classes": json.dumps(bundle["classes"]),
                    }
                )
                pd.DataFrame(scores).to_csv(out / "seed_scores.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
