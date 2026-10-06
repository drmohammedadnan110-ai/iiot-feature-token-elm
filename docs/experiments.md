# Experiment commands

Run commands from the repository root after installation. Paths in YAML resolve from the current working directory; `extends` resolves relative to its config file. Outputs must be fresh. Record source versions, access terms and grouping/label decisions before running.

## Matched seeds and ablations

```bash
python scripts/run_experiments.py --config configs/reference.yaml --out runs/five_seeds --seeds 7 13 21 42 99
# Add --test only after fixing the experiment plan and selected configuration.
python scripts/run_experiments.py --config configs/reference.yaml --out runs/ablations --seeds 7 13 21 42 99 --ablations --test
iiot statistics --csv runs/ablations/seed_scores.csv --reference proposed --out runs/ablations/statistics.json
```

The runner preserves one prepared split and changes training seeds. Ablations are full proposed, no attention (token-wise feed-forward replacement), gamma zero (weighted cross-entropy), no class weighting, and Gaussian ELM initialization. The no-attention replacement has a different parameter count, so it is not a capacity-matched causal experiment. The runner does not re-tune every ablation; use separate `search` runs on development data if your declared protocol requires tuning. It never uses TEST scores to choose a model.

## Baselines and paired prediction bootstrap

```bash
iiot baseline --data data/prepared/edge --model logistic --seed 7 --out runs/logistic7 --test
iiot baseline --data data/prepared/edge --model extra_trees --seed 7 --out runs/extra_trees7 --test
iiot bootstrap --a runs/reference_seed7_test/predictions.csv --b runs/logistic7/test_predictions.csv --schema data/prepared/edge/schema.json --draws 2000 --out runs/bootstrap.json
python -m pip install -e ".[boosting]"
iiot baseline --data data/prepared/edge --model catboost --seed 7 --out runs/catboost7
```

Baseline categorical features are one-hot encoded from the TRAIN vocabulary, including the UNK slot. Probabilities or decision scores are calibrated on VALIDATION. `--parameters config.json` accepts model options. The CLI seed controls the estimator random state. Estimator objects are not pickled; reproduce the fit from configuration and provenance. One-hot matrices and kernel SVMs can be expensive for large data. Optional boosting packages require their own compatible installation and were not exercised in the included run.

## External datasets

Create a reviewed schema and prepared directory for each dataset. Declare the actual target level rather than forcing every dataset into the Edge taxonomy.

```bash
iiot prepare --csv data/raw/toniot.csv --schema configs/toniot.reviewed.json --out data/prepared/toniot
iiot train --config configs/toniot.yaml --out runs/toniot7
iiot test --bundle runs/toniot7 --data data/prepared/toniot --out runs/toniot7_test
```

The same procedure applies to `configs/xiiotid.yaml` and `configs/ciciot2023.yaml`. These are **dataset-native retraining** experiments, not source-only transfer.

Direct transfer requires a source model trained on a defensible common feature/label space and an explicit JSON mapping. An illustrative mapping for two source features is:

```json
{
  "features": {
    "flow_duration": {"column": "duration_ms", "unit": "milliseconds to seconds", "scale": 0.001, "offset": 0, "meaning": "same flow timeout and bidirectional observation definition"},
    "protocol": {"column": "proto", "values": {"TCP": "tcp"}, "meaning": "IANA transport protocol"}
  },
  "label_column": "traffic_type",
  "labels": {"benign": "Normal", "ddos": "DDoS"}
}
```

Every source predictor must appear exactly once, and every target label must map explicitly into source classes. The program can check completeness but cannot establish semantic validity; that requires dataset documentation. Missing numeric values use frozen source imputation; categorical unknowns use the source UNK. No target fitting or target temperature calibration occurs.

```bash
iiot transfer --bundle runs/common_schema_source --csv data/target_test.csv --mapping configs/reviewed_transfer.json --out runs/transfer
```

## Unknown attacks and efficiency

```bash
iiot leave-one-out --config configs/reference.yaml --heldout MITM --out runs/unknown_mitm
iiot profile --bundle runs/reference_seed7 --csv data/prepared/edge/test.csv --batch-size 1 --warmup 10 --repetitions 50 --out runs/profile_cpu.json
```

Repeat the holdout command for each of the 14 attack classes under a fixed declared protocol. The held-out attack never enters fitting or calibration. Ordinary `test` is for closed-set tasks; the leave-one-out runner handles unknown labels separately.

Profiling measures preprocessing, encoder, **CPU** ELM, softmax and prediction-frame construction together, excluding model loading and file I/O. With `--device cuda`, the encoder runs on CUDA while the ELM remains NumPy/SciPy CPU. Throughput uses measured elapsed time, batch size and repetitions. It is not the reciprocal of a separately rounded latency. Profiling does not emulate a PLC or industrial gateway.

## Artifacts and resource bounds

Each trained bundle contains `bundle.json`, `preprocessor.json`, `encoder.npz`, `elm.npz`, `elm.json`, the temporary training checkpoint, validation predictions/metrics, and epoch history. The bundle records software environment, seeds, source/split/schema hashes, configuration and elapsed training time. Model state loads through numeric NPZ only. Digest checks detect mismatches, not malicious metadata replacement.

Memory for full CSV parsing, transitive grouping, TRAIN/VALIDATION arrays and DataLoader tensor copies remains proportional to corpus size. The ELM adds a float64 L-by-L Gram matrix (about 32 MiB at L=2048), plus model tensors, batch activations and solver workspace. Attention memory also depends on batch size and number of predictors. Each HPO trial retains weights and validation predictions on disk. Start with an explicitly labeled subset to estimate resource requirements, then run the predeclared full protocol; do not present subset results as full-corpus results.
