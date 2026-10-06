# Paper and implementation audit

Reviewed on 2026-10-06 against the supplied manuscript, *Industrial IoT Intrusion Detection Using Feature-Token Attention and a Ridge-Regularized Extreme Learning Machine*, its 20 extracted tables, and `implementation_contract.md`.

This repository is a reference implementation of the described method. The manuscript does not supply enough information for exact numerical reproduction. No original dataset snapshot, split manifest, prediction file, fitted preprocessing artifact, training log, or checkpoint was available for this audit. Arithmetic checks below verify internal consistency of printed numbers; they do not verify that the reported experiments were performed or reproduce their performance.

## Method that can be implemented from the paper

1. Fit numerical median imputation and standardization, categorical vocabularies, and class-frequency weights on TRAIN only. Missing and unseen categorical values share an UNK representation. Keep feature order fixed.
2. Convert individual numerical and categorical features into learned tokens with feature identity embeddings. Process tokens with multi-head attention, residual transformations, and feed-forward layers, then pool them into an observation representation.
3. Train the encoder and a temporary classification head on TRAIN with Adam and class-weighted focal loss. Use VALIDATION Macro-F1 for encoder checkpoint selection.
4. Discard the temporary head and freeze the selected encoder. Generate fixed representations, apply a fixed random hidden map, and fit the ELM output matrix from TRAIN using the ridge objective `||H beta - Y||_F^2 + lambda ||beta||_F^2`. An implementation should solve the linear system rather than explicitly form a matrix inverse.
5. Select configurations using 100 TPE trials and VALIDATION Macro-F1, fit a positive scalar temperature on VALIDATION, freeze the bundle, and evaluate TEST. A positive temperature preserves score argmax.

The paper specifies five training seeds: 7, 13, 21, 42, and 99. Table 5 defines the search domains, including Gaussian, orthogonal, and structured trigonometric random maps. The paper does not provide the selected configuration or fully define those maps.

## Table 1 arithmetic: verified from the supplied table

| Quantity | Recomputed value | Finding |
|---|---:|---|
| Sum of 15 class counts | 1,890,671 | Matches printed total |
| TRAIN sum | 1,210,030 | Matches printed total |
| VALIDATION sum | 302,507 | Matches printed total |
| TEST sum | 378,134 | Matches printed total |
| TRAIN + VALIDATION + TEST | 1,890,671 | Matches corpus total |
| Per-class partition sums | All 15 match | No row arithmetic mismatch |
| Class percentages at three decimals | All 15 match | Computed as count / 1,890,671 |
| Sum of printed percentages | 99.999% | Ordinary rounding; total row states 100.000% |

Normal has 1,367,959 records (72.353% after rounding), and MITM has 1,734 (0.092%), including 347 TEST records. These are verified table calculations, not verified counts of accessible raw records.

Protocol A's printed aggregate of 1,909,671 exceeds Protocol B by 19,000. Section 3.1 says the historical source's class counts themselves sum to 1,890,671. That statement does **not** establish that 19,000 records were removed to create Protocol B, nor that the two evaluations share record membership. The original historical source accounting was not independently reconstructed in this audit.

## Other arithmetic checks

Table 8's printed per-class precision, recall, and F1 average to 0.9420733, 0.9494000, and 0.9455867. Its support-weighted recall and F1 are 0.9867367 and 0.9868274. These round to the aggregate values printed in Tables 7 and 8.

For the proposed model, Table 14's five printed scores have mean 0.94558 and sample SD 0.00121532. A two-sided Student-t interval with four degrees of freedom is [0.94407098, 0.94708902], agreeing with the printed [0.9441, 0.9471]. The corresponding TabM, FT-Transformer, CatBoost, and SAINT means, sample SDs, and intervals also reconcile to displayed precision. Table 15's paired mean differences, t statistics, confidence intervals, and Cohen dz reconcile with these seed-aligned scores. These checks do not independently validate pairing, model runs, raw predictions, or population-level generalization.

**Confirmed numerical inconsistency:** Table 14's published-configuration rerun scores are 0.9237, 0.9260, 0.9251, 0.9272, and 0.9249. Their mean is **0.92538**, rounding to **0.9254**, rather than the printed **0.9230**. Its sample SD 0.00130652 and interval [0.92375774, 0.92700226] agree with the printed SD and interval, which are centered on 0.92538. Preserve the source value when transcribing the paper and label the discrepancy; do not silently replace reported results.

Tables 7 and 16 use comparator point estimates such as TabM 0.9385, while Table 14 reports TabM's five-seed mean as 0.9379. Section 6.2 distinguishes point comparisons from matched-seed analysis, so those values should not be substituted for each other. The proposed model's 0.9456 is identified in the abstract as the five-seed mean; per-table run aggregation still needs an explicit manifest.

## Contradictions and unresolved claims

| Location | Conflicting statements or missing evidence | Treatment in this repository |
|---|---|---|
| Sections 3.4 and 5.1; Section 7.1 | The paper says environment details are undocumented. Section 5.1 then lists Windows 11 Pro, Python 3.11.6, PyTorch 2.4.1, scikit-learn 1.5.2, CUDA 12.1, RTX 4090 24 GB, i9-13900K, 32 GB RAM, and 32 CPU threads, with deterministic CUDA where supported, referring to Supplementary Table S1. | Treat these as reported, contradictory environment claims. The supplied extracted tables contain Tables 1-20, not an independently inspectable S1. Record the actual environment of each new run. |
| Sections 6.13-6.14 / Tables 19-20 versus Sections 7.1-7.2 | The results report placeholder harmonization and identifier-removal experiments, but the limitations say no such controlled results exist and future work proposes them. | Both claims are present. Do not assert verified independence from identifiers or serialization artifacts. Exact feature lists, transformations, logs, and predictions are needed. |
| Section 5.4 versus Figure 4 discussion | Maximum epochs and patience are said to be undocumented; Figure 4 describes 80 epochs with a best validation point at epoch 75. | A plotted trajectory does not establish the maximum epoch policy for all runs, patience, or stopping logic. Defaults must be labeled implementation choices. |
| Tables 6-8 and 11 | Binary specificity is 0.9895 in Table 6, while Normal recall is 0.9940 in Table 8. Attack-to-Normal FNR is 0.0140 in Table 6 and 0.0190 in Tables 7/11. | A shared set of hard multiclass predictions collapsed to binary would have specificity equal to Normal recall. Separate thresholds, models, runs, or averaging could explain differences, but are not aligned in the supplied records. |
| Tables 7-8 | For one shared confusion matrix, Table 8's Normal precision 0.9978 and recall 0.9940 imply an attack-to-Normal FNR near 0.00574 using Table 1 supports, rather than 0.0190. | This is a conditional consistency check, not a definitive raw-count correction: per-seed averaging or a different operating point could alter the relation. Require confusion matrices and an explicit aggregation rule. |

The hardware listed above is desktop hardware. Table 16 does not demonstrate operation on an industrial gateway or PLC. Its throughput is explicitly the reciprocal of CPU per-sample latency, not an independently measured sustained throughput. No reported timing or memory number is a benchmark of this implementation.

## Missing information needed for exact reproduction

| Area | Missing or ambiguous information |
|---|---|
| Data provenance | Archive and member filenames, exact release/snapshot, hashes, raw row identities, exclusions, label normalization, duplicate handling, and how the working corpus was formed. |
| Partitions | Record membership, split algorithm and split seed, stratification details, grouping keys, and temporal/device/host/session/capture separation. Class counts and five training seeds cannot reconstruct membership. |
| Schema | Ordered retained predictors, data types and units, categorical columns, identifiers retained or excluded, derived features, binary handling, placeholder normalization, missing-value sentinels, infinite-value policy, and all-missing-column behavior. |
| Tokenizer and encoder | Exact numerical token formula, categorical vocabulary normalization, pooling/fusion operation, normalization placement, feed-forward activation, temporary-head architecture, and representation width. |
| Optimization | Final selected parameters, complete trial records, TPE seed and sampler/pruning settings, whether trials are scored with the temporary head or final ELM, epoch budget, patience, checkpoint ties, Adam settings, focal reduction and weight normalization. |
| ELM | Exact initialization distributions and scales, rectangular orthogonal construction, structured trigonometric definition, bias distribution, numerical precision and solver behavior. |
| Calibration | Explicit objective and optimizer, temperature bounds, tolerances, fitted temperature per run, and whether calibration shares all validation data with HPO. |
| Metrics | Prediction/confusion files, exact macro label set, absent-class and zero-division rules, PR-AUC definition, multiclass Brier convention, ECE bins, binary scoring/thresholding, run aggregation, and source data for figures. |
| Comparators | Final model configurations and tuning budgets; common partition roles alone do not establish equal search effort. |
| Unseen attacks | Unknownness score, known/unknown label definitions, held-out removal from both TRAIN and VALIDATION, threshold-selection rule, known-class subset, and aggregation across seeds and attacks. |
| External data and transfer | Dataset variant/files/snapshots, split membership, target taxonomy, shared feature names and units, feature extraction equivalence, class mappings, handling of unmapped classes, and source-only adaptation/calibration rules. |
| Execution and efficiency | Repository commit, complete dependency lock, deterministic flags, driver/runtime details, warm-up, synchronization, inference batch sizes, latency distributions, memory measurement, and whether preprocessing or HPO is included. |

## Contract review and claim boundary

The implementation contract correctly separates TRAIN fitting, VALIDATION selection/calibration, and explicit frozen TEST evaluation. It also explicitly forbids invented original splits, schemas, selected hyperparameters, checkpoints, and reproduction claims. These are appropriate boundaries for a research implementation.

Reference choices needed to fill the gaps above must appear in configuration and documentation. Synthetic integration results must identify the data as synthetic and must remain separate from the manuscript's reported results. A successful synthetic run verifies pipeline mechanics, not Edge-IIoTset accuracy or external generalization. Training and evaluation should record file hashes, schema/order, split policy, configuration, seed, actual dependencies, actual hardware, and frozen-artifact identity.

Fit-boundary checks should verify TRAIN-only preprocessing and ELM fitting, a genuinely frozen encoder in evaluation mode, deterministic initialization families, stable UNK mapping, positive-temperature argmax preservation, and save/load prediction parity. Data checks should fail on row overlap and available group overlap, while clearly stating that passing those checks cannot prove independence of unrecorded capture conditions. Paired statistics require the same declared TEST manifest and seed keys. Reusing a fixed TEST partition across seeds measures training variability on that partition, not uncertainty across future industrial environments.

Dataset-native retraining and source-only transfer must be separate runs with separate provenance. Transfer cannot be claimed without an explicit compatible feature/unit/label mapping. Leave-one-attack-out evaluation must exclude the held-out attack from fitting, vocabulary/statistic fitting, model selection, and temperature/threshold calibration; TEST may supply final labels only for evaluation. The paper's unknownness and threshold rules are unavailable, so any implementation of that experiment must identify its own rule.

At this review stage, the contract is suitable for a clearly labeled reference implementation. Exact paper reproduction and validation of its accuracy, robustness, calibration, or efficiency remain unestablished.
