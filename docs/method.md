# Method and implementation choices

The paper defines the two-stage architecture but leaves several implementation details unspecified. The choices below make the reference executable; they are not recovered author settings.

## Mathematical path

For numeric feature j, the token is `x_j * w_j + b_j + e_j`. Categorical feature j uses a learned category embedding plus identity `e_j`. Features occur in schema order: numeric first, then categorical. Missing numeric values are converted to the TRAIN median (zero for an entirely missing training column), then standardized using TRAIN imputed population mean and standard deviation. Constant columns use scale one. Nonfinite numeric values are missing. Categories are stripped, explicitly aliased if configured, and mapped to a sorted TRAIN vocabulary; missing and unseen values have index zero.

The tokenizer is followed by pre-layer-normalized self-attention residual blocks with GELU feed-forward networks. A final layer norm and mean over feature tokens produce z. There is no positional time model or CLS token. Inputs are individual observations; no claim of sequence learning is made. A temporary linear K-class head trains with Adam and mean weighted focal loss:

`L = mean[-alpha[y] * (1 - softmax(logits)[y])**gamma * log_softmax(logits)[y]]`

`alpha[k] = N_train / (K * n_train[k])`. Class weights enter the encoder objective, not the ELM ridge objective. The earliest epoch achieving the largest validation Macro-F1 is retained. The temporary head is removed, and encoder parameters and dropout behavior are frozen for representation extraction.

For random projection W and bias b, `H = activation(Z W + b)`. The ELM minimizes `||H beta - Y||² + lambda ||beta||²`. Batches accumulate `G = sum(H.T H)` and `B = sum(H.T Y)`. Cholesky solves `(G + lambda I) beta = B` in float64. The code never forms an explicit inverse and does not allocate a corpus-wide one-hot Y or hidden matrix H. Prediction scores are `H beta`.

Random maps are deterministic for a declared seed:

- Gaussian: independent normal entries with standard deviation `1/sqrt(d)`.
- Orthogonal: QR-normalized, sign-canonicalized square blocks, concatenated/truncated to the required hidden width. For L > d, all columns cannot be mutually orthogonal; each block is orthogonal.
- Structured trigonometric: signed/permuted orthonormal DCT-II basis blocks. This is one explicit interpretation of the paper's undefined structured trigonometric family.

All families use uniform `[-1, 1]` biases and GELU, ReLU, or tanh activation. Positive ridge is required. There is no separately unpenalized intercept. Numeric model states, random weights, biases and fitted output weights are saved for prediction.

Temperature T is fitted by bounded scalar optimization of validation negative log-likelihood in log-temperature space `[-6, 6]`, including T=1 as a fallback candidate. Thus T is positive, cannot change argmax, and cannot worsen the fitted validation NLL relative to T=1. It is applied to ELM scores, not the removed temporary head.

## Splits and protocol

File SHA-256 and original row index form stable row identifiers. Canonicalized predictor fingerprints join duplicate observations. Supplied capture/device/session grouping columns join further observations transitively. These identifiers never enter predictors. A configured group column with missing identifiers is rejected. Identical predictors with conflicting labels are reported and kept in the same group, not silently relabeled.

Independent observations receive nested stratified 80/20 and 80/20 splits (64/16/20 overall). Linked groups use nested five-fold stratified group allocation; ratios become approximate. The explicit split-manifest option accepts reviewed memberships. A new reference split is never claimed to recover manuscript Protocol B.

This grouping uses predictor fingerprints plus declared metadata, not a universal duplicate detector. Missing capture or host metadata limits what independence can be established. Hash collisions can over-group records; file integrity uses separate SHA-256. Split preparation inspects labels to stratify and audit support. Training reads only TRAIN and VALIDATION values, plus the saved partition manifest for integrity checks. It does not open the TEST feature file.

Default width 96, two layers, four heads, expansion 3, dropout .1, learning rate .0003, weight decay .0001, 100 maximum epochs, patience 15, batch 512, gamma 2, gradient clipping 5, hidden width 512, orthogonal map and ridge .01 are reference choices. `configs/reference.yaml` exposes every choice. Table 5's search bounds are implemented in `search.py`; a search scores each complete encoder/ELM trial by validation Macro-F1. Calibration and model selection reuse validation and therefore can overfit it. HPO is sequential and retains every trial; resume/pruning are not implemented.

## Evaluation conventions

Macro-F1 and balanced accuracy average over the full declared class list; an absent class contributes zero recall/F1. Undefined one-vs-rest ROC/PR metrics are null. PR area uses average precision, not trapezoidal area. ECE uses 15 equal-width confidence bins; confidence one enters the final bin. Multiclass Brier is the summed squared probability error per observation. Binary metrics collapse multiclass argmax into Normal versus any attack; no independent binary threshold is fitted. Normal recall and collapsed binary specificity therefore agree for one shared prediction set.

Paired tests operate on identical seeds, TEST checksum and class order. Their uncertainty concerns training randomness on a fixed test set, not variability between industrial deployments. Group bootstrap instead resamples matched observation groups for one seed. Neither analysis establishes independent population generalization.

For unknown-attack experiments, one attack is excluded from TRAIN and VALIDATION before every fitted stage. Novelty is `1 - max calibrated probability`; the operational threshold is a known-validation quantile for a declared false-unknown rate. AUROC/AUPR and FPR@95TPR are retrospective TEST metrics. This score/threshold is a documented reference choice absent from the paper; multiclass confidence is not guaranteed to identify new attacks.
