# Implementation contract

Implement the supplied paper's supervised feature-token encoder, frozen representations, ridge ELM, and validation-fitted temperature. Input is one tabular network observation; output is class scores and calibrated probabilities. Intended use is offline IDS research. Publishing code is not deployment of an operational security control.

Fit preprocessing, class weights, encoder and ELM using TRAIN. Use VALIDATION for checkpoint selection, TPE and temperature. A separate explicit test command evaluates the frozen bundle. Never synthesize paper-reproduction claims, final hyperparameters, original feature lists, record memberships, or pretrained checkpoints.

The source omits final hyperparameters, exact record memberships, feature schema, epoch limit and patience. Choose documented reference defaults; expose schema and all parameters in configuration. Match the Table 5 search space. Provide fixed seeds 7, 13, 21, 42, 99. Treat duplicate/group integrity and fitted preprocessing as testable contracts. Keep full raw data and secrets outside Git.

Acceptance: numerical ELM solve agrees with an independent ridge solution; all initialization families are deterministic; encoder is unchanged during ELM fitting; no validation values enter preprocessing; UNK mapping is stable; calibration preserves argmax; prediction loading equals in-memory inference; partition and paired-statistic alignment errors fail clearly. Run an explicitly synthetic integration example, record actual results, and keep reported paper tables separate.

No latency or accuracy promise is made. Dataset-native retraining and source-only transfer must remain separate. Transfer requires explicit feature/unit and label mappings; missing original mappings cannot be invented. Unknown-attack evaluation must exclude the held-out class from both fitting and calibration. Checkpoint loading uses tensor/numeric formats without arbitrary pickle loading.
