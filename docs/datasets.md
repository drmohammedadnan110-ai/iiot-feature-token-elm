# Dataset sources and reproducibility requirements

Access checks below were performed on 2026-10-06 using public primary-source pages. No large datasets were downloaded, no provider forms were submitted, and no original paper split was recovered. Landing-page access is not verification of archive contents. A failed tool fetch does not prove that a dataset is globally unavailable.

## Official sources and observed access

| Dataset | Primary source | Observed status |
|---|---|---|
| Edge-IIoTset | [Dataset DOI](https://doi.org/10.21227/mbc1-1h68) | DOI opening failed in the web tool. Archive contents, current download requirements, and checksum were not verified. |
| ToN-IoT / TON_IoT | [UNSW dataset page](https://research.unsw.edu.au/projects/toniot-datasets) | Landing page accessible. Its download link goes to UNSW SharePoint; the fetch redirected to Microsoft sign-in and failed. No authenticated download was attempted. |
| X-IIoTID | [Author repository](https://github.com/Alhawawreh/X-IIoTID) and [dataset link file](https://github.com/Alhawawreh/X-IIoTID/blob/main/dataset_file.txt) | Repository and metadata accessible. The link file points to the [CloudStor location](https://cloudstor.aarnet.edu.au/plus/s/uCa6M7IDI1S8VrE); that location was inaccessible through the web tool. Dataset-file availability is unverified. |
| CICIoT2023 | [UNB/CIC dataset page](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | Landing page accessible. The [download endpoint](https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/) displayed a registration form and a server error. No registration or archive download was performed. |

The UNSW page describes separate telemetry, operating-system, and network data, with processed CSVs, sampled train/test data, and a feature/statistics directory. Therefore, “ToN-IoT” alone is not a sufficient input specification. The page grants academic research use, requests citations, and directs commercial users to the author. Consult its current terms before using the data. These observations come from the [official UNSW page](https://research.unsw.edu.au/projects/toniot-datasets).

The X-IIoTID author repository describes 820,834 instances and 68 columns/features **including three label levels**. Those levels must not be treated as interchangeable prediction targets or accidentally included as predictors. It describes network, log, resource, and IDS-derived information; cross-dataset equivalence cannot be assumed from similar column names. These are provider-reported metadata, not counts independently measured here. See the [author README](https://github.com/Alhawawreh/X-IIoTID).

UNB describes CICIoT2023 as 33 attacks in seven attack categories across a 105-device topology. Its directories include PCAP, extracted CSVs, an example notebook, and collection/processing material. Choosing a binary, attack-category, or fine-grained target changes the task; the supplied model paper does not identify the exact external target mapping. See the [official CICIoT2023 description](https://www.unb.ca/cic/datasets/iotdataset-2023.html).

No third-party mirror is designated as equivalent to an official release. This repository's code license does not grant rights to redistribute dataset files. Keep raw files outside Git and record the provider's applicable usage and citation requirements separately.

## Edge-IIoTset paper corpus

The supplied paper reports a working corpus of 1,890,671 records in 15 classes and TRAIN/VALIDATION/TEST totals of 1,210,030 / 302,507 / 378,134. Its Table 1 arithmetic is internally consistent; see `paper_audit.md`. The class names are Normal, DDoS_UDP, DDoS_ICMP, SQL_Injection, DDoS_TCP, DDoS_HTTP, Password, Vulnerability_Scanner, Port_Scanning, XSS, Uploading, Backdoor, OS_Fingerprinting, Ransomware, and MITM.

These class totals do not identify the raw rows. The paper does not provide exact source files, a source hash, filtering or duplicate policy, ordered predictors, split membership, or a split-generation algorithm. Its fixed seeds 7, 13, 21, 42, and 99 identify training runs, not a documented partition seed. Reconstructing a 64/16/20 split is a new reference experiment even if aggregate counts are similar. Never label that split the original Protocol B partition.

The manuscript calls each input a flow/network observation, but the actual observation grain and aggregation procedure must be checked against the acquired files and extraction metadata. Do not infer independent sessions or flows from one CSV row per sample. Duplicate and correlated capture records can cross a random row split even when all fitted preprocessing is TRAIN-only.

## Required local data manifest

Before a real-data run, record:

| Field | Required information |
|---|---|
| Source identity | Dataset, official URL, release/snapshot, acquisition date, archive and extracted filenames, file sizes, and SHA-256 hashes. |
| Usage and citation | Provider terms, required citations, and whether redistributed or derived files are permitted. |
| Observation grain | Packet, flow, window, device telemetry record, or another explicitly defined unit; extraction tool/version and aggregation rules. |
| Labels | Target column, ordered class names, raw-to-model mapping, Normal class, exclusions, and all alternate label columns to drop. |
| Predictor schema | Ordered column names, numeric/categorical/binary roles, units, allowed missing values, sentinels, parse rules, and all identifier/derived-identifier exclusions. |
| Record identity | Stable row keys or content fingerprints; source-file and original-row lineage; duplicate/conflicting-label policy. |
| Grouping | Available device, host, session, capture, and timestamp metadata; chosen independence unit and known unavailable metadata. |
| Partitions | Fixed TRAIN/VALIDATION/TEST membership or deterministic generation procedure, split seed, stratification/group rules, per-class counts, and membership hashes. |
| Fitted processing | TRAIN-derived statistics/vocabulary, versioned transformation choices, and a record of handling unknown categories and unusable columns. |

Fit transforms only after defining partitions. Partition checks should reject overlapping row identities/content and available configured groups. A dataset may need explicit deduplication or a group-aware split before training. If the paper's exact split files later become available, validate their integrity and preserve their manifest rather than quietly regenerating them.

Keep label columns, split membership, raw addresses, and capture metadata out of predictors unless an explicit research protocol documents their use. Record which potentially identifying fields were retained. Dropping obvious identifiers does not establish removal of every provenance shortcut.

## External evaluation contracts

**Dataset-native retraining:** define each dataset's own schema, label taxonomy, source snapshot, and partitions. Fit its preprocessing and model on its TRAIN partition, select/calibrate on its VALIDATION partition, and evaluate its held-out TEST. The paper reports ToN-IoT, X-IIoTID, and CICIoT2023 Macro-F1 values, but supplies no exact external file, split, or label manifests. Those values remain paper-reported and unreproduced here.

**Source-only transfer:** provide a reviewed mapping from source and target fields to a common schema, including units, aggregation/window meaning, missing-field handling, protocol encodings, and compatible label mappings. Fit the source model on that same common schema. Freeze source preprocessing, vocabulary, model, temperature, and thresholds before using target TEST. Do not infer semantic equivalence from names such as duration, rate, or service. Do not use target labels to choose a mapping or target statistics to refit preprocessing while labeling the run source-only. Record any adaptation as a different experiment.

The paper's Edge-IIoTset-to-ToN-IoT and Edge-IIoTset-to-X-IIoTID mappings are not supplied. A 15-class source model cannot automatically evaluate a target's different taxonomy by renaming classes. No original transfer result can be reproduced until the exact mapping and denominator/class inclusion policy are obtained or a clearly new protocol is declared.

**Leave-one-attack-out:** remove the designated unknown attack from TRAIN and VALIDATION before fitting any preprocessing or selecting/calibrating the model. Preserve unknown observations only for final evaluation. Declare the unknownness score and a threshold fitted without unknown TEST labels. Record known-class metrics and unknown-detection metrics separately. The paper does not specify its full score/threshold procedure.

## Current evidence boundary

Public provider metadata and paper-table arithmetic have been checked. Official archive bytes, original record membership, real-data execution, and the manuscript's performance have not been verified. A synthetic example or user-supplied CSV can exercise the pipeline under its declared schema; neither establishes reproduction of any of the four benchmarks.
