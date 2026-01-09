# TraficAnalysis (CIC-IDS-2017)

End-to-end data pipeline for the **CIC-IDS-2017** intrusion detection dataset:
from raw CSV ingestion → parquet “bronze” → auditing/EDA → partitioned “processed” dataset → split definition.

This repository focuses on **reproducible data preparation**. Feature engineering and modeling come next.

---

## Dataset

This project uses the **CIC-IDS-2017** dataset from the Canadian Institute for Cybersecurity (University of New Brunswick).

The data is taken from the official **MachineLearningCSV (non-PCA)** version. Each row represents a single network flow with aggregated traffic statistics.

### What’s inside

- Traffic from **5 working days** (Monday–Friday), with some days split into multiple attack scenarios.
- Normal traffic (`BENIGN`) and multiple attack types (DoS/DDoS, PortScan, Brute Force, Botnet, Web Attacks, Infiltration).
- Flow-level features (duration, packets, bytes, IATs, TCP flags, etc.).
- Metadata fields (IP addresses, ports, protocol, timestamps).
- Target column: `Label`.

### How it is used

- Flow features are used for model training.
- Metadata is kept for replay, visualization, and alerts, but is **not used as model input** to avoid leakage.
- The dataset is stored via **Git LFS** and used without modifying the original files.

---

## Raw files used

We use the following **MachineLearningCSV** files (one file per day/scenario):

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

### Where they are stored

Raw files are placed under:

- `data/raw/CICIDS-2017/MachineLearningCVE/`
- (optional mirror) `data/raw/CICIDS-2017/TrafficLabelling/`

The pipeline reads from `data/raw/...` and writes all derived artifacts into `data/interim/` and `data/processed/`.

---

## Project stages (data pipeline)

The pipeline is executed end-to-end by:

```bash
python run_data_pipeline.py
````

It runs the following stages:

### 1) Build raw manifest

**Goal:** inventory raw CSV files and record basic stats.

Outputs:

* `reports/raw_manifest.json`
* `reports/raw_manifest.csv`

### 2) Ingest raw CSV → bronze parquet

**Goal:** convert each raw CSV into a parquet file with stable column naming and minimal normalization.

Outputs:

* `data/interim/bronze/*.parquet` (one per input CSV)
* `data/interim/bronze/*.schema.json` (sidecar schema per parquet)
* `data/interim/bronze_ingest_summary.json`

### 3) Audit / EDA on bronze

**Goal:** generate dataset sanity checks and quick visuals (labels, missingness, correlations).

Outputs:

* `reports/data_audit.json`
* `reports/figures/label_distribution.png`
* `reports/figures/missing_rate_top30.png`
* `reports/figures/corr_heatmap.png`

### 4) Build processed dataset (partitioned parquet)

**Goal:** produce a single dataset folder partitioned by file/source for faster downstream reading.

Outputs:

* `data/processed/dataset/` (partitioned parquet dataset)
* `data/processed/dataset_meta.json`

### 5) Create split definition (by file prefix)

**Goal:** define reproducible train/test splits (e.g., by day/scenario).

Outputs:

* `data/processed/splits/split_by_file_prefix.json`

---

## Repository layout

* `configs/` — configuration files (pipeline settings)
* `data/raw/` — original dataset files (tracked via Git LFS)
* `data/interim/` — intermediate artifacts (bronze)
* `data/processed/` — processed dataset + splits (model-ready structure)
* `reports/` — audit results and figures
* `scripts/` — step-by-step entry points for individual stages
* `src/` — core implementation (ingest, audit, build, splits)

---

## Notes

* **Do not edit raw CSVs.** The pipeline assumes raw files are immutable.
* Metadata columns (IPs/ports/timestamps) are kept for analysis/replay but should be excluded from model features to avoid leakage.
* If a specific raw file produces unusually high missing rates or missing labels, validate that file independently (encoding/header/row consistency) before training.

```
