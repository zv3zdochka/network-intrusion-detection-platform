## Dataset

This project uses the **CIC-IDS-2017** dataset from the Canadian Institute for Cybersecurity (University of New Brunswick).

The data is taken from the official **MachineLearningCSV (non-PCA)** version. Each row represents a single network flow with aggregated traffic statistics.

### What’s inside

* Traffic from **5 working days** (Monday–Friday), with some days split into multiple attack scenarios.
* Normal traffic (`BENIGN`) and multiple attack types (DoS/DDoS, PortScan, Brute Force, Botnet, Web Attacks, Infiltration).
* Flow-level features (duration, packets, bytes, IATs, TCP flags, etc.).
* Metadata fields (IP addresses, ports, protocol, timestamps).
* Target column: `Label`.

### How it is used

* Flow features are used for model training.
* Metadata is kept for replay, visualization, and alerts, but is **not used as model input** to avoid data leakage.

The dataset is stored via **Git LFS** and used without modifying the original files.

