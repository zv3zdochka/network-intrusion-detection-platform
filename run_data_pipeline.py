#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.common import read_yaml
from src.data.manifest import build_manifest
from src.data.ingest import ingest_folder
from src.data.audit import audit_bronze
from src.data.build import build_processed_dataset
from src.data.splits import make_split_by_file_prefix


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CICIDS-2017 data pipeline (raw CSV -> bronze parquet -> audit -> processed -> splits)")
    ap.add_argument("--config", default="configs/data_pipeline.yaml", help="Path to YAML config")
    ap.add_argument("--skip-manifest", action="store_true")
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-splits", action="store_true")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))

    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])

    timestamp_col = cfg["data"]["timestamp_col"]
    label_col = cfg["data"]["label_col"]
    meta_cols = cfg["data"].get("meta_cols", [])

    chunksize = int(cfg["ingest"]["chunksize"])

    if not args.skip_manifest:
        print("[1/5] Building manifest...")
        build_manifest(raw_dir, reports_dir / "raw_manifest.json", reports_dir / "raw_manifest.csv", count_rows=True)

    if not args.skip_ingest:
        print("[2/5] Ingesting CSV -> bronze parquet...")
        ingest_folder(
            raw_dir,
            interim_dir,
            chunksize=chunksize,
            timestamp_col=timestamp_col,
            label_col=label_col,
            meta_cols=meta_cols,
            normalize_labels=bool(cfg["data"].get("normalize_labels", True)),
            replace_infinite=bool(cfg["data"].get("replace_infinite", True)),
            coerce_numeric_flag=bool(cfg["data"].get("coerce_numeric", True)),
            downcast_float32=bool(cfg["data"].get("downcast_float32", True)),
        )

    # Note: in ingest we standardize column names to "safe" snake_case,
    # so label/timestamp are expected to be "label" and "timestamp" here.
    if not args.skip_audit:
        print("[3/5] Auditing bronze dataset...")
        audit_bronze(
            interim_dir / "bronze",
            reports_dir,
            label_col="label",
            timestamp_col="timestamp",
            sample_rows=int(cfg["audit"]["sample_rows"]),
            max_corr_features=int(cfg["audit"]["max_corr_features"]),
        )

    if not args.skip_build:
        print("[4/5] Building processed dataset (partitioned parquet)...")
        build_processed_dataset(interim_dir / "bronze", processed_dir, partition_cols=["source_file"],
                                out_name="dataset")

    if not args.skip_splits:
        print("[5/5] Creating split definition (by file prefix)...")
        sp = cfg["splits"]["by_file_prefix"]
        make_split_by_file_prefix(
            processed_dir / "dataset",
            processed_dir / "splits",
            train_prefixes=sp.get("train_prefixes", []),
            val_prefixes=sp.get("val_prefixes", []),
            test_prefixes=sp.get("test_prefixes", []),
        )

    print("Done.")


if __name__ == "__main__":
    main()
