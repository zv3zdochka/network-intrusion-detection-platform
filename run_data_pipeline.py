#!/usr/bin/env python3
"""
============================================================
CIC-IDS-2017 Data Pipeline
============================================================

Run the full data processing pipeline:
1. Create manifest
2. Ingest and merge raw CSV files (bronze)
3. Audit and EDA
4. Cleaning and preprocessing
5. Train/val/test split creation

Usage:
    python run_data_pipeline.py
    python run_data_pipeline.py --steps 1,2,3
    python run_data_pipeline.py --config path/to/config.yaml
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the project src directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.common import load_config, setup_logging, get_project_root, ensure_dir
from src.data.manifest import create_manifest
from src.data.ingest import merge_csv_files, load_bronze_data
from src.data.audit import run_audit, run_eda, generate_report
from src.data.build import create_feature_schema, clean_data, preprocess_data, save_processed_data
from src.data.splits import create_splits, save_splits


def run_step_1_manifest(config: dict) -> dict:
    """Step 1: Create a data manifest."""
    print("\n" + "=" * 60)
    print("STEP 1: Creating Data Manifest")
    print("=" * 60)

    manifest = create_manifest(config)
    return {"manifest": manifest}


def run_step_2_ingest(config: dict) -> dict:
    """Step 2: Ingest and merge raw data (bronze layer)."""
    print("\n" + "=" * 60)
    print("STEP 2: Ingesting Raw Data (Bronze Layer)")
    print("=" * 60)

    bronze_path = merge_csv_files(config)
    return {"bronze_path": bronze_path}


def run_step_3_audit_eda(config: dict) -> dict:
    """Step 3: Run data audit and EDA."""
    print("\n" + "=" * 60)
    print("STEP 3: Data Audit and EDA")
    print("=" * 60)

    df = load_bronze_data(config)

    audit_results = run_audit(df, config)
    eda_files = run_eda(df, config)
    report_path = generate_report(audit_results, eda_files, config)

    return {
        "audit_results": audit_results,
        "eda_files": eda_files,
        "report_path": report_path,
    }


def run_step_4_build(config: dict) -> dict:
    """Step 4: Clean and preprocess data."""
    print("\n" + "=" * 60)
    print("STEP 4: Data Cleaning and Preprocessing")
    print("=" * 60)

    df = load_bronze_data(config)

    schema = create_feature_schema(df, config)
    df_clean = clean_data(df, config, schema)
    df_processed, label_mapping, preprocessor = preprocess_data(
        df_clean, config, schema, fit=True
    )

    saved = save_processed_data(
        df_processed, schema, label_mapping, preprocessor, config
    )

    return {
        "schema": schema,
        "label_mapping": label_mapping,
        "saved_files": saved,
        "processed_df": df_processed,
    }


def run_step_5_splits(config: dict, df=None) -> dict:
    """Step 5: Create train/val/test splits."""
    print("\n" + "=" * 60)
    print("STEP 5: Creating Train/Validation/Test Splits")
    print("=" * 60)

    if df is None:
        root = get_project_root()
        data_path = root / config["paths"]["processed_data"] / "processed_data.parquet"
        import pandas as pd

        df = pd.read_parquet(data_path)

    splits = create_splits(df, config)
    saved = save_splits(splits, config)

    return {
        "splits": {k: len(v) for k, v in splits.items()},
        "saved_files": saved,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CIC-IDS-2017 Data Processing Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: configs/data_pipeline.yaml)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated list of steps to run (default: 1,2,3,4,5)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))

    # Create required directories
    root = get_project_root()
    for path_key in ["interim_data", "processed_data", "reports", "artifacts"]:
        ensure_dir(root / config["paths"][path_key])

    # Parse steps
    steps = [int(s.strip()) for s in args.steps.split(",")]

    print("\n" + "=" * 60)
    print("CIC-IDS-2017 DATA PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Steps to run: {steps}")
    print(f"Project root: {root}")

    start_time = time.time()
    results = {}
    df_processed = None

    try:
        # Step 1: Manifest
        if 1 in steps:
            step_start = time.time()
            results["step_1"] = run_step_1_manifest(config)
            print(f"Step 1 completed in {time.time() - step_start:.1f}s")

        # Step 2: Ingestion
        if 2 in steps:
            step_start = time.time()
            results["step_2"] = run_step_2_ingest(config)
            print(f"Step 2 completed in {time.time() - step_start:.1f}s")

        # Step 3: Audit and EDA
        if 3 in steps:
            step_start = time.time()
            results["step_3"] = run_step_3_audit_eda(config)
            print(f"Step 3 completed in {time.time() - step_start:.1f}s")

        # Step 4: Cleaning and preprocessing
        if 4 in steps:
            step_start = time.time()
            results["step_4"] = run_step_4_build(config)
            df_processed = results["step_4"].get("processed_df")
            print(f"Step 4 completed in {time.time() - step_start:.1f}s")

        # Step 5: Splits
        if 5 in steps:
            step_start = time.time()
            results["step_5"] = run_step_5_splits(config, df_processed)
            print(f"Step 5 completed in {time.time() - step_start:.1f}s")

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")

        print("\nSummary:")

        if "step_3" in results:
            report_path = results["step_3"].get("report_path")
            print(f"- Report: {report_path}")

        if "step_5" in results:
            splits = results["step_5"].get("splits", {})
            print(f"- Train: {splits.get('train', 0):,} rows")
            print(f"- Val: {splits.get('val', 0):,} rows")
            print(f"- Test: {splits.get('test', 0):,} rows")

        print("\nOutputs:")
        print(f"- Interim data: {root / config['paths']['interim_data']}")
        print(f"- Processed data: {root / config['paths']['processed_data']}")
        print(f"- Reports: {root / config['paths']['reports']}")
        print(f"- Artifacts: {root / config['paths']['artifacts']}")

    except Exception as e:
        print("\n" + "=" * 60)
        print("PIPELINE FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
