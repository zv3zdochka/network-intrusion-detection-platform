#!/usr/bin/env python3
"""
Train/validation/test split creation script.
"""

import sys
from pathlib import Path

import pandas as pd

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_config, create_splits, save_splits
from src.data.common import get_project_root

if __name__ == "__main__":
    config = load_config()
    root = get_project_root()

    print("Loading processed data...")
    data_path = root / config["paths"]["processed_data"] / "processed_data.parquet"
    df = pd.read_parquet(data_path)

    print(f"Loaded {len(df):,} rows.")

    print("Creating splits...")
    splits = create_splits(df, config)

    print("Saving splits...")
    saved = save_splits(splits, config)

    print("Completed successfully.")
    for name, path in saved.items():
        print(f"{name}: {path}")
