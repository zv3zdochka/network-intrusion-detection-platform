#!/usr/bin/env python3
"""
Data cleaning and preprocessing script.
"""

import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_config, clean_data, preprocess_data, create_feature_schema
from src.data.ingest import load_bronze_data
from src.data.build import save_processed_data

if __name__ == "__main__":
    config = load_config()

    print("Loading bronze data...")
    df = load_bronze_data(config)

    print("Creating feature schema...")
    schema = create_feature_schema(df, config)

    print("Cleaning data...")
    df_clean = clean_data(df, config, schema)

    print("Preprocessing data...")
    df_processed, label_mapping, preprocessor = preprocess_data(
        df_clean, config, schema, fit=True
    )

    print("Saving processed data...")
    saved = save_processed_data(
        df_processed, schema, label_mapping, preprocessor, config
    )

    print("Completed successfully.")
    for name, path in saved.items():
        print(f"{name}: {path}")
