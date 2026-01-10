#!/usr/bin/env python3
"""
Скрипт очистки и препроцессинга данных
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_config, clean_data, preprocess_data, create_feature_schema
from src.data.ingest import load_bronze_data
from src.data.build import save_processed_data

if __name__ == "__main__":
    config = load_config()

    print("Loading bronze data...")
    df = load_bronze_data(config)

    print("\nCreating feature schema...")
    schema = create_feature_schema(df, config)

    print("\nCleaning data...")
    df_clean = clean_data(df, config, schema)

    print("\nPreprocessing data...")
    df_processed, label_mapping, preprocessor = preprocess_data(
        df_clean, config, schema, fit=True
    )

    print("\nSaving processed data...")
    saved = save_processed_data(
        df_processed, schema, label_mapping, preprocessor, config
    )

    print(f"\n✅ All done!")
    for name, path in saved.items():
        print(f"   {name}: {path}")