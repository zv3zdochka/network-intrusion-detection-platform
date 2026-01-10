#!/usr/bin/env python3
"""
Bronze layer ingestion script: load and merge raw CSV files.
"""

import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_config, merge_csv_files

if __name__ == "__main__":
    config = load_config()
    output_file = merge_csv_files(config)
    print(f"Bronze data created: {output_file}")
