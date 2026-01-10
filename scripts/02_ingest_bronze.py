#!/usr/bin/env python3
"""
Скрипт загрузки и объединения данных (bronze layer)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_config, merge_csv_files

if __name__ == "__main__":
    config = load_config()
    output_file = merge_csv_files(config)
    print(f"\nBronze data created: {output_file}")