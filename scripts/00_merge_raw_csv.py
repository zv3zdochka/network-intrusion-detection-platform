#!/usr/bin/env python3
"""
Скрипт объединения сырых CSV файлов
"""

import sys
from pathlib import Path

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_config, merge_csv_files

if __name__ == "__main__":
    config = load_config()
    merge_csv_files(config)