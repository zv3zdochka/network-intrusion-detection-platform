#!/usr/bin/env python3
"""
Скрипт создания манифеста данных
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_config, create_manifest

if __name__ == "__main__":
    config = load_config()
    manifest = create_manifest(config)
    print("\nManifest created successfully!")