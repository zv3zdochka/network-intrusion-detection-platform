#!/usr/bin/env python3
"""
Data manifest creation script.
"""

import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_config, create_manifest

if __name__ == "__main__":
    config = load_config()
    create_manifest(config)
    print("Manifest created successfully.")
