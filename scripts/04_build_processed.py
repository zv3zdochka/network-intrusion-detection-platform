#!/usr/bin/env python
from pathlib import Path
import argparse
from src.data.build import build_processed_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bronze-dir", default="data/interim/bronze")
    ap.add_argument("--processed-dir", default="data/processed")
    args = ap.parse_args()
    build_processed_dataset(Path(args.bronze_dir), Path(args.processed_dir), partition_cols=["source_file"])


if __name__ == "__main__":
    main()
