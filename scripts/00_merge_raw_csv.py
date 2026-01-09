#!/usr/bin/env python
"""
Optional utility: merges raw CICIDS-2017 CSV files into a single CSV.

Recommendation:
- For large CICIDS-2017 files, prefer parquet ingest + partitioned dataset.
- Use this only if you explicitly need a single CSV for external tooling.

Usage:
  python scripts/00_merge_raw_csv.py --raw-dir data/raw/CICIDS-2017/TrafficLabelling --out data/interim/merged.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/CICIDS-2017/TrafficLabelling")
    ap.add_argument("--out", default="data/interim/merged.csv")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    first = True
    for fp in files:
        for chunk in pd.read_csv(fp, chunksize=args.chunksize, low_memory=False):
            chunk["source_file"] = fp.name
            chunk.to_csv(out, mode="w" if first else "a", header=first, index=False)
            first = False

    print(f"Saved merged CSV: {out}")


if __name__ == "__main__":
    main()
