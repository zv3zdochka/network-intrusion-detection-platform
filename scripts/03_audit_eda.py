#!/usr/bin/env python
from pathlib import Path
import argparse
from src.data.audit import audit_bronze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bronze-dir", default="data/interim/bronze")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--sample-rows", type=int, default=100_000)
    args = ap.parse_args()
    audit_bronze(Path(args.bronze_dir), Path(args.reports_dir), sample_rows=args.sample_rows)


if __name__ == "__main__":
    main()
