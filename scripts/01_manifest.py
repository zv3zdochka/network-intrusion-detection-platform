from pathlib import Path
import argparse
from src.data.manifest import build_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/CICIDS-2017/TrafficLabelling")
    ap.add_argument("--out-json", default="reports/raw_manifest.json")
    ap.add_argument("--out-csv", default="reports/raw_manifest.csv")
    ap.add_argument("--no-row-count", action="store_true")
    args = ap.parse_args()
    build_manifest(Path(args.raw_dir), Path(args.out_json), Path(args.out_csv), count_rows=not args.no_row_count)


if __name__ == "__main__":
    main()
