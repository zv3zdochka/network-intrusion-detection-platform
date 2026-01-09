from pathlib import Path
import argparse
from src.data.ingest import ingest_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/CICIDS-2017/TrafficLabelling")
    ap.add_argument("--interim-dir", default="data/interim")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()
    ingest_folder(Path(args.raw_dir), Path(args.interim_dir), chunksize=args.chunksize)


if __name__ == "__main__":
    main()
