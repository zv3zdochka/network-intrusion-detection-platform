from pathlib import Path
import argparse
from src.data.splits import make_split_by_file_prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dataset-dir", default="data/processed/dataset")
    ap.add_argument("--out-dir", default="data/processed/splits")
    ap.add_argument("--train-prefixes", nargs="*", default=["Monday", "Tuesday", "Wednesday", "Thursday"])
    ap.add_argument("--val-prefixes", nargs="*", default=[])
    ap.add_argument("--test-prefixes", nargs="*", default=["Friday"])
    args = ap.parse_args()
    make_split_by_file_prefix(
        Path(args.processed_dataset_dir),
        Path(args.out_dir),
        train_prefixes=args.train_prefixes,
        val_prefixes=args.val_prefixes,
        test_prefixes=args.test_prefixes,
    )


if __name__ == "__main__":
    main()
