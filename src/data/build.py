from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .common import ensure_dir, write_json

try:
    import pyarrow.dataset as ds  # type: ignore
except Exception:
    ds = None


def build_processed_dataset(
    bronze_dir: Path,
    processed_dir: Path,
    partition_cols: Optional[List[str]] = None,
    out_name: str = "dataset",
) -> Dict:
    """
    Builds a processed parquet dataset (folder) from bronze parquet files.

    IMPORTANT:
    The bronze folder may contain sidecar files (e.g., *.schema.json). We must
    explicitly include only *.parquet inputs; otherwise pyarrow will try to open
    non-parquet files and fail.
    """
    if ds is None:
        raise RuntimeError("pyarrow is required. Install with: pip install pyarrow")

    bronze_dir = bronze_dir.resolve()
    processed_dir = processed_dir.resolve()
    out_dir = ensure_dir(processed_dir / out_name)

    partition_cols = partition_cols or ["source_file"]

    parquet_files = sorted(bronze_dir.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in bronze dir: {bronze_dir}")

    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")

    # Write without materializing the full table in memory
    kwargs = dict(
        base_dir=str(out_dir),
        format="parquet",
        partitioning=partition_cols,
    )

    # Overwrite behavior differs across pyarrow versions
    try:
        ds.write_dataset(dataset, existing_data_behavior="delete_matching", **kwargs)
    except TypeError:
        ds.write_dataset(dataset, **kwargs)

    meta = {
        "bronze_dir": str(bronze_dir),
        "processed_dataset_dir": str(out_dir),
        "partition_cols": partition_cols,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "n_input_files": len(parquet_files),
    }
    write_json(processed_dir / f"{out_name}_meta.json", meta)
    return meta
