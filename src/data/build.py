from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .common import ensure_dir, write_json

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.dataset as ds  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    ds = None
    pq = None


def build_processed_dataset(
        bronze_dir: Path,
        processed_dir: Path,
        *,
        partition_cols: Optional[List[str]] = None,
        out_name: str = "dataset",
) -> Dict:
    """
    Builds a processed parquet dataset (folder) from bronze parquet files.
    By default, partitions by source_file to keep things scalable and split-friendly.
    """
    if ds is None or pq is None:
        raise RuntimeError("pyarrow is required. Install with: pip install pyarrow")

    bronze_dir = bronze_dir.resolve()
    processed_dir = processed_dir.resolve()
    out_dir = ensure_dir(processed_dir / out_name)

    partition_cols = partition_cols or ["source_file"]

    dataset = ds.dataset(str(bronze_dir), format="parquet")

    # Write to dataset folder (partitioned)
    pq.write_to_dataset(
        dataset.to_table(),
        root_path=str(out_dir),
        partition_cols=partition_cols,
        compression="zstd",
    )

    meta = {
        "bronze_dir": str(bronze_dir),
        "processed_dataset_dir": str(out_dir),
        "partition_cols": partition_cols,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }
    write_json(processed_dir / f"{out_name}_meta.json", meta)
    return meta
