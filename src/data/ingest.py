from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin1")

from .common import (
    coerce_numeric,
    ensure_dir,
    make_unique,
    normalize_label,
    replace_inf,
    slugify_column,
    write_json,
)

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None


def detect_csv_encoding(
    path: Path,
    encodings: tuple[str, ...] = DEFAULT_CSV_ENCODINGS,
    sample_bytes: int = 1_000_000,
) -> str:
    """Best-effort encoding detection by decoding a byte sample."""
    data = path.read_bytes()[:sample_bytes]
    for enc in encodings:
        try:
            data.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin1"


def _read_csv_chunks(
    path: Path,
    *,
    chunksize: int,
    encoding: str,
    use_python_engine: bool = False,
) -> pd.io.parsers.TextFileReader:
    """Create a chunk reader with resilient decoding."""
    kwargs = dict(
        chunksize=chunksize,
        low_memory=False,
        encoding=encoding,
    )
    if use_python_engine:
        kwargs["engine"] = "python"
    # pandas>=1.5 supports encoding_errors
    try:
        kwargs["encoding_errors"] = "replace"
    except Exception:
        pass
    return pd.read_csv(path, **kwargs)


@dataclass
class IngestResult:
    source_csv: str
    output_parquet: str
    rows: int
    cols: int
    dropped_cols: List[str]


def standardize_columns(columns: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns:
      - safe_columns: standardized, slugified and de-duplicated column names
      - mapping_original_to_safe: mapping from stripped original names to safe names
    """
    orig = [c.strip() for c in columns]
    orig_unique = make_unique(orig)
    safe = [slugify_column(c) for c in orig_unique]
    safe_unique = make_unique(safe)
    mapping = {o: s for o, s in zip(orig_unique, safe_unique)}
    return safe_unique, mapping


def ingest_csv_to_parquet(
    csv_path: Path,
    out_parquet: Path,
    *,
    chunksize: int = 200_000,
    timestamp_col: str = "Timestamp",
    label_col: str = "Label",
    meta_cols: Optional[List[str]] = None,
    normalize_labels: bool = True,
    replace_infinite: bool = True,
    coerce_numeric_flag: bool = True,
    downcast_float32: bool = True,
    add_source_file_col: bool = True,
) -> IngestResult:
    if pq is None or pa is None:
        raise RuntimeError("pyarrow is required. Install with: pip install pyarrow")

    meta_cols = meta_cols or []
    ensure_dir(out_parquet.parent)

    encoding = detect_csv_encoding(csv_path)

    def _run_with_reader(reader: pd.io.parsers.TextFileReader) -> Tuple[int, int, Dict[str, str]]:
        writer = None
        canonical_schema = None
        total_rows = 0
        cols_written = 0

        col_mapping: Dict[str, str] = {}
        safe_cols: Optional[List[str]] = None

        # These must be fixed after first chunk based on mapping
        timestamp_safe = None
        label_safe = None
        meta_safe: List[str] = []

        for i, chunk in enumerate(reader):
            if i == 0:
                safe_cols, mapping = standardize_columns(list(chunk.columns))
                col_mapping = mapping
                chunk.columns = safe_cols

                # IMPORTANT: use mapping (after make_unique) so we don't miss duplicates
                timestamp_safe = mapping.get(timestamp_col.strip(), slugify_column(timestamp_col))
                label_safe = mapping.get(label_col.strip(), slugify_column(label_col))
                meta_safe = [mapping.get(c.strip(), slugify_column(c)) for c in meta_cols]
            else:
                if safe_cols is None:
                    raise RuntimeError("Internal error: safe_cols not initialized")
                chunk.columns = safe_cols

            assert timestamp_safe is not None and label_safe is not None

            if add_source_file_col:
                chunk["source_file"] = csv_path.name

            # Timestamp parsing
            if timestamp_safe in chunk.columns:
                chunk[timestamp_safe] = pd.to_datetime(chunk[timestamp_safe], errors="coerce")

            # Label normalization WITHOUT turning NaN into "nan"
            if label_safe in chunk.columns and normalize_labels:
                s = chunk[label_safe]
                s = s.map(lambda v: normalize_label(v) if pd.notna(v) else pd.NA)
                s = s.replace({"nan": pd.NA, "NaN": pd.NA, "": pd.NA})
                chunk[label_safe] = s

            # numeric columns = all except meta/label/timestamp/source_file
            exclude = set(meta_safe + [label_safe, timestamp_safe, "source_file"])
            numeric_cols = [c for c in chunk.columns if c not in exclude]

            if replace_infinite:
                replace_inf(chunk, numeric_cols)

            if coerce_numeric_flag:
                coerce_numeric(chunk, numeric_cols, downcast_float32=downcast_float32)

            # keep schema stable across chunks: force numeric to float
            if numeric_cols:
                target = "float32" if downcast_float32 else "float64"
                for c in numeric_cols:
                    chunk[c] = chunk[c].astype(target)

            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                canonical_schema = table.schema
                writer = pq.ParquetWriter(out_parquet, canonical_schema, compression="zstd")
            else:
                table = table.cast(canonical_schema, safe=False)

            writer.write_table(table)

            total_rows += len(chunk)
            cols_written = table.num_columns

        if writer is not None:
            writer.close()

        return total_rows, cols_written, col_mapping

    # First attempt (fast path)
    try:
        reader = _read_csv_chunks(csv_path, chunksize=chunksize, encoding=encoding)
        total_rows, cols, col_mapping = _run_with_reader(reader)
    except UnicodeDecodeError:
        # Fallback: permissive read for any bytes
        if out_parquet.exists():
            out_parquet.unlink(missing_ok=True)
        reader = _read_csv_chunks(csv_path, chunksize=chunksize, encoding="latin1", use_python_engine=True)
        total_rows, cols, col_mapping = _run_with_reader(reader)

    # Save schema mapping next to parquet
    schema_path = out_parquet.with_suffix(".schema.json")
    write_json(
        schema_path,
        {
            "source_csv": str(csv_path),
            "output_parquet": str(out_parquet),
            "column_mapping_original_to_safe": col_mapping,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        },
    )

    return IngestResult(
        source_csv=str(csv_path),
        output_parquet=str(out_parquet),
        rows=int(total_rows),
        cols=int(cols),
        dropped_cols=[],
    )


def ingest_folder(
    raw_dir: Path,
    interim_dir: Path,
    *,
    chunksize: int = 200_000,
    timestamp_col: str = "Timestamp",
    label_col: str = "Label",
    meta_cols: Optional[List[str]] = None,
    normalize_labels: bool = True,
    replace_infinite: bool = True,
    coerce_numeric_flag: bool = True,
    downcast_float32: bool = True,
) -> Dict:
    raw_dir = raw_dir.resolve()
    out_dir = ensure_dir(interim_dir / "bronze")

    files = sorted(raw_dir.glob("*.csv"))
    results: List[Dict] = []

    for fp in files:
        out_pq = out_dir / (fp.stem + ".parquet")
        res = ingest_csv_to_parquet(
            fp,
            out_pq,
            chunksize=chunksize,
            timestamp_col=timestamp_col,
            label_col=label_col,
            meta_cols=meta_cols,
            normalize_labels=normalize_labels,
            replace_infinite=replace_infinite,
            coerce_numeric_flag=coerce_numeric_flag,
            downcast_float32=downcast_float32,
        )
        results.append(res.__dict__)

    summary = {
        "raw_dir": str(raw_dir),
        "bronze_dir": str(out_dir),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "files_processed": len(results),
        "total_rows": sum(r["rows"] for r in results),
        "results": results,
    }
    write_json(interim_dir / "bronze_ingest_summary.json", summary)
    return summary
