from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin1")


def detect_csv_encoding(path: Path, encodings: tuple[str, ...] = DEFAULT_CSV_ENCODINGS, sample_bytes: int = 1_000_000) -> str:
    """Best-effort encoding detection by decoding a byte sample.

    CICIDS-2017 CSVs sometimes contain cp1252 characters (e.g., byte 0x96).
    We choose the first encoding that successfully decodes the sample.
    """
    data = path.read_bytes()[:sample_bytes]
    for enc in encodings:
        try:
            data.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin1"


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
except Exception as e:
    pa = None
    pq = None


def _read_csv_chunks(path: Path, *, chunksize: int, encoding: str, use_python_engine: bool = False) -> pd.io.parsers.TextFileReader:
    """Create a pandas chunk reader with resilient decoding.

    We prefer the C engine for speed, but CICIDS CSVs sometimes contain
    cp1252/latin1 bytes. Using encoding_errors='replace' prevents hard failures.
    """
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
    Returns (safe_columns, mapping_original_to_safe).
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
        raise RuntimeError(
            "pyarrow is required for parquet ingest. Install with: pip install pyarrow"
        )

    meta_cols = meta_cols or []

    ensure_dir(out_parquet.parent)

    encoding = detect_csv_encoding(csv_path)
    # Use resilient decoding; fallback to python engine if needed
    try:
        reader = _read_csv_chunks(csv_path, chunksize=chunksize, encoding=encoding)
    except TypeError:
        # very old pandas without encoding_errors support
        reader = pd.read_csv(csv_path, chunksize=chunksize, low_memory=False, encoding=encoding)

    writer = None
    canonical_schema = None
    total_rows = 0
    dropped_cols: List[str] = []

    col_mapping: Dict[str, str] = {}
    safe_cols: Optional[List[str]] = None

    try:
        for i, chunk in enumerate(reader):
            # Standardize column names on first chunk
            if i == 0:
                safe_cols, mapping = standardize_columns(list(chunk.columns))
                col_mapping = mapping
                chunk.columns = safe_cols

                # Normalize config col names to safe versions if needed
                # (users often pass "Timestamp", "Label" as in raw CSV)
                timestamp_safe = slugify_column(timestamp_col)
                label_safe = slugify_column(label_col)

                meta_safe = [slugify_column(c) for c in meta_cols]
            else:
                # Pandas yields the same set/order of columns for each chunk.
                # Reuse the *exact* safe names from the first chunk to guarantee
                # stable naming (and correct handling of duplicates).
                if safe_cols is None:
                    raise RuntimeError("Internal error: safe_cols is not initialized")
                chunk.columns = safe_cols

            # Recompute safe names each chunk
            timestamp_safe = slugify_column(timestamp_col)
            label_safe = slugify_column(label_col)
            meta_safe = [slugify_column(c) for c in meta_cols]

            if add_source_file_col:
                chunk["source_file"] = csv_path.name

            # Timestamp parsing
            if timestamp_safe in chunk.columns:
                chunk[timestamp_safe] = pd.to_datetime(chunk[timestamp_safe], errors="coerce")
            else:
                # Keep going, but note
                pass

            # Label normalization
            if label_safe in chunk.columns and normalize_labels:
                chunk[label_safe] = chunk[label_safe].astype(str).map(normalize_label)

            # Determine numeric columns (everything except meta/label/timestamp/source_file)
            exclude = set(meta_safe + [label_safe, timestamp_safe, "source_file"])
            numeric_cols = [c for c in chunk.columns if c not in exclude]

            if replace_infinite:
                replace_inf(chunk, numeric_cols)

            if coerce_numeric_flag:
                coerce_numeric(chunk, numeric_cols, downcast_float32=downcast_float32)

            # IMPORTANT: keep numeric dtypes stable across chunks.
            # CICIDS columns may appear as int in one chunk and float in another
            # (e.g., if NaNs occur only in some chunks). ParquetWriter requires
            # a consistent schema, so we force all numeric columns to float.
            if numeric_cols:
                target = "float32" if downcast_float32 else "float64"
                for c in numeric_cols:
                    # safe conversion: ints -> float, floats stay float
                    chunk[c] = chunk[c].astype(target)

            # Write chunk
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                canonical_schema = table.schema
                writer = pq.ParquetWriter(out_parquet, canonical_schema, compression="zstd")
            else:
                # As an extra guard, cast to the canonical schema.
                # This avoids crashes if Pandas/Arrow infer subtly different types.
                table = table.cast(canonical_schema, safe=False)
            writer.write_table(table)

            total_rows += len(chunk)
    except UnicodeDecodeError:
        # Retry with a permissive encoding/engine.
        if writer is not None:
            writer.close()
        if out_parquet.exists():
            out_parquet.unlink()
        reader = _read_csv_chunks(csv_path, chunksize=chunksize, encoding="latin1", use_python_engine=True)
        writer = None
        canonical_schema = None
        total_rows = 0
        for i, chunk in enumerate(reader):
            # Standardize column names on first chunk
            if i == 0:
                safe_cols, mapping = standardize_columns(list(chunk.columns))
                col_mapping = mapping
                chunk.columns = safe_cols
            else:
                if safe_cols is None:
                    raise RuntimeError("Internal error: safe_cols is not initialized")
                chunk.columns = safe_cols

            # Recompute safe names each chunk
            timestamp_safe = slugify_column(timestamp_col)
            label_safe = slugify_column(label_col)
            meta_safe = [slugify_column(c) for c in meta_cols]

            if add_source_file_col:
                chunk["source_file"] = csv_path.name

            if timestamp_safe in chunk.columns:
                chunk[timestamp_safe] = pd.to_datetime(chunk[timestamp_safe], errors="coerce")

            if label_safe in chunk.columns and normalize_labels:
                chunk[label_safe] = chunk[label_safe].astype(str).map(normalize_label)

            exclude = set(meta_safe + [label_safe, timestamp_safe, "source_file"])
            numeric_cols = [c for c in chunk.columns if c not in exclude]

            if replace_infinite:
                replace_inf(chunk, numeric_cols)
            if coerce_numeric_flag:
                coerce_numeric(chunk, numeric_cols, downcast_float32=downcast_float32)
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


    if writer is not None:
        writer.close()

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

    # Quick column count (from last chunk schema)
    cols = table.num_columns if "table" in locals() else 0

    return IngestResult(
        source_csv=str(csv_path),
        output_parquet=str(out_parquet),
        rows=total_rows,
        cols=cols,
        dropped_cols=dropped_cols,
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
