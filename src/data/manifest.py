from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .common import ensure_dir, write_json


DEFAULT_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin1")


def detect_csv_encoding(path: Path, encodings: tuple[str, ...] = DEFAULT_CSV_ENCODINGS, sample_bytes: int = 200_000) -> str:
    data = path.read_bytes()[:sample_bytes]
    for enc in encodings:
        try:
            data.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin1"



@dataclass
class FileManifestRow:
    filename: str
    path: str
    size_bytes: int
    rows: Optional[int]
    cols: Optional[int]
    has_label: Optional[bool]


def _count_rows_fast(csv_path: Path) -> int:
    """
    Row counting by scanning lines. Fast-ish and memory safe, but still O(file_size).
    Subtract 1 for header.
    """
    n = 0
    with csv_path.open("rb") as f:
        for _ in f:
            n += 1
    return max(0, n - 1)


def build_manifest(raw_dir: Path, out_json: Path, out_csv: Optional[Path] = None, count_rows: bool = True) -> Dict:
    raw_dir = raw_dir.resolve()
    files = sorted(raw_dir.glob("*.csv"))
    rows: List[FileManifestRow] = []

    for fp in files:
        size = fp.stat().st_size
        # Read header only
        try:
            enc = detect_csv_encoding(fp)
            try:
                header = pd.read_csv(fp, nrows=0, encoding=enc, encoding_errors="replace")
            except TypeError:
                header = pd.read_csv(fp, nrows=0, encoding=enc)
            # CICFlowMeter CSV headers sometimes contain leading/trailing spaces
            # (e.g., " Label"), so normalize before checks.
            norm_cols = [str(c).strip() for c in header.columns]
            cols = len(norm_cols)
            has_label = any(c.lower() == "label" for c in norm_cols)
        except Exception:
            cols = None
            has_label = None

        nrows = _count_rows_fast(fp) if count_rows else None

        rows.append(
            FileManifestRow(
                filename=fp.name,
                path=str(fp),
                size_bytes=size,
                rows=nrows,
                cols=cols,
                has_label=has_label,
            )
        )

    manifest = {
        "raw_dir": str(raw_dir),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "files": [r.__dict__ for r in rows],
        "total_files": len(rows),
        "total_rows": sum(r.rows for r in rows if r.rows is not None),
        "total_size_bytes": sum(r.size_bytes for r in rows),
    }

    ensure_dir(out_json.parent)
    write_json(out_json, manifest)

    if out_csv is not None:
        ensure_dir(out_csv.parent)
        pd.DataFrame([r.__dict__ for r in rows]).to_csv(out_csv, index=False)

    return manifest
