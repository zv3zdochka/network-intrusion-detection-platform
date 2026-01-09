from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .common import ensure_dir, write_json


def make_split_by_file_prefix(
        processed_dataset_dir: Path,
        out_dir: Path,
        *,
        train_prefixes: List[str],
        val_prefixes: List[str],
        test_prefixes: List[str],
) -> Dict:
    """
    Split by source_file prefix (e.g., Monday..., Friday...).
    Stores split definition as file lists (not row indices).
    """
    processed_dataset_dir = processed_dataset_dir.resolve()
    out_dir = ensure_dir(out_dir)

    # Discover available source_file partitions by reading folder names.
    # When partitioned by source_file, folders look like: source_file=Monday-WorkingHours.../
    source_files: List[str] = []
    for p in processed_dataset_dir.rglob("*"):
        if p.is_dir() and p.name.startswith("source_file="):
            source_files.append(p.name.split("=", 1)[1])

    source_files = sorted(set(source_files))

    def pick(prefixes: List[str]) -> List[str]:
        picked = []
        for sf in source_files:
            for pref in prefixes:
                if sf.startswith(pref):
                    picked.append(sf)
                    break
        return sorted(set(picked))

    train_files = pick(train_prefixes)
    val_files = pick(val_prefixes)
    test_files = pick(test_prefixes)

    # Default: anything not assigned goes to train (conservative), but we keep explicit too
    assigned = set(train_files) | set(val_files) | set(test_files)
    unassigned = [sf for sf in source_files if sf not in assigned]

    split = {
        "strategy": "by_file_prefix",
        "train_prefixes": train_prefixes,
        "val_prefixes": val_prefixes,
        "test_prefixes": test_prefixes,
        "train_files": train_files + unassigned,
        "val_files": val_files,
        "test_files": test_files,
        "all_files": source_files,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }

    write_json(out_dir / "split_by_file_prefix.json", split)
    return split
