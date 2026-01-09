from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, TextIO

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify_column(name: str) -> str:
    """
    Make a safe column name:
    - normalize common whitespace (incl. NBSP)
    - strip
    - lower
    - replace non-alnum with underscore
    - collapse underscores
    """
    s = (name or "")
    # Some CICIDS files contain non-breaking spaces in headers.
    s = s.replace("\u00a0", " ").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"


def make_unique(names: List[str]) -> List[str]:
    """
    Ensure all column names are unique and stable.
    If duplicates exist, append __dupN.
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for n in names:
        base = n
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__dup{seen[base]}")
    return out


def normalize_label(x: str) -> str:
    """Normalize raw CICIDS label strings.

    The CICIDS-2017 CSVs are often encoded as cp1252 and sometimes include
    an en dash between tokens (commonly seen as 'Web Attack – XSS').
    If read with the wrong encoding, that dash may become the replacement
    character (\uFFFD). We normalize these cases to a simple hyphen.

    Note: keep this function *pure* and safe to call on any scalar.
    """
    if x is None:
        return ""
    s = str(x).strip()

    # Normalize various dash / replacement artefacts
    s = s.replace("\uFFFD", "-")  # replacement char
    s = s.replace("�", "-")  # sometimes appears in terminal/fonts
    s = s.replace("–", "-").replace("—", "-")  # en/em dash

    # Normalize spacing around hyphens and collapse whitespace
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def replace_inf(df: pd.DataFrame, cols: Iterable[str]) -> None:
    df[list(cols)] = df[list(cols)].replace([np.inf, -np.inf], np.nan)


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str], downcast_float32: bool = True) -> None:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # Downcast only floats; keep ints if possible
        if downcast_float32:
            # pandas will convert to float64 if NaN present; we downcast to float32
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype("float32")


def read_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to read configs/data_pipeline.yaml. Install with: pip install pyyaml"
        ) from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
