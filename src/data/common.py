"""
Common utilities for data pipeline operations.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def get_project_root() -> Path:
    """Return the project root directory."""
    current = Path(__file__).resolve()
    # Detect the root by presence of "configs" or "run_data_pipeline.py"
    for parent in current.parents:
        if (parent / "configs").exists() or (parent / "run_data_pipeline.py").exists():
            return parent
    return current.parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if config_path is None:
        config_path = get_project_root() / "configs" / "data_pipeline.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(level: str = "INFO", log_format: Optional[str] = None) -> logging.Logger:
    """Configure and return a logger instance."""
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger("DataPipeline")
    return logger


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(file_path: Path) -> str:
    """Compute a file hash (integrity check)."""
    import xxhash

    hasher = xxhash.xxh64()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def format_number(n: int) -> str:
    """Format an integer with thousands separators."""
    return f"{n:,}"
