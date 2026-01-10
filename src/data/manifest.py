"""
Создание и управление манифестом данных
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .common import get_project_root, load_config, get_file_hash, ensure_dir


# Список кодировок для попытки чтения
ENCODINGS_TO_TRY = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']


def read_csv_safe(file_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Безопасное чтение CSV с автоопределением кодировки"""
    for encoding in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=False,
                nrows=nrows
            )
            return df
        except UnicodeDecodeError:
            continue

    # Fallback с заменой символов
    return pd.read_csv(
        file_path,
        encoding='utf-8',
        encoding_errors='replace',
        low_memory=False,
        nrows=nrows
    )


def count_lines_safe(file_path: Path) -> int:
    """Подсчёт строк с обработкой кодировки"""
    for encoding in ENCODINGS_TO_TRY:
        try:
            with open(file_path, 'r', encoding=encoding, errors='strict') as f:
                return sum(1 for _ in f) - 1  # минус заголовок
        except UnicodeDecodeError:
            continue

    # Fallback с заменой
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return sum(1 for _ in f) - 1


def create_manifest(
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Создать манифест исходных данных

    Манифест содержит:
    - Список файлов
    - Размеры и хэши
    - Базовую статистику
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_path = root / config["paths"]["raw_data"]

    if output_path is None:
        output_path = root / config["paths"]["interim_data"]

    ensure_dir(output_path)

    # Собираем информацию о файлах
    csv_files = sorted(raw_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")

    files_info = []
    total_rows = 0
    all_columns = None

    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}...")

        # Базовая информация
        file_info = {
            "filename": csv_file.name,
            "path": str(csv_file.relative_to(root)),
            "size_bytes": csv_file.stat().st_size,
            "size_mb": round(csv_file.stat().st_size / (1024 * 1024), 2),
            "hash": get_file_hash(csv_file),
        }

        # Читаем для проверки колонок
        df = read_csv_safe(csv_file, nrows=5)
        df.columns = df.columns.str.strip()

        # Подсчёт строк
        row_count = count_lines_safe(csv_file)

        file_info["row_count"] = row_count
        file_info["columns"] = list(df.columns)
        total_rows += row_count

        # Определяем день недели из имени файла
        day = extract_day_from_filename(csv_file.name)
        file_info["day"] = day

        # Проверяем консистентность колонок
        if all_columns is None:
            all_columns = set(df.columns)
        else:
            if set(df.columns) != all_columns:
                file_info["column_mismatch"] = True

        files_info.append(file_info)

    # Формируем манифест
    manifest = {
        "created_at": datetime.now().isoformat(),
        "source": str(raw_path),
        "total_files": len(csv_files),
        "total_rows": total_rows,
        "columns": list(all_columns) if all_columns else [],
        "column_count": len(all_columns) if all_columns else 0,
        "files": files_info,
    }

    # Сохраняем
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Manifest saved to: {manifest_path}")
    print(f"   Total files: {manifest['total_files']}")
    print(f"   Total rows: {manifest['total_rows']:,}")

    return manifest


def extract_day_from_filename(filename: str) -> str:
    """Извлечь день недели из имени файла"""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    filename_lower = filename.lower()

    for day in days:
        if day.lower() in filename_lower:
            return day

    return "Unknown"


def load_manifest(manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    """Загрузить манифест"""
    if manifest_path is None:
        config = load_config()
        root = get_project_root()
        manifest_path = root / config["paths"]["interim_data"] / "manifest.json"

    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)