"""
Загрузка и объединение сырых данных
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from .common import get_project_root, load_config, ensure_dir
from .manifest import load_manifest, extract_day_from_filename


# Список кодировок для попытки чтения
ENCODINGS_TO_TRY = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']


def read_csv_with_encoding(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Попытка прочитать CSV с разными кодировками

    Args:
        file_path: Путь к файлу
        **kwargs: Дополнительные параметры для pd.read_csv

    Returns:
        DataFrame
    """
    last_error = None

    for encoding in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=False,
                **kwargs
            )
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            # Другие ошибки пробрасываем сразу
            raise e

    # Если ничего не сработало, пробуем с errors='replace'
    try:
        df = pd.read_csv(
            file_path,
            encoding='utf-8',
            encoding_errors='replace',
            low_memory=False,
            **kwargs
        )
        print(f"   ⚠️ {file_path.name}: read with replacement characters")
        return df
    except Exception:
        raise last_error


def load_raw_data(
    config: Optional[Dict[str, Any]] = None,
    sample_frac: Optional[float] = None
) -> pd.DataFrame:
    """
    Загрузить все сырые CSV файлы и объединить

    Args:
        config: Конфигурация
        sample_frac: Доля сэмплирования (для отладки)

    Returns:
        Объединённый DataFrame
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_path = root / config["paths"]["raw_data"]

    csv_files = sorted(raw_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")

    dfs = []

    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        try:
            df = read_csv_with_encoding(csv_file)

            # Очищаем названия колонок
            df.columns = df.columns.str.strip()

            # Добавляем информацию об источнике
            df['_source_file'] = csv_file.name
            df['_day'] = extract_day_from_filename(csv_file.name)

            if sample_frac is not None:
                df = df.sample(frac=sample_frac, random_state=42)

            dfs.append(df)

        except Exception as e:
            print(f"\n❌ Error reading {csv_file.name}: {e}")
            raise

    # Объединяем
    combined = pd.concat(dfs, ignore_index=True)

    print(f"\n✅ Loaded {len(combined):,} rows from {len(csv_files)} files")

    return combined


def merge_csv_files(
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None
) -> Path:
    """
    Объединить все CSV в один файл (bronze layer)

    Returns:
        Путь к объединённому файлу
    """
    if config is None:
        config = load_config()

    root = get_project_root()

    if output_path is None:
        output_path = root / config["paths"]["interim_data"]

    ensure_dir(output_path)

    # Загружаем данные
    df = load_raw_data(config)

    # Сохраняем в parquet (эффективнее чем CSV)
    output_file = output_path / "bronze_combined.parquet"
    df.to_parquet(output_file, index=False, engine='pyarrow')

    print(f"✅ Saved bronze data to: {output_file}")
    print(f"   Size: {output_file.stat().st_size / (1024*1024):.1f} MB")

    return output_file


def load_bronze_data(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Загрузить bronze данные из parquet"""
    if config is None:
        config = load_config()

    root = get_project_root()
    bronze_path = root / config["paths"]["interim_data"] / "bronze_combined.parquet"

    if not bronze_path.exists():
        raise FileNotFoundError(
            f"Bronze data not found: {bronze_path}\n"
            "Run ingestion step first."
        )

    return pd.read_parquet(bronze_path)