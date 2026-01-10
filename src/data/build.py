"""
Очистка данных, Feature Engineering и препроцессинг
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder

from .common import get_project_root, load_config, ensure_dir
from .ingest import load_bronze_data


def create_feature_schema(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Создать схему признаков (feature contract)

    Returns:
        Словарь со схемой признаков
    """
    if config is None:
        config = load_config()

    drop_cols = config["ingestion"]["drop_columns"]
    target_col = config["ingestion"]["target_column"]

    # Все колонки
    all_cols = df.columns.tolist()

    # Служебные колонки (добавленные нами)
    meta_cols = [c for c in all_cols if c.startswith('_')]

    # Обработка дублирующегося столбца Fwd Header Length
    # Переименовываем .1 версию
    duplicate_cols = [c for c in all_cols if '.1' in c]

    # Колонки для удаления
    cols_to_drop = drop_cols + meta_cols + [target_col] + duplicate_cols

    # Признаки
    feature_cols = [c for c in all_cols if c not in cols_to_drop]

    # Типы признаков
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    # Проверяем на дублирующиеся названия
    seen = set()
    unique_numeric = []
    for col in numeric_features:
        if col not in seen:
            seen.add(col)
            unique_numeric.append(col)

    # Статистики для каждого признака
    feature_stats = {}
    for col in unique_numeric:
        valid_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) > 0:
            feature_stats[col] = {
                "dtype": str(df[col].dtype),
                "min": float(valid_data.min()),
                "max": float(valid_data.max()),
                "mean": float(valid_data.mean()),
                "std": float(valid_data.std()) if valid_data.std() == valid_data.std() else 0.0,
                "median": float(valid_data.median()),
                "q01": float(valid_data.quantile(0.01)),
                "q99": float(valid_data.quantile(0.99)),
                "null_count": int(df[col].isna().sum()),
                "inf_count": int(((df[col] == np.inf) | (df[col] == -np.inf)).sum())
            }

    schema = {
        "version": "1.0",
        "target_column": target_col,
        "feature_columns": unique_numeric,
        "categorical_columns": categorical_features,
        "drop_columns": cols_to_drop,
        "duplicate_columns": duplicate_cols,
        "meta_columns": meta_cols,
        "total_features": len(unique_numeric),
        "feature_stats": feature_stats
    }

    return schema


def clean_data(
    df: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Очистить данные

    Шаги:
    1. Удаление полностью пустых/битых строк
    2. Удаление дубликатов
    3. Удаление дублирующихся колонок
    4. Обработка Inf
    5. Обработка NaN
    6. Клиппинг выбросов
    """
    if config is None:
        config = load_config()

    if df is None:
        df = load_bronze_data(config)

    # ВАЖНО: создаём копию чтобы избежать SettingWithCopyWarning
    df = df.copy()

    cleaning_config = config["cleaning"]
    target_col = config["ingestion"]["target_column"]

    print("="*60)
    print("ОЧИСТКА ДАННЫХ")
    print("="*60)

    initial_rows = len(df)
    print(f"\n📊 Исходных строк: {initial_rows:,}")

    # 0. Удаление полностью пустых строк (где даже Label пустой)
    empty_mask = df[target_col].isna()
    if empty_mask.sum() > 0:
        print(f"\n📊 Удаление пустых строк (Label is NaN)...")
        print(f"   Найдено пустых строк: {empty_mask.sum():,}")
        df = df[~empty_mask].copy()
        print(f"   После удаления: {len(df):,}")

    # 1. Удаление дубликатов
    if cleaning_config["remove_duplicates"]:
        before_dup = len(df)
        # Убираем мета-колонки при проверке дубликатов
        check_cols = [c for c in df.columns if not c.startswith('_')]
        df = df.drop_duplicates(subset=check_cols, keep='first').copy()
        removed = before_dup - len(df)
        print(f"\n📊 Удаление дубликатов...")
        print(f"   Удалено: {removed:,}")
        print(f"   Осталось: {len(df):,}")

    # 2. Удаление дублирующихся колонок (например, Fwd Header Length.1)
    dup_cols = [c for c in df.columns if '.1' in c]
    if dup_cols:
        print(f"\n📊 Удаление дублирующихся колонок: {dup_cols}")
        df = df.drop(columns=dup_cols)

    # Получаем числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.startswith('_')]

    # 3. Обработка Inf
    inf_replacement = cleaning_config["inf_replacement"]
    print(f"\n📊 Обработка Inf (стратегия: {inf_replacement})...")

    total_inf_replaced = 0
    for col in numeric_cols:
        inf_mask = (df[col] == np.inf) | (df[col] == -np.inf)
        inf_count = inf_mask.sum()

        if inf_count > 0:
            total_inf_replaced += inf_count

            if inf_replacement == "nan":
                df.loc[inf_mask, col] = np.nan
            elif inf_replacement == "clip":
                valid_data = df.loc[~inf_mask, col]
                if len(valid_data) > 0:
                    max_val = valid_data.max()
                    min_val = valid_data.min()
                    df.loc[df[col] == np.inf, col] = max_val
                    df.loc[df[col] == -np.inf, col] = min_val
            elif inf_replacement == "median":
                valid_data = df.loc[~inf_mask, col]
                if len(valid_data) > 0:
                    median_val = valid_data.median()
                    df.loc[inf_mask, col] = median_val

    print(f"   Заменено Inf значений: {total_inf_replaced:,}")

    # 4. Обработка NaN
    nan_strategy = cleaning_config["nan_strategy"]
    print(f"\n📊 Обработка NaN (стратегия: {nan_strategy})...")

    nan_before = df[numeric_cols].isna().sum().sum()

    if nan_strategy == "median":
        for col in numeric_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                median_val = df[col].median()
                # Если медиана тоже NaN, используем 0
                if pd.isna(median_val):
                    median_val = 0
                df.loc[df[col].isna(), col] = median_val
    elif nan_strategy == "mean":
        for col in numeric_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                mean_val = df[col].mean()
                if pd.isna(mean_val):
                    mean_val = 0
                df.loc[df[col].isna(), col] = mean_val
    elif nan_strategy == "zero":
        for col in numeric_cols:
            df.loc[df[col].isna(), col] = 0
    elif nan_strategy == "drop":
        df = df.dropna(subset=numeric_cols).copy()

    nan_after = df[numeric_cols].isna().sum().sum()
    print(f"   NaN до: {nan_before:,}, после: {nan_after:,}")

    # 5. Клиппинг выбросов
    if cleaning_config["clip_outliers"]:
        lower_pct = cleaning_config["clip_lower_percentile"]
        upper_pct = cleaning_config["clip_upper_percentile"]
        print(f"\n📊 Клиппинг выбросов ({lower_pct}-{upper_pct} percentile)...")

        clipped_cols = 0
        for col in numeric_cols:
            lower = df[col].quantile(lower_pct)
            upper = df[col].quantile(upper_pct)

            # Проверяем что границы валидны
            if pd.notna(lower) and pd.notna(upper) and lower < upper:
                before_clip = ((df[col] < lower) | (df[col] > upper)).sum()
                if before_clip > 0:
                    df.loc[:, col] = df[col].clip(lower, upper)
                    clipped_cols += 1

        print(f"   Обработано колонок: {clipped_cols}")

    print(f"\n✅ Итого строк после очистки: {len(df):,}")
    print(f"   Удалено всего: {initial_rows - len(df):,} ({100*(initial_rows - len(df))/initial_rows:.1f}%)")

    return df


def preprocess_data(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
    fit: bool = True,
    preprocessor: Optional[Any] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """
    Препроцессинг данных: скейлинг и кодирование

    Args:
        df: DataFrame
        config: Конфигурация
        schema: Схема признаков
        fit: Обучать ли препроцессор (True для train)
        preprocessor: Готовый препроцессор (для val/test)

    Returns:
        (processed_df, label_mapping, preprocessor)
    """
    if config is None:
        config = load_config()

    # Создаём копию
    df = df.copy()

    if schema is None:
        schema = create_feature_schema(df, config)

    print("\n" + "="*60)
    print("ПРЕПРОЦЕССИНГ")
    print("="*60)

    target_col = config["ingestion"]["target_column"]
    feature_cols = schema["feature_columns"]

    # Фильтруем только существующие колонки
    feature_cols = [c for c in feature_cols if c in df.columns]

    preprocessing_config = config["preprocessing"]
    labels_config = config["labels"]

    # 1. Нормализация названий классов (убираем лишние пробелы, приводим к единому виду)
    print("\n📊 Нормализация меток...")
    df[target_col] = df[target_col].str.strip()

    # Унификация названий Web Attack (разные варианты написания)
    label_fixes = {
        'Web Attack Brute Force': 'Web Attack – Brute Force',
        'Web Attack XSS': 'Web Attack – XSS',
        'Web Attack Sql Injection': 'Web Attack – Sql Injection',
        'Web Attack - Brute Force': 'Web Attack – Brute Force',
        'Web Attack - XSS': 'Web Attack – XSS',
        'Web Attack - Sql Injection': 'Web Attack – Sql Injection',
    }
    df[target_col] = df[target_col].replace(label_fixes)

    print(f"   Уникальных классов: {df[target_col].nunique()}")

    # 2. Создание бинарных и мультикласс меток
    print("\n📊 Создание меток...")

    # Бинарная классификация
    binary_mapping = labels_config["binary_mapping"]
    df["label_binary"] = df[target_col].apply(
        lambda x: binary_mapping.get(x, binary_mapping["default"])
    )

    # Мультиклассовая классификация
    multiclass_mapping = labels_config["multiclass_mapping"]
    max_class = max(multiclass_mapping.values())

    def get_multiclass_label(x):
        if x in multiclass_mapping:
            return multiclass_mapping[x]
        # Пробуем найти частичное совпадение
        for key, val in multiclass_mapping.items():
            if key.lower() in x.lower() or x.lower() in key.lower():
                return val
        return max_class + 1  # unknown

    df["label_multiclass"] = df[target_col].apply(get_multiclass_label)

    # Проверяем unknown классы
    unknown_count = (df["label_multiclass"] == max_class + 1).sum()
    if unknown_count > 0:
        unknown_labels = df[df["label_multiclass"] == max_class + 1][target_col].unique()
        print(f"   ⚠️ Неизвестных классов: {unknown_count} ({unknown_labels[:5]})")

    label_mapping = {
        "binary": binary_mapping,
        "multiclass": multiclass_mapping,
        "binary_column": "label_binary",
        "multiclass_column": "label_multiclass"
    }

    binary_dist = df['label_binary'].value_counts()
    print(f"   Binary - Benign: {binary_dist.get(0, 0):,}, Attack: {binary_dist.get(1, 0):,}")
    print(f"   Multiclass: {df['label_multiclass'].nunique()} unique classes")

    # 3. Проверка на NaN/Inf перед скейлингом
    print(f"\n📊 Проверка данных перед скейлингом...")

    # Заменяем оставшиеся проблемные значения
    for col in feature_cols:
        # Заменяем inf на NaN, затем NaN на 0
        df.loc[:, col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isna().sum() > 0:
            df.loc[:, col] = df[col].fillna(0)

    nan_count = df[feature_cols].isna().sum().sum()
    inf_count = sum(((df[col] == np.inf) | (df[col] == -np.inf)).sum() for col in feature_cols)
    print(f"   NaN: {nan_count}, Inf: {inf_count}")

    # 4. Скейлинг
    print(f"\n📊 Скейлинг (метод: {preprocessing_config['scaler']})...")

    scaler_type = preprocessing_config["scaler"]

    if fit:
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler_type}")

        # Обучаем скейлер
        scaled_values = scaler.fit_transform(df[feature_cols])
        df.loc[:, feature_cols] = scaled_values
        preprocessor = {"scaler": scaler, "feature_cols": feature_cols}
    else:
        if preprocessor is None:
            raise ValueError("Preprocessor required when fit=False")
        scaler = preprocessor["scaler"]
        scaled_values = scaler.transform(df[feature_cols])
        df.loc[:, feature_cols] = scaled_values

    print(f"   Scaled {len(feature_cols)} features")

    # 5. Финальная проверка
    print(f"\n📊 Финальная проверка...")
    print(f"   Строк: {len(df):,}")
    print(f"   Признаков: {len(feature_cols)}")
    print(f"   Колонок всего: {len(df.columns)}")

    return df, label_mapping, preprocessor


def save_processed_data(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    label_mapping: Dict[str, Any],
    preprocessor: Any,
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None
) -> Dict[str, Path]:
    """Сохранить обработанные данные и артефакты"""
    if config is None:
        config = load_config()

    root = get_project_root()
    if output_path is None:
        output_path = root / config["paths"]["processed_data"]

    ensure_dir(output_path)
    artifacts_path = root / config["paths"]["artifacts"]
    ensure_dir(artifacts_path)

    saved_files = {}

    # 1. Данные
    data_path = output_path / "processed_data.parquet"
    df.to_parquet(data_path, index=False)
    saved_files["data"] = data_path
    print(f"\n💾 Data saved to: {data_path}")
    print(f"   Size: {data_path.stat().st_size / (1024*1024):.1f} MB")

    # 2. Схема признаков
    schema_path = artifacts_path / "feature_schema.json"
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    saved_files["schema"] = schema_path
    print(f"💾 Schema saved to: {schema_path}")

    # 3. Маппинг меток
    labels_path = artifacts_path / "label_mapping.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    saved_files["labels"] = labels_path
    print(f"💾 Labels saved to: {labels_path}")

    # 4. Препроцессор
    if config["preprocessing"]["save_preprocessor"]:
        preprocessor_path = artifacts_path / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        saved_files["preprocessor"] = preprocessor_path
        print(f"💾 Preprocessor saved to: {preprocessor_path}")

    return saved_files