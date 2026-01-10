"""
Разбиение данных на train/val/test
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .common import get_project_root, load_config, ensure_dir


def create_splits(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Разбить данные на train/val/test

    Поддерживает:
    - stratified: стратифицированное случайное разбиение (рекомендуется)
    - temporal: разбиение по дням
    - random: простое случайное разбиение

    Returns:
        {"train": df_train, "val": df_val, "test": df_test}
    """
    if config is None:
        config = load_config()

    split_config = config["splitting"]
    strategy = split_config["strategy"]

    print("="*60)
    print("РАЗБИЕНИЕ ДАННЫХ")
    print("="*60)
    print(f"\n📊 Стратегия: {strategy}")

    if strategy == "temporal":
        splits = _temporal_split(df, split_config)
    elif strategy in ["stratified", "random"]:
        splits = _stratified_split(df, split_config, config)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    # Статистика
    print("\n📊 Результаты разбиения:")
    total = len(df)

    for name, split_df in splits.items():
        pct = 100 * len(split_df) / total
        print(f"   {name}: {len(split_df):,} строк ({pct:.1f}%)")

        if "label_binary" in split_df.columns:
            n_benign = (split_df["label_binary"] == 0).sum()
            n_attack = (split_df["label_binary"] == 1).sum()
            pct_benign = 100 * n_benign / len(split_df)
            pct_attack = 100 * n_attack / len(split_df)
            print(f"      - Benign: {n_benign:,} ({pct_benign:.1f}%)")
            print(f"      - Attack: {n_attack:,} ({pct_attack:.1f}%)")

        if "label_multiclass" in split_df.columns:
            n_classes = split_df["label_multiclass"].nunique()
            print(f"      - Multiclass: {n_classes} классов")

    # Проверка баланса
    _check_split_balance(splits)

    return splits


def _stratified_split(
    df: pd.DataFrame,
    split_config: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """Стратифицированное разбиение"""
    test_size = split_config["test_size"]
    val_size = split_config["val_size"]
    random_state = split_config["random_state"]
    stratify = split_config.get("stratify", True)
    stratify_col = split_config.get("stratify_column", "label_binary")

    print(f"   Test size: {test_size}")
    print(f"   Val size: {val_size}")
    print(f"   Stratify by: {stratify_col if stratify else 'None'}")

    # Колонка для стратификации
    if stratify and stratify_col in df.columns:
        stratify_data = df[stratify_col]
    else:
        stratify_data = None
        if stratify:
            print(f"   ⚠️ Column '{stratify_col}' not found, using random split")

    # Сначала отделяем test
    df_temp, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_data
    )

    # Затем val от оставшегося
    val_adjusted = val_size / (1 - test_size)

    if stratify and stratify_col in df_temp.columns:
        stratify_data_temp = df_temp[stratify_col]
    else:
        stratify_data_temp = None

    df_train, df_val = train_test_split(
        df_temp,
        test_size=val_adjusted,
        random_state=random_state,
        stratify=stratify_data_temp
    )

    return {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True)
    }


def _temporal_split(
    df: pd.DataFrame,
    split_config: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """Разбиение по дням недели"""
    temporal_mapping = split_config["temporal_mapping"]

    if "_day" not in df.columns:
        raise ValueError("Column '_day' not found. Run ingestion first.")

    splits = {}

    for split_name, days in temporal_mapping.items():
        mask = df["_day"].isin(days)
        splits[split_name] = df[mask].copy().reset_index(drop=True)
        print(f"   {split_name}: дни {days} -> {mask.sum():,} строк")

    return splits


def _check_split_balance(splits: Dict[str, pd.DataFrame]) -> None:
    """Проверить баланс классов в сплитах"""
    print("\n📊 Проверка баланса классов:")

    attack_ratios = {}
    for name, df in splits.items():
        if "label_binary" in df.columns:
            attack_ratio = (df["label_binary"] == 1).mean()
            attack_ratios[name] = attack_ratio

    if attack_ratios:
        ratios = list(attack_ratios.values())
        max_diff = max(ratios) - min(ratios)

        if max_diff < 0.05:
            print("   ✅ Классы сбалансированы между сплитами")
        elif max_diff < 0.15:
            print("   ⚠️ Небольшой дисбаланс между сплитами")
            for name, ratio in attack_ratios.items():
                print(f"      {name}: {ratio*100:.1f}% атак")
        else:
            print("   🔴 Сильный дисбаланс между сплитами!")
            for name, ratio in attack_ratios.items():
                print(f"      {name}: {ratio*100:.1f}% атак")
            print("   💡 Рекомендуется использовать strategy: stratified")


def save_splits(
    splits: Dict[str, pd.DataFrame],
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None
) -> Dict[str, Path]:
    """Сохранить сплиты"""
    if config is None:
        config = load_config()

    root = get_project_root()
    if output_path is None:
        output_path = root / config["paths"]["processed_data"] / "splits"

    ensure_dir(output_path)

    saved_files = {}
    split_info = {}

    for name, df in splits.items():
        file_path = output_path / f"{name}.parquet"
        df.to_parquet(file_path, index=False)
        saved_files[name] = file_path

        split_info[name] = {
            "rows": len(df),
            "file": str(file_path.name)
        }

        if "label_binary" in df.columns:
            binary_dist = df["label_binary"].value_counts().to_dict()
            split_info[name]["binary_distribution"] = {
                "benign": int(binary_dist.get(0, 0)),
                "attack": int(binary_dist.get(1, 0))
            }

        if "label_multiclass" in df.columns:
            multi_dist = df["label_multiclass"].value_counts().to_dict()
            split_info[name]["multiclass_distribution"] = {
                str(k): int(v) for k, v in multi_dist.items()
            }

        print(f"💾 {name} saved to: {file_path}")

    # Сохраняем метаданные сплитов
    meta_path = output_path / "split_metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            "strategy": config["splitting"]["strategy"],
            "random_state": config["splitting"]["random_state"],
            "test_size": config["splitting"]["test_size"],
            "val_size": config["splitting"]["val_size"],
            "stratify": config["splitting"].get("stratify", True),
            "splits": split_info
        }, f, indent=2, ensure_ascii=False)

    saved_files["metadata"] = meta_path

    return saved_files


def load_splits(
    config: Optional[Dict[str, Any]] = None,
    splits_path: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """Загрузить сплиты"""
    if config is None:
        config = load_config()

    root = get_project_root()
    if splits_path is None:
        splits_path = root / config["paths"]["processed_data"] / "splits"

    splits = {}
    for split_name in ["train", "val", "test"]:
        file_path = splits_path / f"{split_name}.parquet"
        if file_path.exists():
            splits[split_name] = pd.read_parquet(file_path)
            print(f"📂 Loaded {split_name}: {len(splits[split_name]):,} rows")

    return splits