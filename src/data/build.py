"""
Data cleaning, feature engineering, and preprocessing.
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
    Create a feature schema (feature contract).

    Returns:
        A dictionary containing the feature schema.
    """
    if config is None:
        config = load_config()

    drop_cols = config["ingestion"]["drop_columns"]
    target_col = config["ingestion"]["target_column"]

    # All columns
    all_cols = df.columns.tolist()

    # Internal metadata columns (added by this pipeline)
    meta_cols = [c for c in all_cols if c.startswith('_')]

    # Handle duplicated columns (e.g., "Fwd Header Length.1")
    duplicate_cols = [c for c in all_cols if '.1' in c]

    # Columns to drop
    cols_to_drop = drop_cols + meta_cols + [target_col] + duplicate_cols

    # Feature columns
    feature_cols = [c for c in all_cols if c not in cols_to_drop]

    # Feature types
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    # Ensure unique feature names (defensive, in case of duplicates)
    seen = set()
    unique_numeric = []
    for col in numeric_features:
        if col not in seen:
            seen.add(col)
            unique_numeric.append(col)

    # Basic statistics for each numeric feature
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
    Clean the data.

    Steps:
    1. Remove fully empty/broken rows
    2. Remove duplicates
    3. Drop duplicated columns
    4. Handle Inf
    5. Handle NaN
    6. Clip outliers
    """
    if config is None:
        config = load_config()

    if df is None:
        df = load_bronze_data(config)

    # Important: work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    cleaning_config = config["cleaning"]
    target_col = config["ingestion"]["target_column"]

    print("=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    initial_rows = len(df)
    print(f"Initial rows: {initial_rows:,}")

    # 0. Drop fully empty rows (where even the Label is missing)
    empty_mask = df[target_col].isna()
    if empty_mask.sum() > 0:
        print("Dropping rows with empty labels (Label is NaN)...")
        print(f"Found empty-label rows: {empty_mask.sum():,}")
        df = df[~empty_mask].copy()
        print(f"Rows after drop: {len(df):,}")

    # 1. Remove duplicates
    if cleaning_config["remove_duplicates"]:
        before_dup = len(df)
        # Exclude pipeline metadata columns from duplicate checking
        check_cols = [c for c in df.columns if not c.startswith('_')]
        df = df.drop_duplicates(subset=check_cols, keep='first').copy()
        removed = before_dup - len(df)
        print("Removing duplicates...")
        print(f"Removed: {removed:,}")
        print(f"Remaining: {len(df):,}")

    # 2. Drop duplicated columns (e.g., "Fwd Header Length.1")
    dup_cols = [c for c in df.columns if '.1' in c]
    if dup_cols:
        print(f"Dropping duplicated columns: {dup_cols}")
        df = df.drop(columns=dup_cols)

    # Numeric columns (excluding metadata columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.startswith('_')]

    # 3. Handle Inf
    inf_replacement = cleaning_config["inf_replacement"]
    print(f"Handling Inf (strategy: {inf_replacement})...")

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

    print(f"Inf values replaced: {total_inf_replaced:,}")

    # 4. Handle NaN
    nan_strategy = cleaning_config["nan_strategy"]
    print(f"Handling NaN (strategy: {nan_strategy})...")

    nan_before = df[numeric_cols].isna().sum().sum()

    if nan_strategy == "median":
        for col in numeric_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                median_val = df[col].median()
                # If the median is also NaN, fall back to 0
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
    print(f"NaN before: {nan_before:,}, after: {nan_after:,}")

    # 5. Outlier clipping
    if cleaning_config["clip_outliers"]:
        lower_pct = cleaning_config["clip_lower_percentile"]
        upper_pct = cleaning_config["clip_upper_percentile"]
        print(f"Clipping outliers ({lower_pct}-{upper_pct} percentile)...")

        clipped_cols = 0
        for col in numeric_cols:
            lower = df[col].quantile(lower_pct)
            upper = df[col].quantile(upper_pct)

            # Ensure valid bounds
            if pd.notna(lower) and pd.notna(upper) and lower < upper:
                before_clip = ((df[col] < lower) | (df[col] > upper)).sum()
                if before_clip > 0:
                    df.loc[:, col] = df[col].clip(lower, upper)
                    clipped_cols += 1

        print(f"Columns clipped: {clipped_cols}")

    removed_total = initial_rows - len(df)
    removed_pct = 100 * removed_total / initial_rows if initial_rows > 0 else 0.0
    print(f"Final rows after cleaning: {len(df):,}")
    print(f"Total removed: {removed_total:,} ({removed_pct:.1f}%)")

    return df


def preprocess_data(
        df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        schema: Optional[Dict[str, Any]] = None,
        fit: bool = True,
        preprocessor: Optional[Any] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """
    Preprocess data: scaling and label encoding.

    Args:
        df: DataFrame
        config: Configuration
        schema: Feature schema
        fit: Whether to fit the preprocessor (True for train)
        preprocessor: Pre-fitted preprocessor (for val/test)

    Returns:
        (processed_df, label_mapping, preprocessor)
    """
    if config is None:
        config = load_config()

    # Work on a copy
    df = df.copy()

    if schema is None:
        schema = create_feature_schema(df, config)

    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    target_col = config["ingestion"]["target_column"]
    feature_cols = schema["feature_columns"]

    # Keep only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    preprocessing_config = config["preprocessing"]
    labels_config = config["labels"]

    # 1. Normalize label strings (strip whitespace, unify variants)
    print("Normalizing labels...")
    df[target_col] = df[target_col].str.strip()

    # Unify "Web Attack" naming variants
    label_fixes = {
        'Web Attack Brute Force': 'Web Attack – Brute Force',
        'Web Attack XSS': 'Web Attack – XSS',
        'Web Attack Sql Injection': 'Web Attack – Sql Injection',
        'Web Attack - Brute Force': 'Web Attack – Brute Force',
        'Web Attack - XSS': 'Web Attack – XSS',
        'Web Attack - Sql Injection': 'Web Attack – Sql Injection',
    }
    df[target_col] = df[target_col].replace(label_fixes)

    print(f"Unique classes: {df[target_col].nunique()}")

    # 2. Create binary and multiclass labels
    print("Creating labels...")

    binary_mapping = labels_config["binary_mapping"]
    df["label_binary"] = df[target_col].apply(
        lambda x: binary_mapping.get(x, binary_mapping["default"])
    )

    multiclass_mapping = labels_config["multiclass_mapping"]
    max_class = max(multiclass_mapping.values())

    def get_multiclass_label(x):
        if x in multiclass_mapping:
            return multiclass_mapping[x]
        # Attempt partial match
        for key, val in multiclass_mapping.items():
            if key.lower() in x.lower() or x.lower() in key.lower():
                return val
        return max_class + 1  # unknown

    df["label_multiclass"] = df[target_col].apply(get_multiclass_label)

    # Unknown class check
    unknown_count = (df["label_multiclass"] == max_class + 1).sum()
    if unknown_count > 0:
        unknown_labels = df[df["label_multiclass"] == max_class + 1][target_col].unique()
        print(f"Unknown class rows: {unknown_count} (examples: {unknown_labels[:5]})")

    label_mapping = {
        "binary": binary_mapping,
        "multiclass": multiclass_mapping,
        "binary_column": "label_binary",
        "multiclass_column": "label_multiclass"
    }

    binary_dist = df['label_binary'].value_counts()
    print(f"Binary distribution - Benign: {binary_dist.get(0, 0):,}, Attack: {binary_dist.get(1, 0):,}")
    print(f"Multiclass: {df['label_multiclass'].nunique()} unique classes")

    # 3. Sanity check before scaling (ensure no Inf/NaN remain)
    print("Checking data before scaling...")

    for col in feature_cols:
        # Replace inf with NaN, then fill NaN with 0
        df.loc[:, col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isna().sum() > 0:
            df.loc[:, col] = df[col].fillna(0)

    nan_count = df[feature_cols].isna().sum().sum()
    inf_count = sum(((df[col] == np.inf) | (df[col] == -np.inf)).sum() for col in feature_cols)
    print(f"Remaining NaN: {nan_count}, remaining Inf: {inf_count}")

    # 4. Scaling
    scaler_type = preprocessing_config["scaler"]
    print(f"Scaling (method: {scaler_type})...")

    if fit:
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler_type}")

        scaled_values = scaler.fit_transform(df[feature_cols])
        df.loc[:, feature_cols] = scaled_values
        preprocessor = {"scaler": scaler, "feature_cols": feature_cols}
    else:
        if preprocessor is None:
            raise ValueError("Preprocessor required when fit=False")
        scaler = preprocessor["scaler"]
        scaled_values = scaler.transform(df[feature_cols])
        df.loc[:, feature_cols] = scaled_values

    print(f"Scaled features: {len(feature_cols)}")

    # 5. Final summary
    print("Final check...")
    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Total columns: {len(df.columns)}")

    return df, label_mapping, preprocessor


def save_processed_data(
        df: pd.DataFrame,
        schema: Dict[str, Any],
        label_mapping: Dict[str, Any],
        preprocessor: Any,
        config: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None
) -> Dict[str, Path]:
    """Save processed data and related artifacts."""
    if config is None:
        config = load_config()

    root = get_project_root()
    if output_path is None:
        output_path = root / config["paths"]["processed_data"]

    ensure_dir(output_path)
    artifacts_path = root / config["paths"]["artifacts"]
    ensure_dir(artifacts_path)

    saved_files = {}

    # 1. Data
    data_path = output_path / "processed_data.parquet"
    df.to_parquet(data_path, index=False)
    saved_files["data"] = data_path
    print(f"Data saved to: {data_path}")
    print(f"Size: {data_path.stat().st_size / (1024 * 1024):.1f} MB")

    # 2. Feature schema
    schema_path = artifacts_path / "feature_schema.json"
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    saved_files["schema"] = schema_path
    print(f"Schema saved to: {schema_path}")

    # 3. Label mapping
    labels_path = artifacts_path / "label_mapping.json"
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    saved_files["labels"] = labels_path
    print(f"Labels saved to: {labels_path}")

    # 4. Preprocessor
    if config["preprocessing"]["save_preprocessor"]:
        preprocessor_path = artifacts_path / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        saved_files["preprocessor"] = preprocessor_path
        print(f"Preprocessor saved to: {preprocessor_path}")

    return saved_files
