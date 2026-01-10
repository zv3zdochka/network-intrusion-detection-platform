#!/usr/bin/env python3
"""
Скрипт обучения моделей (локальный запуск)
"""

import sys
from pathlib import Path

# Добавляем корень проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import yaml
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')


def load_data(data_path: Path, artifacts_path: Path):
    """Загрузить данные"""
    print("📂 Loading data...")

    train_df = pd.read_parquet(data_path / "splits" / "train.parquet")
    val_df = pd.read_parquet(data_path / "splits" / "val.parquet")
    test_df = pd.read_parquet(data_path / "splits" / "test.parquet")

    with open(artifacts_path / "feature_schema.json", 'r') as f:
        feature_schema = json.load(f)

    feature_cols = feature_schema['feature_columns']

    X_train = train_df[feature_cols].values
    y_train = train_df['label_binary'].values

    X_val = val_df[feature_cols].values
    y_val = val_df['label_binary'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['label_binary'].values

    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_model(model, name, X_train, y_train, X_val, y_val, X_test, y_test):
    """Обучить и оценить модель"""
    print(f"\n🚀 Training: {name}")

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    results = {'name': name, 'training_time': train_time}

    for split_name, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        results[f'{split_name}_f1'] = f1_score(y, y_pred)
        results[f'{split_name}_roc_auc'] = roc_auc_score(y, y_proba)
        results[f'{split_name}_pr_auc'] = average_precision_score(y, y_proba)

    print(f"   Val F1: {results['val_f1']:.4f}, Test F1: {results['test_f1']:.4f}")

    return model, results


def main():
    # Пути
    root = Path(__file__).parent.parent
    data_path = root / "data" / "processed"
    artifacts_path = root / "artifacts"
    output_path = root / "models"
    output_path.mkdir(exist_ok=True)

    # Загрузка данных
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data(
        data_path, artifacts_path
    )

    # Scale pos weight для XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Модели для обучения
    models_config = [
        ("RF_baseline", RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                               n_jobs=-1, random_state=42)),
        ("RF_deep", RandomForestClassifier(n_estimators=200, max_depth=20,
                                           class_weight='balanced', n_jobs=-1, random_state=42)),
        ("XGB_baseline", xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                           scale_pos_weight=scale_pos_weight,
                                           use_label_encoder=False, eval_metric='logloss',
                                           n_jobs=-1, random_state=42, verbosity=0)),
        ("XGB_deep", xgb.XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.05,
                                       scale_pos_weight=scale_pos_weight,
                                       use_label_encoder=False, eval_metric='logloss',
                                       n_jobs=-1, random_state=42, verbosity=0)),
        ("LGBM_baseline", lgb.LGBMClassifier(n_estimators=100, num_leaves=31,
                                             class_weight='balanced', n_jobs=-1,
                                             random_state=42, verbose=-1)),
        ("LGBM_deep", lgb.LGBMClassifier(n_estimators=200, num_leaves=63, max_depth=10,
                                         class_weight='balanced', n_jobs=-1,
                                         random_state=42, verbose=-1)),
    ]

    # Обучение
    all_results = []
    all_models = {}

    for name, model in tqdm(models_config, desc="Training models"):
        trained_model, results = train_model(
            model, name, X_train, y_train, X_val, y_val, X_test, y_test
        )
        all_results.append(results)
        all_models[name] = trained_model

    # Результаты
    results_df = pd.DataFrame(all_results).sort_values('val_f1', ascending=False)
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(results_df[['name', 'training_time', 'val_f1', 'test_f1', 'val_roc_auc']].to_string(index=False))

    # Сохранение лучшей модели
    best_name = results_df.iloc[0]['name']
    best_model = all_models[best_name]

    joblib.dump(best_model, output_path / f"best_model_{best_name}.joblib")
    results_df.to_csv(output_path / "experiment_results.csv", index=False)

    print(f"\n✅ Best model saved: {best_name}")
    print(f"   Val F1: {results_df.iloc[0]['val_f1']:.4f}")
    print(f"   Test F1: {results_df.iloc[0]['test_f1']:.4f}")


if __name__ == "__main__":
    main()