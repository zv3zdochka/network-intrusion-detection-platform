"""
Offline analysis routes - analyze uploaded datasets
"""

import os
import json
import tempfile
import threading
from pathlib import Path
from flask import Blueprint, render_template, jsonify, request, current_app
import pandas as pd
import numpy as np

offline_bp = Blueprint('offline', __name__)

# Global state
_offline_state = {
    'running': False,
    'progress': 0,
    'total': 0,
    'results': None,
    'error': None,
    'project_root': None
}

@offline_bp.route('/')
def offline_page():
    return render_template('offline.html')

def _load_dataframe(temp_path: Path, filename_lower: str, max_rows: int = None) -> pd.DataFrame:
    """Load CSV/Parquet. Modified to read FULL file if max_rows is None."""
    if filename_lower.endswith('.csv'):
        # Если max_rows передан (для обратной совместимости), используем его, иначе читаем всё
        df = pd.read_csv(
            temp_path,
            nrows=max_rows,
            encoding='utf-8',
            encoding_errors='replace'
        )
    else:
        df = pd.read_parquet(temp_path)
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)

    df.columns = df.columns.astype(str).str.strip()
    return df

# ... (функции _map_feature_columns, _extract_ground_truth, _find_transformer оставляем без изменений) ...
# Я их не дублирую здесь, чтобы не занимать место, они не меняются.
def _map_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    # ... (код без изменений)
    missing = [c for c in feature_cols if c not in df.columns]
    if not missing:
        return feature_cols

    df_cols_lower = {c.lower(): c for c in df.columns}
    mapped = []
    still_missing = []

    for fc in feature_cols:
        if fc in df.columns:
            mapped.append(fc)
        else:
            key = fc.lower()
            if key in df_cols_lower:
                mapped.append(df_cols_lower[key])
            else:
                still_missing.append(fc)

    if still_missing:
        raise KeyError(f"Missing columns: {still_missing[:10]}...")

    return mapped

def _extract_ground_truth(df: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    # ... (код без изменений)
    for col in ("label_binary", "Label_Binary", "LABEL_BINARY"):
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).values
            y = (y > 0).astype(int)
            return y, col

    label_col = None
    for col in ("Label", "label", "Class", "class", " Label"):
        c = col.strip()
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        return None, None

    s = df[label_col]
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.9:
        y = s_num.fillna(0).astype(float).values
        y = (y > 0).astype(int)
        return y, label_col

    labels = s.astype(str).str.strip().str.upper()
    benign_tokens = {
        "BENIGN", "NORMAL", "NORMAL.", "NORMAL TRAFFIC",
        "0", "0.0"
    }
    y = (~labels.isin(benign_tokens)).astype(int).values
    return y, label_col

def _find_transformer(preprocessor_obj):
    # ... (код без изменений)
    info = {
        "preprocessor_type": type(preprocessor_obj).__name__,
        "transformer_found": False,
        "transformer_type": None,
        "dict_keys_preview": None,
    }

    if preprocessor_obj is None:
        return None, info

    if hasattr(preprocessor_obj, "transform"):
        info["transformer_found"] = True
        info["transformer_type"] = type(preprocessor_obj).__name__
        return preprocessor_obj, info

    if isinstance(preprocessor_obj, dict):
        keys = list(preprocessor_obj.keys())
        info["dict_keys_preview"] = keys[:30]

        common_keys = ["pipeline", "preprocessor", "transformer", "column_transformer",
                       "ct", "sklearn_pipeline", "processor", "model_pipeline"]
        for k in common_keys:
            obj = preprocessor_obj.get(k)
            if obj is not None and hasattr(obj, "transform"):
                info["transformer_found"] = True
                info["transformer_type"] = type(obj).__name__
                return obj, info

        for v in preprocessor_obj.values():
            if hasattr(v, "transform"):
                info["transformer_found"] = True
                info["transformer_type"] = type(v).__name__
                return v, info

        imputer = preprocessor_obj.get("imputer")
        scaler = preprocessor_obj.get("scaler")
        if imputer is not None and scaler is not None:
             class _Seq:
                def __init__(self, a, b): self.a = a; self.b = b
                def transform(self, X): return self.b.transform(self.a.transform(X))

             info["transformer_found"] = True
             info["transformer_type"] = "Sequential"
             return _Seq(imputer, scaler), info

    return None, info

@offline_bp.route('/analyze', methods=['POST'])
def analyze_dataset():
    """Analyze uploaded dataset"""
    global _offline_state

    if _offline_state['running']:
        return jsonify({'status': 'error', 'message': 'Analysis already running'})

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})

    filename_lower = file.filename.lower()
    if not (filename_lower.endswith('.csv') or filename_lower.endswith('.parquet')):
        return jsonify({'status': 'error', 'message': 'File must be CSV or Parquet'})

    # Save uploaded file
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename
    file.save(str(temp_path))

    # УДАЛЕНО получение max_rows и threshold из формы
    # Теперь ставим значения по умолчанию
    max_rows = None # Читаем всё
    threshold = 0.5 # Стандартный порог для модели

    project_root = current_app.config['PROJECT_ROOT']

    # Reset state
    _offline_state['running'] = True
    _offline_state['progress'] = 0
    _offline_state['total'] = 0
    _offline_state['results'] = None
    _offline_state['error'] = None
    _offline_state['project_root'] = project_root

    def run_analysis():
        global _offline_state
        try:
            proj_root = _offline_state['project_root']

            # Читаем весь файл (max_rows=None)
            df = _load_dataframe(temp_path, filename_lower, max_rows=None)
            rows_loaded = int(len(df))
            _offline_state['total'] = rows_loaded

            # Load feature schema
            schema_path = proj_root / 'artifacts' / 'feature_schema.json'
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            feature_cols = schema['feature_columns']

            # Map feature cols
            feature_cols = _map_feature_columns(df, feature_cols)

            # Ground truth
            y_true, y_source = _extract_ground_truth(df)

            # Features as DF
            X_df = df[feature_cols].copy()

            # Numeric conversion
            for c in X_df.columns:
                X_df[c] = pd.to_numeric(X_df[c], errors='coerce')

            X_df = X_df.replace([np.inf, -np.inf], np.nan)

            # Load predictor
            import sys
            if str(proj_root) not in sys.path:
                sys.path.insert(0, str(proj_root))
            from src.inference import Predictor

            predictor = Predictor(
                model_path=str(proj_root / 'training_artifacts' / 'best_model_XGB_regularized.joblib'),
                preprocessor_path=str(proj_root / 'artifacts' / 'preprocessor.joblib'),
                feature_schema_path=str(proj_root / 'artifacts' / 'feature_schema.json'),
                threshold=threshold
            )
            predictor.load()

            # Preprocessing
            transformer, prep_info = _find_transformer(getattr(predictor, "preprocessor", None))
            preprocessor_used = False

            if transformer is not None:
                try:
                    X_proc = transformer.transform(X_df)
                    preprocessor_used = True
                except Exception:
                    X_proc = X_df.values.astype(np.float32)
            else:
                X_proc = X_df.values.astype(np.float32)

            X_proc = np.nan_to_num(X_proc, nan=0.0, posinf=0.0, neginf=0.0)

            # Batch prediction
            batch_size = 1000
            predictions = []
            probabilities = []

            n_samples = int(X_proc.shape[0])
            for i in range(0, n_samples, batch_size):
                batch = X_proc[i:i + batch_size]
                preds, probs, _ = predictor.predict_batch(batch)
                predictions.extend(preds.tolist())
                probabilities.extend(probs.tolist())
                _offline_state['progress'] = min(i + int(batch.shape[0]), n_samples)

            predictions = np.array(predictions, dtype=int)
            probabilities = np.array(probabilities, dtype=float)

            # Summary Results
            results = {
                'total_flows': int(len(predictions)),
                'benign_count': int((predictions == 0).sum()),
                'attack_count': int((predictions == 1).sum()),
                'attack_rate': float((predictions == 1).mean() * 100.0),
                'avg_confidence': float(
                    np.where(predictions == 1, probabilities, 1.0 - probabilities).mean() * 100.0
                ),
                'meta': {
                    'uploaded_filename': temp_path.name,
                    'rows_loaded': rows_loaded,
                    'threshold': float(threshold),
                    'label_source_col': y_source,
                    'preprocessor_used': bool(preprocessor_used)
                }
            }

            # Metrics if labels exist
            if y_true is not None and len(y_true) == len(predictions):
                tp = int(((predictions == 1) & (y_true == 1)).sum())
                fp = int(((predictions == 1) & (y_true == 0)).sum())
                tn = int(((predictions == 0) & (y_true == 0)).sum())
                fn = int(((predictions == 0) & (y_true == 1)).sum())

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0.0

                results.update({
                    'has_labels': True,
                    'true_benign': int((y_true == 0).sum()),
                    'true_attack': int((y_true == 1).sum()),
                    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                    'precision': round(precision * 100.0, 2),
                    'recall': round(recall * 100.0, 2),
                    'f1': round(f1 * 100.0, 2),
                    'accuracy': round(accuracy * 100.0, 2),
                })
            else:
                results['has_labels'] = False

            # Probability distribution
            hist, bins = np.histogram(probabilities, bins=10, range=(0, 1))
            results['prob_distribution'] = {
                'bins': [round(b, 1) for b in bins.tolist()],
                'counts': [int(x) for x in hist.tolist()]
            }

            _offline_state['results'] = results

        except Exception as e:
            import traceback
            _offline_state['error'] = f"{str(e)}\n{traceback.format_exc()}"
        finally:
            _offline_state['running'] = False
            try:
                os.remove(temp_path)
                os.rmdir(temp_dir)
            except Exception:
                pass

    thread = threading.Thread(target=run_analysis, daemon=True)
    thread.start()

    return jsonify({'status': 'success', 'message': 'Analysis started'})

@offline_bp.route('/progress')
def get_progress():
    global _offline_state
    return jsonify({
        'running': _offline_state['running'],
        'progress': _offline_state['progress'],
        'total': _offline_state['total'],
        'error': _offline_state['error'],
        'results': _offline_state['results']
    })