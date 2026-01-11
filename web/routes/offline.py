"""
Offline analysis routes - analyze uploaded datasets
"""

import os
import json
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from flask import Blueprint, render_template, jsonify, request, current_app
import pandas as pd
import numpy as np

offline_bp = Blueprint('offline', __name__)

# Global state for offline analysis
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
    """Offline analysis page"""
    return render_template('offline.html')


@offline_bp.route('/analyze', methods=['POST'])
def analyze_dataset():
    """Analyze uploaded dataset"""
    global _offline_state

    if _offline_state['running']:
        return jsonify({'status': 'error', 'message': 'Analysis already running'})

    # Check for file upload
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})

    # Check file extension
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.parquet')):
        return jsonify({'status': 'error', 'message': 'File must be CSV or Parquet'})

    # Save uploaded file
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename
    file.save(str(temp_path))

    # Get parameters
    max_rows = int(request.form.get('max_rows', 10000))
    threshold = float(request.form.get('threshold', 0.5))

    # Store project root before starting thread
    project_root = current_app.config['PROJECT_ROOT']

    # Start analysis in background
    _offline_state['running'] = True
    _offline_state['progress'] = 0
    _offline_state['total'] = 0
    _offline_state['results'] = None
    _offline_state['error'] = None
    _offline_state['project_root'] = project_root

    def run_analysis():
        global _offline_state
        try:
            # Use stored project_root
            proj_root = _offline_state['project_root']

            # Load data
            if filename.endswith('.csv'):
                df = pd.read_csv(temp_path, nrows=max_rows, encoding='utf-8', encoding_errors='replace')
            else:
                df = pd.read_parquet(temp_path)
                if len(df) > max_rows:
                    df = df.head(max_rows)

            _offline_state['total'] = len(df)

            # Load feature schema
            schema_path = proj_root / 'artifacts' / 'feature_schema.json'
            with open(schema_path) as f:
                schema = json.load(f)

            feature_cols = schema['feature_columns']

            # Check if columns exist - strip whitespace from column names
            df.columns = df.columns.str.strip()

            missing_cols = [c for c in feature_cols if c not in df.columns]
            if missing_cols:
                # Try case-insensitive match
                df_cols_lower = {c.lower(): c for c in df.columns}
                feature_cols_mapped = []
                still_missing = []

                for fc in feature_cols:
                    if fc in df.columns:
                        feature_cols_mapped.append(fc)
                    elif fc.lower() in df_cols_lower:
                        feature_cols_mapped.append(df_cols_lower[fc.lower()])
                    else:
                        still_missing.append(fc)

                if still_missing:
                    _offline_state['error'] = f"Missing columns: {still_missing[:5]}..."
                    _offline_state['running'] = False
                    return

                feature_cols = feature_cols_mapped

            # Extract features
            X = df[feature_cols].values.astype(np.float64)

            # Replace inf/nan
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

            # Check for label column
            label_col = None
            for col in ['Label', 'label', 'class', 'Class', ' Label']:
                if col in df.columns:
                    label_col = col
                    break

            y_true = None
            if label_col:
                # Convert labels to binary
                labels = df[label_col].astype(str).str.strip().str.upper()
                y_true = (~labels.isin(['BENIGN', '0', 'NORMAL'])).astype(int).values

            # Load predictor
            import sys
            sys.path.insert(0, str(proj_root))
            from src.inference import Predictor

            predictor = Predictor(
                model_path=str(proj_root / 'training_artifacts' / 'best_model_XGB_regularized.joblib'),
                preprocessor_path=str(proj_root / 'artifacts' / 'preprocessor.joblib'),
                feature_schema_path=str(proj_root / 'artifacts' / 'feature_schema.json'),
                threshold=threshold
            )
            predictor.load()

            # Batch prediction
            batch_size = 1000
            predictions = []
            probabilities = []

            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                preds, probs, _ = predictor.predict_batch(batch)
                predictions.extend(preds.tolist())
                probabilities.extend(probs.tolist())
                _offline_state['progress'] = min(i + batch_size, len(X))

            predictions = np.array(predictions)
            probabilities = np.array(probabilities)

            # Calculate metrics
            results = {
                'total_flows': len(predictions),
                'benign_count': int((predictions == 0).sum()),
                'attack_count': int((predictions == 1).sum()),
                'attack_rate': float((predictions == 1).mean() * 100),
                'avg_confidence': float(np.where(predictions == 1, probabilities, 1 - probabilities).mean() * 100)
            }

            # If we have ground truth
            if y_true is not None:
                tp = int(((predictions == 1) & (y_true == 1)).sum())
                fp = int(((predictions == 1) & (y_true == 0)).sum())
                tn = int(((predictions == 0) & (y_true == 0)).sum())
                fn = int(((predictions == 0) & (y_true == 1)).sum())

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0

                results['has_labels'] = True
                results['true_benign'] = int((y_true == 0).sum())
                results['true_attack'] = int((y_true == 1).sum())
                results['tp'] = tp
                results['fp'] = fp
                results['tn'] = tn
                results['fn'] = fn
                results['precision'] = round(precision * 100, 2)
                results['recall'] = round(recall * 100, 2)
                results['f1'] = round(f1 * 100, 2)
                results['accuracy'] = round(accuracy * 100, 2)
            else:
                results['has_labels'] = False

            # Probability distribution
            results['prob_distribution'] = {
                'bins': list(np.arange(0, 1.1, 0.1).round(1)),
                'counts': [int(x) for x in np.histogram(probabilities, bins=10, range=(0, 1))[0]]
            }

            _offline_state['results'] = results

        except Exception as e:
            import traceback
            _offline_state['error'] = f"{str(e)}\n{traceback.format_exc()}"
        finally:
            _offline_state['running'] = False
            # Cleanup temp file
            try:
                os.remove(temp_path)
                os.rmdir(temp_dir)
            except:
                pass

    # Run in background thread
    thread = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'success', 'message': 'Analysis started'})


@offline_bp.route('/progress')
def get_progress():
    """Get analysis progress"""
    global _offline_state

    return jsonify({
        'running': _offline_state['running'],
        'progress': _offline_state['progress'],
        'total': _offline_state['total'],
        'error': _offline_state['error'],
        'results': _offline_state['results']
    })
