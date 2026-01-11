"""
Service layer for offline dataset analysis
"""

import threading
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class OfflineAnalysisTask:
    """
    Background task for analyzing datasets
    """

    def __init__(self, file_path: Path, max_rows: int, threshold: float,
                 project_root: Path):
        self.file_path = file_path
        self.max_rows = max_rows
        self.threshold = threshold
        self.project_root = project_root

        self.running = False
        self.progress = 0
        self.total = 0
        self.results = None
        self.error = None

        self._thread = None

    def start(self):
        """Start analysis in background"""
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Run the analysis"""
        try:
            # Load data
            filename = str(self.file_path).lower()
            if filename.endswith('.csv'):
                df = pd.read_csv(self.file_path, nrows=self.max_rows)
            else:
                df = pd.read_parquet(self.file_path)
                if len(df) > self.max_rows:
                    df = df.head(self.max_rows)

            self.total = len(df)

            # Load feature schema
            import json
            with open(self.project_root / 'artifacts' / 'feature_schema.json') as f:
                schema = json.load(f)

            feature_cols = schema['feature_columns']

            # Check columns
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                self.error = f"Missing columns: {missing[:5]}..."
                self.running = False
                return

            # Extract features
            X = df[feature_cols].values

            # Check for labels
            label_col = None
            for col in ['Label', 'label', 'class', 'Class']:
                if col in df.columns:
                    label_col = col
                    break

            y_true = None
            if label_col:
                labels = df[label_col].astype(str).str.upper()
                y_true = (~labels.isin(['BENIGN', '0', 'NORMAL'])).astype(int).values

            # Load predictor
            from src.inference import Predictor

            predictor = Predictor(
                model_path=str(self.project_root / 'training_artifacts' / 'best_model_XGB_regularized.joblib'),
                preprocessor_path=str(self.project_root / 'artifacts' / 'preprocessor.joblib'),
                feature_schema_path=str(self.project_root / 'artifacts' / 'feature_schema.json'),
                threshold=self.threshold
            )
            predictor.load()

            # Predict in batches
            batch_size = 1000
            predictions = []
            probabilities = []

            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                preds, probs, _ = predictor.predict_batch(batch)
                predictions.extend(preds.tolist())
                probabilities.extend(probs.tolist())
                self.progress = min(i + batch_size, len(X))

            predictions = np.array(predictions)
            probabilities = np.array(probabilities)

            # Calculate results
            self.results = {
                'total_flows': len(predictions),
                'benign_count': int((predictions == 0).sum()),
                'attack_count': int((predictions == 1).sum()),
                'attack_rate': float((predictions == 1).mean() * 100),
                'avg_confidence': float(np.where(predictions == 1, probabilities, 1 - probabilities).mean() * 100)
            }

            # Add metrics if labels available
            if y_true is not None:
                tp = int(((predictions == 1) & (y_true == 1)).sum())
                fp = int(((predictions == 1) & (y_true == 0)).sum())
                tn = int(((predictions == 0) & (y_true == 0)).sum())
                fn = int(((predictions == 0) & (y_true == 1)).sum())

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / len(predictions)

                self.results.update({
                    'has_labels': True,
                    'true_benign': int((y_true == 0).sum()),
                    'true_attack': int((y_true == 1).sum()),
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'fn': fn,
                    'precision': round(precision * 100, 2),
                    'recall': round(recall * 100, 2),
                    'f1': round(f1 * 100, 2),
                    'accuracy': round(accuracy * 100, 2)
                })
            else:
                self.results['has_labels'] = False

            # Probability distribution
            hist, bins = np.histogram(probabilities, bins=10, range=(0, 1))
            self.results['prob_distribution'] = {
                'bins': [round(b, 1) for b in bins.tolist()],
                'counts': [int(x) for x in hist.tolist()]
            }

        except Exception as e:
            import traceback
            self.error = f"{str(e)}\n{traceback.format_exc()}"

        finally:
            self.running = False

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'running': self.running,
            'progress': self.progress,
            'total': self.total,
            'error': self.error,
            'results': self.results
        }


class OfflineService:
    """
    Manages offline analysis tasks
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.current_task: Optional[OfflineAnalysisTask] = None
        self._initialized = True

    def start_analysis(self, file_path: Path, max_rows: int, threshold: float,
                       project_root: Path) -> Dict[str, Any]:
        """Start a new analysis"""

        if self.current_task and self.current_task.running:
            return {'status': 'error', 'message': 'Analysis already running'}

        self.current_task = OfflineAnalysisTask(
            file_path=file_path,
            max_rows=max_rows,
            threshold=threshold,
            project_root=project_root
        )

        self.current_task.start()
        return {'status': 'success', 'message': 'Analysis started'}

    def get_progress(self) -> Dict[str, Any]:
        """Get current analysis progress"""

        if self.current_task is None:
            return {
                'running': False,
                'progress': 0,
                'total': 0,
                'error': None,
                'results': None
            }

        return self.current_task.get_status()