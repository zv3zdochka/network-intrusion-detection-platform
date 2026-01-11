"""
Model predictor for inference.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib


@dataclass
class PredictionResult:
    """Single prediction result."""
    flow_index: int
    prediction: int
    probability: float
    is_attack: bool
    inference_time_ms: float
    features: Optional[Dict[str, float]] = None
    true_label: Optional[int] = None


class Predictor:
    """
    Load trained model and preprocessor, run predictions.
    """

    def __init__(
            self,
            model_path: Union[str, Path],
            preprocessor_path: Union[str, Path],
            feature_schema_path: Union[str, Path],
            threshold: float = 0.5
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.feature_schema_path = Path(feature_schema_path)
        self.threshold = threshold

        self.model = None
        self.preprocessor = None
        self.feature_cols = None
        self.is_loaded = False

    def load(self) -> "Predictor":
        """Load model, preprocessor, and feature schema."""
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        with open(self.feature_schema_path, "r") as f:
            schema = json.load(f)
        self.feature_cols = schema["feature_columns"]

        self.is_loaded = True
        return self

    def predict_single(
            self,
            features: np.ndarray,
            flow_index: int = 0,
            true_label: Optional[int] = None,
            return_features: bool = False
    ) -> PredictionResult:
        """
        Predict single flow.

        Args:
            features: Feature array (1, n_features) or (n_features,)
            flow_index: Flow identifier
            true_label: Ground truth label (optional)
            return_features: Include features in result
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        start_time = time.perf_counter()

        proba = self.model.predict_proba(features)[0]
        attack_prob = proba[1] if len(proba) > 1 else proba[0]
        prediction = 1 if attack_prob >= self.threshold else 0

        inference_time = (time.perf_counter() - start_time) * 1000

        result = PredictionResult(
            flow_index=flow_index,
            prediction=prediction,
            probability=float(attack_prob),
            is_attack=prediction == 1,
            inference_time_ms=inference_time,
            true_label=true_label
        )

        if return_features and self.feature_cols:
            result.features = {
                col: float(features[0, i])
                for i, col in enumerate(self.feature_cols)
            }

        return result

    def predict_batch(
            self,
            features: np.ndarray,
            flow_indices: Optional[List[int]] = None,
            true_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predict batch of flows.

        Args:
            features: Feature array (n_samples, n_features)
            flow_indices: Flow identifiers
            true_labels: Ground truth labels

        Returns:
            predictions, probabilities, inference_time_ms
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        proba = self.model.predict_proba(features)
        attack_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        predictions = (attack_proba >= self.threshold).astype(int)

        inference_time = (time.perf_counter() - start_time) * 1000

        return predictions, attack_proba, inference_time

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        if not self.is_loaded:
            return {"loaded": False}

        info = {
            "loaded": True,
            "model_path": str(self.model_path),
            "model_type": type(self.model).__name__,
            "n_features": len(self.feature_cols),
            "threshold": self.threshold
        }

        if hasattr(self.model, "n_estimators"):
            info["n_estimators"] = self.model.n_estimators
        if hasattr(self.model, "max_depth"):
            info["max_depth"] = self.model.max_depth

        return info
