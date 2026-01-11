"""
Full inference pipeline combining preprocessing and prediction.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .predictor import Predictor, PredictionResult


@dataclass
class Alert:
    """Alert generated from prediction."""
    id: int
    timestamp: datetime
    flow_index: int
    prediction: int
    probability: float
    true_label: Optional[int]
    is_correct: Optional[bool]
    inference_time_ms: float
    features: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "flow_index": self.flow_index,
            "prediction": self.prediction,
            "probability": self.probability,
            "true_label": self.true_label,
            "is_correct": self.is_correct,
            "inference_time_ms": self.inference_time_ms
        }


@dataclass
class InferenceStats:
    """Statistics for inference run."""
    total_flows: int = 0
    total_alerts: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    inference_times: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        correct = self.true_positives + self.true_negatives
        return correct / total if total > 0 else 0.0

    @property
    def latency_p50(self) -> float:
        if not self.inference_times:
            return 0.0
        return float(np.percentile(self.inference_times, 50))

    @property
    def latency_p95(self) -> float:
        if not self.inference_times:
            return 0.0
        return float(np.percentile(self.inference_times, 95))

    @property
    def latency_p99(self) -> float:
        if not self.inference_times:
            return 0.0
        return float(np.percentile(self.inference_times, 99))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_flows": self.total_flows,
            "total_alerts": self.total_alerts,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "latency_p50_ms": self.latency_p50,
            "latency_p95_ms": self.latency_p95,
            "latency_p99_ms": self.latency_p99
        }


class InferencePipeline:
    """
    Complete inference pipeline.
    """

    def __init__(
            self,
            predictor: Predictor,
            alert_threshold: float = 0.5
    ):
        self.predictor = predictor
        self.alert_threshold = alert_threshold
        self.stats = InferenceStats()
        self.alerts: List[Alert] = []
        self._alert_counter = 0

    def reset(self):
        """Reset pipeline state."""
        self.stats = InferenceStats()
        self.alerts = []
        self._alert_counter = 0

    def process_flow(
            self,
            features: np.ndarray,
            flow_index: int,
            true_label: Optional[int] = None,
            store_alert: bool = True
    ) -> Optional[Alert]:
        """
        Process single flow through pipeline.

        Returns Alert if prediction is attack, None otherwise.
        """
        result = self.predictor.predict_single(
            features=features,
            flow_index=flow_index,
            true_label=true_label
        )

        self.stats.total_flows += 1
        self.stats.inference_times.append(result.inference_time_ms)

        # Update confusion matrix
        if true_label is not None:
            if result.prediction == 1 and true_label == 1:
                self.stats.true_positives += 1
            elif result.prediction == 1 and true_label == 0:
                self.stats.false_positives += 1
            elif result.prediction == 0 and true_label == 0:
                self.stats.true_negatives += 1
            else:
                self.stats.false_negatives += 1

        # Generate alert if attack detected
        if result.is_attack:
            self._alert_counter += 1
            self.stats.total_alerts += 1

            is_correct = None
            if true_label is not None:
                is_correct = result.prediction == true_label

            alert = Alert(
                id=self._alert_counter,
                timestamp=datetime.now(),
                flow_index=flow_index,
                prediction=result.prediction,
                probability=result.probability,
                true_label=true_label,
                is_correct=is_correct,
                inference_time_ms=result.inference_time_ms
            )

            if store_alert:
                self.alerts.append(alert)

            return alert

        return None

    def process_batch(
            self,
            features: np.ndarray,
            flow_indices: Optional[List[int]] = None,
            true_labels: Optional[np.ndarray] = None,
            store_alerts: bool = True
    ) -> List[Alert]:
        """
        Process batch of flows through pipeline.

        Returns list of alerts for detected attacks.
        """
        n_samples = features.shape[0]

        if flow_indices is None:
            flow_indices = list(range(n_samples))

        predictions, probabilities, batch_time = self.predictor.predict_batch(
            features=features,
            true_labels=true_labels
        )

        per_sample_time = batch_time / n_samples

        self.stats.total_flows += n_samples
        self.stats.inference_times.extend([per_sample_time] * n_samples)

        alerts = []

        for i in range(n_samples):
            pred = predictions[i]
            prob = probabilities[i]
            true_label = true_labels[i] if true_labels is not None else None
            flow_idx = flow_indices[i]

            # Update confusion matrix
            if true_label is not None:
                if pred == 1 and true_label == 1:
                    self.stats.true_positives += 1
                elif pred == 1 and true_label == 0:
                    self.stats.false_positives += 1
                elif pred == 0 and true_label == 0:
                    self.stats.true_negatives += 1
                else:
                    self.stats.false_negatives += 1

            # Generate alert
            if pred == 1:
                self._alert_counter += 1
                self.stats.total_alerts += 1

                is_correct = None
                if true_label is not None:
                    is_correct = pred == true_label

                alert = Alert(
                    id=self._alert_counter,
                    timestamp=datetime.now(),
                    flow_index=flow_idx,
                    prediction=pred,
                    probability=float(prob),
                    true_label=true_label,
                    is_correct=is_correct,
                    inference_time_ms=per_sample_time
                )

                alerts.append(alert)

                if store_alerts:
                    self.alerts.append(alert)

        return alerts

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return self.stats.to_dict()

    def get_alerts(
            self,
            limit: Optional[int] = None,
            offset: int = 0,
            only_incorrect: bool = False
    ) -> List[Alert]:
        """Get alerts with pagination."""
        alerts = self.alerts

        if only_incorrect:
            alerts = [a for a in alerts if a.is_correct is False]

        if offset:
            alerts = alerts[offset:]

        if limit:
            alerts = alerts[:limit]

        return alerts
