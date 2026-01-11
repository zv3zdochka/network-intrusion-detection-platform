"""Unit tests for inference module."""

import json
import pytest
import numpy as np
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import Predictor, InferencePipeline


@pytest.fixture
def predictor():
    p = Predictor(
        model_path="training_artifacts/best_model_XGB_regularized.joblib",
        preprocessor_path="artifacts/preprocessor.joblib",
        feature_schema_path="artifacts/feature_schema.json"
    )
    p.load()
    return p


@pytest.fixture
def feature_cols():
    with open("artifacts/feature_schema.json", "r") as f:
        schema = json.load(f)
    return schema["feature_columns"]


class TestPredictor:

    def test_load(self, predictor):
        assert predictor.is_loaded
        assert predictor.model is not None
        assert len(predictor.feature_cols) > 0

    def test_predict_single(self, predictor, feature_cols):
        features = np.random.randn(len(feature_cols)).astype(np.float32)
        result = predictor.predict_single(features, flow_index=0)

        assert result.prediction in [0, 1]
        assert 0 <= result.probability <= 1
        assert result.inference_time_ms > 0

    def test_predict_batch(self, predictor, feature_cols):
        batch_size = 100
        features = np.random.randn(batch_size, len(feature_cols)).astype(np.float32)

        predictions, probabilities, inference_time = predictor.predict_batch(features)

        assert len(predictions) == batch_size
        assert len(probabilities) == batch_size
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= p <= 1 for p in probabilities)


class TestInferencePipeline:

    def test_process_flow(self, predictor, feature_cols):
        pipeline = InferencePipeline(predictor)

        features = np.random.randn(len(feature_cols)).astype(np.float32)
        pipeline.process_flow(features, flow_index=0, true_label=0)

        stats = pipeline.get_stats()
        assert stats["total_flows"] == 1

    def test_process_batch(self, predictor, feature_cols):
        pipeline = InferencePipeline(predictor)

        batch_size = 50
        features = np.random.randn(batch_size, len(feature_cols)).astype(np.float32)
        labels = np.random.randint(0, 2, batch_size)

        alerts = pipeline.process_batch(features, true_labels=labels)

        stats = pipeline.get_stats()
        assert stats["total_flows"] == batch_size
        assert stats["total_alerts"] == len(alerts)

    def test_metrics_calculation(self, predictor, feature_cols):
        pipeline = InferencePipeline(predictor)

        # Process enough data for meaningful metrics
        for _ in range(10):
            features = np.random.randn(100, len(feature_cols)).astype(np.float32)
            labels = np.random.randint(0, 2, 100)
            pipeline.process_batch(features, true_labels=labels)

        stats = pipeline.get_stats()

        assert "precision" in stats
        assert "recall" in stats
        assert "f1" in stats
        assert 0 <= stats["precision"] <= 1
        assert 0 <= stats["recall"] <= 1