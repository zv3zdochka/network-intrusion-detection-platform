#!/usr/bin/env python3
"""
End-to-end test for the simulation pipeline.

Tests:
1. Model loading
2. Data loading
3. Inference pipeline
4. Metrics collection
5. Database storage
6. No data loss verification
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.inference import Predictor, InferencePipeline
from src.simulation import FlowReplay, SimulationRunner, SimulationConfig, MetricsCollector
from src.database import Repository, Session
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration = 0.0
        self.details = {}


def run_test(name: str, test_func) -> TestResult:
    """Run single test and capture result."""
    result = TestResult(name)
    start = time.time()

    try:
        details = test_func()
        result.passed = True
        result.details = details or {}
    except AssertionError as e:
        result.passed = False
        result.error = str(e)
    except Exception as e:
        result.passed = False
        result.error = f"{type(e).__name__}: {str(e)}"

    result.duration = time.time() - start
    return result


def test_model_loading() -> dict:
    """Test model and preprocessor loading."""
    predictor = Predictor(
        model_path="training_artifacts/best_model_XGB_regularized.joblib",
        preprocessor_path="artifacts/preprocessor.joblib",
        feature_schema_path="artifacts/feature_schema.json"
    )
    predictor.load()

    info = predictor.get_model_info()

    assert info["loaded"], "Model not loaded"
    assert info["n_features"] > 0, "No features loaded"

    return info


def test_data_loading() -> dict:
    """Test data loading from parquet."""
    with open("artifacts/feature_schema.json", "r") as f:
        schema = json.load(f)

    replay = FlowReplay(
        data_path="data/processed/splits/test.parquet",
        feature_cols=schema["feature_columns"],
        batch_size=100
    )
    replay.load()

    assert replay.n_samples > 0, "No samples loaded"
    assert len(replay.feature_cols) > 0, "No features"

    return replay.get_info()


def test_single_inference() -> dict:
    """Test single flow inference."""
    with open("artifacts/feature_schema.json", "r") as f:
        schema = json.load(f)

    predictor = Predictor(
        model_path="training_artifacts/best_model_XGB_regularized.joblib",
        preprocessor_path="artifacts/preprocessor.joblib",
        feature_schema_path="artifacts/feature_schema.json"
    )
    predictor.load()

    replay = FlowReplay(
        data_path="data/processed/splits/test.parquet",
        feature_cols=schema["feature_columns"],
        batch_size=1
    )
    replay.load()

    features, label = replay.get_single_flow(0)
    result = predictor.predict_single(features, flow_index=0, true_label=label)

    assert result.prediction in [0, 1], "Invalid prediction"
    assert 0 <= result.probability <= 1, "Invalid probability"
    assert result.inference_time_ms > 0, "Invalid inference time"

    return {
        "prediction": result.prediction,
        "probability": result.probability,
        "true_label": label,
        "inference_time_ms": result.inference_time_ms
    }


def test_batch_inference() -> dict:
    """Test batch inference."""
    with open("artifacts/feature_schema.json", "r") as f:
        schema = json.load(f)

    predictor = Predictor(
        model_path="training_artifacts/best_model_XGB_regularized.joblib",
        preprocessor_path="artifacts/preprocessor.joblib",
        feature_schema_path="artifacts/feature_schema.json"
    )
    predictor.load()

    replay = FlowReplay(
        data_path="data/processed/splits/test.parquet",
        feature_cols=schema["feature_columns"],
        batch_size=1000
    )
    replay.load()

    batch = replay.get_batch()
    predictions, probabilities, inference_time = predictor.predict_batch(
        batch.features,
        true_labels=batch.labels
    )

    assert len(predictions) == batch.size, "Prediction count mismatch"
    assert len(probabilities) == batch.size, "Probability count mismatch"
    assert inference_time > 0, "Invalid inference time"

    return {
        "batch_size": batch.size,
        "predictions_sum": int(predictions.sum()),
        "inference_time_ms": inference_time,
        "per_sample_ms": inference_time / batch.size
    }


def test_inference_pipeline() -> dict:
    """Test full inference pipeline."""
    with open("artifacts/feature_schema.json", "r") as f:
        schema = json.load(f)

    predictor = Predictor(
        model_path="training_artifacts/best_model_XGB_regularized.joblib",
        preprocessor_path="artifacts/preprocessor.joblib",
        feature_schema_path="artifacts/feature_schema.json"
    )
    predictor.load()

    pipeline = InferencePipeline(predictor)

    replay = FlowReplay(
        data_path="data/processed/splits/test.parquet",
        feature_cols=schema["feature_columns"],
        batch_size=500
    )
    replay.load()

    # Process 5 batches
    for _ in range(5):
        batch = replay.get_batch()
        if batch is None:
            break
        pipeline.process_batch(
            batch.features,
            batch.indices,
            batch.labels
        )

    stats = pipeline.get_stats()

    assert stats["total_flows"] > 0, "No flows processed"
    assert stats["f1"] > 0.9, f"F1 too low: {stats['f1']}"

    return stats


def test_metrics_collector() -> dict:
    """Test metrics collection."""
    collector = MetricsCollector()
    collector.start()

    # Simulate updates
    for i in range(10):
        collector.update(
            flows=100,
            alerts=5,
            tp=4, fp=1, tn=90, fn=5,
            latencies=[0.5 + i * 0.1] * 100
        )
        time.sleep(0.1)

    snapshot = collector.take_snapshot()
    report = collector.get_final_report()

    assert report["summary"]["total_flows"] == 1000, "Flow count mismatch"
    assert report["summary"]["total_alerts"] == 50, "Alert count mismatch"
    assert report["classification"]["precision"] > 0, "Invalid precision"

    return report


def test_database_operations() -> dict:
    """Test database CRUD operations."""
    import tempfile
    import os
    import time

    # Use temporary database
    db_path = tempfile.mktemp(suffix=".db")

    repo = None
    try:
        repo = Repository(f"sqlite:///{db_path}")

        # Create simulation run
        run = repo.create_simulation_run(
            data_source="test.parquet",
            speed=1.0,
            batch_size=100
        )
        assert run.id > 0, "Failed to create run"

        # Create alerts
        alerts_data = [
            {"flow_index": i, "prediction": 1, "probability": 0.9, "true_label": 1, "is_correct": True}
            for i in range(10)
        ]
        count = repo.create_alerts_batch(run.id, alerts_data)
        assert count == 10, "Failed to create alerts"

        # Get alerts
        alerts = repo.get_alerts(run.id, limit=100)
        assert len(alerts) == 10, "Alert count mismatch"

        # Update run
        repo.update_simulation_run(
            run.id,
            status="completed",
            total_flows=1000,
            precision=0.95
        )

        updated_run = repo.get_simulation_run(run.id)
        assert updated_run.status == "completed", "Status not updated"

        result = {
            "run_id": run.id,
            "alerts_created": count,
            "alerts_retrieved": len(alerts)
        }

    finally:
        # Close repository and database connection
        if repo:
            repo.close()

        # Wait a bit for Windows to release file lock
        time.sleep(0.1)

        # Try to remove temp file
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except PermissionError:
            # On Windows, sometimes the file is still locked
            pass

    return result


def test_no_data_loss() -> dict:
    """Verify no data loss during simulation."""
    with open("artifacts/feature_schema.json", "r") as f:
        schema = json.load(f)

    predictor = Predictor(
        model_path="training_artifacts/best_model_XGB_regularized.joblib",
        preprocessor_path="artifacts/preprocessor.joblib",
        feature_schema_path="artifacts/feature_schema.json"
    )
    predictor.load()

    replay = FlowReplay(
        data_path="data/processed/splits/test.parquet",
        feature_cols=schema["feature_columns"],
        batch_size=500
    )
    replay.load()

    config = SimulationConfig(
        speed=100,
        batch_size=500,
        max_flows=5000,
        verbose=False
    )

    runner = SimulationRunner(predictor, replay, config)
    report = runner.run()

    expected_flows = min(5000, replay.n_samples)
    actual_flows = report["summary"]["total_flows"]

    assert actual_flows == expected_flows, f"Data loss: expected {expected_flows}, got {actual_flows}"

    # Verify confusion matrix adds up
    cm = report["classification"]
    cm_total = cm["true_positives"] + cm["false_positives"] + cm["true_negatives"] + cm["false_negatives"]

    assert cm_total == actual_flows, f"Confusion matrix mismatch: {cm_total} vs {actual_flows}"

    return {
        "expected_flows": expected_flows,
        "actual_flows": actual_flows,
        "confusion_matrix_total": cm_total
    }


def main():
    """Run all tests."""
    tests = [
        ("Model Loading", test_model_loading),
        ("Data Loading", test_data_loading),
        ("Single Inference", test_single_inference),
        ("Batch Inference", test_batch_inference),
        ("Inference Pipeline", test_inference_pipeline),
        ("Metrics Collector", test_metrics_collector),
        ("Database Operations", test_database_operations),
        ("No Data Loss", test_no_data_loss),
    ]

    print("=" * 70)
    print("E2E TEST SUITE")
    print("=" * 70)
    print()

    results = []

    for name, test_func in tests:
        print(f"Running: {name}...", end=" ", flush=True)
        result = run_test(name, test_func)
        results.append(result)

        if result.passed:
            print(f"PASSED ({result.duration:.2f}s)")
        else:
            print(f"FAILED ({result.duration:.2f}s)")
            print(f"  Error: {result.error}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
