"""Unit tests for replay module."""

import json
import pytest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import FlowReplay


@pytest.fixture
def feature_cols():
    with open("artifacts/feature_schema.json", "r") as f:
        schema = json.load(f)
    return schema["feature_columns"]


@pytest.fixture
def replay(feature_cols):
    r = FlowReplay(
        data_path="data/processed/splits/test.parquet",
        feature_cols=feature_cols,
        batch_size=100
    )
    r.load()
    return r


class TestFlowReplay:

    def test_load(self, replay):
        assert replay._is_loaded
        assert replay.n_samples > 0

    def test_get_batch(self, replay):
        batch = replay.get_batch()

        assert batch is not None
        assert batch.size == 100
        assert batch.features.shape[0] == 100
        assert len(batch.labels) == 100

    def test_iter_batches(self, replay):
        batches = list(replay.iter_batches(max_batches=5))

        assert len(batches) == 5
        for batch in batches:
            assert batch.size > 0

    def test_progress(self, replay):
        assert replay.progress == 0.0

        replay.get_batch()

        assert replay.progress > 0.0
        assert replay.progress < 1.0

    def test_reset(self, replay):
        replay.get_batch()
        replay.get_batch()

        assert replay.current_index > 0

        replay.reset()

        assert replay.current_index == 0

    def test_get_attack_samples(self, replay):
        attacks = replay.get_attack_samples(n_samples=50)

        assert attacks.size > 0
        assert all(label == 1 for label in attacks.labels)