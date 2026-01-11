"""
Replay flows from dataset with timing control.
"""

import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FlowBatch:
    """Batch of flows for processing."""
    features: np.ndarray
    labels: np.ndarray
    indices: List[int]
    batch_id: int
    timestamp: float

    @property
    def size(self) -> int:
        return len(self.indices)


class FlowReplay:
    """
    Replay flows from parquet dataset.
    """

    def __init__(
            self,
            data_path: Union[str, Path],
            feature_cols: List[str],
            label_col: str = "label_binary",
            batch_size: int = 100,
            shuffle: bool = False,
            random_state: int = 42
    ):
        self.data_path = Path(data_path)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state

        self.data: Optional[pd.DataFrame] = None
        self.n_samples = 0
        self.current_index = 0
        self._is_loaded = False

    def load(self) -> "FlowReplay":
        """Load data from parquet file."""
        self.data = pd.read_parquet(self.data_path)
        self.n_samples = len(self.data)

        if self.shuffle:
            self.data = self.data.sample(
                frac=1,
                random_state=self.random_state
            ).reset_index(drop=True)

        self._is_loaded = True
        self.current_index = 0

        return self

    def reset(self):
        """Reset replay to beginning."""
        self.current_index = 0

        if self.shuffle and self.data is not None:
            self.data = self.data.sample(
                frac=1,
                random_state=self.random_state
            ).reset_index(drop=True)

    def get_batch(self) -> Optional[FlowBatch]:
        """Get next batch of flows."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")

        if self.current_index >= self.n_samples:
            return None

        end_index = min(self.current_index + self.batch_size, self.n_samples)
        batch_data = self.data.iloc[self.current_index:end_index]

        features = batch_data[self.feature_cols].values.astype(np.float32)
        labels = batch_data[self.label_col].values.astype(np.int64)
        indices = list(range(self.current_index, end_index))

        batch = FlowBatch(
            features=features,
            labels=labels,
            indices=indices,
            batch_id=self.current_index // self.batch_size,
            timestamp=time.time()
        )

        self.current_index = end_index

        return batch

    def iter_batches(
            self,
            max_batches: Optional[int] = None,
            speed: float = 1.0,
            delay_per_batch: float = 0.0
    ) -> Generator[FlowBatch, None, None]:
        """
        Iterate over batches with optional timing control.

        Args:
            max_batches: Maximum number of batches to yield
            speed: Replay speed multiplier (2.0 = 2x faster)
            delay_per_batch: Base delay between batches in seconds
        """
        batch_count = 0

        while True:
            batch = self.get_batch()

            if batch is None:
                break

            yield batch

            batch_count += 1

            if max_batches and batch_count >= max_batches:
                break

            if delay_per_batch > 0 and speed > 0:
                time.sleep(delay_per_batch / speed)

    def get_single_flow(self, index: int) -> Tuple[np.ndarray, int]:
        """Get single flow by index."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")

        row = self.data.iloc[index]
        features = row[self.feature_cols].values.astype(np.float32)
        label = int(row[self.label_col])

        return features, label

    def get_attack_samples(self, n_samples: int = 100) -> FlowBatch:
        """Get batch of attack samples for injection."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")

        attacks = self.data[self.data[self.label_col] == 1]

        if len(attacks) < n_samples:
            sample = attacks
        else:
            sample = attacks.sample(n=n_samples, random_state=self.random_state)

        features = sample[self.feature_cols].values.astype(np.float32)
        labels = sample[self.label_col].values.astype(np.int64)
        indices = sample.index.tolist()

        return FlowBatch(
            features=features,
            labels=labels,
            indices=indices,
            batch_id=-1,
            timestamp=time.time()
        )

    @property
    def progress(self) -> float:
        """Get current progress as fraction."""
        if self.n_samples == 0:
            return 0.0
        return self.current_index / self.n_samples

    @property
    def remaining(self) -> int:
        """Get number of remaining samples."""
        return max(0, self.n_samples - self.current_index)

    def get_info(self) -> dict:
        """Get replay info."""
        return {
            "data_path": str(self.data_path),
            "n_samples": self.n_samples,
            "n_features": len(self.feature_cols),
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "is_loaded": self._is_loaded,
            "current_index": self.current_index,
            "progress": self.progress
        }