"""
Random Forest model.
"""

import time
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier."""

    def __init__(
            self,
            name: str = "RandomForest",
            task: str = "binary",
            n_estimators: int = 100,
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            max_features: str = "sqrt",
            class_weight: str = "balanced",
            n_jobs: int = -1,
            random_state: int = 42,
            **kwargs,
    ):
        super().__init__(name=name, task=task, random_state=random_state)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs

        self.params.update(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "class_weight": class_weight,
            }
        )

        self.model = self._create_model()

    def _create_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0,
        )

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[list] = None,
            **kwargs,
    ) -> "RandomForestModel":
        """Fit the model."""
        self.feature_names = feature_names

        print(f"Training {self.name}...")
        print(f"Parameters: n_estimators={self.n_estimators}, max_depth={self.max_depth}")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        self.is_fitted = True

        print(f"Training completed in {self.training_time:.1f}s")

        return self
