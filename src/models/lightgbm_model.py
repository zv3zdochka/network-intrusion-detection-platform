"""
LightGBM model.
"""

import time
from typing import Optional

import numpy as np

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from .base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM classifier."""

    def __init__(
            self,
            name: str = "LightGBM",
            task: str = "binary",
            n_estimators: int = 100,
            max_depth: int = -1,
            num_leaves: int = 31,
            learning_rate: float = 0.1,
            subsample: float = 0.8,
            colsample_bytree: float = 0.8,
            min_child_samples: int = 20,
            reg_alpha: float = 0,
            reg_lambda: float = 0,
            class_weight: str = "balanced",
            n_jobs: int = -1,
            random_state: int = 42,
            **kwargs,
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm")

        super().__init__(name=name, task=task, random_state=random_state)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.class_weight = class_weight
        self.n_jobs = n_jobs

        self.params.update(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "min_child_samples": min_child_samples,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "class_weight": class_weight,
            }
        )

        self.model = self._create_model()

    def _create_model(self) -> "lgb.LGBMClassifier":
        objective = "binary" if self.task == "binary" else "multiclass"

        return lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            class_weight=self.class_weight,
            objective=objective,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=-1,
        )

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[list] = None,
            early_stopping_rounds: int = 50,
            **kwargs,
    ) -> "LightGBMModel":
        """Fit the model."""
        self.feature_names = feature_names

        print(f"Training {self.name}...")
        print(
            "Parameters: "
            f"n_estimators={self.n_estimators}, "
            f"num_leaves={self.num_leaves}, "
            f"lr={self.learning_rate}"
        )

        start_time = time.time()

        # Disable evaluation logging
        callbacks = [lgb.log_evaluation(period=0)]

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        else:
            self.model.fit(X_train, y_train, callbacks=callbacks)

        self.training_time = time.time() - start_time
        self.is_fitted = True

        print(f"Training completed in {self.training_time:.1f}s")

        return self
