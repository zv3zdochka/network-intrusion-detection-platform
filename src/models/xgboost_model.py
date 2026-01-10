"""
XGBoost model.
"""

import time
from typing import Optional

import numpy as np

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost classifier."""

    def __init__(
            self,
            name: str = "XGBoost",
            task: str = "binary",
            n_estimators: int = 100,
            max_depth: int = 6,
            learning_rate: float = 0.1,
            subsample: float = 0.8,
            colsample_bytree: float = 0.8,
            min_child_weight: int = 1,
            reg_alpha: float = 0,
            reg_lambda: float = 1,
            scale_pos_weight: Optional[float] = None,
            use_gpu: bool = False,
            n_jobs: int = -1,
            random_state: int = 42,
            **kwargs,
    ):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost")

        super().__init__(name=name, task=task, random_state=random_state)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs

        self.params.update(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "min_child_weight": min_child_weight,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "scale_pos_weight": scale_pos_weight,
            }
        )

        self.model = self._create_model()

    def _create_model(self) -> "xgb.XGBClassifier":
        objective = "binary:logistic" if self.task == "binary" else "multi:softprob"

        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "objective": objective,
            "eval_metric": "logloss" if self.task == "binary" else "mlogloss",
            "use_label_encoder": False,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": 0,
        }

        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight

        if self.use_gpu:
            params["tree_method"] = "gpu_hist"
            params["predictor"] = "gpu_predictor"

        return xgb.XGBClassifier(**params)

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[list] = None,
            early_stopping_rounds: int = 50,
            **kwargs,
    ) -> "XGBoostModel":
        """Fit the model."""
        self.feature_names = feature_names

        # Auto scale_pos_weight if not provided
        if self.scale_pos_weight is None and self.task == "binary":
            n_neg = np.sum(y_train == 0)
            n_pos = np.sum(y_train == 1)
            self.model.set_params(scale_pos_weight=n_neg / n_pos)

        print(f"Training {self.name}...")
        print(
            "Parameters: "
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"lr={self.learning_rate}"
        )

        start_time = time.time()

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self.history["eval_metric"] = self.model.evals_result()
        else:
            self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        self.is_fitted = True

        print(f"Training completed in {self.training_time:.1f}s")

        return self
