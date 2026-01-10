"""
Base model class shared across all model implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Base class for all models."""

    def __init__(
            self,
            name: str,
            task: str = "binary",  # "binary" or "multiclass"
            random_state: int = 42,
            **kwargs,
    ):
        self.name = name
        self.task = task
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.training_time = 0
        self.feature_names = None
        self.params = kwargs
        self.history = {}

    @abstractmethod
    def _create_model(self) -> Any:
        """Create and return an underlying model instance."""
        raise NotImplementedError

    @abstractmethod
    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs,
    ) -> "BaseModel":
        """Fit the model."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importances (if available) as a DataFrame."""
        if not self.is_fitted or self.feature_names is None:
            return None

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            coef = self.model.coef_
            importance = np.abs(coef).mean(axis=0) if len(coef.shape) > 1 else np.abs(coef)
        else:
            return None

        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
        )

    def save(self, path: Union[str, Path]) -> Path:
        """Save the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "name": self.name,
            "task": self.task,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "training_time": self.training_time,
            "feature_names": self.feature_names,
            "params": self.params,
            "history": self.history,
        }

        joblib.dump(model_data, path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseModel":
        """Load a saved model from disk."""
        path = Path(path)
        model_data = joblib.load(path)

        instance = cls.__new__(cls)
        instance.model = model_data["model"]
        instance.name = model_data["name"]
        instance.task = model_data["task"]
        instance.random_state = model_data["random_state"]
        instance.is_fitted = model_data["is_fitted"]
        instance.training_time = model_data["training_time"]
        instance.feature_names = model_data["feature_names"]
        instance.params = model_data["params"]
        instance.history = model_data.get("history", {})

        return instance

    def get_params_dict(self) -> Dict[str, Any]:
        """Return model metadata and hyperparameters as a dictionary."""
        return {
            "name": self.name,
            "task": self.task,
            "random_state": self.random_state,
            **self.params,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', task='{self.task}')"
