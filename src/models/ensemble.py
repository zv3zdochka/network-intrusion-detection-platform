"""
Ensemble model - combines multiple models
"""

import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from .base import BaseModel


class EnsembleModel(BaseModel):
    """Ансамбль моделей (soft voting)"""

    def __init__(
            self,
            name: str = "Ensemble",
            task: str = "binary",
            models: Optional[List[BaseModel]] = None,
            weights: Optional[List[float]] = None,
            voting: str = "soft",  # "soft" или "hard"
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(name=name, task=task, random_state=random_state)

        self.models = models or []
        self.weights = weights
        self.voting = voting

        self.params.update({
            'voting': voting,
            'n_models': len(self.models),
            'model_names': [m.name for m in self.models]
        })

    def _create_model(self):
        return None  # Ансамбль не имеет единой модели

    def add_model(self, model: BaseModel, weight: float = 1.0):
        """Добавить модель в ансамбль"""
        if not model.is_fitted:
            raise ValueError(f"Model {model.name} is not fitted!")

        self.models.append(model)

        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)

        self.params['n_models'] = len(self.models)
        self.params['model_names'] = [m.name for m in self.models]

    def fit(
            self,
            X_train: np.ndarray = None,
            y_train: np.ndarray = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[list] = None,
            **kwargs
    ) -> "EnsembleModel":
        """
        Fit ансамбля - для уже обученных моделей просто проверяем,
        для необученных - обучаем
        """
        self.feature_names = feature_names

        print(f"🎯 Building Ensemble with {len(self.models)} models...")

        start_time = time.time()

        for i, model in enumerate(self.models):
            if not model.is_fitted:
                print(f"   Training model {i + 1}/{len(self.models)}: {model.name}")
                model.fit(X_train, y_train, X_val, y_val, feature_names)

        # Нормализуем веса
        if self.weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        else:
            self.weights = [1.0 / len(self.models)] * len(self.models)

        self.training_time = time.time() - start_time
        self.is_fitted = True

        print(f"   ✅ Ensemble ready with {len(self.models)} models")
        print(f"   Weights: {[f'{w:.2f}' for w in self.weights]}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Предсказать вероятности (взвешенное среднее)"""
        if not self.is_fitted:
            raise ValueError("Ensemble is not fitted!")

        probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            probas.append(proba)

        # Взвешенное среднее
        weighted_proba = np.zeros_like(probas[0])
        for proba, weight in zip(probas, self.weights):
            weighted_proba += proba * weight

        return weighted_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказать классы"""
        if self.voting == "soft":
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        else:
            # Hard voting
            predictions = np.array([m.predict(X) for m in self.models])
            # Majority vote
            from scipy import stats
            return stats.mode(predictions, axis=0)[0].flatten()

    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Получить предсказания каждой модели отдельно"""
        return {model.name: model.predict_proba(X) for model in self.models}

    def save(self, path: Union[str, Path]) -> Path:
        """Сохранить ансамбль"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем каждую модель отдельно
        model_paths = []
        for i, model in enumerate(self.models):
            model_path = path.parent / f"{path.stem}_model_{i}{path.suffix}"
            model.save(model_path)
            model_paths.append(str(model_path))

        # Сохраняем метаданные ансамбля
        ensemble_data = {
            'name': self.name,
            'task': self.task,
            'weights': self.weights,
            'voting': self.voting,
            'model_paths': model_paths,
            'model_classes': [m.__class__.__name__ for m in self.models],
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'feature_names': self.feature_names,
            'params': self.params
        }

        joblib.dump(ensemble_data, path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EnsembleModel":
        """Загрузить ансамбль"""
        from . import RandomForestModel, XGBoostModel, LightGBMModel, NeuralNetModel

        model_class_map = {
            'RandomForestModel': RandomForestModel,
            'XGBoostModel': XGBoostModel,
            'LightGBMModel': LightGBMModel,
            'NeuralNetModel': NeuralNetModel
        }

        path = Path(path)
        ensemble_data = joblib.load(path)

        # Загружаем модели
        models = []
        for model_path, model_class_name in zip(
                ensemble_data['model_paths'],
                ensemble_data['model_classes']
        ):
            model_class = model_class_map.get(model_class_name, BaseModel)
            model = model_class.load(model_path)
            models.append(model)

        instance = cls(
            name=ensemble_data['name'],
            task=ensemble_data['task'],
            models=models,
            weights=ensemble_data['weights'],
            voting=ensemble_data['voting']
        )
        instance.is_fitted = ensemble_data['is_fitted']
        instance.training_time = ensemble_data['training_time']
        instance.feature_names = ensemble_data['feature_names']

        return instance