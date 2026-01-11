"""
Анализатор трафика в реальном времени
Использует Predictor и InferencePipeline из src/inference
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque
import threading
import numpy as np

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Импортируем ваш Predictor
try:
    from src.inference import Predictor, InferencePipeline
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Predictor: {e}")
    PREDICTOR_AVAILABLE = False


class TrafficAnalyzer:
    """
    Анализирует сетевой трафик с использованием Predictor из src/inference
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        feature_schema_path: Optional[str] = None,
        threshold: float = 0.5,
        history_size: int = 1000
    ):
        """
        Args:
            model_path: Путь к модели (.pkl или .joblib)
            preprocessor_path: Путь к препроцессору (.pkl)
            feature_schema_path: Путь к схеме признаков (.json)
            threshold: Порог для классификации атаки
            history_size: Размер истории предсказаний
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.feature_schema_path = feature_schema_path
        self.threshold = threshold

        # Predictor и Pipeline
        self.predictor: Optional[Predictor] = None
        self.pipeline: Optional[InferencePipeline] = None

        # Метаданные
        self.feature_cols: List[str] = []
        self.n_features: int = 78

        # История и статистика
        self._prediction_history = deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._flow_counter = 0

        self._stats = {
            'total_predictions': 0,
            'benign_count': 0,
            'attack_count': 0,
            'errors': 0
        }

        # Загружаем модель
        if model_path:
            self.load_model(model_path, preprocessor_path, feature_schema_path)

    def load_model(
        self,
        model_path: str,
        preprocessor_path: Optional[str] = None,
        feature_schema_path: Optional[str] = None
    ):
        """Загружает модель используя Predictor из src/inference"""

        if not PREDICTOR_AVAILABLE:
            raise RuntimeError(
                "Predictor not available. Make sure src/inference is accessible."
            )

        print(f"[Analyzer] Loading model...")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model_dir = model_path.parent

        # Автоопределение путей если не указаны
        if preprocessor_path is None:
            possible = [
                model_dir / 'preprocessor.pkl',
                model_dir / 'preprocessor.joblib',
                model_dir / 'scaler.pkl',
            ]
            for p in possible:
                if p.exists():
                    preprocessor_path = str(p)
                    break

        if feature_schema_path is None:
            possible = [
                model_dir / 'feature_schema.json',
                model_dir / 'features.json',
                model_dir / 'schema.json',
            ]
            for p in possible:
                if p.exists():
                    feature_schema_path = str(p)
                    break

        print(f"  Model: {model_path}")
        print(f"  Preprocessor: {preprocessor_path or 'Not found'}")
        print(f"  Feature Schema: {feature_schema_path or 'Not found'}")

        if not preprocessor_path or not feature_schema_path:
            raise FileNotFoundError(
                "preprocessor_path and feature_schema_path are required.\n"
                f"Looking in: {model_dir}"
            )

        try:
            # Создаём Predictor
            self.predictor = Predictor(
                model_path=str(model_path),
                preprocessor_path=preprocessor_path,
                feature_schema_path=feature_schema_path,
                threshold=self.threshold
            )
            self.predictor.load()

            # Создаём Pipeline
            self.pipeline = InferencePipeline(
                predictor=self.predictor,
                alert_threshold=self.threshold
            )

            # Сохраняем метаданные
            self.feature_cols = self.predictor.feature_cols or []
            self.n_features = len(self.feature_cols) if self.feature_cols else 78

            print(f"  Features: {self.n_features}")
            print(f"  Model type: {type(self.predictor.model).__name__}")
            print(f"[Analyzer] Model loaded successfully!")

        except Exception as e:
            import traceback
            print(f"[Analyzer] Error loading model: {e}")
            traceback.print_exc()
            raise

    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Препроцессинг признаков через ваш preprocessor"""

        # Обеспечиваем 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Очистка данных
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        features = np.clip(features, -1e15, 1e15).astype(np.float64)

        # Применяем препроцессор если загружен
        if self.predictor and self.predictor.preprocessor:
            try:
                features = self.predictor.preprocessor.transform(features)
            except Exception as e:
                print(f"Warning: Preprocessor error: {e}")

        return features

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Выполняет предсказание

        Args:
            features: numpy array признаков [n_features] или [batch, n_features]

        Returns:
            Словарь с результатом предсказания
        """
        if self.predictor is None or not self.predictor.is_loaded:
            return {
                'error': 'Model not loaded',
                'prediction': None,
                'is_attack': False
            }

        try:
            # Препроцессинг
            if features.ndim == 1:
                features = features.reshape(1, -1)

            features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
            features = np.clip(features, -1e15, 1e15).astype(np.float64)

            batch_size = features.shape[0]

            # Индексы потоков
            with self._lock:
                flow_indices = list(range(self._flow_counter, self._flow_counter + batch_size))
                self._flow_counter += batch_size

            # Предсказание через pipeline
            alerts = self.pipeline.process_batch(
                features=features,
                flow_indices=flow_indices,
                true_labels=None,
                store_alerts=True
            )

            # Получаем вероятности напрямую для всех потоков
            predictions, probabilities, _ = self.predictor.predict_batch(
                features=features,
                flow_indices=flow_indices
            )

            # Формируем результаты
            results = []
            alert_indices = {a.flow_index for a in alerts}

            for i in range(batch_size):
                flow_idx = flow_indices[i]
                is_attack = flow_idx in alert_indices
                pred = int(predictions[i])
                prob = float(probabilities[i])

                result = {
                    'prediction': pred,
                    'class_name': 'ATTACK' if is_attack else 'BENIGN',
                    'confidence': prob if is_attack else (1 - prob),
                    'probability': prob,
                    'is_attack': is_attack,
                    'flow_index': flow_idx,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)

                # Статистика
                with self._lock:
                    self._stats['total_predictions'] += 1
                    if is_attack:
                        self._stats['attack_count'] += 1
                    else:
                        self._stats['benign_count'] += 1
                    self._prediction_history.append(result)

            return results[0] if len(results) == 1 else {'predictions': results}

        except Exception as e:
            self._stats['errors'] += 1
            import traceback
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'prediction': None,
                'is_attack': False
            }

    def get_alerts(self, limit: int = 100) -> List[Dict]:
        """Получает последние алерты"""
        if self.pipeline is None:
            return []

        alerts = self.pipeline.get_alerts(limit=limit)
        return [a.to_dict() for a in alerts]

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику"""
        with self._lock:
            total = max(self._stats['total_predictions'], 1)

            stats = {
                **self._stats.copy(),
                'benign_rate': self._stats['benign_count'] / total,
                'attack_rate': self._stats['attack_count'] / total,
                'model_loaded': self.predictor is not None and self.predictor.is_loaded,
                'n_features': self.n_features,
                'history_size': len(self._prediction_history)
            }

            # Добавляем статистику из pipeline если есть
            if self.pipeline:
                pipeline_stats = self.pipeline.get_stats()
                stats['pipeline'] = pipeline_stats

            return stats

    def get_recent_predictions(self, n: int = 100) -> List[Dict[str, Any]]:
        """Возвращает последние предсказания"""
        with self._lock:
            return list(self._prediction_history)[-n:]

    def reset(self):
        """Сбрасывает состояние"""
        with self._lock:
            self._flow_counter = 0
            self._prediction_history.clear()
            self._stats = {
                'total_predictions': 0,
                'benign_count': 0,
                'attack_count': 0,
                'errors': 0
            }

        if self.pipeline:
            self.pipeline.reset()

    def get_model_info(self) -> Dict[str, Any]:
        """Информация о модели"""
        if self.predictor:
            return self.predictor.get_model_info()
        return {'loaded': False}


# === Заглушка для тестирования ===

class DummyPredictor:
    """Заглушка для тестирования без модели"""

    def __init__(self, attack_ratio: float = 0.1):
        self.attack_ratio = attack_ratio
        self.is_loaded = True
        self.feature_cols = [f'feature_{i}' for i in range(78)]
        self.preprocessor = None
        self.threshold = 0.5
        self.model = None

    def load(self):
        return self

    def predict_batch(self, features, flow_indices=None, true_labels=None):
        n = features.shape[0]
        probabilities = np.random.random(n)
        # Делаем атаки редкими
        probabilities = probabilities * 0.3  # Максимум 30%
        # Иногда делаем атаку
        attack_mask = np.random.random(n) < self.attack_ratio
        probabilities[attack_mask] = np.random.uniform(0.6, 0.95, attack_mask.sum())

        predictions = (probabilities >= self.threshold).astype(int)
        inference_time = 0.1
        return predictions, probabilities, inference_time

    def predict_single(self, features, flow_index=0, true_label=None, return_features=False):
        from dataclasses import dataclass

        @dataclass
        class Result:
            flow_index: int
            prediction: int
            probability: float
            is_attack: bool
            inference_time_ms: float
            true_label: Optional[int] = None

        prob = np.random.random() * 0.3
        if np.random.random() < self.attack_ratio:
            prob = np.random.uniform(0.6, 0.95)

        pred = 1 if prob >= self.threshold else 0

        return Result(
            flow_index=flow_index,
            prediction=pred,
            probability=prob,
            is_attack=pred == 1,
            inference_time_ms=0.1,
            true_label=true_label
        )

    def get_model_info(self):
        return {'loaded': True, 'type': 'DummyPredictor'}


class DummyPipeline:
    """Заглушка InferencePipeline"""

    def __init__(self, predictor):
        self.predictor = predictor
        self._alerts = []
        self._alert_counter = 0
        self.stats = type('Stats', (), {
            'total_flows': 0,
            'total_alerts': 0,
            'to_dict': lambda s: {'total_flows': s.total_flows, 'total_alerts': s.total_alerts}
        })()

    def process_batch(self, features, flow_indices, true_labels=None, store_alerts=True):
        predictions, probabilities, _ = self.predictor.predict_batch(features, flow_indices)

        alerts = []
        for i, (pred, prob, flow_idx) in enumerate(zip(predictions, probabilities, flow_indices)):
            self.stats.total_flows += 1

            if pred == 1:
                self._alert_counter += 1
                self.stats.total_alerts += 1

                alert = type('Alert', (), {
                    'id': self._alert_counter,
                    'flow_index': flow_idx,
                    'prediction': pred,
                    'probability': float(prob),
                    'is_attack': True,
                    'timestamp': datetime.now(),
                    'true_label': None,
                    'is_correct': None,
                    'inference_time_ms': 0.1,
                    'to_dict': lambda s: {
                        'id': s.id,
                        'flow_index': s.flow_index,
                        'prediction': s.prediction,
                        'probability': s.probability
                    }
                })()

                alerts.append(alert)
                if store_alerts:
                    self._alerts.append(alert)

        return alerts

    def get_alerts(self, limit=100):
        return self._alerts[-limit:]

    def get_stats(self):
        return self.stats.to_dict()

    def reset(self):
        self._alerts = []
        self._alert_counter = 0
        self.stats.total_flows = 0
        self.stats.total_alerts = 0


def create_dummy_analyzer(attack_ratio: float = 0.1) -> TrafficAnalyzer:
    """Создаёт тестовый анализатор со случайными предсказаниями"""
    analyzer = TrafficAnalyzer()
    analyzer.predictor = DummyPredictor(attack_ratio)
    analyzer.pipeline = DummyPipeline(analyzer.predictor)
    analyzer.feature_cols = analyzer.predictor.feature_cols
    analyzer.n_features = 78
    return analyzer


# === Тест ===

if __name__ == "__main__":
    print("Testing TrafficAnalyzer...")
    print("=" * 50)

    analyzer = create_dummy_analyzer(attack_ratio=0.15)

    # Тест
    test_features = np.random.randn(10, 78).astype(np.float32)

    print("\nProcessing 10 test flows...")
    result = analyzer.predict(test_features)

    if 'predictions' in result:
        for pred in result['predictions']:
            status = "🚨 ATTACK" if pred['is_attack'] else "✅ BENIGN"
            print(f"  Flow {pred['flow_index']}: {status} ({pred['confidence']:.1%})")

    print(f"\nStats: {analyzer.get_stats()}")