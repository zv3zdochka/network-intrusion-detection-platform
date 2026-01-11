"""
Анализатор трафика с использованием нейронной сети
Загружает модель и выполняет предсказания
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import deque
import threading

# Попытка импорта PyTorch
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed")

# Попытка импорта sklearn для препроцессинга
try:
    import joblib
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not installed")


class TrafficAnalyzer:
    """
    Анализирует сетевой трафик с использованием обученной нейронной сети
    """

    def __init__(
            self,
            model_path: Optional[str] = None,
            scaler_path: Optional[str] = None,
            config_path: Optional[str] = None,
            threshold: float = 0.5,
            device: str = 'auto',
            history_size: int = 1000
    ):
        """
        Args:
            model_path: Путь к файлу модели (.pt или .pth)
            scaler_path: Путь к скейлеру (pickle/joblib)
            config_path: Путь к конфигурации модели (JSON)
            threshold: Порог для классификации
            device: 'cpu', 'cuda' или 'auto'
            history_size: Размер истории предсказаний
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        self.threshold = threshold
        self.history_size = history_size

        # Определяем устройство
        if device == 'auto':
            self.device = 'cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
        else:
            self.device = device

        self.model = None
        self.scaler = None
        self.config = {}
        self.class_names = ['BENIGN', 'ATTACK']

        # История предсказаний для статистики
        self._prediction_history = deque(maxlen=history_size)
        self._lock = threading.Lock()

        # Статистика
        self._stats = {
            'total_predictions': 0,
            'benign_count': 0,
            'attack_count': 0,
            'errors': 0
        }

        # Загружаем модель если путь указан
        if model_path:
            self.load_model(model_path, scaler_path, config_path)

    def load_model(
            self,
            model_path: str,
            scaler_path: Optional[str] = None,
            config_path: Optional[str] = None
    ):
        """
        Загружает модель и связанные файлы

        Args:
            model_path: Путь к модели
            scaler_path: Путь к скейлеру
            config_path: Путь к конфигурации
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to load the model")

        # Загружаем конфигурацию
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.class_names = self.config.get('class_names', self.class_names)

        # Загружаем скейлер
        if scaler_path and os.path.exists(scaler_path):
            if SKLEARN_AVAILABLE:
                self.scaler = joblib.load(scaler_path)
            else:
                print("Warning: sklearn not available, skipping scaler")

        # Загружаем модель
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)

            # Проверяем формат checkpoint
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Нужно создать модель и загрузить веса
                    self._create_model_from_config(checkpoint)
                elif 'state_dict' in checkpoint:
                    self._create_model_from_config(checkpoint)
                else:
                    # Возможно это уже модель целиком
                    self.model = checkpoint
            else:
                self.model = checkpoint

            if self.model:
                self.model.to(self.device)
                self.model.eval()

            print(f"Model loaded from {model_path} on {self.device}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def _create_model_from_config(self, checkpoint: dict):
        """Создаёт модель на основе конфигурации в checkpoint"""
        # Простая MLP модель как fallback
        # В реальном проекте здесь нужно использовать ту же архитектуру что при обучении

        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            input_size = config.get('input_size', 78)
            hidden_sizes = config.get('hidden_sizes', [128, 64])
            num_classes = config.get('num_classes', 2)
        else:
            # Пытаемся определить по state_dict
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
            if state_dict:
                # Определяем размер входа по первому слою
                first_layer = list(state_dict.keys())[0]
                if 'weight' in first_layer:
                    input_size = state_dict[first_layer].shape[1]
                else:
                    input_size = 78
            else:
                input_size = 78
            hidden_sizes = [128, 64]
            num_classes = 2

        # Создаём простую модель
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))

        self.model = nn.Sequential(*layers)

        # Загружаем веса
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
        if state_dict:
            try:
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Could not load state dict: {e}")

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """
        Препроцессинг признаков перед подачей в модель

        Args:
            features: Массив признаков [n_samples, n_features] или [n_features]

        Returns:
            Преобразованные признаки
        """
        # Обеспечиваем 2D форму
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Заменяем inf и nan
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

        # Применяем скейлер если есть
        if self.scaler is not None:
            features = self.scaler.transform(features)

        return features.astype(np.float32)

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Выполняет предсказание для набора признаков

        Args:
            features: Массив признаков

        Returns:
            Словарь с результатами предсказания
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'prediction': None,
                'probabilities': None
            }

        try:
            # Препроцессинг
            processed = self.preprocess(features)

            # Конвертируем в тензор
            with torch.no_grad():
                tensor = torch.FloatTensor(processed).to(self.device)
                outputs = self.model(tensor)

                # Применяем softmax для получения вероятностей
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions = outputs.argmax(dim=1).cpu().numpy()

            # Обновляем статистику
            with self._lock:
                self._stats['total_predictions'] += len(predictions)
                for pred in predictions:
                    if pred == 0:
                        self._stats['benign_count'] += 1
                    else:
                        self._stats['attack_count'] += 1

            # Формируем результат
            results = []
            for i in range(len(predictions)):
                pred_class = int(predictions[i])
                probs = probabilities[i].tolist()

                result = {
                    'prediction': pred_class,
                    'class_name': self.class_names[pred_class] if pred_class < len(
                        self.class_names) else f'class_{pred_class}',
                    'probabilities': {
                        self.class_names[j] if j < len(self.class_names) else f'class_{j}': float(probs[j])
                        for j in range(len(probs))
                    },
                    'confidence': float(max(probs)),
                    'is_attack': pred_class != 0,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)

                # Добавляем в историю
                with self._lock:
                    self._prediction_history.append(result)

            if len(results) == 1:
                return results[0]
            return {'predictions': results}

        except Exception as e:
            self._stats['errors'] += 1
            return {
                'error': str(e),
                'prediction': None,
                'probabilities': None
            }

    def predict_flow(self, flow_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Предсказание для потока с признаками в виде словаря

        Args:
            flow_features: Словарь признаков потока

        Returns:
            Результат предсказания
        """
        # Получаем список признаков в правильном порядке
        from .feature_extractor import FeatureConfig

        feature_values = []
        for name in FeatureConfig.FEATURE_NAMES:
            value = flow_features.get(name, 0.0)
            feature_values.append(float(value))

        features = np.array(feature_values, dtype=np.float32)
        return self.predict(features)

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику анализатора"""
        with self._lock:
            total = self._stats['total_predictions']
            return {
                **self._stats,
                'benign_rate': self._stats['benign_count'] / total if total > 0 else 0,
                'attack_rate': self._stats['attack_count'] / total if total > 0 else 0,
                'model_loaded': self.model is not None,
                'device': self.device,
                'history_size': len(self._prediction_history)
            }

    def get_recent_predictions(self, n: int = 100) -> List[Dict[str, Any]]:
        """Возвращает последние n предсказаний"""
        with self._lock:
            return list(self._prediction_history)[-n:]

    def get_attack_summary(self, time_window_seconds: float = 60.0) -> Dict[str, Any]:
        """
        Возвращает сводку атак за последние N секунд

        Args:
            time_window_seconds: Временное окно в секундах

        Returns:
            Сводка по атакам
        """
        now = datetime.now()
        attacks = []
        benign = 0

        with self._lock:
            for pred in self._prediction_history:
                try:
                    pred_time = datetime.fromisoformat(pred['timestamp'])
                    delta = (now - pred_time).total_seconds()

                    if delta <= time_window_seconds:
                        if pred.get('is_attack', False):
                            attacks.append(pred)
                        else:
                            benign += 1
                except:
                    pass

        return {
            'time_window_seconds': time_window_seconds,
            'total_flows': len(attacks) + benign,
            'attack_count': len(attacks),
            'benign_count': benign,
            'attack_rate': len(attacks) / (len(attacks) + benign) if (len(attacks) + benign) > 0 else 0,
            'attacks': attacks[-10:]  # Последние 10 атак
        }


# Заглушка модели для тестирования без реальной модели
class DummyModel:
    """Заглушка модели для тестирования"""

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes

    def __call__(self, x):
        batch_size = x.shape[0]
        # Возвращаем случайные логиты
        return torch.randn(batch_size, self.num_classes)

    def eval(self):
        pass

    def to(self, device):
        return self


def create_dummy_analyzer() -> TrafficAnalyzer:
    """Создаёт анализатор с заглушкой для тестирования"""
    analyzer = TrafficAnalyzer()
    if TORCH_AVAILABLE:
        analyzer.model = DummyModel()
        analyzer.model.eval()
    return analyzer 