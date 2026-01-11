"""
Конфигурация модуля реального времени
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class CaptureConfig:
    """Конфигурация захвата пакетов"""
    interface: Optional[str] = None
    bpf_filter: str = "ip"
    max_queue_size: int = 10000
    promiscuous: bool = True


@dataclass
class FlowConfig:
    """Конфигурация агрегации потоков"""
    timeout_seconds: float = 120.0
    activity_timeout: float = 5.0
    max_flows: int = 100000
    cleanup_interval: float = 30.0


@dataclass
class AnalyzerConfig:
    """Конфигурация анализатора"""
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None
    config_path: Optional[str] = None
    threshold: float = 0.5
    device: str = "auto"
    batch_size: int = 32


@dataclass
class AlertConfig:
    """Конфигурация оповещений"""
    enabled: bool = True
    threshold: int = 10
    window_seconds: float = 60.0
    cooldown_seconds: float = 300.0
    email_enabled: bool = False
    email_recipients: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None


@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    level: str = "INFO"
    file: Optional[str] = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    rotation: str = "daily"
    retention_days: int = 30


@dataclass
class PipelineConfig:
    """Полная конфигурация pipeline"""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Общие настройки
    analysis_interval: float = 5.0
    results_history_size: int = 1000
    save_results: bool = False
    results_path: str = "./results"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Создаёт конфигурацию из словаря"""
        config = cls()

        if 'capture' in data:
            config.capture = CaptureConfig(**data['capture'])
        if 'flow' in data:
            config.flow = FlowConfig(**data['flow'])
        if 'analyzer' in data:
            config.analyzer = AnalyzerConfig(**data['analyzer'])
        if 'alert' in data:
            config.alert = AlertConfig(**data['alert'])
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])

        # Общие настройки
        for key in ['analysis_interval', 'results_history_size',
                    'save_results', 'results_path']:
            if key in data:
                setattr(config, key, data[key])

        return config

    @classmethod
    def from_json(cls, filepath: str) -> 'PipelineConfig':
        """Загружает конфигурацию из JSON файла"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_env(cls) -> 'PipelineConfig':
        """Создаёт конфигурацию из переменных окружения"""
        config = cls()

        # Capture
        if os.environ.get('CAPTURE_INTERFACE'):
            config.capture.interface = os.environ['CAPTURE_INTERFACE']
        if os.environ.get('CAPTURE_FILTER'):
            config.capture.bpf_filter = os.environ['CAPTURE_FILTER']

        # Analyzer
        if os.environ.get('MODEL_PATH'):
            config.analyzer.model_path = os.environ['MODEL_PATH']
        if os.environ.get('SCALER_PATH'):
            config.analyzer.scaler_path = os.environ['SCALER_PATH']
        if os.environ.get('ANALYZER_DEVICE'):
            config.analyzer.device = os.environ['ANALYZER_DEVICE']

        # Alert
        if os.environ.get('ALERT_WEBHOOK'):
            config.alert.webhook_url = os.environ['ALERT_WEBHOOK']

        # Logging
        if os.environ.get('LOG_FILE'):
            config.logging.file = os.environ['LOG_FILE']
        if os.environ.get('LOG_LEVEL'):
            config.logging.level = os.environ['LOG_LEVEL']

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        return asdict(self)

    def save_json(self, filepath: str):
        """Сохраняет конфигурацию в JSON файл"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> List[str]:
        """
        Валидирует конфигурацию

        Returns:
            Список ошибок (пустой если всё ок)
        """
        errors = []

        # Проверяем пути к модели
        if self.analyzer.model_path:
            if not os.path.exists(self.analyzer.model_path):
                errors.append(f"Model file not found: {self.analyzer.model_path}")

        if self.analyzer.scaler_path:
            if not os.path.exists(self.analyzer.scaler_path):
                errors.append(f"Scaler file not found: {self.analyzer.scaler_path}")

        # Проверяем интерфейс
        if self.capture.interface:
            try:
                from .capture import PacketCapture
                interfaces = PacketCapture.list_interfaces()
                if self.capture.interface not in interfaces:
                    errors.append(f"Interface not found: {self.capture.interface}")
            except Exception:
                pass

        # Проверяем значения
        if self.flow.timeout_seconds <= 0:
            errors.append("Flow timeout must be positive")

        if self.analyzer.threshold < 0 or self.analyzer.threshold > 1:
            errors.append("Analyzer threshold must be between 0 and 1")

        return errors


# Конфигурация по умолчанию
DEFAULT_CONFIG = PipelineConfig()


def create_example_config(filepath: str = "config.example.json"):
    """Создаёт пример конфигурационного файла"""
    config = PipelineConfig(
        capture=CaptureConfig(
            interface="eth0",
            bpf_filter="ip",
            max_queue_size=10000
        ),
        analyzer=AnalyzerConfig(
            model_path="./models/intrusion_detector.pt",
            scaler_path="./models/scaler.pkl",
            device="auto"
        ),
        alert=AlertConfig(
            enabled=True,
            threshold=10,
            window_seconds=60.0
        )
    )
    config.save_json(filepath)
    print(f"Example config saved to {filepath}")