"""
Вспомогательные функции и утилиты
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import asdict
from collections import deque
import threading


def setup_logging(
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        format_string: Optional[str] = None
) -> logging.Logger:
    """
    Настраивает логирование

    Args:
        log_file: Путь к файлу логов (None = только консоль)
        level: Уровень логирования
        format_string: Формат сообщений

    Returns:
        Настроенный logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logger = logging.getLogger('realtime_analyzer')
    logger.setLevel(level)

    # Очищаем существующие handlers
    logger.handlers.clear()

    # Консольный handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # Файловый handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def result_to_dict(result) -> Dict[str, Any]:
    """Конвертирует AnalysisResult в словарь"""
    if hasattr(result, '__dataclass_fields__'):
        return asdict(result)
    elif hasattr(result, '__dict__'):
        return result.__dict__.copy()
    return dict(result)


def save_results_json(
        results: List[Any],
        filepath: str,
        append: bool = False
):
    """
    Сохраняет результаты в JSON файл

    Args:
        results: Список результатов
        filepath: Путь к файлу
        append: Дописывать в существующий файл
    """
    data = [result_to_dict(r) for r in results]

    if append and os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing = json.load(f)
        data = existing + data

    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def save_results_csv(
        results: List[Any],
        filepath: str,
        append: bool = False
):
    """
    Сохраняет результаты в CSV файл

    Args:
        results: Список результатов
        filepath: Путь к файлу
        append: Дописывать в существующий файл
    """
    import csv

    if not results:
        return

    data = [result_to_dict(r) for r in results]

    # Получаем заголовки (исключаем вложенные словари)
    headers = [k for k in data[0].keys()
               if not isinstance(data[0][k], (dict, list))]

    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

    mode = 'a' if append else 'w'
    write_header = not append or not os.path.exists(filepath)

    with open(filepath, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerows(data)


def protocol_name(proto_num: int) -> str:
    """Возвращает имя протокола по номеру"""
    protocols = {
        1: 'ICMP',
        6: 'TCP',
        17: 'UDP',
        41: 'IPv6',
        47: 'GRE',
        50: 'ESP',
        51: 'AH',
        58: 'ICMPv6',
        89: 'OSPF',
        132: 'SCTP'
    }
    return protocols.get(proto_num, f'Proto-{proto_num}')


def format_bytes(num_bytes: int) -> str:
    """Форматирует байты в читаемый формат"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Форматирует продолжительность"""
    if seconds < 0.001:
        return f"{seconds * 1000000:.1f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_timestamp(ts: float) -> str:
    """Форматирует Unix timestamp в читаемую дату"""
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


class AlertManager:
    """Менеджер оповещений об атаках"""

    def __init__(
            self,
            alert_threshold: int = 10,
            alert_window_seconds: float = 60.0,
            cooldown_seconds: float = 300.0,
            on_alert: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Args:
            alert_threshold: Порог количества атак для оповещения
            alert_window_seconds: Временное окно для подсчёта атак
            cooldown_seconds: Период "охлаждения" между оповещениями
            on_alert: Callback при срабатывании оповещения
        """
        self.alert_threshold = alert_threshold
        self.alert_window_seconds = alert_window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.on_alert = on_alert

        self._attacks: deque = deque(maxlen=10000)
        self._last_alert_time: Optional[datetime] = None
        self._alert_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_attack(self, attack_info: Dict[str, Any]) -> bool:
        """
        Добавляет информацию об атаке

        Returns:
            True если было отправлено оповещение
        """
        now = datetime.now()
        attack_info['detected_at'] = now.isoformat()

        with self._lock:
            self._attacks.append(attack_info)

            # Очищаем старые атаки
            cutoff = now.timestamp() - self.alert_window_seconds
            recent_attacks = [
                a for a in self._attacks
                if datetime.fromisoformat(a['detected_at']).timestamp() > cutoff
            ]

            # Проверяем нужно ли оповещение
            if len(recent_attacks) >= self.alert_threshold:
                # Проверяем cooldown
                if self._last_alert_time is None or \
                        (now - self._last_alert_time).total_seconds() > self.cooldown_seconds:

                    self._last_alert_time = now
                    alert_data = self._create_alert(recent_attacks)
                    self._alert_history.append(alert_data)

                    if self.on_alert:
                        self.on_alert(alert_data)

                    return True

        return False

    def _create_alert(self, attacks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Создаёт объект оповещения"""
        # Группируем по типу атаки
        attack_types: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        targets: Dict[str, int] = {}

        for attack in attacks:
            # Тип атаки
            attack_type = attack.get('class_name', 'unknown')
            attack_types[attack_type] = attack_types.get(attack_type, 0) + 1

            # Источник
            src = attack.get('src_ip', 'unknown')
            sources[src] = sources.get(src, 0) + 1

            # Цель
            dst = attack.get('dst_ip', 'unknown')
            targets[dst] = targets.get(dst, 0) + 1

        return {
            'timestamp': datetime.now().isoformat(),
            'attack_count': len(attacks),
            'window_seconds': self.alert_window_seconds,
            'attack_types': attack_types,
            'top_sources': dict(sorted(sources.items(), key=lambda x: -x[1])[:5]),
            'top_targets': dict(sorted(targets.items(), key=lambda x: -x[1])[:5]),
            'severity': self._calculate_severity(len(attacks)),
            'message': f"Detected {len(attacks)} attacks in the last {self.alert_window_seconds}s"
        }

    def _calculate_severity(self, attack_count: int) -> str:
        """Определяет уровень серьёзности"""
        if attack_count >= self.alert_threshold * 5:
            return 'critical'
        elif attack_count >= self.alert_threshold * 2:
            return 'high'
        elif attack_count >= self.alert_threshold:
            return 'medium'
        return 'low'

    def get_alert_history(self, n: int = 100) -> List[Dict[str, Any]]:
        """Возвращает историю оповещений"""
        with self._lock:
            return self._alert_history[-n:]

    def get_recent_attacks(self, n: int = 100) -> List[Dict[str, Any]]:
        """Возвращает последние атаки"""
        with self._lock:
            return list(self._attacks)[-n:]

    def clear(self):
        """Очищает историю"""
        with self._lock:
            self._attacks.clear()
            self._alert_history.clear()


class RateLimiter:
    """Ограничитель скорости для обработки"""

    def __init__(self, max_rate: float, window_seconds: float = 1.0):
        """
        Args:
            max_rate: Максимальное количество операций за окно
            window_seconds: Размер окна в секундах
        """
        self.max_rate = max_rate
        self.window_seconds = window_seconds
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """
        Пытается получить разрешение на операцию

        Returns:
            True если операция разрешена
        """
        now = datetime.now().timestamp()

        with self._lock:
            # Очищаем старые timestamps
            cutoff = now - self.window_seconds
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

            # Проверяем лимит
            if len(self._timestamps) < self.max_rate:
                self._timestamps.append(now)
                return True

            return False

    def wait(self) -> float:
        """
        Ждёт до освобождения слота

        Returns:
            Время ожидания в секундах
        """
        import time

        start = datetime.now().timestamp()
        while not self.acquire():
            time.sleep(0.01)
        return datetime.now().timestamp() - start


class MovingAverage:
    """Скользящее среднее для метрик"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._values: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def add(self, value: float):
        """Добавляет значение"""
        with self._lock:
            self._values.append(value)

    def get(self) -> float:
        """Возвращает текущее среднее"""
        with self._lock:
            if not self._values:
                return 0.0
            return sum(self._values) / len(self._values)

    def get_stats(self) -> Dict[str, float]:
        """Возвращает статистику"""
        import statistics

        with self._lock:
            if not self._values:
                return {'mean': 0, 'min': 0, 'max': 0, 'std': 0}

            values = list(self._values)
            return {
                'mean': statistics.mean(values),
                'min': min(values),
                'max': max(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0
            }


class MetricsCollector:
    """Сборщик метрик для мониторинга"""

    def __init__(self):
        self._metrics: Dict[str, MovingAverage] = {}
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self, name: str, value: float, window_size: int = 100):
        """Записывает значение метрики"""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = MovingAverage(window_size)
            self._metrics[name].add(value)

    def increment(self, name: str, value: int = 1):
        """Увеличивает счётчик"""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def get_metric(self, name: str) -> Dict[str, float]:
        """Возвращает статистику метрики"""
        with self._lock:
            if name in self._metrics:
                return self._metrics[name].get_stats()
            return {}

    def get_counter(self, name: str) -> int:
        """Возвращает значение счётчика"""
        with self._lock:
            return self._counters.get(name, 0)

    def get_all(self) -> Dict[str, Any]:
        """Возвращает все метрики"""
        with self._lock:
            return {
                'metrics': {name: m.get_stats() for name, m in self._metrics.items()},
                'counters': self._counters.copy()
            }


def validate_ip(ip: str) -> bool:
    """Проверяет валидность IP-адреса"""
    import socket
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        try:
            socket.inet_pton(socket.AF_INET6, ip)
            return True
        except socket.error:
            return False


def is_private_ip(ip: str) -> bool:
    """Проверяет, является ли IP приватным"""
    import ipaddress
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def get_geo_info(ip: str) -> Optional[Dict[str, Any]]:
    """
    Получает геолокацию по IP (требует geoip2)

    Returns:
        Словарь с информацией о геолокации или None
    """
    try:
        import geoip2.database
        # Путь к базе GeoLite2
        db_path = os.environ.get('GEOIP_DB', '/usr/share/GeoIP/GeoLite2-City.mmdb')
        if os.path.exists(db_path):
            with geoip2.database.Reader(db_path) as reader:
                response = reader.city(ip)
                return {
                    'country': response.country.name,
                    'country_code': response.country.iso_code,
                    'city': response.city.name,
                    'latitude': response.location.latitude,
                    'longitude': response.location.longitude
                }
    except Exception:
        pass
    return None