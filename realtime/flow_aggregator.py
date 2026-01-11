"""
Агрегатор сетевых потоков
Собирает пакеты в потоки (flows) для анализа
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from .capture import PacketInfo


@dataclass
class FlowStats:
    """Статистика потока"""
    # Временные метки
    start_time: float = 0.0
    last_time: float = 0.0

    # Прямое направление (forward)
    fwd_packets: int = 0
    fwd_bytes: int = 0
    fwd_payload_bytes: int = 0
    fwd_packet_lengths: List[int] = field(default_factory=list)
    fwd_iat: List[float] = field(default_factory=list)  # Inter-arrival times
    fwd_last_time: float = 0.0

    # Обратное направление (backward)
    bwd_packets: int = 0
    bwd_bytes: int = 0
    bwd_payload_bytes: int = 0
    bwd_packet_lengths: List[int] = field(default_factory=list)
    bwd_iat: List[float] = field(default_factory=list)
    bwd_last_time: float = 0.0

    # Флаги TCP
    fwd_psh_flags: int = 0
    bwd_psh_flags: int = 0
    fwd_urg_flags: int = 0
    bwd_urg_flags: int = 0
    fin_count: int = 0
    syn_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    ack_count: int = 0
    urg_count: int = 0
    ece_count: int = 0
    cwr_count: int = 0

    # Init/Fin packets
    fwd_init_win_bytes: int = 0
    bwd_init_win_bytes: int = 0

    # Active/Idle times
    active_times: List[float] = field(default_factory=list)
    idle_times: List[float] = field(default_factory=list)

    # Header lengths
    fwd_header_length: int = 0
    bwd_header_length: int = 0

    # Метаданные потока
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: int = 0


class FlowAggregator:
    """
    Агрегирует пакеты в сетевые потоки
    """

    def __init__(
            self,
            flow_timeout: float = 120.0,  # Таймаут неактивного потока (секунды)
            activity_timeout: float = 5.0,  # Таймаут активности
            max_flows: int = 100000,  # Максимум потоков в памяти
            on_flow_complete: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Args:
            flow_timeout: Время неактивности после которого поток считается завершённым
            activity_timeout: Таймаут для разделения периодов активности
            max_flows: Максимальное количество потоков в памяти
            on_flow_complete: Callback при завершении потока
        """
        self.flow_timeout = flow_timeout
        self.activity_timeout = activity_timeout
        self.max_flows = max_flows
        self.on_flow_complete = on_flow_complete

        self._flows: Dict[tuple, FlowStats] = {}
        self._lock = threading.Lock()
        self._stats = defaultdict(int)

        # Поток для очистки старых flows
        self._cleanup_running = False
        self._cleanup_thread: Optional[threading.Thread] = None

    def add_packet(self, packet: PacketInfo) -> Optional[str]:
        """
        Добавляет пакет в соответствующий поток

        Returns:
            ID потока или None если ошибка
        """
        flow_key = packet.get_flow_key()
        direction = packet.get_direction()

        with self._lock:
            # Создаём новый поток если не существует
            if flow_key not in self._flows:
                if len(self._flows) >= self.max_flows:
                    self._cleanup_oldest_flow()

                self._flows[flow_key] = FlowStats(
                    start_time=packet.timestamp,
                    src_ip=flow_key[0],
                    src_port=flow_key[1],
                    dst_ip=flow_key[2],
                    dst_port=flow_key[3],
                    protocol=flow_key[4]
                )
                self._stats['flows_created'] += 1

            flow = self._flows[flow_key]
            self._update_flow(flow, packet, direction)
            flow.last_time = packet.timestamp

        self._stats['packets_processed'] += 1
        return str(flow_key)

    def _update_flow(self, flow: FlowStats, packet: PacketInfo, direction: str):
        """Обновляет статистику потока на основе пакета"""

        if direction == 'forward':
            # Inter-arrival time
            if flow.fwd_last_time > 0:
                iat = packet.timestamp - flow.fwd_last_time
                flow.fwd_iat.append(iat)
            flow.fwd_last_time = packet.timestamp

            # Счётчики
            flow.fwd_packets += 1
            flow.fwd_bytes += packet.length
            flow.fwd_payload_bytes += packet.payload_length
            flow.fwd_packet_lengths.append(packet.length)
            flow.fwd_header_length += packet.length - packet.payload_length

            # TCP флаги
            if packet.flags.get('PSH'):
                flow.fwd_psh_flags += 1
            if packet.flags.get('URG'):
                flow.fwd_urg_flags += 1

        else:  # backward
            if flow.bwd_last_time > 0:
                iat = packet.timestamp - flow.bwd_last_time
                flow.bwd_iat.append(iat)
            flow.bwd_last_time = packet.timestamp

            flow.bwd_packets += 1
            flow.bwd_bytes += packet.length
            flow.bwd_payload_bytes += packet.payload_length
            flow.bwd_packet_lengths.append(packet.length)
            flow.bwd_header_length += packet.length - packet.payload_length

            if packet.flags.get('PSH'):
                flow.bwd_psh_flags += 1
            if packet.flags.get('URG'):
                flow.bwd_urg_flags += 1

        # Общие флаги TCP
        if packet.flags.get('FIN'):
            flow.fin_count += 1
        if packet.flags.get('SYN'):
            flow.syn_count += 1
        if packet.flags.get('RST'):
            flow.rst_count += 1
        if packet.flags.get('PSH'):
            flow.psh_count += 1
        if packet.flags.get('ACK'):
            flow.ack_count += 1
        if packet.flags.get('URG'):
            flow.urg_count += 1
        if packet.flags.get('ECE'):
            flow.ece_count += 1
        if packet.flags.get('CWR'):
            flow.cwr_count += 1

    def _cleanup_oldest_flow(self):
        """Удаляет самый старый поток"""
        if not self._flows:
            return

        oldest_key = min(self._flows.keys(),
                         key=lambda k: self._flows[k].last_time)
        completed_flow = self._export_flow(oldest_key)
        del self._flows[oldest_key]

        if self.on_flow_complete and completed_flow:
            self.on_flow_complete(completed_flow)

        self._stats['flows_expired'] += 1

    def check_timeouts(self) -> List[Dict[str, Any]]:
        """
        Проверяет и завершает потоки с истёкшим таймаутом

        Returns:
            Список завершённых потоков
        """
        current_time = time.time()
        completed_flows = []
        expired_keys = []

        with self._lock:
            for key, flow in self._flows.items():
                if current_time - flow.last_time > self.flow_timeout:
                    expired_keys.append(key)

            for key in expired_keys:
                flow_data = self._export_flow(key)
                if flow_data:
                    completed_flows.append(flow_data)
                del self._flows[key]
                self._stats['flows_expired'] += 1

        # Вызываем callbacks
        for flow_data in completed_flows:
            if self.on_flow_complete:
                self.on_flow_complete(flow_data)

        return completed_flows

    def _export_flow(self, flow_key: tuple) -> Optional[Dict[str, Any]]:
        """Экспортирует поток в словарь признаков"""
        if flow_key not in self._flows:
            return None

        flow = self._flows[flow_key]
        return self._flow_to_dict(flow, flow_key)

    def _flow_to_dict(self, flow: FlowStats, flow_key: tuple) -> Dict[str, Any]:
        """Конвертирует FlowStats в словарь"""
        duration = flow.last_time - flow.start_time if flow.last_time > flow.start_time else 0
        total_packets = flow.fwd_packets + flow.bwd_packets
        total_bytes = flow.fwd_bytes + flow.bwd_bytes

        return {
            'flow_key': flow_key,
            'src_ip': flow.src_ip,
            'dst_ip': flow.dst_ip,
            'src_port': flow.src_port,
            'dst_port': flow.dst_port,
            'protocol': flow.protocol,
            'timestamp': flow.start_time,
            'duration': duration,

            # Счётчики
            'total_fwd_packets': flow.fwd_packets,
            'total_bwd_packets': flow.bwd_packets,
            'total_fwd_bytes': flow.fwd_bytes,
            'total_bwd_bytes': flow.bwd_bytes,
            'fwd_payload_bytes': flow.fwd_payload_bytes,
            'bwd_payload_bytes': flow.bwd_payload_bytes,

            # Длины пакетов
            'fwd_packet_lengths': flow.fwd_packet_lengths.copy(),
            'bwd_packet_lengths': flow.bwd_packet_lengths.copy(),

            # IAT
            'fwd_iat': flow.fwd_iat.copy(),
            'bwd_iat': flow.bwd_iat.copy(),

            # TCP флаги
            'fwd_psh_flags': flow.fwd_psh_flags,
            'bwd_psh_flags': flow.bwd_psh_flags,
            'fwd_urg_flags': flow.fwd_urg_flags,
            'bwd_urg_flags': flow.bwd_urg_flags,
            'fin_count': flow.fin_count,
            'syn_count': flow.syn_count,
            'rst_count': flow.rst_count,
            'psh_count': flow.psh_count,
            'ack_count': flow.ack_count,
            'urg_count': flow.urg_count,
            'ece_count': flow.ece_count,
            'cwr_count': flow.cwr_count,

            # Header lengths
            'fwd_header_length': flow.fwd_header_length,
            'bwd_header_length': flow.bwd_header_length,

            # Вычисляемые поля
            'total_packets': total_packets,
            'total_bytes': total_bytes,
            'packets_per_second': total_packets / duration if duration > 0 else 0,
            'bytes_per_second': total_bytes / duration if duration > 0 else 0,
        }

    def get_active_flows(self) -> List[Dict[str, Any]]:
        """Возвращает все активные потоки"""
        with self._lock:
            return [self._flow_to_dict(flow, key)
                    for key, flow in self._flows.items()]

    def get_flow_count(self) -> int:
        """Возвращает количество активных потоков"""
        return len(self._flows)

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику агрегатора"""
        return {
            **dict(self._stats),
            'active_flows': len(self._flows)
        }

    def start_cleanup_thread(self, interval: float = 10.0):
        """Запускает фоновый поток очистки"""
        self._cleanup_running = True

        def cleanup_loop():
            while self._cleanup_running:
                time.sleep(interval)
                self.check_timeouts()

        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name="FlowCleanup"
        )
        self._cleanup_thread.start()

    def stop_cleanup_thread(self):
        """Останавливает фоновый поток очистки"""
        self._cleanup_running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2.0)

    def clear(self):
        """Очищает все потоки"""
        with self._lock:
            self._flows.clear()