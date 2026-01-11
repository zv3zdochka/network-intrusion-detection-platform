"""
Модуль захвата сетевых пакетов
Использует scapy для перехвата трафика
Поддержка Windows через Npcap
"""

import threading
import queue
import time
import platform
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict

# Определяем ОС
IS_WINDOWS = platform.system() == 'Windows'

# Настройка для Windows
if IS_WINDOWS:
    # Важно для Windows - указываем использовать Npcap
    import os
    # Добавляем путь к Npcap
    npcap_path = r'C:\Windows\System32\Npcap'
    if os.path.exists(npcap_path):
        os.environ['PATH'] = npcap_path + ';' + os.environ.get('PATH', '')

try:
    # Для Windows важен порядок импорта
    if IS_WINDOWS:
        from scapy.arch.windows import get_windows_if_list
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw, conf, get_if_list
    from scapy.layers.inet import IP as IPLayer
    SCAPY_AVAILABLE = True

    if IS_WINDOWS:
        # Настройки для Windows
        conf.use_pcap = True
        conf.use_npcap = True

except ImportError as e:
    SCAPY_AVAILABLE = False
    print(f"Warning: scapy not installed or Npcap missing. Error: {e}")
    print("Install scapy: pip install scapy")
    print("Install Npcap from: https://npcap.com/#download")


@dataclass
class PacketInfo:
    """Структура для хранения информации о пакете"""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # 6=TCP, 17=UDP, 1=ICMP
    length: int
    payload_length: int
    flags: Dict[str, bool] = field(default_factory=dict)
    raw_packet: Any = None

    def get_flow_key(self) -> tuple:
        """Возвращает ключ потока (bidirectional)"""
        if (self.src_ip, self.src_port) < (self.dst_ip, self.dst_port):
            return (self.src_ip, self.src_port, self.dst_ip, self.dst_port, self.protocol)
        else:
            return (self.dst_ip, self.dst_port, self.src_ip, self.src_port, self.protocol)

    def get_direction(self) -> str:
        """Определяет направление пакета в потоке"""
        if (self.src_ip, self.src_port) < (self.dst_ip, self.dst_port):
            return 'forward'
        return 'backward'


class PacketCapture:
    """
    Класс для захвата сетевых пакетов в реальном времени
    Поддерживает Windows (Npcap) и Linux
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        packet_queue: Optional[queue.Queue] = None,
        max_queue_size: int = 10000,
        bpf_filter: Optional[str] = None
    ):
        """
        Args:
            interface: Сетевой интерфейс для захвата (None = первый активный)
            packet_queue: Очередь для пакетов (создаётся автоматически если None)
            max_queue_size: Максимальный размер очереди
            bpf_filter: BPF фильтр (например, "tcp port 80")
        """
        if not SCAPY_AVAILABLE:
            raise RuntimeError(
                "scapy is required for packet capture.\n"
                "Install: pip install scapy\n"
                "On Windows, also install Npcap: https://npcap.com/#download"
            )

        self.interface = interface
        self.packet_queue = packet_queue or queue.Queue(maxsize=max_queue_size)
        self.bpf_filter = bpf_filter

        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._stats = defaultdict(int)
        self._start_time: Optional[float] = None

        # Автоматически выбираем интерфейс если не указан
        if self.interface is None:
            self.interface = self._get_default_interface()

    def _get_default_interface(self) -> Optional[str]:
        """Получает интерфейс по умолчанию"""
        try:
            if IS_WINDOWS:
                interfaces = get_windows_if_list()
                # Ищем активный интерфейс с IP
                for iface in interfaces:
                    if iface.get('ips') and any(ip for ip in iface['ips'] if not ip.startswith('169.254')):
                        return iface['name']
                # Если не нашли, берём первый
                if interfaces:
                    return interfaces[0]['name']
            else:
                interfaces = get_if_list()
                # Исключаем loopback
                for iface in interfaces:
                    if iface != 'lo' and not iface.startswith('lo'):
                        return iface
        except Exception as e:
            print(f"Warning: Could not auto-detect interface: {e}")
        return None

    @staticmethod
    def list_interfaces() -> List[Dict[str, Any]]:
        """Возвращает список доступных сетевых интерфейсов"""
        if not SCAPY_AVAILABLE:
            return []

        try:
            if IS_WINDOWS:
                interfaces = get_windows_if_list()
                return [
                    {
                        'name': iface.get('name', ''),
                        'description': iface.get('description', ''),
                        'ips': iface.get('ips', []),
                        'mac': iface.get('mac', '')
                    }
                    for iface in interfaces
                ]
            else:
                return [{'name': iface} for iface in get_if_list()]
        except Exception as e:
            print(f"Error listing interfaces: {e}")
            return []

    @staticmethod
    def list_interface_names() -> List[str]:
        """Возвращает только имена интерфейсов"""
        interfaces = PacketCapture.list_interfaces()
        return [iface['name'] for iface in interfaces if iface.get('name')]

    def _parse_packet(self, packet) -> Optional[PacketInfo]:
        """Парсит scapy пакет в PacketInfo"""
        try:
            if not packet.haslayer(IP):
                return None

            ip_layer = packet[IP]

            # Базовая информация
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            protocol = ip_layer.proto
            length = len(packet)

            src_port = 0
            dst_port = 0
            flags = {}
            payload_length = 0

            # TCP
            if packet.haslayer(TCP):
                tcp = packet[TCP]
                src_port = tcp.sport
                dst_port = tcp.dport
                flags = {
                    'FIN': bool(tcp.flags & 0x01),
                    'SYN': bool(tcp.flags & 0x02),
                    'RST': bool(tcp.flags & 0x04),
                    'PSH': bool(tcp.flags & 0x08),
                    'ACK': bool(tcp.flags & 0x10),
                    'URG': bool(tcp.flags & 0x20),
                    'ECE': bool(tcp.flags & 0x40),
                    'CWR': bool(tcp.flags & 0x80),
                }
                if packet.haslayer(Raw):
                    payload_length = len(packet[Raw].load)

            # UDP
            elif packet.haslayer(UDP):
                udp = packet[UDP]
                src_port = udp.sport
                dst_port = udp.dport
                if packet.haslayer(Raw):
                    payload_length = len(packet[Raw].load)

            # ICMP
            elif packet.haslayer(ICMP):
                protocol = 1

            return PacketInfo(
                timestamp=float(packet.time),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                length=length,
                payload_length=payload_length,
                flags=flags,
                raw_packet=packet
            )

        except Exception as e:
            self._stats['parse_errors'] += 1
            return None

    def _packet_callback(self, packet):
        """Callback для обработки захваченного пакета"""
        self._stats['total_packets'] += 1

        packet_info = self._parse_packet(packet)
        if packet_info is None:
            self._stats['skipped_packets'] += 1
            return

        try:
            self.packet_queue.put_nowait(packet_info)
            self._stats['queued_packets'] += 1
        except queue.Full:
            self._stats['dropped_packets'] += 1

    def _capture_loop(self):
        """Основной цикл захвата пакетов"""
        try:
            print(f"Starting capture on interface: {self.interface}")
            print(f"Filter: {self.bpf_filter or 'none'}")

            sniff(
                iface=self.interface,
                prn=self._packet_callback,
                filter=self.bpf_filter,
                store=False,
                stop_filter=lambda _: not self._running
            )
        except PermissionError:
            print("\nERROR: Permission denied!")
            if IS_WINDOWS:
                print("Try running as Administrator")
            else:
                print("Try running with sudo")
            self._running = False
        except Exception as e:
            print(f"\nCapture error: {e}")
            self._running = False

    def start(self):
        """Запускает захват пакетов в отдельном потоке"""
        if self._running:
            print("Capture already running")
            return

        if self.interface is None:
            print("ERROR: No network interface specified or detected")
            print("Available interfaces:")
            for iface in self.list_interfaces():
                print(f"  - {iface.get('name', 'unknown')}: {iface.get('description', '')}")
            return

        self._running = True
        self._start_time = time.time()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="PacketCapture"
        )
        self._capture_thread.start()

        # Даём время на запуск
        time.sleep(0.5)

        if self._running:
            print(f"Packet capture started on interface: {self.interface}")

    def stop(self):
        """Останавливает захват пакетов"""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=3.0)
        print("Packet capture stopped")

    def is_running(self) -> bool:
        """Проверяет, запущен ли захват"""
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику захвата"""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            **dict(self._stats),
            'uptime_seconds': uptime,
            'packets_per_second': self._stats['total_packets'] / uptime if uptime > 0 else 0,
            'queue_size': self.packet_queue.qsize(),
            'is_running': self._running,
            'interface': self.interface
        }

    def get_packet(self, timeout: float = 1.0) -> Optional[PacketInfo]:
        """Получает пакет из очереди"""
        try:
            return self.packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# Тестирование модуля
if __name__ == "__main__":
    print("=" * 60)
    print("Network Interface Detection")
    print("=" * 60)
    print(f"Operating System: {platform.system()}")
    print(f"Scapy available: {SCAPY_AVAILABLE}")
    print()

    print("Available interfaces:")
    interfaces = PacketCapture.list_interfaces()
    for i, iface in enumerate(interfaces):
        print(f"  [{i}] {iface.get('name', 'unknown')}")
        if iface.get('description'):
            print(f"      Description: {iface['description']}")
        if iface.get('ips'):
            print(f"      IPs: {', '.join(iface['ips'][:3])}")

    if not interfaces:
        print("  No interfaces found!")
        print("  On Windows, make sure Npcap is installed: https://npcap.com")
        exit(1)

    print()
    print("=" * 60)
    print("Testing Packet Capture")
    print("=" * 60)

    # Выбираем интерфейс
    selected = interfaces[0]['name'] if interfaces else None
    print(f"Using interface: {selected}")

    capture = PacketCapture(interface=selected, bpf_filter="ip")
    capture.start()

    if not capture.is_running():
        print("Failed to start capture")
        exit(1)

    try:
        print("\nCapturing packets (Ctrl+C to stop)...\n")
        for i in range(50):
            packet = capture.get_packet(timeout=1.0)
            if packet:
                print(f"Packet: {packet.src_ip}:{packet.src_port} -> "
                      f"{packet.dst_ip}:{packet.dst_port} "
                      f"(proto={packet.protocol}, len={packet.length})")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        capture.stop()
        print("\nStats:", capture.get_stats())