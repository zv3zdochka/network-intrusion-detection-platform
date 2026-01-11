"""
Основной pipeline для анализа трафика в реальном времени
Объединяет все компоненты
"""

import time
import threading
import queue
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from .capture import PacketCapture, PacketInfo
from .flow_aggregator import FlowAggregator
from .feature_extractor import FeatureExtractor
from .analyzer import TrafficAnalyzer


@dataclass
class AnalysisResult:
    """Результат анализа потока"""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    timestamp: str
    duration: float
    total_packets: int
    total_bytes: int
    prediction: int
    class_name: str
    confidence: float
    is_attack: bool
    probabilities: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)


class RealtimePipeline:
    """
    Pipeline для анализа сетевого трафика в реальном времени

    Поток данных:
    PacketCapture -> FlowAggregator -> FeatureExtractor -> TrafficAnalyzer -> Results
    """

    def __init__(
            self,
            interface: Optional[str] = None,
            model_path: Optional[str] = None,
            scaler_path: Optional[str] = None,
            config_path: Optional[str] = None,
            bpf_filter: Optional[str] = None,
            flow_timeout: float = 120.0,
            analysis_interval: float = 5.0,
            on_attack_detected: Optional[Callable[[AnalysisResult], None]] = None,
            on_flow_analyzed: Optional[Callable[[AnalysisResult], None]] = None
    ):
        """
        Args:
            interface: Сетевой интерфейс
            model_path: Путь к модели
            scaler_path: Путь к скейлеру
            config_path: Путь к конфигурации
            bpf_filter: BPF фильтр для пакетов
            flow_timeout: Таймаут потока в секундах
            analysis_interval: Интервал анализа активных потоков
            on_attack_detected: Callback при обнаружении атаки
            on_flow_analyzed: Callback после анализа потока
        """
        # Callbacks
        self.on_attack_detected = on_attack_detected
        self.on_flow_analyzed = on_flow_analyzed

        # Инициализация компонентов
        self.packet_queue = queue.Queue(maxsize=10000)

        self.capture = PacketCapture(
            interface=interface,
            packet_queue=self.packet_queue,
            bpf_filter=bpf_filter
        )

        self.aggregator = FlowAggregator(
            flow_timeout=flow_timeout,
            on_flow_complete=self._on_flow_complete
        )

        self.feature_extractor = FeatureExtractor()

        self.analyzer = TrafficAnalyzer(
            model_path=model_path,
            scaler_path=scaler_path,
            config_path=config_path
        )

        # Параметры
        self.analysis_interval = analysis_interval

        # Потоки
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._analysis_thread: Optional[threading.Thread] = None

        # Результаты
        self._results_queue = queue.Queue(maxsize=1000)
        self._recent_results: List[AnalysisResult] = []
        self._max_results = 1000
        self._lock = threading.Lock()

        # Статистика
        self._stats = {
            'start_time': None,
            'packets_processed': 0,
            'flows_analyzed': 0,
            'attacks_detected': 0
        }

    def _on_flow_complete(self, flow_data: Dict[str, Any]):
        """Callback при завершении потока"""
        self._analyze_flow(flow_data)

    def _analyze_flow(self, flow_data: Dict[str, Any]) -> Optional[AnalysisResult]:
        """Анализирует поток и возвращает результат"""
        try:
            # Извлекаем признаки
            features = self.feature_extractor.extract(flow_data)
            features_array = self.feature_extractor.extract_array(flow_data)

            # Получаем предсказание
            prediction = self.analyzer.predict(features_array)

            if 'error' in prediction:
                return None

            # Формируем результат
            result = AnalysisResult(
                flow_id=str(flow_data.get('flow_key', '')),
                src_ip=flow_data.get('src_ip', ''),
                dst_ip=flow_data.get('dst_ip', ''),
                src_port=flow_data.get('src_port', 0),
                dst_port=flow_data.get('dst_port', 0),
                protocol=flow_data.get('protocol', 0),
                timestamp=datetime.now().isoformat(),
                duration=flow_data.get('duration', 0),
                total_packets=flow_data.get('total_packets', 0),
                total_bytes=flow_data.get('total_bytes', 0),
                prediction=prediction.get('prediction', -1),
                class_name=prediction.get('class_name', 'unknown'),
                confidence=prediction.get('confidence', 0.0),
                is_attack=prediction.get('is_attack', False),
                probabilities=prediction.get('probabilities', {}),
                features=features
            )

            # Сохраняем результат
            with self._lock:
                self._recent_results.append(result)
                if len(self._recent_results) > self._max_results:
                    self._recent_results.pop(0)
                self._stats['flows_analyzed'] += 1

            # Помещаем в очередь результатов
            try:
                self._results_queue.put_nowait(result)
            except queue.Full:
                pass

            # Callbacks
            if self.on_flow_analyzed:
                self.on_flow_analyzed(result)

            if result.is_attack:
                self._stats['attacks_detected'] += 1
                if self.on_attack_detected:
                    self.on_attack_detected(result)

            return result

        except Exception as e:
            print(f"Error analyzing flow: {e}")
            return None

    def _packet_processing_loop(self):
        """Цикл обработки пакетов"""
        while self._running:
            try:
                packet = self.packet_queue.get(timeout=0.1)
                self.aggregator.add_packet(packet)
                self._stats['packets_processed'] += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing packet: {e}")

    def _periodic_analysis_loop(self):
        """Цикл периодического анализа активных потоков"""
        while self._running:
            time.sleep(self.analysis_interval)

            # Проверяем таймауты потоков
            self.aggregator.check_timeouts()

            # Опционально: анализируем активные потоки
            # (для обнаружения длительных атак)
            active_flows = self.aggregator.get_active_flows()
            for flow_data in active_flows:
                # Анализируем только потоки с достаточным количеством пакетов
                if flow_data.get('total_packets', 0) >= 10:
                    self._analyze_flow(flow_data)

    def start(self):
        """Запускает pipeline"""
        if self._running:
            return

        self._running = True
        self._stats['start_time'] = datetime.now().isoformat()

        # Запускаем захват пакетов
        self.capture.start()

        # Запускаем очистку потоков
        self.aggregator.start_cleanup_thread(interval=30.0)

        # Запускаем обработку пакетов
        self._processing_thread = threading.Thread(
            target=self._packet_processing_loop,
            daemon=True,
            name="PacketProcessing"
        )
        self._processing_thread.start()

        # Запускаем периодический анализ
        self._analysis_thread = threading.Thread(
            target=self._periodic_analysis_loop,
            daemon=True,
            name="PeriodicAnalysis"
        )
        self._analysis_thread.start()

        print("Realtime pipeline started")

    def stop(self):
        """Останавливает pipeline"""
        self._running = False

        self.capture.stop()
        self.aggregator.stop_cleanup_thread()

        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        if self._analysis_thread:
            self._analysis_thread.join(timeout=2.0)

        print("Realtime pipeline stopped")

    def is_running(self) -> bool:
        """Проверяет, запущен ли pipeline"""
        return self._running

    def get_result(self, timeout: float = 1.0) -> Optional[AnalysisResult]:
        """Получает результат из очереди"""
        try:
            return self._results_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_recent_results(self, n: int = 100) -> List[AnalysisResult]:
        """Возвращает последние n результатов"""
        with self._lock:
            return self._recent_results[-n:]

    def get_recent_attacks(self, n: int = 100) -> List[AnalysisResult]:
        """Возвращает последние n обнаруженных атак"""
        with self._lock:
            attacks = [r for r in self._recent_results if r.is_attack]
            return attacks[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику pipeline"""
        capture_stats = self.capture.get_stats()
        aggregator_stats = self.aggregator.get_stats()
        analyzer_stats = self.analyzer.get_stats()

        return {
            'pipeline': self._stats.copy(),
            'capture': capture_stats,
            'aggregator': aggregator_stats,
            'analyzer': analyzer_stats,
            'is_running': self._running,
            'results_queue_size': self._results_queue.qsize()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Возвращает краткую сводку состояния"""
        stats = self.get_stats()

        with self._lock:
            recent_attacks = sum(1 for r in self._recent_results[-100:] if r.is_attack)
            recent_total = min(100, len(self._recent_results))

        return {
            'is_running': self._running,
            'uptime_seconds': stats['capture'].get('uptime_seconds', 0),
            'packets_per_second': stats['capture'].get('packets_per_second', 0),
            'active_flows': stats['aggregator'].get('active_flows', 0),
            'total_packets': stats['pipeline'].get('packets_processed', 0),
            'total_flows_analyzed': stats['pipeline'].get('flows_analyzed', 0),
            'total_attacks': stats['pipeline'].get('attacks_detected', 0),
            'recent_attack_rate': recent_attacks / recent_total if recent_total > 0 else 0
        }


# CLI интерфейс для тестирования
def main():
    """Тестовый запуск pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Realtime Network Traffic Analyzer')
    parser.add_argument('--interface', '-i', type=str, default=None,
                        help='Network interface to capture')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model file')
    parser.add_argument('--filter', '-f', type=str, default='ip',
                        help='BPF filter')
    parser.add_argument('--duration', '-d', type=int, default=60,
                        help='Duration in seconds')

    args = parser.parse_args()

    def on_attack(result: AnalysisResult):
        print(f"\n🚨 ATTACK DETECTED!")
        print(f"   {result.src_ip}:{result.src_port} -> {result.dst_ip}:{result.dst_port}")
        print(f"   Type: {result.class_name}, Confidence: {result.confidence:.2%}")

    def on_flow(result: AnalysisResult):
        status = "⚠️" if result.is_attack else "✅"
        print(f"{status} Flow: {result.src_ip}:{result.src_port} -> "
              f"{result.dst_ip}:{result.dst_port} | "
              f"{result.class_name} ({result.confidence:.1%})")

    # Создаём pipeline
    pipeline = RealtimePipeline(
        interface=args.interface,
        model_path=args.model,
        bpf_filter=args.filter,
        on_attack_detected=on_attack,
        on_flow_analyzed=on_flow
    )

    # Если нет модели, используем заглушку
    if not args.model:
        from .analyzer import create_dummy_analyzer
        pipeline.analyzer = create_dummy_analyzer()
        print("Using dummy analyzer (no model provided)")

    print(f"\nStarting capture on interface: {args.interface or 'all'}")
    print(f"BPF filter: {args.filter}")
    print(f"Duration: {args.duration} seconds")
    print("-" * 50)

    pipeline.start()

    try:
        for _ in range(args.duration):
            time.sleep(1)
            summary = pipeline.get_summary()
            print(f"\rPackets: {summary['total_packets']} | "
                  f"Flows: {summary['total_flows_analyzed']} | "
                  f"Attacks: {summary['total_attacks']} | "
                  f"Rate: {summary['packets_per_second']:.1f} pps", end='')
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        pipeline.stop()

        print("\n" + "=" * 50)
        print("Final Statistics:")
        stats = pipeline.get_stats()
        print(f"  Total packets: {stats['pipeline']['packets_processed']}")
        print(f"  Flows analyzed: {stats['pipeline']['flows_analyzed']}")
        print(f"  Attacks detected: {stats['pipeline']['attacks_detected']}")


if __name__ == "__main__":
    main()