"""
Основной pipeline для анализа трафика в реальном времени
"""

import time
import threading
import queue
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np

from .capture import PacketCapture, PacketInfo
from .flow_aggregator import FlowAggregator
from .feature_extractor import FeatureExtractor
from .analyzer import TrafficAnalyzer, create_dummy_analyzer


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
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        feature_schema_path: Optional[str] = None,
        bpf_filter: Optional[str] = None,
        flow_timeout: float = 120.0,
        analysis_interval: float = 5.0,
        threshold: float = 0.5,
        on_attack_detected: Optional[Callable[[AnalysisResult], None]] = None,
        on_flow_analyzed: Optional[Callable[[AnalysisResult], None]] = None,
        debug: bool = False
    ):
        self.debug = debug

        # Callbacks
        self.on_attack_detected = on_attack_detected
        self.on_flow_analyzed = on_flow_analyzed

        # Компоненты
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

        # Загружаем feature_schema если есть
        if feature_schema_path:
            self._load_feature_schema(feature_schema_path)

        # Анализатор
        if model_path:
            self.analyzer = TrafficAnalyzer(
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                feature_schema_path=feature_schema_path,
                threshold=threshold
            )
        else:
            self.analyzer = create_dummy_analyzer()

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
            'attacks_detected': 0,
            'analysis_errors': 0
        }

        # Отладка
        self._first_flow_logged = False

    def _load_feature_schema(self, schema_path: str):
        """Загружает схему признаков и обновляет feature_extractor"""
        import json
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)

            feature_cols = schema.get('feature_columns', [])
            if feature_cols:
                self.feature_extractor = FeatureExtractor(feature_names=feature_cols)
                print(f"  Loaded {len(feature_cols)} features from schema")
        except Exception as e:
            print(f"  Warning: Could not load feature schema: {e}")

    def _on_flow_complete(self, flow_data: Dict[str, Any]):
        """Callback при завершении потока"""
        self._analyze_flow(flow_data)

    def _analyze_flow(self, flow_data: Dict[str, Any]) -> Optional[AnalysisResult]:
        """Анализирует поток"""
        try:
            # Отладка первого потока
            if self.debug and not self._first_flow_logged:
                print(f"\n[DEBUG] First flow data:")
                print(f"  src: {flow_data.get('src_ip')}:{flow_data.get('src_port')}")
                print(f"  dst: {flow_data.get('dst_ip')}:{flow_data.get('dst_port')}")
                print(f"  packets: {flow_data.get('total_packets')}")
                print(f"  duration: {flow_data.get('duration')}")

            # Извлекаем признаки
            features = self.feature_extractor.extract(flow_data)
            features_array = self.feature_extractor.extract_array(flow_data)

            if self.debug and not self._first_flow_logged:
                print(f"  features shape: {features_array.shape}")
                print(f"  expected features: {self.analyzer.n_features if hasattr(self.analyzer, 'n_features') else 'unknown'}")
                print(f"  first 5 features: {features_array[:5]}")
                self._first_flow_logged = True

            # Предсказание
            prediction = self.analyzer.predict(features_array)

            if 'error' in prediction:
                if self.debug:
                    print(f"[DEBUG] Prediction error: {prediction.get('error')}")
                self._stats['analysis_errors'] += 1
                return None

            # Результат
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
                probabilities={'probability': prediction.get('probability', 0.0)},
                features=features
            )

            # Сохраняем
            with self._lock:
                self._recent_results.append(result)
                if len(self._recent_results) > self._max_results:
                    self._recent_results.pop(0)
                self._stats['flows_analyzed'] += 1

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
            import traceback
            self._stats['analysis_errors'] += 1
            if self.debug:
                print(f"[DEBUG] Error analyzing flow: {e}")
                traceback.print_exc()
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
                if self.debug:
                    print(f"Error processing packet: {e}")

    def _periodic_analysis_loop(self):
        """Периодический анализ активных потоков"""
        while self._running:
            time.sleep(self.analysis_interval)

            # Проверяем таймауты
            completed = self.aggregator.check_timeouts()

            if self.debug and completed:
                print(f"[DEBUG] Completed {len(completed)} flows by timeout")

            # Анализируем активные потоки с достаточным количеством пакетов
            active_flows = self.aggregator.get_active_flows()

            if self.debug and active_flows:
                print(f"[DEBUG] Active flows: {len(active_flows)}")

            for flow_data in active_flows:
                if flow_data.get('total_packets', 0) >= 5:  # Снизил порог
                    self._analyze_flow(flow_data)

    def start(self):
        """Запускает pipeline"""
        if self._running:
            return

        self._running = True
        self._stats['start_time'] = datetime.now().isoformat()

        self.capture.start()
        self.aggregator.start_cleanup_thread(interval=30.0)

        self._processing_thread = threading.Thread(
            target=self._packet_processing_loop,
            daemon=True,
            name="PacketProcessing"
        )
        self._processing_thread.start()

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
        return self._running

    def get_recent_results(self, n: int = 100) -> List[AnalysisResult]:
        with self._lock:
            return self._recent_results[-n:]

    def get_recent_attacks(self, n: int = 100) -> List[AnalysisResult]:
        with self._lock:
            attacks = [r for r in self._recent_results if r.is_attack]
            return attacks[-n:]

    def get_stats(self) -> Dict[str, Any]:
        return {
            'pipeline': self._stats.copy(),
            'capture': self.capture.get_stats(),
            'aggregator': self.aggregator.get_stats(),
            'analyzer': self.analyzer.get_stats(),
            'is_running': self._running
        }

    def get_summary(self) -> Dict[str, Any]:
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
            'analysis_errors': stats['pipeline'].get('analysis_errors', 0),
            'recent_attack_rate': recent_attacks / recent_total if recent_total > 0 else 0
        }