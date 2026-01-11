"""
Service layer for real-time analysis
"""

import threading
import queue
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from pathlib import Path


class RealtimeService:
    """
    Manages real-time traffic analysis pipeline
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.pipeline = None
        self.running = False
        self.flows = []
        self.events_queue = queue.Queue(maxsize=1000)
        self.stats = {}
        self._flow_lock = threading.Lock()
        self._initialized = True

    def start(self, interface: str, model_path: str, preprocessor_path: str,
              schema_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Start real-time capture"""

        if self.running:
            return {'status': 'error', 'message': 'Already running'}

        try:
            from realtime import RealtimePipeline

            # Clear previous state
            with self._flow_lock:
                self.flows = []

            # Clear queue
            while not self.events_queue.empty():
                try:
                    self.events_queue.get_nowait()
                except:
                    pass

            def on_flow(result):
                flow_data = {
                    'timestamp': result.timestamp,
                    'src_ip': result.src_ip,
                    'src_port': result.src_port,
                    'dst_ip': result.dst_ip,
                    'dst_port': result.dst_port,
                    'protocol': result.protocol,
                    'packets': result.total_packets,
                    'bytes': result.total_bytes,
                    'is_attack': result.is_attack,
                    'confidence': result.confidence,
                    'class_name': result.class_name
                }

                with self._flow_lock:
                    self.flows.append(flow_data)
                    if len(self.flows) > 100:
                        self.flows = self.flows[-100:]

                try:
                    self.events_queue.put_nowait({'type': 'flow', 'data': flow_data})
                except queue.Full:
                    pass

            def on_attack(result):
                attack_data = {
                    'timestamp': result.timestamp,
                    'src_ip': result.src_ip,
                    'src_port': result.src_port,
                    'dst_ip': result.dst_ip,
                    'dst_port': result.dst_port,
                    'confidence': result.confidence
                }
                try:
                    self.events_queue.put_nowait({'type': 'attack', 'data': attack_data})
                except queue.Full:
                    pass

            self.pipeline = RealtimePipeline(
                interface=interface,
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                feature_schema_path=schema_path,
                threshold=threshold,
                on_flow_analyzed=on_flow,
                on_attack_detected=on_attack
            )

            self.pipeline.start()

            import time
            time.sleep(1)

            if not self.pipeline.is_running():
                return {'status': 'error', 'message': 'Failed to start capture'}

            self.running = True
            return {'status': 'success', 'message': 'Capture started'}

        except Exception as e:
            import traceback
            return {'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}

    def stop(self) -> Dict[str, Any]:
        """Stop real-time capture"""

        if not self.running:
            return {'status': 'error', 'message': 'Not running'}

        try:
            if self.pipeline:
                self.pipeline.stop()
                self.pipeline = None

            self.running = False
            return {'status': 'success', 'message': 'Capture stopped'}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""

        if self.pipeline and self.running:
            try:
                summary = self.pipeline.get_summary()
                return {
                    'running': True,
                    'packets': summary.get('total_packets', 0),
                    'flows': summary.get('total_flows_analyzed', 0),
                    'attacks': summary.get('total_attacks', 0),
                    'packets_per_sec': summary.get('packets_per_second', 0),
                    'active_flows': summary.get('active_flows', 0),
                    'attack_rate': summary.get('recent_attack_rate', 0) * 100
                }
            except:
                pass

        return {
            'running': False,
            'packets': 0,
            'flows': 0,
            'attacks': 0,
            'packets_per_sec': 0,
            'active_flows': 0,
            'attack_rate': 0
        }

    def get_flows(self, limit: int = 50) -> list:
        """Get recent flows"""
        with self._flow_lock:
            return list(self.flows[-limit:])

    def get_event(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get next event from queue"""
        try:
            return self.events_queue.get(timeout=timeout)
        except queue.Empty:
            return None