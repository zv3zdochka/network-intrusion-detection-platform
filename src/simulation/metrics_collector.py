"""
Collect and aggregate metrics during simulation.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

import numpy as np


@dataclass
class MetricsSnapshot:
    """Snapshot of metrics at a point in time."""
    timestamp: datetime
    elapsed_seconds: float
    flows_processed: int
    flows_per_second: float
    alerts_generated: int
    alerts_per_second: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "flows_processed": self.flows_processed,
            "flows_per_second": round(self.flows_per_second, 1),
            "alerts_generated": self.alerts_generated,
            "alerts_per_second": round(self.alerts_per_second, 2),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "latency_p99_ms": round(self.latency_p99_ms, 2)
        }


class MetricsCollector:
    """
    Collect metrics during simulation run.
    """

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval

        self.start_time: Optional[float] = None
        self.snapshots: List[MetricsSnapshot] = []

        # Cumulative counters
        self.flows_processed = 0
        self.alerts_generated = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

        # Latency tracking
        self.latencies: List[float] = []

        # For rate calculation
        self._last_snapshot_time: Optional[float] = None
        self._last_flows_count = 0
        self._last_alerts_count = 0

    def start(self):
        """Start metrics collection."""
        self.start_time = time.time()
        self._last_snapshot_time = self.start_time
        self._last_flows_count = 0
        self._last_alerts_count = 0

    def update(
            self,
            flows: int,
            alerts: int,
            tp: int,
            fp: int,
            tn: int,
            fn: int,
            latencies: List[float]
    ):
        """Update metrics with new batch results."""
        self.flows_processed += flows
        self.alerts_generated += alerts
        self.true_positives += tp
        self.false_positives += fp
        self.true_negatives += tn
        self.false_negatives += fn
        self.latencies.extend(latencies)

    def take_snapshot(self) -> MetricsSnapshot:
        """Take current metrics snapshot."""
        now = time.time()
        elapsed = now - self.start_time if self.start_time else 0

        # Calculate rates
        time_delta = now - self._last_snapshot_time if self._last_snapshot_time else 1
        flows_delta = self.flows_processed - self._last_flows_count
        alerts_delta = self.alerts_generated - self._last_alerts_count

        flows_per_sec = flows_delta / time_delta if time_delta > 0 else 0
        alerts_per_sec = alerts_delta / time_delta if time_delta > 0 else 0

        # Calculate metrics
        precision = self._safe_div(
            self.true_positives,
            self.true_positives + self.false_positives
        )
        recall = self._safe_div(
            self.true_positives,
            self.true_positives + self.false_negatives
        )
        f1 = self._safe_div(2 * precision * recall, precision + recall)

        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        accuracy = self._safe_div(self.true_positives + self.true_negatives, total)

        # Latency percentiles
        if self.latencies:
            p50 = float(np.percentile(self.latencies, 50))
            p95 = float(np.percentile(self.latencies, 95))
            p99 = float(np.percentile(self.latencies, 99))
        else:
            p50 = p95 = p99 = 0.0

        snapshot = MetricsSnapshot(
            timestamp=datetime.now(),
            elapsed_seconds=elapsed,
            flows_processed=self.flows_processed,
            flows_per_second=flows_per_sec,
            alerts_generated=self.alerts_generated,
            alerts_per_second=alerts_per_sec,
            true_positives=self.true_positives,
            false_positives=self.false_positives,
            true_negatives=self.true_negatives,
            false_negatives=self.false_negatives,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99
        )

        self.snapshots.append(snapshot)

        # Update last values
        self._last_snapshot_time = now
        self._last_flows_count = self.flows_processed
        self._last_alerts_count = self.alerts_generated

        return snapshot

    def get_final_report(self) -> Dict[str, Any]:
        """Get final metrics report."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        precision = self._safe_div(
            self.true_positives,
            self.true_positives + self.false_positives
        )
        recall = self._safe_div(
            self.true_positives,
            self.true_positives + self.false_negatives
        )
        f1 = self._safe_div(2 * precision * recall, precision + recall)

        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        accuracy = self._safe_div(self.true_positives + self.true_negatives, total)

        throughput = self.flows_processed / elapsed if elapsed > 0 else 0

        report = {
            "summary": {
                "total_flows": self.flows_processed,
                "total_alerts": self.alerts_generated,
                "elapsed_seconds": round(elapsed, 2),
                "throughput_flows_per_sec": round(throughput, 1)
            },
            "classification": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4)
            },
            "latency": {
                "p50_ms": round(float(np.percentile(self.latencies, 50)), 2) if self.latencies else 0,
                "p95_ms": round(float(np.percentile(self.latencies, 95)), 2) if self.latencies else 0,
                "p99_ms": round(float(np.percentile(self.latencies, 99)), 2) if self.latencies else 0,
                "mean_ms": round(float(np.mean(self.latencies)), 2) if self.latencies else 0,
                "max_ms": round(float(np.max(self.latencies)), 2) if self.latencies else 0
            },
            "snapshots_count": len(self.snapshots)
        }

        return report

    def export_snapshots(self, path: str):
        """Export snapshots to JSON file."""
        data = [s.to_dict() for s in self.snapshots]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.snapshots = []
        self.flows_processed = 0
        self.alerts_generated = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.latencies = []
        self._last_snapshot_time = None
        self._last_flows_count = 0
        self._last_alerts_count = 0