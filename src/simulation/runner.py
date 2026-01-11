"""
Simulation runner orchestrating the full pipeline.
"""

import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.inference import Predictor, InferencePipeline
from .replay import FlowReplay, FlowBatch
from .metrics_collector import MetricsCollector, MetricsSnapshot


@dataclass
class SimulationConfig:
    """Simulation configuration."""
    speed: float = 1.0
    batch_size: int = 100
    max_flows: Optional[int] = None
    max_duration: Optional[float] = None
    update_interval: float = 1.0
    verbose: bool = True
    store_alerts: bool = True


class SimulationRunner:
    """
    Run simulation of flow processing.
    """

    def __init__(
            self,
            predictor: Predictor,
            replay: FlowReplay,
            config: Optional[SimulationConfig] = None
    ):
        self.predictor = predictor
        self.replay = replay
        self.config = config or SimulationConfig()

        self.pipeline = InferencePipeline(predictor)
        self.metrics = MetricsCollector(update_interval=self.config.update_interval)

        self._running = False
        self._callbacks: List[Callable[[MetricsSnapshot], None]] = []

    def add_callback(self, callback: Callable[[MetricsSnapshot], None]):
        """Add callback for metrics updates."""
        self._callbacks.append(callback)

    def run(self) -> Dict[str, Any]:
        """
        Run simulation.

        Returns:
            Final metrics report
        """
        self._running = True
        self.pipeline.reset()
        self.metrics.reset()
        self.replay.reset()

        self.metrics.start()
        start_time = time.time()
        last_update = start_time

        flows_limit = self.config.max_flows
        duration_limit = self.config.max_duration

        if self.config.verbose:
            self._print_header()

        try:
            for batch in self.replay.iter_batches(speed=self.config.speed):
                if not self._running:
                    break

                # Check limits
                if flows_limit and self.metrics.flows_processed >= flows_limit:
                    break

                if duration_limit and (time.time() - start_time) >= duration_limit:
                    break

                # Process batch
                self._process_batch(batch)

                # Periodic update
                now = time.time()
                if now - last_update >= self.config.update_interval:
                    snapshot = self.metrics.take_snapshot()

                    if self.config.verbose:
                        self._print_progress(snapshot)

                    for callback in self._callbacks:
                        callback(snapshot)

                    last_update = now

        except KeyboardInterrupt:
            if self.config.verbose:
                print("\nSimulation interrupted.")

        finally:
            self._running = False

        # Final snapshot
        final_snapshot = self.metrics.take_snapshot()

        if self.config.verbose:
            self._print_final(final_snapshot)

        return self.metrics.get_final_report()

    def _process_batch(self, batch: FlowBatch):
        """Process single batch."""
        alerts = self.pipeline.process_batch(
            features=batch.features,
            flow_indices=batch.indices,
            true_labels=batch.labels,
            store_alerts=self.config.store_alerts
        )

        # Update metrics
        tp = fp = tn = fn = 0
        latencies = []

        for i in range(batch.size):
            pred = 1 if any(a.flow_index == batch.indices[i] for a in alerts) else 0
            true = batch.labels[i]

            if pred == 1 and true == 1:
                tp += 1
            elif pred == 1 and true == 0:
                fp += 1
            elif pred == 0 and true == 0:
                tn += 1
            else:
                fn += 1

        # Estimate per-flow latency from batch
        batch_latency = sum(a.inference_time_ms for a in alerts) if alerts else 0.1
        per_flow = batch_latency / batch.size if batch.size > 0 else 0.1
        latencies = [per_flow] * batch.size

        self.metrics.update(
            flows=batch.size,
            alerts=len(alerts),
            tp=tp, fp=fp, tn=tn, fn=fn,
            latencies=latencies
        )

    def stop(self):
        """Stop simulation."""
        self._running = False

    def get_alerts(self, limit: int = 100) -> List[Dict]:
        """Get generated alerts."""
        alerts = self.pipeline.get_alerts(limit=limit)
        return [a.to_dict() for a in alerts]

    def _print_header(self):
        """Print simulation header."""
        print("=" * 70)
        print("SIMULATION STARTED")
        print("=" * 70)
        print(f"Data source: {self.replay.data_path.name}")
        print(f"Total flows: {self.replay.n_samples:,}")
        print(f"Speed: x{self.config.speed}")
        print(f"Batch size: {self.config.batch_size}")
        if self.config.max_flows:
            print(f"Max flows: {self.config.max_flows:,}")
        if self.config.max_duration:
            print(f"Max duration: {self.config.max_duration}s")
        print("=" * 70)
        print()

    def _print_progress(self, snapshot: MetricsSnapshot):
        """Print progress line."""
        elapsed = f"{snapshot.elapsed_seconds:6.1f}s"
        flows = f"{snapshot.flows_processed:>10,}"
        rate = f"{snapshot.flows_per_second:>8.0f}/s"
        alerts = f"{snapshot.alerts_generated:>8,}"
        f1 = f"{snapshot.f1:.4f}"
        latency = f"{snapshot.latency_p95_ms:>6.2f}ms"

        progress = self.replay.progress * 100
        bar_len = 20
        filled = int(bar_len * self.replay.progress)
        bar = "=" * filled + "-" * (bar_len - filled)

        line = f"[{elapsed}] [{bar}] {progress:5.1f}% | Flows: {flows} ({rate}) | Alerts: {alerts} | F1: {f1} | p95: {latency}"

        sys.stdout.write(f"\r{line}")
        sys.stdout.flush()

    def _print_final(self, snapshot: MetricsSnapshot):
        """Print final report."""
        print("\n")
        print("=" * 70)
        print("SIMULATION COMPLETED")
        print("=" * 70)
        print()
        print("Summary:")
        print(f"  Total flows processed: {snapshot.flows_processed:,}")
        print(f"  Total alerts generated: {snapshot.alerts_generated:,}")
        print(f"  Elapsed time: {snapshot.elapsed_seconds:.1f}s")
        print(f"  Throughput: {snapshot.flows_per_second:.0f} flows/sec")
        print()
        print("Classification Metrics:")
        print(f"  Precision: {snapshot.precision:.4f}")
        print(f"  Recall:    {snapshot.recall:.4f}")
        print(f"  F1 Score:  {snapshot.f1:.4f}")
        print(f"  Accuracy:  {snapshot.accuracy:.4f}")
        print()
        print("Confusion Matrix:")
        print(f"  TP: {snapshot.true_positives:>10,}  FP: {snapshot.false_positives:>10,}")
        print(f"  FN: {snapshot.false_negatives:>10,}  TN: {snapshot.true_negatives:>10,}")
        print()
        print("Latency:")
        print(f"  p50: {snapshot.latency_p50_ms:.2f}ms")
        print(f"  p95: {snapshot.latency_p95_ms:.2f}ms")
        print(f"  p99: {snapshot.latency_p99_ms:.2f}ms")
        print("=" * 70)