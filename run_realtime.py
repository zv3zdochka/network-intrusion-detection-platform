#!/usr/bin/env python3
"""
Real-time Network Traffic Analyzer
With logging, charts, and attack detection
"""

import argparse
import signal
import sys
import time
import platform
import os
import json
import warnings
from pathlib import Path
from datetime import datetime
from collections import deque
from threading import Thread
import csv

# Подавляем предупреждения XGBoost
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning)

IS_WINDOWS = platform.system() == 'Windows'
sys.path.insert(0, '.')

from realtime import RealtimePipeline
from realtime.capture import PacketCapture
from realtime.utils import setup_logging, protocol_name

# Глобальные переменные
pipeline = None
results_saver = None


class ResultsSaver:
    """Сохраняет результаты в файлы"""

    def __init__(self, output_dir: str = "realtime_testing_artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Создаём папку для текущей сессии
        self.session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / self.session_time
        self.session_dir.mkdir(exist_ok=True)

        # Файлы
        self.flows_file = self.session_dir / "flows.csv"
        self.attacks_file = self.session_dir / "attacks.csv"
        self.stats_file = self.session_dir / "stats.json"
        self.log_file = self.session_dir / "session.log"

        # Инициализируем CSV
        self._init_csv()

        # Данные для графиков
        self.timeline_data = {
            'timestamps': deque(maxlen=1000),
            'packets_per_sec': deque(maxlen=1000),
            'flows_per_sec': deque(maxlen=1000),
            'attacks_total': deque(maxlen=1000),
            'attack_rate': deque(maxlen=1000),
        }

        self._flows_count = 0
        self._attacks_count = 0
        self._last_flows = 0
        self._last_time = time.time()

        print(f"[Saver] Session directory: {self.session_dir}")

    def _init_csv(self):
        """Инициализирует CSV файлы с заголовками"""

        # Flows CSV
        with open(self.flows_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port',
                'protocol', 'duration', 'packets', 'bytes',
                'prediction', 'class_name', 'confidence', 'is_attack'
            ])

        # Attacks CSV
        with open(self.attacks_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port',
                'protocol', 'duration', 'packets', 'bytes',
                'confidence', 'probability'
            ])

    def save_flow(self, result):
        """Сохраняет результат анализа потока"""
        self._flows_count += 1

        with open(self.flows_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.timestamp,
                result.src_ip,
                result.src_port,
                result.dst_ip,
                result.dst_port,
                result.protocol,
                f"{result.duration:.4f}",
                result.total_packets,
                result.total_bytes,
                result.prediction,
                result.class_name,
                f"{result.confidence:.4f}",
                result.is_attack
            ])

        if result.is_attack:
            self._attacks_count += 1
            self._save_attack(result)

    def _save_attack(self, result):
        """Сохраняет атаку"""
        with open(self.attacks_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.timestamp,
                result.src_ip,
                result.src_port,
                result.dst_ip,
                result.dst_port,
                result.protocol,
                f"{result.duration:.4f}",
                result.total_packets,
                result.total_bytes,
                f"{result.confidence:.4f}",
                result.probabilities.get('probability', 0)
            ])

        # Логируем в файл
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{result.timestamp}] ATTACK: "
                    f"{result.src_ip}:{result.src_port} -> "
                    f"{result.dst_ip}:{result.dst_port} "
                    f"({result.confidence:.1%})\n")

    def update_timeline(self, stats: dict):
        """Обновляет данные для графиков"""
        now = time.time()
        elapsed = now - self._last_time

        if elapsed > 0:
            flows_per_sec = (self._flows_count - self._last_flows) / elapsed
        else:
            flows_per_sec = 0

        self.timeline_data['timestamps'].append(datetime.now())
        self.timeline_data['packets_per_sec'].append(
            stats.get('packets_per_second', 0)
        )
        self.timeline_data['flows_per_sec'].append(flows_per_sec)
        self.timeline_data['attacks_total'].append(self._attacks_count)

        total_flows = self._flows_count or 1
        self.timeline_data['attack_rate'].append(
            self._attacks_count / total_flows * 100
        )

        self._last_flows = self._flows_count
        self._last_time = now

    def save_final_stats(self, stats: dict):
        """Сохраняет финальную статистику"""
        final_stats = {
            'session_id': self.session_time,
            'start_time': stats.get('pipeline', {}).get('start_time'),
            'end_time': datetime.now().isoformat(),
            'runtime_seconds': stats.get('capture', {}).get('uptime_seconds', 0),
            'total_packets': stats.get('capture', {}).get('total_packets', 0),
            'total_flows': self._flows_count,
            'total_attacks': self._attacks_count,
            'attack_rate': self._attacks_count / max(self._flows_count, 1),
            'packets_per_second': stats.get('capture', {}).get('packets_per_second', 0),
            'pipeline_stats': stats.get('pipeline', {}),
            'analyzer_stats': stats.get('analyzer', {}),
        }

        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, default=str)

        print(f"\n[Saver] Results saved to: {self.session_dir}")

    def generate_charts(self):
        """Генерирует графики"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Без GUI
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print("[Saver] matplotlib not installed, skipping charts")
            return

        if len(self.timeline_data['timestamps']) < 2:
            print("[Saver] Not enough data for charts")
            return

        timestamps = list(self.timeline_data['timestamps'])

        # Создаём figure с несколькими графиками
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Network Traffic Analysis - Session {self.session_time}',
                     fontsize=14, fontweight='bold')

        # 1. Packets per second
        ax1 = axes[0, 0]
        ax1.plot(timestamps, list(self.timeline_data['packets_per_sec']),
                 'b-', linewidth=1.5, label='Packets/s')
        ax1.set_ylabel('Packets per second')
        ax1.set_title('Network Traffic Rate')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        # 2. Flows per second
        ax2 = axes[0, 1]
        ax2.plot(timestamps, list(self.timeline_data['flows_per_sec']),
                 'g-', linewidth=1.5, label='Flows/s')
        ax2.set_ylabel('Flows per second')
        ax2.set_title('Flow Analysis Rate')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        # 3. Cumulative attacks
        ax3 = axes[1, 0]
        ax3.fill_between(timestamps, list(self.timeline_data['attacks_total']),
                         color='red', alpha=0.3)
        ax3.plot(timestamps, list(self.timeline_data['attacks_total']),
                 'r-', linewidth=2, label='Total Attacks')
        ax3.set_ylabel('Cumulative Attacks')
        ax3.set_title('Detected Attacks Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        # 4. Attack rate percentage
        ax4 = axes[1, 1]
        ax4.plot(timestamps, list(self.timeline_data['attack_rate']),
                 'orange', linewidth=1.5, label='Attack Rate %')
        ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1% threshold')
        ax4.set_ylabel('Attack Rate (%)')
        ax4.set_title('Attack Rate Percentage')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax4.set_ylim(bottom=0)

        plt.tight_layout()

        # Сохраняем
        chart_path = self.session_dir / "traffic_analysis.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[Saver] Chart saved to: {chart_path}")

        # Отдельный график атак если они есть
        if self._attacks_count > 0:
            self._generate_attack_chart()

    def _generate_attack_chart(self):
        """Генерирует детальный график атак"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd

        try:
            # Читаем атаки
            attacks_df = pd.read_csv(self.attacks_file)
            if len(attacks_df) == 0:
                return

            attacks_df['timestamp'] = pd.to_datetime(attacks_df['timestamp'])

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Attack Analysis', fontsize=14, fontweight='bold')

            # 1. Attacks by source IP
            ax1 = axes[0]
            src_counts = attacks_df['src_ip'].value_counts().head(10)
            src_counts.plot(kind='barh', ax=ax1, color='red', alpha=0.7)
            ax1.set_xlabel('Number of Attacks')
            ax1.set_title('Top Attack Sources')

            # 2. Attacks by destination port
            ax2 = axes[1]
            dst_counts = attacks_df['dst_port'].value_counts().head(10)
            dst_counts.plot(kind='barh', ax=ax2, color='orange', alpha=0.7)
            ax2.set_xlabel('Number of Attacks')
            ax2.set_title('Top Targeted Ports')

            plt.tight_layout()

            chart_path = self.session_dir / "attack_analysis.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[Saver] Attack chart saved to: {chart_path}")

        except Exception as e:
            print(f"[Saver] Error generating attack chart: {e}")


def signal_handler(sig, frame):
    print("\n\nStopping...")
    global pipeline, results_saver

    if pipeline:
        pipeline.stop()

    if results_saver:
        stats = pipeline.get_stats() if pipeline else {}
        results_saver.save_final_stats(stats)
        results_saver.generate_charts()

    sys.exit(0)


def print_attack_alert(result):
    print()
    print("🚨" + "=" * 68)
    print("    ATTACK DETECTED!")
    print("=" * 70)
    print(f"  Time:       {result.timestamp}")
    print(f"  Source:     {result.src_ip}:{result.src_port}")
    print(f"  Target:     {result.dst_ip}:{result.dst_port}")
    print(f"  Protocol:   {protocol_name(result.protocol)}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Traffic:    {result.total_packets} packets, {result.total_bytes} bytes")
    print("=" * 70)
    print()


def print_flow_result(result):
    if result.is_attack:
        status = "🚨ATTACK"
    else:
        status = "  BENIGN"

    proto = protocol_name(result.protocol)
    ts = datetime.now().strftime('%H:%M:%S')

    print(f"{ts} {status} {result.src_ip:>15}:{result.src_port:<5} -> "
          f"{result.dst_ip:>15}:{result.dst_port:<5} {proto:4} "
          f"{result.confidence:5.1%} {result.total_packets:5}pkts")


def check_admin():
    if IS_WINDOWS:
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False
    return os.geteuid() == 0


def find_artifacts(model_path: Path):
    model_dir = model_path.parent
    project_root = Path('.')

    prep_paths = [
        model_dir / 'preprocessor.joblib',
        project_root / 'artifacts' / 'preprocessor.joblib',
    ]
    preprocessor = next((p for p in prep_paths if p.exists()), None)

    schema_paths = [
        model_dir / 'feature_schema.json',
        project_root / 'artifacts' / 'feature_schema.json',
    ]
    schema = next((p for p in schema_paths if p.exists()), None)

    return preprocessor, schema


def select_interface():
    print("\nAvailable interfaces:")
    print("-" * 50)

    interfaces = PacketCapture.list_interfaces()
    main_interfaces = []

    for iface in interfaces:
        name = iface.get('name', '')
        ips = iface.get('ips', [])
        real_ips = [ip for ip in ips
                    if not ip.startswith('169.254')
                    and not ip.startswith('fe80')
                    and ip not in ('127.0.0.1', '::1')]

        if real_ips and not any(x in name for x in ['Npcap', 'WFP', 'Filter', 'Loopback']):
            main_interfaces.append(iface)

    for i, iface in enumerate(main_interfaces):
        name = iface.get('name', 'unknown')
        desc = iface.get('description', '')[:40]
        ips = [ip for ip in iface.get('ips', [])
               if not ip.startswith('169.254') and not ip.startswith('fe80')]
        print(f"  [{i}] {name}: {', '.join(ips[:2])}")

    print()

    if not main_interfaces:
        print("No interfaces found!")
        sys.exit(1)

    try:
        choice = input(f"Select [0]: ").strip()
        idx = int(choice) if choice else 0
        return main_interfaces[idx]['name']
    except:
        return main_interfaces[0]['name']


def main():
    global pipeline, results_saver

    print()
    print("=" * 70)
    print("       REAL-TIME NETWORK TRAFFIC ANALYZER")
    print("=" * 70)

    if not check_admin():
        print("\n⚠️  Run as Administrator for packet capture!")
        print()

    parser = argparse.ArgumentParser(description='Real-time Traffic Analyzer')
    parser.add_argument('-i', '--interface', type=str)
    parser.add_argument('-m', '--model', type=str,
                        default='training_artifacts/best_model_XGB_regularized.joblib')
    parser.add_argument('-p', '--preprocessor', type=str)
    parser.add_argument('-s', '--schema', type=str)
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-f', '--filter', type=str, default='ip')
    parser.add_argument('-d', '--duration', type=int, default=0)
    parser.add_argument('-o', '--output-dir', type=str, default='realtime_testing_artifacts')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('--list-interfaces', action='store_true')
    parser.add_argument('--no-save', action='store_true', help='Disable saving results')

    args = parser.parse_args()

    if args.list_interfaces:
        for iface in PacketCapture.list_interfaces():
            ips = iface.get('ips', [])
            real_ips = [ip for ip in ips if not ip.startswith('169.254')]
            if real_ips:
                print(f"{iface.get('name')}: {', '.join(real_ips[:2])}")
        return

    # Interface
    interface = args.interface or select_interface()

    # Model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    preprocessor_path, schema_path = find_artifacts(model_path)
    preprocessor_path = args.preprocessor or (str(preprocessor_path) if preprocessor_path else None)
    schema_path = args.schema or (str(schema_path) if schema_path else None)

    # Results saver
    if not args.no_save:
        results_saver = ResultsSaver(args.output_dir)

    print()
    print("-" * 70)
    print(f"  Interface: {interface}")
    print(f"  Model: {model_path.name}")
    print(f"  Threshold: {args.threshold}")
    if results_saver:
        print(f"  Output: {results_saver.session_dir}")
    print("-" * 70)

    # Callbacks
    def on_flow(result):
        if results_saver:
            results_saver.save_flow(result)
        if args.verbose:
            print_flow_result(result)

    def on_attack(result):
        if not args.quiet:
            print_attack_alert(result)

    # Pipeline
    pipeline = RealtimePipeline(
        interface=interface,
        model_path=str(model_path),
        preprocessor_path=preprocessor_path,
        feature_schema_path=schema_path,
        bpf_filter=args.filter,
        threshold=args.threshold,
        on_attack_detected=on_attack,
        on_flow_analyzed=on_flow
    )

    signal.signal(signal.SIGINT, signal_handler)

    print("\nStarting... (Ctrl+C to stop)\n")

    pipeline.start()
    time.sleep(1)

    if not pipeline.is_running():
        print("Failed to start!")
        return

    print("[OK] Capture running\n")

    try:
        elapsed = 0
        stats_interval = 3

        while args.duration == 0 or elapsed < args.duration:
            time.sleep(1)
            elapsed += 1

            if elapsed % stats_interval == 0:
                summary = pipeline.get_summary()

                # Update timeline
                if results_saver:
                    results_saver.update_timeline(summary)

                # Print status
                if not args.verbose:
                    attacks = summary['total_attacks']
                    attack_str = f"🚨{attacks}" if attacks > 0 else f"  {attacks}"

                    print(f"\r[{elapsed:4d}s] "
                          f"Pkts: {summary['total_packets']:>8,} | "
                          f"Flows: {summary['total_flows_analyzed']:>5,} | "
                          f"Attacks: {attack_str} | "
                          f"{summary['packets_per_second']:>6.1f} pps",
                          end='', flush=True)

    except KeyboardInterrupt:
        pass

    finally:
        print("\n\nStopping...")
        pipeline.stop()

        # Final stats
        stats = pipeline.get_stats()

        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"  Runtime:      {stats['capture'].get('uptime_seconds', 0):.1f} seconds")
        print(f"  Packets:      {stats['capture'].get('total_packets', 0):,}")
        print(f"  Flows:        {stats['pipeline']['flows_analyzed']:,}")
        print(f"  Attacks:      {stats['pipeline']['attacks_detected']}")

        if stats['pipeline']['flows_analyzed'] > 0:
            rate = stats['pipeline']['attacks_detected'] / stats['pipeline']['flows_analyzed'] * 100
            print(f"  Attack Rate:  {rate:.2f}%")

        attacks = pipeline.get_recent_attacks(5)
        if attacks:
            print("\nLast attacks:")
            for a in attacks:
                print(f"  • {a.src_ip}:{a.src_port} → {a.dst_ip}:{a.dst_port} ({a.confidence:.0%})")

        # Save results
        if results_saver:
            results_saver.save_final_stats(stats)
            results_saver.generate_charts()

        print("=" * 70)


if __name__ == "__main__":
    main()