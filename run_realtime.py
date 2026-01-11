#!/usr/bin/env python3
"""
Скрипт для запуска анализа трафика в реальном времени
"""

import argparse
import signal
import sys
import time
import platform
from pathlib import Path
from datetime import datetime

IS_WINDOWS = platform.system() == 'Windows'

# Добавляем путь к модулю
sys.path.insert(0, '.')

from realtime import RealtimePipeline
from realtime.capture import PacketCapture
from realtime.config import PipelineConfig
from realtime.utils import setup_logging, save_results_json, protocol_name

# Глобальная переменная для pipeline
pipeline = None


def signal_handler(sig, frame):
    """Обработчик сигналов для graceful shutdown"""
    print("\n\n" + "=" * 60)
    print("Received shutdown signal, stopping...")
    global pipeline
    if pipeline:
        pipeline.stop()
    sys.exit(0)


def print_attack_alert(result):
    """Выводит алерт об атаке"""
    print()
    print("!" * 70)
    print("!!! ATTACK DETECTED !!!")
    print("!" * 70)
    print(f"  Time: {result.timestamp}")
    print(f"  Source: {result.src_ip}:{result.src_port}")
    print(f"  Target: {result.dst_ip}:{result.dst_port}")
    print(f"  Protocol: {protocol_name(result.protocol)}")
    print(f"  Attack Type: {result.class_name}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Probability: {result.probabilities.get('probability', 'N/A')}")
    print(f"  Duration: {result.duration:.3f}s")
    print(f"  Packets: {result.total_packets}")
    print(f"  Bytes: {result.total_bytes}")
    print("!" * 70)
    print()


def print_flow_result(result):
    """Выводит результат анализа потока"""
    if result.is_attack:
        status = "[ATTACK]"
    else:
        status = "[BENIGN]"

    proto = protocol_name(result.protocol)

    print(f"{datetime.now().strftime('%H:%M:%S')} {status:9} "
          f"{result.src_ip:>15}:{result.src_port:<5} -> "
          f"{result.dst_ip:>15}:{result.dst_port:<5} "
          f"{proto:4} | {result.class_name:12} "
          f"({result.confidence:5.1%}) | "
          f"{result.total_packets:4} pkts")


def check_admin():
    """Проверяет права администратора"""
    if IS_WINDOWS:
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False
    else:
        import os
        return os.geteuid() == 0


def find_artifacts(model_path: Path):
    """Ищет связанные артефакты для модели"""
    model_dir = model_path.parent
    project_root = Path('.')

    # Ищем preprocessor
    preprocessor_paths = [
        model_dir / 'preprocessor.joblib',
        model_dir / 'preprocessor.pkl',
        project_root / 'artifacts' / 'preprocessor.joblib',
        project_root / 'artifacts' / 'preprocessor.pkl',
    ]
    preprocessor = None
    for p in preprocessor_paths:
        if p.exists():
            preprocessor = p
            break

    # Ищем feature schema
    schema_paths = [
        model_dir / 'feature_schema.json',
        model_dir / 'features.json',
        project_root / 'artifacts' / 'feature_schema.json',
        project_root / 'artifacts' / 'features.json',
    ]
    schema = None
    for p in schema_paths:
        if p.exists():
            schema = p
            break

    return preprocessor, schema


def main():
    global pipeline

    print()
    print("=" * 70)
    print("       REAL-TIME NETWORK TRAFFIC ANALYZER")
    print(f"       Platform: {platform.system()} {platform.release()}")
    print("=" * 70)

    # Проверка прав
    is_admin = check_admin()
    if is_admin:
        print("[OK] Running as Administrator")
    else:
        print("[!!] WARNING: Not running as Administrator!")
        if IS_WINDOWS:
            print("     Right-click PowerShell and 'Run as administrator'")
        print()
        input("Press Enter to continue anyway, or Ctrl+C to exit...")

    parser = argparse.ArgumentParser(
        description='Real-time Network Traffic Analyzer'
    )

    parser.add_argument('-i', '--interface', type=str, default=None,
                        help='Network interface to capture')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Path to trained model (.joblib or .pkl)')
    parser.add_argument('-p', '--preprocessor', type=str, default=None,
                        help='Path to preprocessor (.joblib)')
    parser.add_argument('-s', '--schema', type=str, default=None,
                        help='Path to feature schema (.json)')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='Detection threshold (default: 0.5)')
    parser.add_argument('-f', '--filter', type=str, default='ip',
                        help='BPF filter (default: ip)')
    parser.add_argument('-d', '--duration', type=int, default=0,
                        help='Duration in seconds (0 = infinite)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file for results (JSON)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output (show all flows)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet mode (only show attacks)')
    parser.add_argument('--list-interfaces', action='store_true',
                        help='List available interfaces and exit')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')

    args = parser.parse_args()

    # Список интерфейсов
    if args.list_interfaces:
        print("\nAvailable network interfaces:")
        print("-" * 60)
        interfaces = PacketCapture.list_interfaces()
        for iface in interfaces:
            print(f"\nName: {iface.get('name', 'unknown')}")
            if iface.get('description'):
                print(f"  Description: {iface['description']}")
            if iface.get('ips'):
                print(f"  IPs: {', '.join(iface['ips'])}")
        return

    # Интерфейс
    if not args.interface:
        # Показываем основные интерфейсы для выбора
        print("\nSelect network interface:")
        print("-" * 50)

        interfaces = PacketCapture.list_interfaces()
        main_interfaces = []
        for iface in interfaces:
            name = iface.get('name', '')
            ips = iface.get('ips', [])
            real_ips = [ip for ip in ips
                        if not ip.startswith('169.254')
                        and not ip.startswith('fe80')
                        and ip != '127.0.0.1'
                        and ip != '::1']
            if real_ips and 'Npcap' not in name and 'WFP' not in name and 'Filter' not in name:
                main_interfaces.append(iface)

        for i, iface in enumerate(main_interfaces):
            name = iface.get('name', 'unknown')
            desc = iface.get('description', '')[:50]
            ips = [ip for ip in iface.get('ips', [])
                   if not ip.startswith('169.254') and not ip.startswith('fe80')]
            print(f"  [{i}] {name}")
            print(f"      {desc}")
            print(f"      IP: {', '.join(ips[:2]) if ips else 'No IP'}")
            print()

        if not main_interfaces:
            print("No suitable interfaces found!")
            return

        try:
            choice = input(f"Enter number [0]: ").strip()
            idx = int(choice) if choice else 0
            args.interface = main_interfaces[idx]['name']
        except (ValueError, IndexError, KeyboardInterrupt):
            args.interface = main_interfaces[0]['name']

    # Находим артефакты модели
    model_path = None
    preprocessor_path = args.preprocessor
    schema_path = args.schema

    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path}")
            return

        # Автоматически ищем артефакты если не указаны
        if not preprocessor_path or not schema_path:
            auto_prep, auto_schema = find_artifacts(model_path)
            if not preprocessor_path and auto_prep:
                preprocessor_path = str(auto_prep)
            if not schema_path and auto_schema:
                schema_path = str(auto_schema)

    # Настройка логирования
    logger = setup_logging(level=20)

    # Выводим конфигурацию
    print()
    print("-" * 70)
    print("Configuration:")
    print(f"  Interface: {args.interface}")
    print(f"  Filter: {args.filter}")
    print(f"  Model: {model_path or 'DUMMY (testing)'}")
    if model_path:
        print(f"  Preprocessor: {preprocessor_path or 'Not found!'}")
        print(f"  Feature Schema: {schema_path or 'Not found!'}")
        print(f"  Threshold: {args.threshold}")
    print(f"  Duration: {'infinite' if args.duration == 0 else f'{args.duration}s'}")
    print("-" * 70)

    # Callbacks
    callbacks = {}
    if not args.quiet:
        callbacks['on_attack_detected'] = print_attack_alert
    if args.verbose:
        callbacks['on_flow_analyzed'] = print_flow_result

    # Создаём pipeline
    pipeline = RealtimePipeline(
        interface=args.interface,
        model_path=str(model_path) if model_path else None,
        preprocessor_path=preprocessor_path,
        feature_schema_path=schema_path,
        bpf_filter=args.filter,
        debug=args.debug,
        threshold=args.threshold,
        **callbacks
    )

    # Если нет модели, используем заглушку
    if not model_path:
        from realtime.analyzer import create_dummy_analyzer
        pipeline.analyzer = create_dummy_analyzer(attack_ratio=0.1)
        print()
        print("[!!] No model provided - using DUMMY analyzer for testing")
        print("     Results are RANDOM and not real predictions!")
    else:
        # Проверяем что модель загружена
        if not pipeline.analyzer.predictor or not pipeline.analyzer.predictor.is_loaded:
            print()
            print("[ERROR] Failed to load model!")
            return

        model_info = pipeline.analyzer.get_model_info()
        print()
        print(f"[OK] Model loaded: {model_info.get('model_type', 'Unknown')}")
        print(f"     Features: {model_info.get('n_features', 'Unknown')}")

    # Обработчик сигналов
    signal.signal(signal.SIGINT, signal_handler)
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, signal_handler)

    print()
    print("Starting packet capture...")
    print("Press Ctrl+C to stop")
    print()

    if args.verbose:
        print("TIME     STATUS    SOURCE                  DESTINATION             PROTO  CLASS        CONF")
        print("-" * 100)

    pipeline.start()

    # Ждём запуска
    time.sleep(1.5)

    if not pipeline.is_running():
        print()
        print("[ERROR] Failed to start pipeline!")
        return

    print("[OK] Capture started successfully!")
    print()

    try:
        elapsed = 0

        while args.duration == 0 or elapsed < args.duration:
            time.sleep(1)
            elapsed += 1

            # Статистика каждые 3 секунды
            if elapsed % 3 == 0 and not args.verbose:
                summary = pipeline.get_summary()
                stats_line = (
                    f"\r[{elapsed:5d}s] "
                    f"Packets: {summary['total_packets']:>8,} | "
                    f"Flows: {summary['total_flows_analyzed']:>6,} | "
                    f"Attacks: {summary['total_attacks']:>4} | "
                    f"Rate: {summary['packets_per_second']:>7.1f} pps"
                )
                print(stats_line, end='', flush=True)

    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print("Stopping capture...")

    finally:
        pipeline.stop()

        # Финальная статистика
        print()
        print("=" * 70)
        print("                    FINAL STATISTICS")
        print("=" * 70)

        stats = pipeline.get_stats()

        runtime = stats['capture'].get('uptime_seconds', 0)
        print(f"  Runtime: {runtime:.1f} seconds")
        print()
        print(f"  Packets captured:    {stats['capture'].get('total_packets', 0):>12,}")
        print(f"  Flows analyzed:      {stats['pipeline']['flows_analyzed']:>12,}")
        print(f"  ATTACKS DETECTED:    {stats['pipeline']['attacks_detected']:>12}")

        # Последние атаки
        attacks = pipeline.get_recent_attacks(5)
        if attacks:
            print()
            print("-" * 70)
            print("Last detected attacks:")
            for attack in attacks:
                print(f"  {attack.src_ip}:{attack.src_port} -> "
                      f"{attack.dst_ip}:{attack.dst_port} | "
                      f"{attack.class_name} ({attack.confidence:.0%})")

        # Сохранение
        if args.output:
            results = pipeline.get_recent_results()
            save_results_json(results, args.output)
            print(f"\nResults saved to: {args.output}")

        print()
        print("=" * 70)


if __name__ == "__main__":
    main()