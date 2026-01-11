#!/usr/bin/env python3
"""
Скрипт для запуска анализа трафика в реальном времени
Работает на Windows и Linux
"""

import argparse
import signal
import sys
import time
import platform
from datetime import datetime

IS_WINDOWS = platform.system() == 'Windows'

# Добавляем путь к модулю
sys.path.insert(0, '.')

from realtime import RealtimePipeline
from realtime.capture import PacketCapture
from realtime.config import PipelineConfig
from realtime.utils import setup_logging, save_results_json

# Глобальная переменная для pipeline
pipeline = None


def signal_handler(sig, frame):
    """Обработчик сигналов для graceful shutdown"""
    print("\n\nReceived shutdown signal...")
    global pipeline
    if pipeline:
        pipeline.stop()
    sys.exit(0)


def print_attack_alert(result):
    """Выводит алерт об атаке"""
    print(f"\n{'=' * 60}")
    print(f"🚨 ATTACK DETECTED at {result.timestamp}")
    print(f"   Source: {result.src_ip}:{result.src_port}")
    print(f"   Target: {result.dst_ip}:{result.dst_port}")
    print(f"   Type: {result.class_name}")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Duration: {result.duration:.2f}s")
    print(f"   Packets: {result.total_packets}")
    print(f"{'=' * 60}\n")


def print_flow_result(result):
    """Выводит результат анализа потока"""
    status = "⚠️ ATTACK" if result.is_attack else "✅ BENIGN"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {status} | "
          f"{result.src_ip}:{result.src_port} -> {result.dst_ip}:{result.dst_port} | "
          f"{result.class_name} ({result.confidence:.1%}) | "
          f"{result.total_packets} pkts")


def select_interface_interactive() -> str:
    """Интерактивный выбор интерфейса"""
    interfaces = PacketCapture.list_interfaces()

    if not interfaces:
        print("ERROR: No network interfaces found!")
        if IS_WINDOWS:
            print("\nMake sure Npcap is installed: https://npcap.com/#download")
            print("Install it with 'WinPcap API-compatible Mode' enabled!")
        sys.exit(1)

    print("\nAvailable network interfaces:")
    print("-" * 60)

    for i, iface in enumerate(interfaces):
        name = iface.get('name', 'unknown')
        desc = iface.get('description', '')
        ips = iface.get('ips', [])

        # Фильтруем автоматические IP
        real_ips = [ip for ip in ips if not ip.startswith('169.254') and not ip.startswith('fe80')]

        print(f"  [{i}] {name}")
        if desc:
            print(f"      {desc}")
        if real_ips:
            print(f"      IPs: {', '.join(real_ips[:2])}")
        print()

    # Автовыбор если один интерфейс с реальным IP
    default_idx = 0
    for i, iface in enumerate(interfaces):
        ips = iface.get('ips', [])
        real_ips = [ip for ip in ips if not ip.startswith('169.254') and not ip.startswith('fe80')]
        if real_ips:
            default_idx = i
            break

    try:
        choice = input(f"Select interface number [{default_idx}]: ").strip()
        if choice == '':
            idx = default_idx
        else:
            idx = int(choice)

        if 0 <= idx < len(interfaces):
            return interfaces[idx]['name']
        else:
            print(f"Invalid choice, using default: {interfaces[default_idx]['name']}")
            return interfaces[default_idx]['name']
    except (ValueError, KeyboardInterrupt):
        print(f"\nUsing default: {interfaces[default_idx]['name']}")
        return interfaces[default_idx]['name']


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


def main():
    global pipeline

    print("=" * 60)
    print("Real-time Network Traffic Analyzer")
    print(f"Platform: {platform.system()} {platform.release()}")
    print("=" * 60)

    # Проверка прав
    if not check_admin():
        print("\n⚠️  WARNING: Not running as Administrator!")
        if IS_WINDOWS:
            print("   Packet capture may not work.")
            print("   Right-click and 'Run as administrator'")
        else:
            print("   Run with: sudo python run_realtime.py")
        print()

    parser = argparse.ArgumentParser(
        description='Real-time Network Traffic Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (will ask for interface)
  python run_realtime.py

  # Specify interface
  python run_realtime.py -i "Ethernet"

  # With model
  python run_realtime.py -i "Wi-Fi" -m ./models/model.pt

  # With config file  
  python run_realtime.py -c config.json

  # Filter specific traffic
  python run_realtime.py -i "Ethernet" -f "tcp port 80 or tcp port 443"

  # List interfaces only
  python run_realtime.py --list-interfaces
        """
    )

    parser.add_argument('-i', '--interface', type=str, default=None,
                        help='Network interface to capture')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Path to trained model file (.pt)')
    parser.add_argument('-s', '--scaler', type=str, default=None,
                        help='Path to scaler file (.pkl)')
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to config JSON file')
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
            if iface.get('mac'):
                print(f"  MAC: {iface['mac']}")
        return

    # Загружаем конфигурацию
    if args.config:
        config = PipelineConfig.from_json(args.config)
    else:
        config = PipelineConfig()

    # Выбор интерфейса
    if args.interface:
        config.capture.interface = args.interface
    elif config.capture.interface is None:
        config.capture.interface = select_interface_interactive()

    # Переопределяем из CLI
    if args.filter:
        config.capture.bpf_filter = args.filter
    if args.model:
        config.analyzer.model_path = args.model
    if args.scaler:
        config.analyzer.scaler_path = args.scaler

    # Настройка логирования
    logger = setup_logging(
        log_file=config.logging.file,
        level=getattr(__import__('logging'), config.logging.level)
    )

    # Callbacks
    callbacks = {}

    if not args.quiet:
        callbacks['on_attack_detected'] = print_attack_alert

    if args.verbose and not args.quiet:
        callbacks['on_flow_analyzed'] = print_flow_result

    # Создаём pipeline
    pipeline = RealtimePipeline(
        interface=config.capture.interface,
        model_path=config.analyzer.model_path,
        scaler_path=config.analyzer.scaler_path,
        bpf_filter=config.capture.bpf_filter,
        flow_timeout=config.flow.timeout_seconds,
        analysis_interval=config.analysis_interval,
        **callbacks
    )

    # Если нет модели, используем заглушку
    if not config.analyzer.model_path:
        from realtime.analyzer import create_dummy_analyzer
        pipeline.analyzer = create_dummy_analyzer()
        print("\n⚠️  No model provided, using dummy analyzer for testing")

    # Обработчик сигналов
    signal.signal(signal.SIGINT, signal_handler)
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, signal_handler)

    # Запуск
    print(f"\n{'=' * 60}")
    print("Configuration:")
    print(f"  Interface: {config.capture.interface}")
    print(f"  Filter: {config.capture.bpf_filter}")
    print(f"  Model: {config.analyzer.model_path or 'dummy (testing)'}")
    print(f"  Duration: {'infinite' if args.duration == 0 else f'{args.duration}s'}")
    print(f"{'=' * 60}")
    print("\nStarting capture... (Press Ctrl+C to stop)\n")

    pipeline.start()

    # Ждём запуска
    time.sleep(1)

    if not pipeline.is_running():
        print("\n❌ Failed to start pipeline!")
        print("   Check that Npcap is installed and you have admin rights.")
        return

    try:
        elapsed = 0
        last_stats_time = 0

        while args.duration == 0 or elapsed < args.duration:
            time.sleep(1)
            elapsed += 1

            # Показываем статистику каждые 5 секунд
            if elapsed - last_stats_time >= 5 and not args.quiet:
                summary = pipeline.get_summary()
                stats_line = (
                    f"\r[{elapsed:4d}s] "
                    f"Packets: {summary['total_packets']:,} | "
                    f"Flows: {summary['total_flows_analyzed']:,} | "
                    f"Attacks: {summary['total_attacks']:,} | "
                    f"Rate: {summary['packets_per_second']:.1f} pps   "
                )
                print(stats_line, end='', flush=True)
                last_stats_time = elapsed

    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")

    finally:
        pipeline.stop()

        # Сохраняем результаты
        if args.output:
            results = pipeline.get_recent_results()
            save_results_json(results, args.output)
            print(f"\n💾 Results saved to {args.output}")

        # Финальная статистика
        print("\n" + "=" * 60)
        print("📊 Final Statistics")
        print("=" * 60)
        stats = pipeline.get_stats()
        print(f"  Total packets processed: {stats['pipeline']['packets_processed']:,}")
        print(f"  Total flows analyzed: {stats['pipeline']['flows_analyzed']:,}")
        print(f"  Total attacks detected: {stats['pipeline']['attacks_detected']:,}")

        if stats['capture'].get('parse_errors', 0) > 0:
            print(f"  Capture errors: {stats['capture']['parse_errors']}")

        attacks = pipeline.get_recent_attacks(10)
        if attacks:
            print("\n🚨 Last detected attacks:")
            for attack in attacks:
                print(f"  • {attack.src_ip} -> {attack.dst_ip}: {attack.class_name} "
                      f"({attack.confidence:.0%})")


if __name__ == "__main__":
    main()