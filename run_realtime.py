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
    print("!" * 60)
    print("!!! ATTACK DETECTED !!!")
    print("!" * 60)
    print(f"  Time: {result.timestamp}")
    print(f"  Source: {result.src_ip}:{result.src_port}")
    print(f"  Target: {result.dst_ip}:{result.dst_port}")
    print(f"  Protocol: {protocol_name(result.protocol)}")
    print(f"  Attack Type: {result.class_name}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Duration: {result.duration:.3f}s")
    print(f"  Packets: {result.total_packets}")
    print(f"  Bytes: {result.total_bytes}")
    print("!" * 60)
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
          f"{result.total_packets:4} pkts, {result.total_bytes:6} bytes")


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
            print("     Packet capture may not work.")
            print("     Right-click PowerShell and 'Run as administrator'")
        else:
            print("     Run with: sudo python run_realtime.py")
        print()
        input("Press Enter to continue anyway, or Ctrl+C to exit...")

    parser = argparse.ArgumentParser(
        description='Real-time Network Traffic Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter
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

    # Интерфейс
    if args.interface:
        config.capture.interface = args.interface

    if not config.capture.interface:
        # Показываем основные интерфейсы для выбора
        print("\nSelect network interface:")
        print("-" * 50)

        interfaces = PacketCapture.list_interfaces()
        # Фильтруем только реальные интерфейсы
        main_interfaces = []
        for iface in interfaces:
            name = iface.get('name', '')
            ips = iface.get('ips', [])
            # Только интерфейсы с реальными IP
            real_ips = [ip for ip in ips
                        if not ip.startswith('169.254')
                        and not ip.startswith('fe80')
                        and ip != '127.0.0.1'
                        and ip != '::1']
            if real_ips and 'Npcap' not in name and 'WFP' not in name:
                main_interfaces.append(iface)

        for i, iface in enumerate(main_interfaces):
            name = iface.get('name', 'unknown')
            desc = iface.get('description', '')[:40]
            ips = [ip for ip in iface.get('ips', [])
                   if not ip.startswith('169.254') and not ip.startswith('fe80')]
            ip_str = ', '.join(ips[:2]) if ips else 'No IP'
            print(f"  [{i}] {name}")
            print(f"      {desc}")
            print(f"      IP: {ip_str}")
            print()

        if not main_interfaces:
            print("No suitable interfaces found!")
            return

        try:
            choice = input(f"Enter number [0]: ").strip()
            idx = int(choice) if choice else 0
            config.capture.interface = main_interfaces[idx]['name']
        except (ValueError, IndexError, KeyboardInterrupt):
            config.capture.interface = main_interfaces[0]['name']

    # Переопределяем из CLI
    if args.filter:
        config.capture.bpf_filter = args.filter
    if args.model:
        config.analyzer.model_path = args.model
    if args.scaler:
        config.analyzer.scaler_path = args.scaler

    # Настройка логирования
    logger = setup_logging(level=20)  # INFO

    # Callbacks
    callbacks = {}

    if not args.quiet:
        callbacks['on_attack_detected'] = print_attack_alert

    if args.verbose:
        callbacks['on_flow_analyzed'] = print_flow_result

    # Создаём pipeline
    print()
    print("-" * 70)
    print(f"Interface: {config.capture.interface}")
    print(f"Filter: {config.capture.bpf_filter}")
    print(f"Model: {config.analyzer.model_path or 'DUMMY (testing mode)'}")
    print(f"Duration: {'infinite' if args.duration == 0 else f'{args.duration}s'}")
    print("-" * 70)

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
        print()
        print("[!!] No model provided - using DUMMY analyzer for testing")
        print("     Results are RANDOM and not real predictions!")
        print()

    # Обработчик сигналов
    signal.signal(signal.SIGINT, signal_handler)
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, signal_handler)

    print()
    print("Starting packet capture...")
    print("Press Ctrl+C to stop")
    print()

    if args.verbose:
        print("TIME     STATUS    SOURCE                  DESTINATION             PROTO  CLASS        CONF    PACKETS")
        print("-" * 120)

    pipeline.start()

    # Ждём запуска
    time.sleep(1.5)

    if not pipeline.is_running():
        print()
        print("[ERROR] Failed to start pipeline!")
        print("        Possible causes:")
        print("        1. Not running as Administrator")
        print("        2. Npcap not installed properly")
        print("        3. Interface name is incorrect")
        print()
        print("        Try: python run_realtime.py --list-interfaces")
        return

    print("[OK] Capture started successfully!")
    print()

    try:
        elapsed = 0
        last_stats = None

        while args.duration == 0 or elapsed < args.duration:
            time.sleep(1)
            elapsed += 1

            # Показываем статистику каждые 3 секунды
            if elapsed % 3 == 0 and not args.verbose:
                summary = pipeline.get_summary()

                # Очищаем предыдущую строку и выводим новую
                stats_line = (
                    f"[{elapsed:5d}s] "
                    f"Packets: {summary['total_packets']:>8,} | "
                    f"Flows: {summary['total_flows_analyzed']:>6,} | "
                    f"Attacks: {summary['total_attacks']:>4} | "
                    f"Active: {summary['active_flows']:>4} | "
                    f"Rate: {summary['packets_per_second']:>7.1f} pps"
                )
                print(f"\r{stats_line}", end='', flush=True)
                last_stats = summary

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
        print(f"  Packets processed:   {stats['pipeline']['packets_processed']:>12,}")
        print(f"  Packets dropped:     {stats['capture'].get('dropped_packets', 0):>12,}")
        print(f"  Parse errors:        {stats['capture'].get('parse_errors', 0):>12,}")
        print()
        print(f"  Flows analyzed:      {stats['pipeline']['flows_analyzed']:>12,}")
        print(f"  Active flows:        {stats['aggregator'].get('active_flows', 0):>12,}")
        print()
        print(f"  ATTACKS DETECTED:    {stats['pipeline']['attacks_detected']:>12}")

        if runtime > 0:
            pps = stats['capture'].get('total_packets', 0) / runtime
            print()
            print(f"  Average rate: {pps:.1f} packets/second")

        # Показываем последние атаки
        attacks = pipeline.get_recent_attacks(5)
        if attacks:
            print()
            print("-" * 70)
            print("Last detected attacks:")
            print("-" * 70)
            for attack in attacks:
                print(f"  {attack.timestamp[:19]} | "
                      f"{attack.src_ip}:{attack.src_port} -> "
                      f"{attack.dst_ip}:{attack.dst_port} | "
                      f"{attack.class_name} ({attack.confidence:.0%})")

        # Сохраняем результаты
        if args.output:
            results = pipeline.get_recent_results()
            save_results_json(results, args.output)
            print()
            print(f"Results saved to: {args.output}")

        print()
        print("=" * 70)
        print("Done!")
        print("=" * 70)


if __name__ == "__main__":
    main()