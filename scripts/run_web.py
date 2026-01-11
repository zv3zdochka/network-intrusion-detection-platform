#!/usr/bin/env python3
"""
Script to run the web application
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run Network IDS Web Interface')
    parser.add_argument('-H', '--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('-p', '--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--public', action='store_true', help='Allow external connections (0.0.0.0)')

    args = parser.parse_args()

    host = '0.0.0.0' if args.public else args.host

    from web.app import run_app
    run_app(host=host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
