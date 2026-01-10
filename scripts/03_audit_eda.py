#!/usr/bin/env python3
"""
Скрипт аудита и EDA
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_config, run_audit, run_eda, generate_report
from src.data.ingest import load_bronze_data

if __name__ == "__main__":
    config = load_config()

    print("Loading bronze data...")
    df = load_bronze_data(config)

    print("\nRunning audit...")
    audit_results = run_audit(df, config)

    print("\nRunning EDA...")
    eda_files = run_eda(df, config)

    print("\nGenerating report...")
    report_path = generate_report(audit_results, eda_files, config)

    print(f"\n✅ All done! Report: {report_path}")