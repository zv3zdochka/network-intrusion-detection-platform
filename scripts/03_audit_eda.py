#!/usr/bin/env python3
"""
Audit and EDA runner script.
"""

import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_config, run_audit, run_eda, generate_report
from src.data.ingest import load_bronze_data

if __name__ == "__main__":
    config = load_config()

    print("Loading bronze data...")
    df = load_bronze_data(config)

    print("Running audit...")
    audit_results = run_audit(df, config)

    print("Running EDA...")
    eda_files = run_eda(df, config)

    print("Generating report...")
    report_path = generate_report(audit_results, eda_files, config)

    print(f"Completed successfully. Report saved to: {report_path}")
