#!/usr/bin/env python3
"""
Run simulation on test dataset.

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --speed 5 --max-flows 10000
    python scripts/run_simulation.py --source val --batch-size 200
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.inference import Predictor
from src.simulation import FlowReplay, SimulationRunner, SimulationConfig
from src.database import Repository


def load_config(config_path: str = "configs/simulation.yaml") -> dict:
    """Load simulation configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run flow simulation")
    parser.add_argument("--config", default="configs/simulation.yaml", help="Config file path")
    parser.add_argument("--source", choices=["train", "val", "test"], default="test", help="Data source")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--max-flows", type=int, default=None, help="Maximum flows to process")
    parser.add_argument("--max-duration", type=float, default=None, help="Maximum duration in seconds")
    parser.add_argument("--no-db", action="store_true", help="Disable database storage")
    parser.add_argument("--output", default=None, help="Output report path")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Paths
    artifacts_path = Path(config["paths"]["artifacts"])
    training_path = Path(config["paths"]["training_artifacts"])
    data_path = Path(config["paths"]["data"])
    reports_path = Path(config["paths"]["reports"])
    reports_path.mkdir(parents=True, exist_ok=True)

    # Load feature schema
    with open(artifacts_path / "feature_schema.json", "r") as f:
        schema = json.load(f)
    feature_cols = schema["feature_columns"]

    # Initialize predictor
    print("Loading model...")
    predictor = Predictor(
        model_path=training_path / config["model"]["name"],
        preprocessor_path=artifacts_path / "preprocessor.joblib",
        feature_schema_path=artifacts_path / "feature_schema.json",
        threshold=config["model"]["threshold"]
    )
    predictor.load()
    print(f"Model loaded: {predictor.get_model_info()['model_type']}")

    # Initialize replay
    source_file = f"{args.source}.parquet"
    replay = FlowReplay(
        data_path=data_path / source_file,
        feature_cols=feature_cols,
        batch_size=args.batch_size,
        shuffle=config["replay"]["shuffle"]
    )
    replay.load()
    print(f"Data loaded: {replay.n_samples:,} flows from {source_file}")

    # Simulation config
    sim_config = SimulationConfig(
        speed=args.speed,
        batch_size=args.batch_size,
        max_flows=args.max_flows,
        max_duration=args.max_duration,
        update_interval=config["metrics"]["update_interval"],
        verbose=not args.quiet,
        store_alerts=True
    )

    # Initialize runner
    runner = SimulationRunner(
        predictor=predictor,
        replay=replay,
        config=sim_config
    )

    # Database setup
    repo = None
    run_id = None

    if not args.no_db and config["database"]["enabled"]:
        db_path = f"sqlite:///{config['paths']['database']}"
        repo = Repository(db_path)

        sim_run = repo.create_simulation_run(
            data_source=source_file,
            speed=args.speed,
            batch_size=args.batch_size,
            max_flows=args.max_flows
        )
        run_id = sim_run.id
        print(f"Simulation run ID: {run_id}")

    # Run simulation
    try:
        report = runner.run()

        # Save alerts to database
        if repo and run_id:
            alerts = runner.get_alerts(limit=100000)
            if alerts:
                repo.create_alerts_batch(run_id, alerts)

            # Update simulation run with results
            repo.update_simulation_run(
                run_id=run_id,
                status="completed",
                total_flows=report["summary"]["total_flows"],
                total_alerts=report["summary"]["total_alerts"],
                precision=report["classification"]["precision"],
                recall=report["classification"]["recall"],
                f1=report["classification"]["f1"],
                accuracy=report["classification"]["accuracy"],
                latency_p50=report["latency"]["p50_ms"],
                latency_p95=report["latency"]["p95_ms"],
                report=report
            )

    except Exception as e:
        if repo and run_id:
            repo.update_simulation_run(run_id=run_id, status="failed")
        raise e

    # Save report
    output_path = args.output or (reports_path / f"simulation_report_{run_id or 'local'}.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {output_path}")

    # Export snapshots
    if runner.metrics.snapshots:
        snapshots_path = reports_path / f"snapshots_{run_id or 'local'}.json"
        runner.metrics.export_snapshots(str(snapshots_path))
        print(f"Snapshots saved: {snapshots_path}")


if __name__ == "__main__":
    main()
