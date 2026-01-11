"""
Database repository for CRUD operations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session as SQLAlchemySession

from .models import SimulationRun, Alert, Session


class Repository:
    """
    Repository for database operations.
    """

    def __init__(self, db_path: str = "sqlite:///data/simulation.db"):
        self._db_path = db_path
        Session.init(db_path)

    def close(self):
        """Close database connection."""
        Session.close()

    def create_simulation_run(
            self,
            data_source: str,
            speed: float = 1.0,
            batch_size: int = 100,
            max_flows: Optional[int] = None
    ) -> SimulationRun:
        """Create new simulation run."""
        session = Session.get()

        try:
            run = SimulationRun(
                data_source=data_source,
                speed=speed,
                batch_size=batch_size,
                max_flows=max_flows,
                status="running"
            )

            session.add(run)
            session.commit()
            session.refresh(run)

            run_id = run.id
        finally:
            session.close()

        return self.get_simulation_run(run_id)

    def update_simulation_run(
            self,
            run_id: int,
            status: Optional[str] = None,
            total_flows: Optional[int] = None,
            total_alerts: Optional[int] = None,
            precision: Optional[float] = None,
            recall: Optional[float] = None,
            f1: Optional[float] = None,
            accuracy: Optional[float] = None,
            latency_p50: Optional[float] = None,
            latency_p95: Optional[float] = None,
            report: Optional[Dict] = None
    ) -> SimulationRun:
        """Update simulation run."""
        session = Session.get()

        try:
            run = session.query(SimulationRun).filter(SimulationRun.id == run_id).first()

            if run:
                if status:
                    run.status = status
                    if status in ("completed", "failed"):
                        run.finished_at = datetime.utcnow()
                if total_flows is not None:
                    run.total_flows = total_flows
                if total_alerts is not None:
                    run.total_alerts = total_alerts
                if precision is not None:
                    run.precision = precision
                if recall is not None:
                    run.recall = recall
                if f1 is not None:
                    run.f1 = f1
                if accuracy is not None:
                    run.accuracy = accuracy
                if latency_p50 is not None:
                    run.latency_p50 = latency_p50
                if latency_p95 is not None:
                    run.latency_p95 = latency_p95
                if report is not None:
                    run.report = report

                session.commit()
        finally:
            session.close()

        return self.get_simulation_run(run_id)

    def get_simulation_run(self, run_id: int) -> Optional[SimulationRun]:
        """Get simulation run by ID."""
        session = Session.get()
        try:
            run = session.query(SimulationRun).filter(SimulationRun.id == run_id).first()
            if run:
                # Detach from session
                session.expunge(run)
            return run
        finally:
            session.close()

    def get_all_simulation_runs(self, limit: int = 100) -> List[SimulationRun]:
        """Get all simulation runs."""
        session = Session.get()
        try:
            runs = session.query(SimulationRun).order_by(
                SimulationRun.started_at.desc()
            ).limit(limit).all()
            for run in runs:
                session.expunge(run)
            return runs
        finally:
            session.close()

    def create_alert(
            self,
            simulation_run_id: int,
            flow_index: int,
            prediction: int,
            probability: float,
            true_label: Optional[int] = None,
            is_correct: Optional[bool] = None,
            inference_time_ms: float = 0.0,
            top_features: Optional[Dict] = None
    ) -> Alert:
        """Create alert."""
        session = Session.get()

        try:
            alert = Alert(
                simulation_run_id=simulation_run_id,
                flow_index=flow_index,
                prediction=prediction,
                probability=probability,
                true_label=true_label,
                is_correct=is_correct,
                inference_time_ms=inference_time_ms,
                top_features=top_features
            )

            session.add(alert)
            session.commit()
            alert_id = alert.id
        finally:
            session.close()

        return self.get_alert(alert_id)

    def create_alerts_batch(
            self,
            simulation_run_id: int,
            alerts_data: List[Dict]
    ) -> int:
        """Create multiple alerts in batch."""
        session = Session.get()

        try:
            alerts = [
                Alert(
                    simulation_run_id=simulation_run_id,
                    flow_index=a.get("flow_index", 0),
                    prediction=a.get("prediction", 1),
                    probability=a.get("probability", 0.0),
                    true_label=a.get("true_label"),
                    is_correct=a.get("is_correct"),
                    inference_time_ms=a.get("inference_time_ms", 0.0)
                )
                for a in alerts_data
            ]

            session.bulk_save_objects(alerts)
            session.commit()
            count = len(alerts)
        finally:
            session.close()

        return count

    def get_alert(self, alert_id: int) -> Optional[Alert]:
        """Get alert by ID."""
        session = Session.get()
        try:
            alert = session.query(Alert).filter(Alert.id == alert_id).first()
            if alert:
                session.expunge(alert)
            return alert
        finally:
            session.close()

    def get_alerts(
            self,
            simulation_run_id: int,
            limit: int = 100,
            offset: int = 0,
            only_incorrect: bool = False
    ) -> List[Alert]:
        """Get alerts for simulation run."""
        session = Session.get()

        try:
            query = session.query(Alert).filter(Alert.simulation_run_id == simulation_run_id)

            if only_incorrect:
                query = query.filter(Alert.is_correct == False)

            alerts = query.order_by(Alert.id).offset(offset).limit(limit).all()
            for alert in alerts:
                session.expunge(alert)
            return alerts
        finally:
            session.close()

    def count_alerts(self, simulation_run_id: int) -> int:
        """Count alerts for simulation run."""
        session = Session.get()
        try:
            count = session.query(Alert).filter(
                Alert.simulation_run_id == simulation_run_id
            ).count()
            return count
        finally:
            session.close()

    def delete_simulation_run(self, run_id: int) -> bool:
        """Delete simulation run and its alerts."""
        session = Session.get()
        try:
            run = session.query(SimulationRun).filter(SimulationRun.id == run_id).first()

            if run:
                session.delete(run)
                session.commit()
                return True
            return False
        finally:
            session.close()
