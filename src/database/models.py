"""
SQLAlchemy ORM models for simulation data.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Boolean, DateTime, ForeignKey, Text, JSON
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class SimulationRun(Base):
    """Simulation run record."""
    __tablename__ = "simulation_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="running")

    data_source = Column(String(255))
    speed = Column(Float, default=1.0)
    batch_size = Column(Integer, default=100)
    max_flows = Column(Integer, nullable=True)

    total_flows = Column(Integer, default=0)
    total_alerts = Column(Integer, default=0)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    latency_p50 = Column(Float, nullable=True)
    latency_p95 = Column(Float, nullable=True)

    report = Column(JSON, nullable=True)

    alerts = relationship("Alert", back_populates="simulation_run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<SimulationRun(id={self.id}, status={self.status})>"


class Alert(Base):
    """Alert record."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_run_id = Column(Integer, ForeignKey("simulation_runs.id"), nullable=False)

    timestamp = Column(DateTime, default=datetime.utcnow)
    flow_index = Column(Integer)
    prediction = Column(Integer)
    probability = Column(Float)
    true_label = Column(Integer, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    inference_time_ms = Column(Float)

    top_features = Column(JSON, nullable=True)

    simulation_run = relationship("SimulationRun", back_populates="alerts")

    def __repr__(self):
        return f"<Alert(id={self.id}, prediction={self.prediction}, prob={self.probability:.3f})>"

    def to_dict(self):
        return {
            "id": self.id,
            "simulation_run_id": self.simulation_run_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "flow_index": self.flow_index,
            "prediction": self.prediction,
            "probability": self.probability,
            "true_label": self.true_label,
            "is_correct": self.is_correct,
            "inference_time_ms": self.inference_time_ms
        }


class Session:
    """Database session manager."""

    _engine = None
    _SessionLocal = None
    _database_url = None

    @classmethod
    def init(cls, database_url: str = "sqlite:///data/simulation.db", echo: bool = False):
        """Initialize database connection."""
        cls._database_url = database_url

        # For SQLite, use StaticPool to avoid threading issues
        if "sqlite" in database_url:
            cls._engine = create_engine(
                database_url,
                echo=echo,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )
        else:
            cls._engine = create_engine(database_url, echo=echo)

        cls._SessionLocal = sessionmaker(bind=cls._engine)
        Base.metadata.create_all(cls._engine)

    @classmethod
    def get(cls):
        """Get new session."""
        if cls._SessionLocal is None:
            cls.init()
        return cls._SessionLocal()

    @classmethod
    def close(cls):
        """Close engine and dispose connections."""
        if cls._engine:
            cls._engine.dispose()
            cls._engine = None
            cls._SessionLocal = None
            cls._database_url = None
