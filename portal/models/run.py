"""
Provider run model.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Enum as SQLEnum, Boolean, UniqueConstraint
from sqlalchemy.orm import relationship
from enum import Enum

from .base import BaseModel


class RunStatus(Enum):
    """Status of a provider run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


def _enum_values(enum_cls):
    """Return enum values for SQLAlchemy Enum type mapping."""
    return [member.value for member in enum_cls]


class ProviderRun(BaseModel):
    """Execution record for a provider run."""

    __tablename__ = "provider_runs"
    __table_args__ = (
        UniqueConstraint('run_id', name='uq_provider_runs_run_id'),
    )

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(
        Integer,
        ForeignKey("external_providers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Run identification
    run_id = Column(String(255), nullable=False, unique=True, index=True)  # UUID or similar
    run_mode = Column(String(50), default="batch", nullable=False)  # batch, micro-batch, streaming
    triggered_by = Column(String(255))  # user, schedule, api, etc.

    # Execution details
    status = Column(
        SQLEnum(RunStatus, values_callable=_enum_values, name="run_status"),
        default=RunStatus.PENDING,
        nullable=False,
        index=True,
    )
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Data metrics
    records_ingested = Column(Integer, default=0, nullable=False)
    bytes_ingested = Column(Integer, default=0, nullable=False)
    duration_ms = Column(Integer)  # Total execution time

    # Performance metrics
    throughput_records_sec = Column(Integer)
    throughput_bytes_sec = Column(Integer)
    latency_p50_ms = Column(Integer)
    latency_p95_ms = Column(Integer)
    latency_p99_ms = Column(Integer)

    # Quality metrics
    schema_drift_detected = Column(Boolean, default=False, nullable=False)
    data_quality_score = Column(Integer)  # 0-100
    validation_errors = Column(JSON)  # List of validation failures

    # Resource usage
    cpu_seconds = Column(Integer, default=0, nullable=False)
    memory_mb_peak = Column(Integer, default=0, nullable=False)
    network_bytes = Column(Integer, default=0, nullable=False)

    # Configuration used
    uis_spec = Column(JSON)  # The UIS specification used for this run
    compiler_output = Column(JSON)  # Generated job configuration

    # Error handling
    error_message = Column(Text)
    error_stack_trace = Column(Text)
    retry_count = Column(Integer, default=0, nullable=False)

    # Lineage and tracing
    trace_id = Column(String(255))  # OpenTelemetry trace ID
    parent_run_id = Column(Integer, ForeignKey("provider_runs.id", ondelete="SET NULL"))  # For shadow/canary runs

    # Cost tracking
    estimated_cost_usd = Column(Integer)  # Cost in cents

    # Audit metadata
    created_by = Column(String(255), nullable=False)
    updated_by = Column(String(255))

    # Relationships
    provider = relationship("ExternalProvider", back_populates="runs", passive_deletes=True)
    parent_run = relationship("ProviderRun", remote_side=[id], backref="child_runs")

    def __repr__(self):
        return f"<ProviderRun(id={self.id}, provider_id={self.provider_id}, status={self.status})>"

