"""
HMCo Data Platform Ingestion Runner.

The ingestion runner executes data ingestion jobs based on Unified Ingestion Spec (UIS)
configurations. It supports multiple execution engines (SeaTunnel, Spark, Flink) and
provides comprehensive monitoring, metrics, and observability.
"""

from .runner import IngestionRunner, RunnerConfig
from .job_executor import JobExecutor
from .secret_manager import SecretManager
from .metrics import MetricsCollector
from .tracing import Tracer

__version__ = "1.0.0"

__all__ = [
    'IngestionRunner',
    'RunnerConfig',
    'JobExecutor',
    'SecretManager',
    'MetricsCollector',
    'Tracer'
]


