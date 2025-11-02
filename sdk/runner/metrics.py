"""
Metrics collection and Prometheus integration.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    enabled: bool = True
    prefix: str = "uis_runner"
    registry: Optional[Any] = None  # Prometheus registry

    def __post_init__(self):
        """Initialize Prometheus registry if available."""
        if self.enabled and PROMETHEUS_AVAILABLE:
            self.registry = self.registry or CollectorRegistry()


class MetricsCollector:
    """Collects and exposes metrics for the ingestion runner."""

    def __init__(self, config: MetricsConfig):
        """Initialize metrics collector."""
        self.config = config
        self._metrics = {}
        self._timers = {}

        if not self.config.enabled:
            logger.info("Metrics collection disabled")
            return

        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, disabling metrics collection")
            self.config.enabled = False
            return

        self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        prefix = self.config.prefix

        # Job metrics
        self._metrics["jobs_total"] = Counter(
            f"{prefix}_jobs_total",
            "Total number of jobs executed",
            ["status", "provider_type", "tenant_id"],
            registry=self.config.registry
        )

        self._metrics["job_duration_seconds"] = Histogram(
            f"{prefix}_job_duration_seconds",
            "Job execution duration in seconds",
            ["provider_type", "tenant_id"],
            registry=self.config.registry
        )

        # Data metrics
        self._metrics["records_ingested_total"] = Counter(
            f"{prefix}_records_ingested_total",
            "Total number of records ingested",
            ["provider_type", "tenant_id", "sink_type"],
            registry=self.config.registry
        )

        self._metrics["bytes_ingested_total"] = Counter(
            f"{prefix}_bytes_ingested_total",
            "Total bytes ingested",
            ["provider_type", "tenant_id", "sink_type"],
            registry=self.config.registry
        )

        # Performance metrics
        self._metrics["job_throughput_records_per_second"] = Gauge(
            f"{prefix}_throughput_records_per_second",
            "Records per second throughput",
            ["provider_type", "tenant_id"],
            registry=self.config.registry
        )

        self._metrics["job_throughput_bytes_per_second"] = Gauge(
            f"{prefix}_throughput_bytes_per_second",
            "Bytes per second throughput",
            ["provider_type", "tenant_id"],
            registry=self.config.registry
        )

        # Error metrics
        self._metrics["errors_total"] = Counter(
            f"{prefix}_errors_total",
            "Total number of errors",
            ["error_type", "provider_type", "tenant_id"],
            registry=self.config.registry
        )

        self._metrics["rate_limit_hits_total"] = Counter(
            f"{prefix}_rate_limit_hits_total",
            "Total number of rate limit hits",
            ["provider_type", "tenant_id"],
            registry=self.config.registry
        )

        # Resource metrics
        self._metrics["active_jobs"] = Gauge(
            f"{prefix}_active_jobs",
            "Number of currently active jobs",
            ["tenant_id"],
            registry=self.config.registry
        )

        self._metrics["cpu_usage_percent"] = Gauge(
            f"{prefix}_cpu_usage_percent",
            "CPU usage percentage",
            registry=self.config.registry
        )

        self._metrics["memory_usage_mb"] = Gauge(
            f"{prefix}_memory_usage_mb",
            "Memory usage in MB",
            registry=self.config.registry
        )

        # Schema drift detection
        self._metrics["schema_drift_events_total"] = Counter(
            f"{prefix}_schema_drift_events_total",
            "Total number of schema drift events",
            ["provider_type", "tenant_id", "severity"],
            registry=self.config.registry
        )

        # Cost estimation
        self._metrics["estimated_cost_usd"] = Counter(
            f"{prefix}_estimated_cost_usd",
            "Estimated cost in USD cents",
            ["provider_type", "tenant_id"],
            registry=self.config.registry
        )

        logger.info(f"Initialized Prometheus metrics with prefix: {prefix}")

    @contextmanager
    def time_job(self, provider_type: str, tenant_id: str):
        """Context manager to time job execution."""
        if not self.config.enabled:
            yield
            return

        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_job_duration(duration, provider_type, tenant_id)

    def record_job_start(self, provider_type: str, tenant_id: str):
        """Record job start."""
        if not self.config.enabled:
            return

        self._metrics["active_jobs"].labels(tenant_id=tenant_id).inc()
        logger.debug(f"Job started: {provider_type} for tenant {tenant_id}")

    def record_job_completion(self, status: str, provider_type: str, tenant_id: str):
        """Record job completion."""
        if not self.config.enabled:
            return

        self._metrics["jobs_total"].labels(
            status=status,
            provider_type=provider_type,
            tenant_id=tenant_id
        ).inc()

        self._metrics["active_jobs"].labels(tenant_id=tenant_id).dec()
        logger.debug(f"Job completed: {status} for {provider_type} tenant {tenant_id}")

    def record_job_duration(self, duration_seconds: float, provider_type: str, tenant_id: str):
        """Record job duration."""
        if not self.config.enabled:
            return

        self._metrics["job_duration_seconds"].labels(
            provider_type=provider_type,
            tenant_id=tenant_id
        ).observe(duration_seconds)

        logger.debug(f"Job duration recorded: {duration_seconds:.2f}s for {provider_type}")

    def record_data_ingested(self, records: int, bytes_size: int,
                           provider_type: str, tenant_id: str, sink_type: str):
        """Record data ingestion metrics."""
        if not self.config.enabled:
            return

        self._metrics["records_ingested_total"].labels(
            provider_type=provider_type,
            tenant_id=tenant_id,
            sink_type=sink_type
        ).inc(records)

        self._metrics["bytes_ingested_total"].labels(
            provider_type=provider_type,
            tenant_id=tenant_id,
            sink_type=sink_type
        ).inc(bytes_size)

        # Update throughput gauges (simplified - in reality would track over time windows)
        if records > 0:
            estimated_rps = min(records / 60.0, 10000)  # Cap at 10k RPS
            self._metrics["job_throughput_records_per_second"].labels(
                provider_type=provider_type,
                tenant_id=tenant_id
            ).set(estimated_rps)

        if bytes_size > 0:
            estimated_bps = min(bytes_size / 60.0, 1000000000)  # Cap at 1GB/s
            self._metrics["job_throughput_bytes_per_second"].labels(
                provider_type=provider_type,
                tenant_id=tenant_id
            ).set(estimated_bps)

        logger.debug(f"Data ingested: {records} records, {bytes_size} bytes")

    def record_error(self, error_type: str, provider_type: str, tenant_id: str):
        """Record error occurrence."""
        if not self.config.enabled:
            return

        self._metrics["errors_total"].labels(
            error_type=error_type,
            provider_type=provider_type,
            tenant_id=tenant_id
        ).inc()

        logger.debug(f"Error recorded: {error_type} for {provider_type}")

    def record_rate_limit_hit(self, provider_type: str, tenant_id: str):
        """Record rate limit hit."""
        if not self.config.enabled:
            return

        self._metrics["rate_limit_hits_total"].labels(
            provider_type=provider_type,
            tenant_id=tenant_id
        ).inc()

        logger.debug(f"Rate limit hit for {provider_type}")

    def record_schema_drift(self, provider_type: str, tenant_id: str, severity: str = "warning"):
        """Record schema drift event."""
        if not self.config.enabled:
            return

        self._metrics["schema_drift_events_total"].labels(
            provider_type=provider_type,
            tenant_id=tenant_id,
            severity=severity
        ).inc()

        logger.info(f"Schema drift detected: {severity} for {provider_type}")

    def record_cost_estimate(self, cost_cents: int, provider_type: str, tenant_id: str):
        """Record estimated cost."""
        if not self.config.enabled:
            return

        self._metrics["estimated_cost_usd"].labels(
            provider_type=provider_type,
            tenant_id=tenant_id
        ).inc(cost_cents)

        logger.debug(f"Cost estimate recorded: ${cost_cents/100:.2f} for {provider_type}")

    def update_resource_metrics(self, cpu_percent: float, memory_mb: float):
        """Update resource usage metrics."""
        if not self.config.enabled:
            return

        self._metrics["cpu_usage_percent"].set(cpu_percent)
        self._metrics["memory_usage_mb"].set(memory_mb)

        logger.debug(f"Resource metrics updated: CPU {cpu_percent}%, Memory {memory_mb}MB")

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        if not self.config.enabled or not PROMETHEUS_AVAILABLE:
            return "# Metrics disabled or Prometheus not available"

        try:
            return generate_latest(self.config.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return "# Error generating metrics"

    def get_metrics_json(self) -> Dict[str, Any]:
        """Get metrics in JSON format."""
        if not self.config.enabled:
            return {"status": "disabled"}

        # For now, return basic status
        # In a full implementation, this would extract current metric values
        return {
            "status": "enabled",
            "registry_type": "prometheus",
            "metrics_count": len(self._metrics)
        }

