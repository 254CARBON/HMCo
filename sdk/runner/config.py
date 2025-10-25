"""
Runner configuration and settings.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ExecutionEngine(str, Enum):
    """Supported execution engines."""
    SEATUNNEL = "seatunnel"
    SPARK = "spark"
    FLINK = "flink"


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class RunnerConfig:
    """Configuration for the ingestion runner."""

    # Basic configuration
    runner_id: str = field(default_factory=lambda: os.getenv("RUNNER_ID", "default-runner"))
    tenant_id: str = field(default_factory=lambda: os.getenv("TENANT_ID", "default"))

    # Execution settings
    max_concurrent_jobs: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_JOBS", "10")))
    job_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("JOB_TIMEOUT_SECONDS", "3600")))
    simulate_missing_engines: bool = field(
        default_factory=lambda: os.getenv("SIMULATE_MISSING_ENGINES", "true").lower() == "true"
    )
    simulation_delay_seconds: float = field(
        default_factory=lambda: float(os.getenv("SIMULATION_DELAY_SECONDS", "0.25"))
    )
    simulation_records_per_endpoint: int = field(
        default_factory=lambda: int(os.getenv("SIMULATION_RECORDS_PER_ENDPOINT", "100"))
    )
    simulation_avg_record_size_bytes: int = field(
        default_factory=lambda: int(os.getenv("SIMULATION_AVG_RECORD_SIZE_BYTES", "512"))
    )

    # Vault integration
    vault_enabled: bool = field(default_factory=lambda: os.getenv("VAULT_ENABLED", "false").lower() == "true")
    vault_address: str = field(default_factory=lambda: os.getenv("VAULT_ADDRESS", "http://vault:8200"))
    vault_token: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_TOKEN"))
    vault_mount_path: str = field(default_factory=lambda: os.getenv("VAULT_MOUNT_PATH", "secret"))
    vault_role_id: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_ROLE_ID"))
    vault_secret_id: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_SECRET_ID"))

    # Kubernetes integration
    kubernetes_enabled: bool = field(default_factory=lambda: os.getenv("KUBERNETES_ENABLED", "true").lower() == "true")
    kubernetes_namespace: str = field(default_factory=lambda: os.getenv("KUBERNETES_NAMESPACE", "data-platform"))

    # Metrics and monitoring
    metrics_enabled: bool = field(default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true")
    metrics_port: int = field(default_factory=lambda: int(os.getenv("METRICS_PORT", "9090")))
    metrics_path: str = field(default_factory=lambda: os.getenv("METRICS_PATH", "/metrics"))

    # Tracing
    tracing_enabled: bool = field(default_factory=lambda: os.getenv("TRACING_ENABLED", "true").lower() == "true")
    tracing_endpoint: str = field(default_factory=lambda: os.getenv("TRACING_ENDPOINT", "http://jaeger:14268/api/traces"))
    tracing_service_name: str = field(default_factory=lambda: os.getenv("TRACING_SERVICE_NAME", "ingestion-runner"))

    # Logging
    log_level: LogLevel = field(default_factory=lambda: LogLevel(os.getenv("LOG_LEVEL", "info").lower()))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))

    # Storage configuration
    checkpoint_storage_path: str = field(default_factory=lambda: os.getenv("CHECKPOINT_STORAGE_PATH", "s3://checkpoints/"))
    temp_storage_path: str = field(default_factory=lambda: os.getenv("TEMP_STORAGE_PATH", "/tmp/runner"))

    # Execution engines
    supported_engines: List[ExecutionEngine] = field(default_factory=lambda: [
        ExecutionEngine.SEATUNNEL,
        ExecutionEngine.SPARK,
        ExecutionEngine.FLINK
    ])

    # Engine-specific configuration
    engine_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation."""
        if self.vault_enabled and not (self.vault_token or (self.vault_role_id and self.vault_secret_id)):
            raise ValueError("Vault enabled but no authentication configured (token or role_id/secret_id)")

        if self.max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be at least 1")

        if self.job_timeout_seconds < 60:
            raise ValueError("job_timeout_seconds must be at least 60 seconds")

        if self.simulation_delay_seconds < 0:
            raise ValueError("simulation_delay_seconds must be non-negative")

        if self.simulation_records_per_endpoint < 1:
            raise ValueError("simulation_records_per_endpoint must be at least 1")

        if self.simulation_avg_record_size_bytes < 1:
            raise ValueError("simulation_avg_record_size_bytes must be at least 1")

    @classmethod
    def from_env(cls) -> 'RunnerConfig':
        """Create configuration from environment variables."""
        return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RunnerConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, LogLevel):
                result[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], ExecutionEngine):
                result[key] = [e.value for e in value]
            else:
                result[key] = value
        return result
