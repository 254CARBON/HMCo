"""
Unified Ingestion Spec (UIS) 1.1 core models.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import json


class IngestionMode(str, Enum):
    """Ingestion modes supported by UIS."""
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"
    STREAMING = "streaming"
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"
    GRAPHQL_SUBSCRIPTION = "graphql_subscription"


class ProviderType(str, Enum):
    """Types of data providers."""
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"
    FILE_FTP = "file_ftp"
    DATABASE = "database"
    KAFKA = "kafka"
    S3 = "s3"


class SinkType(str, Enum):
    """Target sink types."""
    ICEBERG = "iceberg"
    CLICKHOUSE = "clickhouse"
    KAFKA = "kafka"
    PARQUET = "parquet"
    POSTGRES = "postgres"
    ELASTICSEARCH = "elasticsearch"


class TransformType(str, Enum):
    """Transform execution types."""
    WASM = "wasm"
    SPARK = "spark"
    FLINK = "flink"
    PYTHON = "python"
    SQL = "sql"


class AuthType(str, Enum):
    """Authentication types."""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    HMAC = "hmac"
    BASIC = "basic"
    BEARER = "bearer"
    NONE = "none"


class PaginationType(str, Enum):
    """Pagination types."""
    CURSOR = "cursor"
    OFFSET = "offset"
    PAGE = "page"
    NONE = "none"


class EndpointConfig(BaseModel):
    """Configuration for a single data endpoint."""

    name: str = Field(..., description="Unique name for this endpoint")
    path: str = Field(..., description="API path or endpoint identifier")
    method: str = Field(default="GET", description="HTTP method")

    # Authentication
    auth: Optional[AuthType] = Field(default=None)
    auth_config: Optional[Dict[str, Any]] = Field(default=None)

    # Headers and parameters
    headers: Optional[Dict[str, str]] = Field(default=None)
    query_params: Optional[Dict[str, Any]] = Field(default=None)
    body_template: Optional[str] = Field(default=None)

    # Pagination
    pagination: PaginationType = Field(default=PaginationType.NONE)
    pagination_config: Optional[Dict[str, Any]] = Field(default=None)

    # Response handling
    response_path: Optional[str] = Field(default=None, description="JSONPath to extract data")
    field_mapping: Optional[Dict[str, str]] = Field(default=None, description="Field name mappings")

    # Rate limiting
    rate_limit_group: Optional[str] = Field(default=None)
    rate_limit_per_second: Optional[int] = Field(default=None)

    # Quality and validation
    sample_size: Optional[int] = Field(default=100, description="Sample size for validation")
    validation_rules: Optional[Dict[str, Any]] = Field(default=None)

    # Status
    is_active: bool = Field(default=True)

    @validator('method')
    def validate_method(cls, v):
        allowed_methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']
        if v.upper() not in allowed_methods:
            raise ValueError(f'Method must be one of: {allowed_methods}')
        return v.upper()


class TransformConfig(BaseModel):
    """Configuration for data transformations."""

    name: str = Field(..., description="Transform name")
    type: TransformType = Field(..., description="Transform execution type")

    # WASM-specific config
    wasm_module: Optional[str] = Field(default=None, description="OCI reference to WASM module")
    wasm_function: Optional[str] = Field(default="transform", description="WASM function name")

    # Spark/Flink config
    spark_config: Optional[Dict[str, Any]] = Field(default=None)
    flink_config: Optional[Dict[str, Any]] = Field(default=None)

    # Python config
    python_module: Optional[str] = Field(default=None)
    python_function: Optional[str] = Field(default=None)

    # SQL config
    sql_query: Optional[str] = Field(default=None)

    # Transform parameters
    parameters: Optional[Dict[str, Any]] = Field(default=None)

    # Schema
    input_schema: Optional[Dict[str, Any]] = Field(default=None)
    output_schema: Optional[Dict[str, Any]] = Field(default=None)


class SinkConfig(BaseModel):
    """Configuration for data sinks."""

    type: SinkType = Field(..., description="Sink type")
    config: Dict[str, Any] = Field(..., description="Sink-specific configuration")

    # Iceberg-specific
    table_name: Optional[str] = Field(default=None)
    partition_by: Optional[List[str]] = Field(default=None)

    # ClickHouse-specific
    clickhouse_table: Optional[str] = Field(default=None)
    clickhouse_cluster: Optional[str] = Field(default=None)

    # Kafka-specific
    kafka_topic: Optional[str] = Field(default=None)
    kafka_key_field: Optional[str] = Field(default=None)


class SLOConfig(BaseModel):
    """Service Level Objective configuration."""

    freshness_target_minutes: Optional[int] = Field(default=None)
    accuracy_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    completeness_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    availability_target: Optional[float] = Field(default=None, ge=0, le=1)

    # Quality gates
    block_on_schema_drift: bool = Field(default=True)
    block_on_quality_drop: bool = Field(default=True)


class ProviderConfig(BaseModel):
    """Configuration for an external data provider."""

    name: str = Field(..., description="Provider name")
    display_name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(default=None)

    provider_type: ProviderType = Field(..., description="Type of provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for APIs")

    # Configuration
    config: Optional[Dict[str, Any]] = Field(default=None)
    credentials_ref: Optional[str] = Field(default=None, description="Vault secret reference")

    # Scheduling
    schedule_cron: Optional[str] = Field(default=None)
    schedule_timezone: str = Field(default="UTC")

    # Data governance
    tenant_id: str = Field(..., description="Tenant identifier")
    owner: str = Field(..., description="Data owner")
    tags: Optional[List[str]] = Field(default=None)

    # Endpoints
    endpoints: List[EndpointConfig] = Field(default_factory=list)

    # Transformations
    transforms: List[TransformConfig] = Field(default_factory=list)

    # Sinks
    sinks: List[SinkConfig] = Field(..., min_items=1, description="Target sinks")

    # Quality and SLOs
    schema_contract: Optional[Dict[str, Any]] = Field(default=None, description="JSON Schema")
    slos: Optional[SLOConfig] = Field(default=None)

    # Runtime options
    mode: IngestionMode = Field(default=IngestionMode.BATCH)
    parallelism: int = Field(default=1, ge=1)
    retry_config: Optional[Dict[str, Any]] = Field(default=None)


class UnifiedIngestionSpec(BaseModel):
    """Unified Ingestion Specification version 1.1."""

    version: str = Field(default="1.1", description="UIS version")
    name: str = Field(..., description="Name of this ingestion spec")
    description: Optional[str] = Field(default=None)

    # Provider configuration
    provider: ProviderConfig = Field(...)

    # Global settings
    global_config: Optional[Dict[str, Any]] = Field(default=None)

    # Metadata
    created_at: Optional[str] = Field(default=None)
    created_by: str = Field(..., description="Creator of this spec")

    @validator('version')
    def validate_version(cls, v):
        if v != "1.1":
            raise ValueError('Only UIS version 1.1 is supported')
        return v

    def to_yaml(self) -> str:
        """Export to YAML format."""
        import yaml

        # Convert the model to dict and handle enum serialization
        data = self.model_dump()

        # Recursively convert enum values to strings
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif hasattr(obj, 'value'):
                # This is an enum
                return obj.value
            else:
                return obj

        data = convert_enums(data)

        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        """Export to JSON format."""
        return self.json(indent=2)

    @classmethod
    def from_yaml(cls, yaml_content: str) -> 'UnifiedIngestionSpec':
        """Create from YAML content."""
        import yaml
        data = yaml.safe_load(yaml_content)
        return cls(**data)

    @classmethod
    def from_json(cls, json_content: str) -> 'UnifiedIngestionSpec':
        """Create from JSON content."""
        data = json.loads(json_content)
        return cls(**data)
