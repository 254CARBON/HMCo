"""
Flink compiler for UIS specifications (streaming processing).
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from spec import UnifiedIngestionSpec, ProviderType, SinkType, EndpointConfig, IngestionMode


class FlinkCompileError(Exception):
    """Exception raised when Flink compilation fails."""
    pass


class FlinkCompiler:
    """Compiles UIS specifications to Flink job configurations for streaming processing."""

    def __init__(self, template_dir: Optional[str] = None):
        """Initialize compiler with optional template directory."""
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent / "templates"

    def compile(self, spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Compile UIS spec to Flink job configuration."""
        if spec.provider.mode not in [IngestionMode.STREAMING, IngestionMode.WEBSOCKET, IngestionMode.WEBHOOK]:
            raise FlinkCompileError(f"Flink compiler only supports streaming modes, got: {spec.provider.mode}")

        sources = self._build_sources_config(spec)
        transforms = self._build_transforms_config(spec)
        sinks = self._build_sinks_config(spec)
        pipelines = self._build_pipeline_spec(spec, sources, transforms, sinks)

        job_config = {
            "job_type": "streaming",
            "flink_config": self._build_flink_config(spec),
            "sources": sources,
            "transforms": transforms,
            "sinks": sinks,
            "pipelines": pipelines,
            "state_management": self._build_state_config(spec),
            "monitoring": self._build_monitoring_config(spec)
        }

        return job_config

    def _build_flink_config(self, spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Build Flink configuration."""
        config = spec.provider.config or {}

        flink_config = {
            # Basic Flink configuration
            "taskmanager.memory.process.size": config.get("taskmanager_memory", "4g"),
            "taskmanager.numberOfTaskSlots": config.get("task_slots", "4"),
            "parallelism.default": spec.provider.parallelism,

            # State management
            "state.backend": "rocksdb",
            "state.checkpoints.dir": f"s3://flink-checkpoints/{spec.name}/",
            "state.savepoints.dir": f"s3://flink-savepoints/{spec.name}/",
            "state.checkpoint-storage": "filesystem",
            "state.checkpoints.num-retained": "10",
            "state.checkpointing.mode": "EXACTLY_ONCE",
            "state.checkpointing.interval": "60000",  # 1 minute
            "state.checkpointing.timeout": "300000",  # 5 minutes

            # Streaming configuration
            "execution.checkpointing.interval": "60000",
            "execution.checkpointing.timeout": "300000",
            "execution.checkpointing.mode": "EXACTLY_ONCE",
            "execution.checkpointing.externalized-checkpoint-retention": "RETAIN_ON_CANCELLATION",

            # Buffer configuration
            "taskmanager.network.memory.fraction": "0.1",
            "taskmanager.network.memory.max": "1gb",
            "taskmanager.memory.segment-size": "32mb",

            # S3 configuration
            "s3.endpoint": config.get("s3_endpoint", "http://minio:9000"),
            "s3.access-key": "{{minio_access_key}}",
            "s3.secret-key": "{{minio_secret_key}}",
            "s3.path.style.access": "true",

            # Kafka configuration
            "kafka.bootstrap.servers": config.get("kafka_bootstrap_servers", "kafka:9092"),
            "kafka.consumer.group-id": f"uis-{spec.name}",
            "kafka.producer.transactional-id-prefix": f"uis-{spec.name}",

            # WebSocket configuration
            "websocket.reconnect.interval": "30000",  # 30 seconds
            "websocket.max.reconnect.attempts": "10",

            # Performance tuning
            "taskmanager.memory.network.fraction": "0.1",
            "taskmanager.memory.managed.fraction": "0.4",
            "taskmanager.memory.network.min": "64mb",
            "taskmanager.memory.network.max": "1gb",

            # Monitoring
            "metrics.reporter.prometheus.class": "org.apache.flink.metrics.prometheus.PrometheusReporter",
            "metrics.reporter.prometheus.port": "9999"
        }

        # Add user-defined Flink configuration
        if config.get("flink_config"):
            flink_config.update(config["flink_config"])

        return flink_config

    def _build_sources_config(self, spec: UnifiedIngestionSpec) -> List[Dict[str, Any]]:
        """Build Flink source configurations."""
        sources = []

        # Handle provider-level sources (like CDC, Kafka topics from config)
        if spec.provider.provider_type == ProviderType.DATABASE:
            source = self._build_cdc_source(spec, None)
            sources.append(source)
        elif spec.provider.provider_type == ProviderType.KAFKA and not spec.provider.endpoints:
            # Handle Kafka topics from provider config
            source = self._build_kafka_source_from_config(spec)
            if source:
                sources.append(source)
        else:
            # Handle endpoint-level sources
            for endpoint in spec.provider.endpoints:
                if spec.provider.provider_type == ProviderType.WEBSOCKET:
                    source = self._build_websocket_source(spec, endpoint)
                elif spec.provider.provider_type == ProviderType.WEBHOOK:
                    source = self._build_webhook_source(spec, endpoint)
                elif spec.provider.provider_type == ProviderType.KAFKA:
                    source = self._build_kafka_source(spec, endpoint)
                else:
                    continue

                sources.append(source)

        return sources

    def _build_websocket_source(self, spec: UnifiedIngestionSpec, endpoint: EndpointConfig) -> Dict[str, Any]:
        """Build WebSocket source configuration."""
        return {
            "type": "websocket",
            "name": f"websocket_{endpoint.name}",
            "url": self._build_url(spec, endpoint),
            "protocol": endpoint.query_params.get("protocol", "ws") if endpoint.query_params else "ws",
            "headers": self._build_headers(spec, endpoint),
            "format": "json",
            "endpoint_name": endpoint.name,
            "reconnect_config": {
                "max_attempts": 10,
                "initial_delay_ms": 1000,
                "max_delay_ms": 30000,
                "backoff_multiplier": 2.0
            },
            "heartbeat": {
                "interval_ms": 30000,
                "timeout_ms": 10000
            },
            "buffer_config": {
                "max_size": 10000,
                "flush_interval_ms": 1000
            }
        }

    def _build_webhook_source(self, spec: UnifiedIngestionSpec, endpoint: EndpointConfig) -> Dict[str, Any]:
        """Build webhook source configuration."""
        return {
            "type": "webhook",
            "name": f"webhook_{endpoint.name}",
            "path": endpoint.path,
            "method": endpoint.method or "POST",
            "headers": self._build_headers(spec, endpoint),
            "format": "json",
            "endpoint_name": endpoint.name,
            "validation": {
                "signature_required": bool(endpoint.auth_config and (
                    endpoint.auth_config.get("signature") or
                    endpoint.auth_config.get("signature_header") or
                    (endpoint.auth and "hmac" in endpoint.auth.value.lower())
                )),
                "signature_header": endpoint.auth_config.get("signature_header") if endpoint.auth_config else None,
                "signature_algorithm": endpoint.auth_config.get("signature_algorithm", "HMAC-SHA256") if endpoint.auth_config else None
            },
            "rate_limiting": {
                "enabled": bool(endpoint.rate_limit_per_second),
                "requests_per_second": endpoint.rate_limit_per_second or 100
            }
        }

    def _build_kafka_source(self, spec: UnifiedIngestionSpec, endpoint: EndpointConfig) -> Dict[str, Any]:
        """Build Kafka source configuration."""
        config = spec.provider.config or {}

        return {
            "type": "kafka",
            "name": f"kafka_{endpoint.name}",
            "bootstrap_servers": config.get("kafka_bootstrap_servers", "kafka:9092"),
            "topics": config.get("kafka_topics", [endpoint.path]),
            "group_id": f"uis-{spec.name}-{endpoint.name}",
            "starting_offsets": config.get("starting_offsets", "latest"),
            "format": "json",
            "schema_registry": config.get("schema_registry_url"),
            "properties": {
                "auto.offset.reset": config.get("auto_offset_reset", "latest"),
                "enable.auto.commit": "false",
                "isolation.level": "read_committed"
            },
            "endpoint_name": endpoint.name
        }

    def _build_cdc_source(self, spec: UnifiedIngestionSpec, endpoint: Optional[EndpointConfig]) -> Dict[str, Any]:
        """Build CDC (Change Data Capture) source configuration."""
        config = spec.provider.config or {}

        return {
            "type": "cdc",
            "name": f"cdc_{spec.provider.name}",
            "connector": config.get("cdc_connector", "debezium"),
            "database": config.get("database_type", "postgres"),
            "hostname": config.get("db_host"),
            "port": config.get("db_port", 5432),
            "database_name": config.get("db_name"),
            "schema_name": config.get("schema_name", "public"),
            "table_names": config.get("table_names", []),
            "username": "{{db_username}}",
            "password": "{{db_password}}",
            "format": "json",
            "debezium_config": {
                "snapshot.mode": "initial",
                "snapshot.locking.mode": "none",
                "include.schema.changes": "false"
            },
            "endpoint_name": endpoint.name if endpoint else spec.provider.name
        }

    def _build_kafka_source_from_config(self, spec: UnifiedIngestionSpec) -> Optional[Dict[str, Any]]:
        """Build Kafka source from provider configuration."""
        config = spec.provider.config or {}

        topics = config.get("kafka_topics", [])
        if not topics:
            return None

        return {
            "type": "kafka",
            "name": f"kafka_{spec.provider.name}",
            "bootstrap_servers": config.get("kafka_bootstrap_servers", "kafka:9092"),
            "topics": topics,
            "group_id": f"uis-{spec.name}",
            "starting_offsets": config.get("starting_offsets", "latest"),
            "format": "json",
            "schema_registry": config.get("schema_registry_url"),
            "properties": {
                "auto.offset.reset": config.get("auto_offset_reset", "latest"),
                "enable.auto.commit": "false",
                "isolation.level": "read_committed"
            },
            "endpoint_name": spec.provider.name
        }

    def _build_transforms_config(self, spec: UnifiedIngestionSpec) -> List[Dict[str, Any]]:
        """Build Flink transform configurations."""
        transforms = []

        # Schema validation transform
        if spec.provider.schema_contract:
            transforms.append({
                "type": "schema_validation",
                "name": "validate_schema",
                "schema": spec.provider.schema_contract,
                "strategy": "FORWARD",  # Continue processing even if validation fails
                "applies_to_endpoints": ["*"]
            })

        # Field mapping transformation
        for endpoint in spec.provider.endpoints:
            if endpoint.field_mapping:
                transforms.append({
                    "type": "field_mapping",
                    "name": f"map_{endpoint.name}",
                    "mapping": endpoint.field_mapping,
                    "drop_unmapped": True,
                    "applies_to_endpoints": [endpoint.name]
                })

        # Custom Flink SQL transforms
        for transform in spec.provider.transforms:
            if transform.type == "flink":
                transforms.append({
                    "type": "flink_sql",
                    "name": transform.name,
                    "sql": transform.sql_query,
                    "parameters": transform.parameters or {},
                    "applies_to_endpoints": (transform.parameters or {}).get("target_endpoints", ["*"])
                })

        # Add watermarking for event time processing
        transforms.append({
            "type": "watermark",
            "name": "add_watermark",
            "timestamp_column": "event_time",
            "max_out_of_orderness_ms": 5000,
            "applies_to_endpoints": ["*"]
        })

        return transforms

    def _build_sinks_config(self, spec: UnifiedIngestionSpec) -> List[Dict[str, Any]]:
        """Build Flink sink configurations."""
        sinks = []

        for sink_config in spec.provider.sinks:
            if sink_config.type == SinkType.KAFKA:
                sink = self._build_kafka_sink(sink_config)
            elif sink_config.type == SinkType.ICEBERG:
                sink = self._build_iceberg_sink(sink_config)
            elif sink_config.type == SinkType.CLICKHOUSE:
                sink = self._build_clickhouse_sink(sink_config)
            else:
                continue

            sinks.append(sink)

        return sinks

    def _build_kafka_sink(self, sink_config) -> Dict[str, Any]:
        """Build Kafka sink configuration."""
        config = sink_config.config or {}

        return {
            "type": "kafka",
            "name": f"kafka_{sink_config.kafka_topic or 'topic'}",
            "bootstrap_servers": config.get("bootstrap_servers", "kafka:9092"),
            "topic": sink_config.kafka_topic or "processed_events",
            "key_field": sink_config.kafka_key_field,
            "format": "json",
            "transactional": True,
            "config": config,
            "properties": {
                "transaction.timeout.ms": "900000",  # 15 minutes
                "batch.size": "16384",
                "linger.ms": "10",
                "compression.type": "snappy"
            }
        }

    def _build_iceberg_sink(self, sink_config) -> Dict[str, Any]:
        """Build Iceberg sink configuration."""
        config = sink_config.config or {}

        return {
            "type": "iceberg",
            "name": f"iceberg_{sink_config.table_name or 'table'}",
            "catalog_name": config.get("catalog", "hive_prod"),
            "database": config.get("database", "default"),
            "table": sink_config.table_name or "streaming_data",
            "format": "parquet",
            "write_mode": "append",
            "options": {
                "write.wap.enabled": "true",
                "write.metadata.delete-after-commit.enabled": "true",
                "write.metadata.previous-versions-max": "5"
            }
        }

    def _build_clickhouse_sink(self, sink_config) -> Dict[str, Any]:
        """Build ClickHouse sink configuration."""
        config = sink_config.config or {}

        return {
            "type": "clickhouse",
            "name": f"clickhouse_{sink_config.clickhouse_table or 'table'}",
            "host": config.get("host", "clickhouse"),
            "port": config.get("port", 8123),
            "database": config.get("database", "default"),
            "table": sink_config.clickhouse_table or "streaming_data",
            "username": "{{clickhouse_username}}",
            "password": "{{clickhouse_password}}",
            "batch_size": 10000,
            "flush_interval_ms": 5000
        }

    def _build_state_config(self, spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Build state management configuration."""
        return {
            "backend": "rocksdb",
            "checkpoint_config": {
                "interval_ms": 60000,
                "timeout_ms": 300000,
                "mode": "EXACTLY_ONCE",
                "num_retained": 10,
                "externalized_retention": "RETAIN_ON_CANCELLATION"
            },
            "savepoint_config": {
                "dir": f"s3://flink-savepoints/{spec.name}/",
                "retention": "RETAIN_ON_CANCELLATION"
            },
            "state_ttl": {
                "enabled": True,
                "time_ms": 86400000,  # 24 hours
                "cleanup_ms": 3600000   # 1 hour
            }
        }

    def _build_monitoring_config(self, spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Build monitoring and observability configuration."""
        return {
            "metrics_enabled": True,
            "metrics_prefix": f"uis.{spec.name}",
            "prometheus": {
                "enabled": True,
                "port": 9999,
                "path": "/metrics"
            },
            "custom_metrics": [
                "source_throughput",
                "sink_throughput",
                "processing_latency_ms",
                "checkpoint_duration_ms",
                "state_size_bytes",
                "error_rate"
            ],
            "alerts": {
                "processing_delay_threshold_ms": 300000,  # 5 minutes
                "error_rate_threshold": 0.01,  # 1%
                "checkpoint_failure_threshold": 3
            }
        }

    def _build_pipeline_spec(
        self,
        spec: UnifiedIngestionSpec,
        sources: List[Dict[str, Any]],
        transforms: List[Dict[str, Any]],
        sinks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build pipeline specifications linking sources to Kafka sinks."""
        if not sources or not sinks:
            return []

        kafka_sinks = [sink for sink in sinks if sink.get("type") == "kafka" and sink.get("topic")]
        if not kafka_sinks:
            return []

        transform_index = self._index_transforms_by_endpoint(transforms)
        pipelines: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        for sink in kafka_sinks:
            target_endpoints = self._determine_sink_targets(sink)
            candidate_sources = self._select_candidate_sources(sources, target_endpoints)

            for source in candidate_sources:
                pipeline_id = f"{source['name']}__{sink['name']}"
                if pipeline_id in seen_ids:
                    continue
                seen_ids.add(pipeline_id)

                transforms_for_source = self._resolve_transforms_for_source(
                    transform_index,
                    source.get("endpoint_name")
                )

                pipeline = {
                    "id": pipeline_id,
                    "name": f"{source['name']} -> {sink['name']}",
                    "mode": spec.provider.mode.value,
                    "description": self._describe_pipeline(source, sink),
                    "source": {
                        "name": source["name"],
                        "type": source["type"],
                        "endpoint": source.get("endpoint_name"),
                        "details": {
                            key: value for key, value in source.items()
                            if key in {"url", "connector", "database", "table_names", "topics"}
                        }
                    },
                    "sink": {
                        "name": sink["name"],
                        "type": sink["type"],
                        "topic": sink.get("topic"),
                        "bootstrap_servers": sink.get("bootstrap_servers"),
                        "key_field": sink.get("key_field")
                    },
                    "transforms": transforms_for_source,
                    "delivery_guarantee": "exactly_once" if sink.get("transactional") else "at_least_once",
                    "parallelism": spec.provider.parallelism
                }

                pipelines.append(pipeline)

        return pipelines

    def _select_candidate_sources(
        self,
        sources: List[Dict[str, Any]],
        target_endpoints: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Select sources that should feed a sink based on configured targets."""
        if target_endpoints:
            target_set = set(target_endpoints)
            selected = [s for s in sources if s.get("endpoint_name") in target_set]
            if selected:
                return selected

        # Default: use streaming-friendly sources
        streaming_source_types = {"websocket", "cdc", "kafka", "webhook"}
        selected = [s for s in sources if s.get("type") in streaming_source_types]
        return selected or sources

    def _index_transforms_by_endpoint(self, transforms: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Index transforms by the endpoints they apply to."""
        index: Dict[str, List[str]] = defaultdict(list)
        for transform in transforms:
            targets = transform.get("applies_to_endpoints") or ["*"]
            if isinstance(targets, str):
                targets = [targets]

            for target in targets:
                index[target].append(transform["name"])

        return index

    def _resolve_transforms_for_source(
        self,
        transform_index: Dict[str, List[str]],
        endpoint: Optional[str]
    ) -> List[str]:
        """Resolve ordered list of transforms for a given source endpoint."""
        ordered: List[str] = []
        for key in (endpoint, "*"):
            if not key:
                continue
            for transform_name in transform_index.get(key, []):
                if transform_name not in ordered:
                    ordered.append(transform_name)

        # Ensure global transforms always considered
        for transform_name in transform_index.get("*", []):
            if transform_name not in ordered:
                ordered.append(transform_name)

        return ordered

    def _determine_sink_targets(self, sink: Dict[str, Any]) -> Optional[List[str]]:
        """Determine explicit endpoint targets defined on a sink."""
        config = sink.get("config") or {}
        possible_keys = [
            "target_endpoints",
            "target_endpoint",
            "endpoints",
            "endpoint",
            "source_endpoints",
            "source_endpoint"
        ]

        for key in possible_keys:
            if key in config and config[key]:
                value = config[key]
                if isinstance(value, str):
                    return [value]
                if isinstance(value, list):
                    return value

        return None

    def _describe_pipeline(self, source: Dict[str, Any], sink: Dict[str, Any]) -> str:
        """Generate a human-readable description for a pipeline."""
        source_desc = f"{source['type']} source '{source['name']}'"
        topic = sink.get("topic")
        if topic:
            sink_desc = f"Kafka topic '{topic}'"
        else:
            sink_desc = f"{sink['type']} sink '{sink['name']}'"
        return f"Route {source_desc} to {sink_desc}."

    def _build_url(self, spec: UnifiedIngestionSpec, endpoint: EndpointConfig) -> str:
        """Build full URL for WebSocket/webhook endpoint."""
        base_url = spec.provider.base_url or ""
        if not base_url.endswith('/'):
            base_url += '/'

        path = endpoint.path
        if path.startswith('/'):
            path = path[1:]

        return base_url + path

    def _build_headers(self, spec: UnifiedIngestionSpec, endpoint: EndpointConfig) -> Dict[str, str]:
        """Build HTTP headers for the request."""
        headers = {}

        # Add static headers from endpoint
        if endpoint.headers:
            headers.update(endpoint.headers)

        # Add authentication headers
        if endpoint.auth and endpoint.auth_config:
            if endpoint.auth.value == "api_key":
                if endpoint.auth_config.get('header_name'):
                    headers[endpoint.auth_config['header_name']] = "{{api_key}}"
                else:
                    headers['Authorization'] = "Bearer {{api_key}}"

        return headers

    def compile_to_json(self, spec: UnifiedIngestionSpec) -> str:
        """Compile UIS spec to Flink JSON configuration."""
        config = self.compile(spec)
        return json.dumps(config, indent=2)

    def compile_to_flink_sql(self, spec: UnifiedIngestionSpec) -> str:
        """Compile UIS spec to Flink SQL statements."""
        sql_statements = []

        # Create source tables
        for source in self.compile(spec)["sources"]:
            if source["type"] == "kafka":
                sql = self._generate_kafka_source_sql(source)
            elif source["type"] == "websocket":
                sql = self._generate_websocket_source_sql(source)
            else:
                continue
            sql_statements.append(sql)

        # Create sink tables
        for sink in self.compile(spec)["sinks"]:
            if sink["type"] == "iceberg":
                sql = self._generate_iceberg_sink_sql(sink)
            elif sink["type"] == "clickhouse":
                sql = self._generate_clickhouse_sink_sql(sink)
            else:
                continue
            sql_statements.append(sql)

        # Create streaming job
        sql_statements.append(self._generate_streaming_job_sql(spec))

        return ";\n\n".join(sql_statements) + ";"

    def _generate_kafka_source_sql(self, source: Dict[str, Any]) -> str:
        """Generate Flink SQL for Kafka source."""
        return f"""
CREATE TABLE {source['name']} (
    event_time TIMESTAMP(3),
    data STRING,
    metadata ROW<timestamp TIMESTAMP(3), source STRING>
) WITH (
    'connector' = 'kafka',
    'topic' = '{",".join(source['topics'])}',
    'properties.bootstrap.servers' = '{source['bootstrap_servers']}',
    'properties.group.id' = '{source['group_id']}',
    'format' = 'json',
    'scan.startup.mode' = '{source['starting_offsets']}'
)
"""

    def _generate_websocket_source_sql(self, source: Dict[str, Any]) -> str:
        """Generate Flink SQL for WebSocket source."""
        return f"""
CREATE TABLE {source['name']} (
    event_time TIMESTAMP(3),
    data STRING,
    metadata ROW<timestamp TIMESTAMP(3), source STRING>
) WITH (
    'connector' = 'websocket',
    'url' = '{source['url']}',
    'format' = 'json'
)
"""

    def _generate_iceberg_sink_sql(self, sink: Dict[str, Any]) -> str:
        """Generate Flink SQL for Iceberg sink."""
        return f"""
CREATE TABLE {sink['name']} (
    event_time TIMESTAMP(3),
    data STRING,
    metadata ROW<timestamp TIMESTAMP(3), source STRING>
) WITH (
    'connector' = 'iceberg',
    'catalog-name' = '{sink['catalog_name']}',
    'database' = '{sink['database']}',
    'table' = '{sink['table']}',
    'format' = 'parquet',
    'write.wap.enabled' = 'true'
)
"""

    def _generate_clickhouse_sink_sql(self, sink: Dict[str, Any]) -> str:
        """Generate Flink SQL for ClickHouse sink."""
        return f"""
CREATE TABLE {sink['name']} (
    event_time TIMESTAMP(3),
    data STRING,
    metadata ROW<timestamp TIMESTAMP(3), source STRING>
) WITH (
    'connector' = 'clickhouse',
    'url' = 'clickhouse://{sink['host']}:{sink['port']}/{sink['database']}',
    'table-name' = '{sink['table']}',
    'sink.batch-size' = '10000'
)
"""

    def _generate_streaming_job_sql(self, spec: UnifiedIngestionSpec) -> str:
        """Generate main streaming job SQL."""
        source_names = [s['name'] for s in self.compile(spec)["sources"]]
        sink_names = [s['name'] for s in self.compile(spec)["sinks"]]

        source_table = source_names[0] if source_names else "unknown_source"
        sink_table = sink_names[0] if sink_names else "unknown_sink"

        return f"""
INSERT INTO {sink_table}
SELECT
    event_time,
    data,
    ROW(event_time, 'flink') as metadata
FROM {source_table}
"""

    def compile_to_flink_args(self, spec: UnifiedIngestionSpec) -> List[str]:
        """Compile UIS spec to Flink job arguments."""
        config = self.compile(spec)

        # Build Flink run arguments
        args = [
            "--jobmanager", "flink-jobmanager:9081",
            "--taskmanager", "flink-taskmanager",
            "--parallelism", str(spec.provider.parallelism),
            "--detached",
            "--job-name", f"uis-{spec.name}",
            "--job-config", json.dumps(config)
        ]

        # Add Flink configuration as --conf arguments
        for key, value in config["flink_config"].items():
            args.extend(["-D", f"{key}={value}"])

        return args

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate generated Flink configuration."""
        errors = []

        # Check required fields
        required_keys = ["job_type", "sources", "sinks"]
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required configuration: {key}")

        # Validate sources
        for source in config.get('sources', []):
            if 'type' not in source or 'name' not in source:
                errors.append(f"Invalid source configuration: {source}")

            # Source-specific validation
            if source["type"] == "websocket" and "url" not in source:
                errors.append("WebSocket source missing URL")
            elif source["type"] == "kafka" and "bootstrap_servers" not in source:
                errors.append("Kafka source missing bootstrap servers")

        # Validate sinks
        for sink in config.get('sinks', []):
            if 'type' not in sink or 'name' not in sink:
                errors.append(f"Invalid sink configuration: {sink}")

        # Validate Flink configuration
        flink_config = config.get('flink_config', {})
        required_flink_keys = [
            'state.backend',
            'state.checkpoints.dir',
            'execution.checkpointing.mode'
        ]

        for key in required_flink_keys:
            if key not in flink_config:
                errors.append(f"Missing required Flink configuration: {key}")

        if not flink_config:
            errors.append("Missing required configuration: flink_config")

        # Validate pipelines if present
        pipelines = config.get("pipelines", [])
        if pipelines:
            source_names = {source.get("name") for source in config.get("sources", [])}
            sink_names = {sink.get("name") for sink in config.get("sinks", [])}

            for pipeline in pipelines:
                if not isinstance(pipeline, dict):
                    errors.append(f"Invalid pipeline configuration: {pipeline}")
                    continue

                source = pipeline.get("source", {})
                sink = pipeline.get("sink", {})

                source_name = source.get("name")
                sink_name = sink.get("name")

                if not source_name:
                    errors.append("Pipeline missing source name")
                elif source_name not in source_names:
                    errors.append(f"Pipeline references unknown source: {source_name}")

                if not sink_name:
                    errors.append("Pipeline missing sink name")
                elif sink_name not in sink_names:
                    errors.append(f"Pipeline references unknown sink: {sink_name}")

                if sink.get("type") == "kafka" and not sink.get("topic"):
                    errors.append(f"Kafka pipeline sink missing topic: {sink_name}")

        return errors
