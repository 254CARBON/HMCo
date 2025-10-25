"""
SeaTunnel compiler for UIS specifications.
"""

import json
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from spec import UnifiedIngestionSpec, ProviderType, SinkType, EndpointConfig, AuthType


class SeaTunnelCompileError(Exception):
    """Exception raised when SeaTunnel compilation fails."""
    pass


class SeaTunnelCompiler:
    """Compiles UIS specifications to SeaTunnel job configurations."""

    def __init__(self, template_dir: Optional[str] = None):
        """Initialize compiler with optional template directory."""
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent / "templates"
        self._endpoint_output_tables: Dict[str, str] = {}
        self._default_sink_table: Optional[str] = None

    def compile(self, spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Compile UIS spec to SeaTunnel job configuration."""
        if spec.provider.provider_type not in [ProviderType.REST_API, ProviderType.FILE_FTP]:
            raise SeaTunnelCompileError(f"SeaTunnel compiler does not support provider type: {spec.provider.provider_type}")

        # Reset compilation state
        self._endpoint_output_tables = {}
        self._default_sink_table = None

        job_config = {
            "env": self._build_env_config(spec),
            "source": self._build_source_config(spec),
            "transform": self._build_transform_config(spec),
            "sink": self._build_sink_config(spec)
        }

        return job_config

    def _build_env_config(self, spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Build SeaTunnel environment configuration."""
        env = {
            "parallelism": spec.provider.parallelism,
            "checkpoint.interval": 10000,
            "checkpoint.timeout": 60000,
            "checkpoint.storage": "hdfs://namenode:9000/seatunnel/checkpoint"
        }

        # Add job-specific environment variables
        if spec.global_config:
            env.update(spec.global_config.get('seatunnel_env', {}))

        return env

    def _build_source_config(self, spec: UnifiedIngestionSpec) -> List[Dict[str, Any]]:
        """Build SeaTunnel source configurations."""
        sources = []

        for index, endpoint in enumerate(spec.provider.endpoints):
            if spec.provider.provider_type == ProviderType.REST_API:
                source = self._build_rest_source(spec, endpoint)
            elif spec.provider.provider_type == ProviderType.FILE_FTP:
                source = self._build_file_source(spec, endpoint)
            else:
                continue

            sources.append(source)
            table_name = source["result_table_name"]
            self._endpoint_output_tables[endpoint.name] = table_name
            if index == 0:
                self._default_sink_table = table_name

        return sources

    def _build_rest_source(self, spec: UnifiedIngestionSpec, endpoint: EndpointConfig) -> Dict[str, Any]:
        """Build REST API source configuration."""
        source = {
            "plugin_name": "Http",
            "result_table_name": f"source_{endpoint.name}",
            "url": self._build_url(spec, endpoint),
            "method": endpoint.method or "GET",
            "headers": self._build_headers(spec, endpoint),
            "params": endpoint.query_params or {}
        }

        # Add pagination configuration
        if endpoint.pagination != "none" and endpoint.pagination_config:
            source["pagination"] = self._build_pagination_config(endpoint)

        # Add format configuration
        source["format"] = "json"
        source["json_field"] = endpoint.response_path or "$.data"

        # Add rate limiting
        if endpoint.rate_limit_per_second:
            source["rate_limit"] = {
                "read_per_second": endpoint.rate_limit_per_second
            }

        return source

    def _build_file_source(self, spec: UnifiedIngestionSpec, endpoint: EndpointConfig) -> Dict[str, Any]:
        """Build file/FTP source configuration."""
        # This would handle CSV, JSON, XML files from FTP/S3/etc
        source = {
            "plugin_name": "File",
            "result_table_name": f"source_{endpoint.name}",
            "path": endpoint.path,
            "format": "csv"  # Default, could be inferred from file extension
        }

        if spec.provider.config and spec.provider.config.get('file_format'):
            source["format"] = spec.provider.config['file_format']
        if spec.provider.config:
            if spec.provider.config.get('delimiter'):
                source["delimiter"] = spec.provider.config['delimiter']
            if spec.provider.config.get('encoding'):
                source["encoding"] = spec.provider.config['encoding']

        return source

    def _build_transform_config(self, spec: UnifiedIngestionSpec) -> List[Dict[str, Any]]:
        """Build SeaTunnel transform configurations."""
        transforms = []

        # Add field mapping transform if needed
        for endpoint in spec.provider.endpoints:
            if endpoint.field_mapping:
                source_table = self._endpoint_output_tables.get(endpoint.name, f"source_{endpoint.name}")
                mapped_table = f"mapped_{endpoint.name}"
                transform = {
                    "plugin_name": "FieldMapper",
                    "source_table_name": source_table,
                    "result_table_name": mapped_table,
                    "field_mapper": endpoint.field_mapping
                }
                transforms.append(transform)
                self._endpoint_output_tables[endpoint.name] = mapped_table
                if spec.provider.endpoints and endpoint.name == spec.provider.endpoints[0].name:
                    self._default_sink_table = mapped_table

        # Add schema validation transform
        if spec.provider.schema_contract and spec.provider.endpoints:
            first_endpoint = spec.provider.endpoints[0]
            source_table = self._endpoint_output_tables.get(first_endpoint.name, f"source_{first_endpoint.name}")
            validated_table = "validated"
            transform = {
                "plugin_name": "SchemaValidator",
                "source_table_name": source_table,
                "result_table_name": validated_table,
                "schema": spec.provider.schema_contract
            }
            transforms.append(transform)
            self._endpoint_output_tables[first_endpoint.name] = validated_table
            self._default_sink_table = validated_table

        return transforms

    def _build_sink_config(self, spec: UnifiedIngestionSpec) -> List[Dict[str, Any]]:
        """Build SeaTunnel sink configurations."""
        sinks = []

        for sink_config in spec.provider.sinks:
            source_table_name = self._resolve_sink_source_table(spec, sink_config)
            if sink_config.type == SinkType.ICEBERG:
                sink = self._build_iceberg_sink(sink_config, source_table_name)
            elif sink_config.type == SinkType.CLICKHOUSE:
                sink = self._build_clickhouse_sink(sink_config, source_table_name)
            elif sink_config.type == SinkType.KAFKA:
                sink = self._build_kafka_sink(sink_config, source_table_name)
            else:
                continue

            sinks.append(sink)

        return sinks

    def _resolve_sink_source_table(self, spec: UnifiedIngestionSpec, sink_config) -> str:
        """Determine which table the sink should read from."""
        if sink_config.config and sink_config.config.get("source_table_name"):
            return sink_config.config["source_table_name"]

        if self._default_sink_table:
            return self._default_sink_table

        if spec.provider.endpoints:
            first_endpoint = spec.provider.endpoints[0]
            return self._endpoint_output_tables.get(first_endpoint.name, f"source_{first_endpoint.name}")

        return "source"

    def _build_iceberg_sink(self, sink_config, source_table_name: str) -> Dict[str, Any]:
        """Build Iceberg sink configuration."""
        config = sink_config.config or {}

        return {
            "plugin_name": "Iceberg",
            "source_table_name": source_table_name,
            "result_table_name": sink_config.table_name or "iceberg_table",
            "catalog_name": config.get("catalog", "hive_prod"),
            "namespace": config.get("namespace", "default"),
            "table": sink_config.table_name or "ingested_data",
            "warehouse": config.get("warehouse"),
            "hadoop_conf_path": config.get("hadoop_conf", "/etc/hadoop/conf"),
            "primary_key": config.get("primary_key", ""),
            "save_mode": "append"
        }

    def _build_clickhouse_sink(self, sink_config, source_table_name: str) -> Dict[str, Any]:
        """Build ClickHouse sink configuration."""
        config = sink_config.config or {}

        return {
            "plugin_name": "Clickhouse",
            "source_table_name": source_table_name,
            "result_table_name": sink_config.clickhouse_table or "clickhouse_table",
            "host": config.get("host", "clickhouse"),
            "port": config.get("port", 8123),
            "database": config.get("database", "default"),
            "table": sink_config.clickhouse_table or "ingested_data",
            "username": config.get("username", "default"),
            "password": config.get("password", ""),
            "bulk_size": config.get("bulk_size", 20000)
        }

    def _build_kafka_sink(self, sink_config, source_table_name: str) -> Dict[str, Any]:
        """Build Kafka sink configuration."""
        config = sink_config.config or {}

        return {
            "plugin_name": "Kafka",
            "source_table_name": source_table_name,
            "result_table_name": sink_config.kafka_topic or "kafka_topic",
            "bootstrap_servers": config.get("bootstrap_servers", "kafka:9092"),
            "topic": sink_config.kafka_topic or "ingested_data",
            "key_field": sink_config.kafka_key_field,
            "format": "json"
        }

    def _build_url(self, spec: UnifiedIngestionSpec, endpoint: EndpointConfig) -> str:
        """Build full URL for REST endpoint."""
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
        if endpoint.auth == AuthType.API_KEY and endpoint.auth_config:
            header_name = endpoint.auth_config.get('header_name', 'Authorization')
            header_format = endpoint.auth_config.get('header_format')
            if not header_format:
                header_format = "Bearer {{api_key}}" if header_name.lower() == "authorization" else "{{api_key}}"

            # Respect explicitly configured headers if present
            if header_name not in headers:
                headers[header_name] = header_format

        elif endpoint.auth == AuthType.BASIC and endpoint.auth_config:
            headers['Authorization'] = "Basic {{basic_auth}}"

        # Add global headers from provider config
        if spec.provider.config and spec.provider.config.get('headers'):
            headers.update(spec.provider.config['headers'])

        return headers

    def _build_pagination_config(self, endpoint: EndpointConfig) -> Dict[str, Any]:
        """Build pagination configuration for SeaTunnel."""
        pagination_config = endpoint.pagination_config or {}

        if endpoint.pagination == "cursor":
            return {
                "type": "cursor",
                "cursor_field": pagination_config.get("cursor_param", "cursor"),
                "page_size": pagination_config.get("page_size", 100),
                "max_pages": pagination_config.get("max_pages", 1000)
            }
        elif endpoint.pagination == "offset":
            return {
                "type": "offset",
                "offset_field": pagination_config.get("offset_param", "offset"),
                "limit_field": pagination_config.get("limit_param", "limit"),
                "page_size": pagination_config.get("page_size", 100),
                "max_pages": pagination_config.get("max_pages", 1000)
            }
        elif endpoint.pagination == "page":
            return {
                "type": "page",
                "page_field": pagination_config.get("page_param", "page"),
                "page_size_field": pagination_config.get("page_size_param", "page_size"),
                "page_size": pagination_config.get("page_size", 100),
                "max_pages": pagination_config.get("max_pages", 1000)
            }

        return {"type": "none"}

    def compile_to_json(self, spec: UnifiedIngestionSpec) -> str:
        """Compile UIS spec to SeaTunnel JSON configuration."""
        config = self.compile(spec)
        return json.dumps(config, indent=2)

    def compile_to_file(self, spec: UnifiedIngestionSpec, output_path: str) -> None:
        """Compile UIS spec and save to file."""
        config_json = self.compile_to_json(spec)

        with open(output_path, 'w') as f:
            f.write(config_json)

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate generated SeaTunnel configuration."""
        errors = []

        # Check required fields
        if 'source' not in config or not config['source']:
            errors.append("No source configuration found")

        if 'sink' not in config or not config['sink']:
            errors.append("No sink configuration found")

        # Validate source configurations
        for source in config.get('source', []):
            if 'plugin_name' not in source:
                errors.append("Source missing plugin_name")
            if 'result_table_name' not in source:
                errors.append("Source missing result_table_name")

        # Validate sink configurations
        for sink in config.get('sink', []):
            if 'plugin_name' not in sink:
                errors.append("Sink missing plugin_name")
            if 'source_table_name' not in sink:
                errors.append("Sink missing source_table_name")

        return errors
