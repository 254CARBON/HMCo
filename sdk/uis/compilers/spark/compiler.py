"""
Spark compiler for UIS specifications (micro-batch processing).
"""

import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from spec import UnifiedIngestionSpec, ProviderType, SinkType, EndpointConfig, IngestionMode


class SparkCompileError(Exception):
    """Exception raised when Spark compilation fails."""
    pass


@dataclass
class SparkJobArtifacts:
    """Artifacts produced when compiling a UIS spec for Spark execution."""

    spark_submit_args: List[str]
    spark_session_conf: Dict[str, Any]
    job_config: Dict[str, Any]
    job_config_path: str


class SparkCompiler:
    """Compiles UIS specifications to Spark job configurations for micro-batch processing."""

    def __init__(self, template_dir: Optional[str] = None):
        """Initialize compiler with optional template directory."""
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent / "templates"

    def compile(self, spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Compile UIS spec to Spark job configuration."""
        if spec.provider.mode != IngestionMode.MICRO_BATCH:
            raise SparkCompileError(f"Spark compiler only supports micro-batch mode, got: {spec.provider.mode}")

        job_config = {
            "job_type": "micro_batch",
            "spark_config": self._build_spark_config(spec),
            "sources": self._build_sources_config(spec),
            "transforms": self._build_transforms_config(spec),
            "sinks": self._build_sinks_config(spec),
            "schedule": self._build_schedule_config(spec),
            "monitoring": self._build_monitoring_config(spec)
        }

        return job_config

    def _build_spark_config(self, spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Build Spark session configuration."""
        config = spec.provider.config or {}

        spark_config = {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.adaptive.localShuffleReader.enabled": "true",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes": "64MB",

            # Micro-batch specific settings
            "spark.sql.streaming.trigger.processingTime": "5 minutes",
            "spark.sql.streaming.checkpointLocation": f"s3://spark-checkpoints/{spec.name}/",
            "spark.sql.streaming.minBatchesToRetain": "10",

            # Memory and performance
            "spark.executor.memory": config.get("executor_memory", "4g"),
            "spark.executor.cores": config.get("executor_cores", "2"),
            "spark.driver.memory": config.get("driver_memory", "2g"),
            "spark.driver.cores": "2",

            # Iceberg integration
            "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            "spark.sql.catalog.spark_catalog": "org.apache.iceberg.spark.SparkSessionCatalog",
            "spark.sql.catalog.spark_catalog.type": "hive",

            # S3/MinIO integration
            "spark.hadoop.fs.s3a.endpoint": config.get("s3_endpoint", "http://minio:9000"),
            "spark.hadoop.fs.s3a.access.key": "{{minio_access_key}}",
            "spark.hadoop.fs.s3a.secret.key": "{{minio_secret_key}}",
            "spark.hadoop.fs.s3a.path.style.access": "true",
            "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",

            # Hive metastore
            "spark.sql.warehouse.dir": config.get("warehouse_dir", "s3://warehouse/"),
            "hive.metastore.uris": config.get("hive_metastore", "thrift://hive-metastore:9083"),

            # Performance optimizations
            "spark.sql.parquet.compression.codec": "snappy",
            "spark.sql.parquet.mergeSchema": "false",
            "spark.sql.broadcastTimeout": "300",
            "spark.sql.autoBroadcastJoinThreshold": "10485760",  # 10MB

            # Monitoring
            "spark.ui.enabled": "true",
            "spark.eventLog.enabled": "true",
            "spark.eventLog.dir": "s3://spark-logs/"
        }

        # Add user-defined Spark configuration
        if config.get("spark_config"):
            spark_config.update(config["spark_config"])

        return spark_config

    def _build_sources_config(self, spec: UnifiedIngestionSpec) -> List[Dict[str, Any]]:
        """Build Spark source configurations."""
        sources = []

        for endpoint in spec.provider.endpoints:
            if spec.provider.provider_type == ProviderType.REST_API:
                source = self._build_rest_source(spec, endpoint)
            elif spec.provider.provider_type == ProviderType.FILE_FTP:
                source = self._build_file_source(spec, endpoint)
            elif spec.provider.provider_type == ProviderType.KAFKA:
                source = self._build_kafka_source(spec, endpoint)
            else:
                continue

            sources.append(source)

        return sources

    def _build_rest_source(self, spec: UnifiedIngestionSpec, endpoint: EndpointConfig) -> Dict[str, Any]:
        """Build REST API source for Spark."""
        return {
            "type": "rest_api",
            "name": f"rest_{endpoint.name}",
            "url": self._build_url(spec, endpoint),
            "method": endpoint.method or "GET",
            "headers": self._build_headers(spec, endpoint),
            "params": endpoint.query_params or {},
            "format": "json",
            "response_path": endpoint.response_path or "$",
            "rate_limit": endpoint.rate_limit_per_second or 10,
            "retry_config": {
                "max_retries": 3,
                "retry_delay_seconds": 60
            }
        }

    def _build_file_source(self, spec: UnifiedIngestionSpec, endpoint: EndpointConfig) -> Dict[str, Any]:
        """Build file source for Spark."""
        config = spec.provider.config or {}

        source = {
            "type": "file",
            "name": f"file_{endpoint.name}",
            "path": endpoint.path,
            "format": config.get("file_format", "csv"),
            "options": {
                "header": "true",
                "inferSchema": "false"  # Use explicit schema for reliability
            }
        }

        # Add format-specific options
        if source["format"] == "csv":
            source["options"].update({
                "delimiter": config.get("delimiter", ","),
                "quote": config.get("quote", '"'),
                "escape": config.get("escape", '"')
            })
        elif source["format"] == "json":
            source["options"]["multiline"] = "true"

        return source

    def _build_kafka_source(self, spec: UnifiedIngestionSpec, endpoint: EndpointConfig) -> Dict[str, Any]:
        """Build Kafka source for Spark."""
        config = spec.provider.config or {}

        return {
            "type": "kafka",
            "name": f"kafka_{endpoint.name}",
            "bootstrap_servers": config.get("kafka_bootstrap_servers", "kafka:9092"),
            "topics": config.get("kafka_topics", [endpoint.path]),
            "starting_offsets": config.get("starting_offsets", "latest"),
            "format": "json",
            "options": {
                "kafka.bootstrap.servers": config.get("kafka_bootstrap_servers", "kafka:9092"),
                "subscribe": ",".join(config.get("kafka_topics", [endpoint.path])),
                "startingOffsets": config.get("starting_offsets", "latest"),
                "failOnDataLoss": "false"
            }
        }

    def _build_transforms_config(self, spec: UnifiedIngestionSpec) -> List[Dict[str, Any]]:
        """Build Spark transform configurations."""
        transforms = []

        # Schema transformation
        if spec.provider.schema_contract:
            transforms.append({
                "type": "schema_validation",
                "name": "validate_schema",
                "schema": spec.provider.schema_contract,
                "mode": "FAILFAST"
            })

        # Field mapping transformation
        for endpoint in spec.provider.endpoints:
            if endpoint.field_mapping:
                transforms.append({
                    "type": "field_mapping",
                    "name": f"map_{endpoint.name}",
                    "mapping": endpoint.field_mapping,
                    "drop_unmapped": True
                })

        # Custom transforms from UIS spec
        for transform in spec.provider.transforms:
            if transform.type == "spark":
                spark_transform = {
                    "type": "custom_spark",
                    "name": transform.name,
                    "sql": transform.sql_query,
                    "parameters": transform.parameters or {}
                }
                transforms.append(spark_transform)

        return transforms

    def _build_sinks_config(self, spec: UnifiedIngestionSpec) -> List[Dict[str, Any]]:
        """Build Spark sink configurations."""
        sinks = []

        for sink_config in spec.provider.sinks:
            if sink_config.type == SinkType.ICEBERG:
                sink = self._build_iceberg_sink(sink_config)
            elif sink_config.type == SinkType.CLICKHOUSE:
                sink = self._build_clickhouse_sink(sink_config)
            elif sink_config.type == SinkType.KAFKA:
                sink = self._build_kafka_sink(sink_config)
            elif sink_config.type == SinkType.PARQUET:
                sink = self._build_parquet_sink(spec, sink_config)
            else:
                continue

            sinks.append(sink)

        return sinks

    def _build_iceberg_sink(self, sink_config) -> Dict[str, Any]:
        """Build Iceberg sink configuration."""
        config = sink_config.config or {}

        return {
            "type": "iceberg",
            "name": f"iceberg_{sink_config.table_name or 'table'}",
            "table": sink_config.table_name or "ingested_data",
            "namespace": config.get("namespace", "default"),
            "catalog": config.get("catalog", "spark_catalog"),
            "mode": "append",
            "options": {
                "path": config.get("table_path"),
                "format": "parquet",
                "partitionBy": ",".join(sink_config.partition_by or ["date"]),
                "write.wap.enabled": "true"  # Write-Audit-Publish for consistency
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
            "table": sink_config.clickhouse_table or "ingested_data",
            "mode": "append",
            "options": {
                "batch_size": 10000,
                "flush_interval": 5000,
                "retry_count": 3
            }
        }

    def _build_kafka_sink(self, sink_config) -> Dict[str, Any]:
        """Build Kafka sink configuration."""
        config = sink_config.config or {}

        return {
            "type": "kafka",
            "name": f"kafka_{sink_config.kafka_topic or 'topic'}",
            "bootstrap_servers": config.get("bootstrap_servers", "kafka:9092"),
            "topic": sink_config.kafka_topic or "ingested_data",
            "key_field": sink_config.kafka_key_field,
            "format": "json",
            "options": {
                "kafka.bootstrap.servers": config.get("bootstrap_servers", "kafka:9092"),
                "topic": sink_config.kafka_topic or "ingested_data",
                "checkpointLocation": f"s3://spark-checkpoints/kafka/{sink_config.kafka_topic or 'topic'}/"
            }
        }

    def _build_parquet_sink(self, spec: UnifiedIngestionSpec, sink_config) -> Dict[str, Any]:
        """Build Parquet sink configuration targeting MinIO/S3."""
        config = sink_config.config or {}

        output_path = config.get("output_path") or config.get("path")
        bucket = config.get("bucket")
        path_prefix = config.get("path_prefix")

        if not output_path and bucket:
            prefix = path_prefix or spec.name
            output_path = f"s3://{bucket}/{prefix}"

        if not output_path:
            raise SparkCompileError("Parquet sink requires an output path or bucket/path_prefix configuration")

        # Build writer options
        sink_options = {
            "compression": config.get("compression", "snappy")
        }

        if config.get("max_records_per_file"):
            sink_options["maxRecordsPerFile"] = str(config["max_records_per_file"])

        if config.get("partition_overwrite_mode"):
            sink_options["partitionOverwriteMode"] = config["partition_overwrite_mode"]

        if config.get("options"):
            sink_options.update({
                k: v for k, v in config["options"].items()
                if v is not None
            })

        partition_by = config.get("partition_by") or sink_config.partition_by
        if partition_by:
            sink_options["partitionBy"] = ",".join(partition_by)

        return {
            "type": "parquet",
            "name": config.get("name") or sink_config.table_name or f"parquet_{spec.name}",
            "format": "parquet",
            "path": output_path,
            "mode": config.get("mode", "append"),
            "options": sink_options
        }

    def _build_schedule_config(self, spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Build scheduling configuration."""
        schedule_config = {
            "trigger_type": "processing_time",
            "trigger_interval": "5 minutes",  # Micro-batch default
            "max_offsets_per_trigger": 1000,
            "checkpoint_location": f"s3://spark-checkpoints/{spec.name}/"
        }

        # Override with provider-specific scheduling
        if spec.provider.schedule_cron:
            schedule_config.update({
                "trigger_type": "cron",
                "cron_expression": spec.provider.schedule_cron,
                "timezone": spec.provider.schedule_timezone
            })

        return schedule_config

    def _build_monitoring_config(self, spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Build monitoring and observability configuration."""
        return {
            "metrics_enabled": True,
            "metrics_prefix": f"uis.{spec.name}",
            "streaming_metrics": {
                "input_rows_per_second": "inputRate",
                "processed_rows_per_second": "processingRate",
                "backlog_rows": "inputRowsPerSecond"
            },
            "custom_metrics": [
                "records_ingested",
                "bytes_ingested",
                "processing_latency_ms",
                "schema_drift_detected"
            ],
            "alerts": {
                "processing_delay_threshold_ms": 300000,  # 5 minutes
                "error_rate_threshold": 0.05  # 5%
            }
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
        if endpoint.auth and endpoint.auth_config:
            if endpoint.auth.value == "api_key":
                if endpoint.auth_config.get('header_name'):
                    headers[endpoint.auth_config['header_name']] = "{{api_key}}"
                else:
                    headers['Authorization'] = "Bearer {{api_key}}"

        return headers

    def compile_to_json(self, spec: UnifiedIngestionSpec) -> str:
        """Compile UIS spec to Spark JSON configuration."""
        config = self.compile(spec)
        return json.dumps(config, indent=2)

    def compile_to_job_artifacts(
        self,
        spec: UnifiedIngestionSpec,
        job_config_path: Optional[Union[str, Path]] = None
    ) -> SparkJobArtifacts:
        """Compile UIS spec to Spark submit arguments, session config, and job config."""
        config = self.compile(spec)

        if job_config_path is None:
            job_config_path_str = f"/opt/spark/jobs/{spec.name}-job-config.json"
        else:
            job_config_path_str = str(job_config_path)

        # Build Spark submit arguments
        args = [
            "--class", "com.hmco.dataplatform.SparkMicroBatchJob",
            "--master", "k8s://https://kubernetes.default.svc.cluster.local:443",
            "--deploy-mode", "cluster",
            "--conf", f"spark.kubernetes.namespace={spec.provider.tenant_id}",
            "--conf", "spark.kubernetes.container.image=hmco/spark-runner:latest",
            "--conf", "spark.kubernetes.authenticate.driver.serviceAccountName=spark",
            "--conf", "spark.kubernetes.authenticate.executor.serviceAccountName=spark",
            "--conf", "spark.kubernetes.driver.request.cores=1",
            "--conf", "spark.kubernetes.driver.limit.cores=2",
            "--conf", "spark.kubernetes.executor.request.cores=1",
            "--conf", "spark.kubernetes.executor.limit.cores=2",
            "--conf", f"spark.kubernetes.driver.label.tenant={spec.provider.tenant_id}",
            "--conf", f"spark.kubernetes.executor.label.tenant={spec.provider.tenant_id}",
            "--conf", f"spark.app.name=uis-{spec.name}",
            "--conf", f"spark.kubernetes.driver.annotation.uis.spec.name={spec.name}",
            "--conf", f"spark.kubernetes.driver.annotation.uis.tenant={spec.provider.tenant_id}",
            "--files", "/etc/spark/conf/log4j.properties",
            "--jars", "/opt/spark/jars/iceberg-spark-runtime-3.3_2.12-1.3.0.jar",
            "--jars", "/opt/spark/jars/bundle-2.12-3.3.0.jar",
            "--job-config", job_config_path_str
        ]

        # Add Spark configuration as --conf arguments
        for key, value in config["spark_config"].items():
            args.extend(["--conf", f"{key}={value}"])

        return SparkJobArtifacts(
            spark_submit_args=args,
            spark_session_conf=config["spark_config"],
            job_config=config,
            job_config_path=job_config_path_str
        )

    def compile_to_spark_args(
        self,
        spec: UnifiedIngestionSpec,
        job_config_path: Optional[Union[str, Path]] = None
    ) -> List[str]:
        """Compile UIS spec to Spark job arguments (compatibility wrapper)."""
        artifacts = self.compile_to_job_artifacts(spec, job_config_path=job_config_path)
        return artifacts.spark_submit_args

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate generated Spark configuration."""
        errors = []

        # Check required fields
        if 'spark_config' not in config:
            errors.append("Missing spark_config section")

        if 'sources' not in config or not config['sources']:
            errors.append("No source configuration found")

        if 'sinks' not in config or not config['sinks']:
            errors.append("No sink configuration found")

        # Validate sources
        for source in config.get('sources', []):
            if 'type' not in source or 'name' not in source:
                errors.append(f"Invalid source configuration: {source}")

        # Validate sinks
        for sink in config.get('sinks', []):
            if 'type' not in sink or 'name' not in sink:
                errors.append(f"Invalid sink configuration: {sink}")
            if sink.get('type') == 'parquet' and not sink.get('path'):
                errors.append("Parquet sink requires an output path")

        # Validate Spark configuration
        spark_config = config.get('spark_config', {})
        required_spark_keys = [
            'spark.sql.adaptive.enabled',
            'spark.sql.warehouse.dir',
            'spark.hadoop.fs.s3a.endpoint'
        ]

        for key in required_spark_keys:
            if key not in spark_config:
                errors.append(f"Missing required Spark configuration: {key}")

        return errors


