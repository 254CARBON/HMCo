"""
Flink configuration templates for streaming processing.
"""

from typing import Dict, Any


class FlinkTemplates:
    """Common Flink configuration templates for streaming processing."""

    @staticmethod
    def get_base_streaming_config() -> Dict[str, Any]:
        """Get base streaming configuration template."""
        return {
            "job_type": "streaming",
            "flink_config": {
                "taskmanager.memory.process.size": "4g",
                "taskmanager.numberOfTaskSlots": "4",
                "parallelism.default": "4",
                "state.backend": "rocksdb",
                "state.checkpoints.dir": "s3://flink-checkpoints/",
                "state.savepoints.dir": "s3://flink-savepoints/",
                "state.checkpointing.mode": "EXACTLY_ONCE",
                "execution.checkpointing.interval": "60000",
                "execution.checkpointing.timeout": "300000"
            },
            "sources": [],
            "transforms": [],
            "sinks": [],
            "state_management": {
                "backend": "rocksdb",
                "checkpoint_config": {
                    "interval_ms": 60000,
                    "timeout_ms": 300000,
                    "mode": "EXACTLY_ONCE"
                }
            },
            "monitoring": {
                "metrics_enabled": True,
                "metrics_prefix": "uis"
            }
        }

    @staticmethod
    def get_websocket_source_template() -> Dict[str, Any]:
        """Get WebSocket source template."""
        return {
            "type": "websocket",
            "name": "websocket_source",
            "url": "",
            "protocol": "ws",
            "headers": {},
            "format": "json",
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

    @staticmethod
    def get_webhook_source_template() -> Dict[str, Any]:
        """Get webhook source template."""
        return {
            "type": "webhook",
            "name": "webhook_source",
            "path": "/webhook",
            "method": "POST",
            "headers": {},
            "format": "json",
            "validation": {
                "signature_required": False,
                "signature_header": None,
                "signature_algorithm": "HMAC-SHA256"
            },
            "rate_limiting": {
                "enabled": False,
                "requests_per_second": 100
            }
        }

    @staticmethod
    def get_kafka_source_template() -> Dict[str, Any]:
        """Get Kafka source template."""
        return {
            "type": "kafka",
            "name": "kafka_source",
            "bootstrap_servers": "kafka:9092",
            "topics": [],
            "group_id": "uis-consumer",
            "starting_offsets": "latest",
            "format": "json",
            "schema_registry": None,
            "properties": {
                "auto.offset.reset": "latest",
                "enable.auto.commit": "false",
                "isolation.level": "read_committed"
            }
        }

    @staticmethod
    def get_cdc_source_template() -> Dict[str, Any]:
        """Get CDC source template."""
        return {
            "type": "cdc",
            "name": "cdc_source",
            "connector": "debezium",
            "database": "postgres",
            "hostname": "",
            "port": 5432,
            "database_name": "",
            "schema_name": "public",
            "table_names": [],
            "username": "{{db_username}}",
            "password": "{{db_password}}",
            "format": "json",
            "debezium_config": {
                "snapshot.mode": "initial",
                "snapshot.locking.mode": "none",
                "include.schema.changes": "false"
            }
        }

    @staticmethod
    def get_kafka_sink_template() -> Dict[str, Any]:
        """Get Kafka sink template."""
        return {
            "type": "kafka",
            "name": "kafka_sink",
            "bootstrap_servers": "kafka:9092",
            "topic": "",
            "key_field": None,
            "format": "json",
            "transactional": True,
            "properties": {
                "transaction.timeout.ms": "900000",
                "batch.size": "16384",
                "linger.ms": "10",
                "compression.type": "snappy"
            }
        }

    @staticmethod
    def get_iceberg_sink_template() -> Dict[str, Any]:
        """Get Iceberg sink template."""
        return {
            "type": "iceberg",
            "name": "iceberg_sink",
            "catalog_name": "hive_prod",
            "database": "default",
            "table": "",
            "format": "parquet",
            "write_mode": "append",
            "options": {
                "write.wap.enabled": "true",
                "write.metadata.delete-after-commit.enabled": "true",
                "write.metadata.previous-versions-max": "5"
            }
        }

    @staticmethod
    def get_clickhouse_sink_template() -> Dict[str, Any]:
        """Get ClickHouse sink template."""
        return {
            "type": "clickhouse",
            "name": "clickhouse_sink",
            "host": "clickhouse",
            "port": 8123,
            "database": "default",
            "table": "",
            "username": "{{clickhouse_username}}",
            "password": "{{clickhouse_password}}",
            "batch_size": 10000,
            "flush_interval_ms": 5000
        }

    @staticmethod
    def get_field_mapping_transform_template() -> Dict[str, Any]:
        """Get field mapping transform template."""
        return {
            "type": "field_mapping",
            "name": "field_mapper",
            "mapping": {},
            "drop_unmapped": True
        }

    @staticmethod
    def get_schema_validation_transform_template() -> Dict[str, Any]:
        """Get schema validation transform template."""
        return {
            "type": "schema_validation",
            "name": "schema_validator",
            "schema": {},
            "strategy": "FORWARD"
        }

    @staticmethod
    def get_watermark_transform_template() -> Dict[str, Any]:
        """Get watermark transform template."""
        return {
            "type": "watermark",
            "name": "add_watermark",
            "timestamp_column": "event_time",
            "max_out_of_orderness_ms": 5000
        }

    @staticmethod
    def get_flink_config_template() -> Dict[str, Any]:
        """Get comprehensive Flink configuration template."""
        return {
            # Memory configuration
            "taskmanager.memory.process.size": "4g",
            "taskmanager.numberOfTaskSlots": "4",
            "taskmanager.memory.network.fraction": "0.1",
            "taskmanager.memory.managed.fraction": "0.4",
            "taskmanager.memory.network.min": "64mb",
            "taskmanager.memory.network.max": "1gb",

            # State management
            "state.backend": "rocksdb",
            "state.checkpoints.dir": "s3://flink-checkpoints/",
            "state.savepoints.dir": "s3://flink-savepoints/",
            "state.checkpointing.mode": "EXACTLY_ONCE",
            "state.checkpointing.interval": "60000",
            "state.checkpointing.timeout": "300000",
            "state.checkpoint-storage": "filesystem",
            "state.checkpoints.num-retained": "10",

            # Checkpointing
            "execution.checkpointing.interval": "60000",
            "execution.checkpointing.timeout": "300000",
            "execution.checkpointing.mode": "EXACTLY_ONCE",
            "execution.checkpointing.externalized-checkpoint-retention": "RETAIN_ON_CANCELLATION",

            # S3 configuration
            "s3.endpoint": "http://minio:9000",
            "s3.access-key": "{{minio_access_key}}",
            "s3.secret-key": "{{minio_secret_key}}",
            "s3.path.style.access": "true",

            # Kafka configuration
            "kafka.bootstrap.servers": "kafka:9092",
            "kafka.consumer.group-id": "uis-consumer",
            "kafka.producer.transactional-id-prefix": "uis-producer",

            # WebSocket configuration
            "websocket.reconnect.interval": "30000",
            "websocket.max.reconnect.attempts": "10",

            # Monitoring
            "metrics.reporter.prometheus.class": "org.apache.flink.metrics.prometheus.PrometheusReporter",
            "metrics.reporter.prometheus.port": "9999",

            # Performance tuning
            "taskmanager.network.memory.fraction": "0.1",
            "taskmanager.memory.segment-size": "32mb",
            "pipeline.operator-chaining": "true",
            "pipeline.max-parallelism": "128"
        }
