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
        """Get Iceberg sink template with exactly-once semantics."""
        return {
            "type": "iceberg",
            "name": "iceberg_sink",
            "catalog_name": "hive_prod",
            "database": "default",
            "table": "",
            "format": "parquet",
            "write_mode": "append",
            "exactly_once": True,  # Enable exactly-once semantics
            "checkpoint_enabled": True,
            "options": {
                "write.wap.enabled": "true",
                "write.metadata.delete-after-commit.enabled": "true",
                "write.metadata.previous-versions-max": "5",
                # Disable upserts to ensure exactly-once semantics with append-only writes
                # Upserts can cause duplicates on replay; append-only guarantees idempotence
                "write.upsert.enabled": "false"
            }
        }

    @staticmethod
    def get_clickhouse_sink_template() -> Dict[str, Any]:
        """Get ClickHouse sink template with exactly-once semantics."""
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
            "flush_interval_ms": 5000,
            "exactly_once": True,  # Enable exactly-once via idempotent writes
            "idempotency": {
                "enabled": True,
                "key_columns": ["event_id", "timestamp"],  # Dedup keys in CH
                "engine": "ReplacingMergeTree",  # Use ReplacingMergeTree for dedup
                "version_column": "event_version"
            }
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
        """Get watermark transform template with bounded out-of-order."""
        return {
            "type": "watermark",
            "name": "add_watermark",
            "timestamp_column": "event_time",
            "max_out_of_orderness_ms": 5000,
            "idle_source_timeout_ms": 60000,  # Mark source as idle after 1 min
            "strategy": "bounded_out_of_orderness"
        }
    
    @staticmethod
    def get_keyed_dedup_transform_template() -> Dict[str, Any]:
        """Get keyed deduplication transform template for exactly-once semantics."""
        return {
            "type": "keyed_dedup",
            "name": "dedup_events",
            "key_fields": ["event_id"],  # Fields to use for deduplication
            "time_window_ms": 600000,  # 10-minute dedup window
            "strategy": "first",  # Keep first occurrence
            "state_ttl_ms": 3600000  # 1-hour state TTL
        }
    
    @staticmethod
    def get_late_data_side_output_template() -> Dict[str, Any]:
        """Get late data side output template for quarantine."""
        return {
            "type": "late_data_side_output",
            "name": "quarantine_late_data",
            "output_tag": "late-data",
            "destination": {
                "type": "iceberg",
                "table": "quarantine.late_events",
                "partition_by": ["processing_date", "source"]
            },
            "metadata": {
                "include_watermark": True,
                "include_lateness_ms": True,
                "include_original_timestamp": True
            }
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

    @staticmethod
    def get_iso_rt_streaming_template() -> Dict[str, Any]:
        """Get ISO real-time streaming pipeline template for CAISO/MISO/SPP."""
        return {
            "job_type": "streaming",
            "name": "iso_rt_lmp_streaming",
            "description": "Real-time LMP 5-minute aggregation from ISO markets to ClickHouse",
            "flink_config": FlinkTemplates.get_flink_config_template(),
            "sources": [
                {
                    "type": "kafka",
                    "name": "iso_rt_lmp_source",
                    "bootstrap_servers": "redpanda:9092",
                    "topics": ["ISO_RT_LMP"],
                    "group_id": "flink-iso-rt-consumer",
                    "starting_offsets": "latest",
                    "format": "json"
                }
            ],
            "transforms": [
                {
                    "type": "watermark",
                    "name": "add_event_time_watermark",
                    "timestamp_column": "timestamp",
                    "max_out_of_orderness_ms": 30000
                },
                {
                    "type": "tumbling_window",
                    "name": "aggregate_5min",
                    "window_size_ms": 300000,
                    "aggregations": [
                        {"field": "lmp", "function": "avg"},
                        {"field": "congestion", "function": "avg"},
                        {"field": "loss", "function": "avg"}
                    ],
                    "group_by": ["iso", "node"]
                }
            ],
            "sinks": [
                {
                    "type": "clickhouse",
                    "name": "rt_lmp_clickhouse_sink",
                    "host": "clickhouse",
                    "port": 8123,
                    "database": "default",
                    "table": "rt_lmp_5m",
                    "batch_size": 5000,
                    "flush_interval_ms": 10000
                }
            ]
        }

    @staticmethod
    def get_outage_streaming_template() -> Dict[str, Any]:
        """Get outage event streaming pipeline template."""
        return {
            "job_type": "streaming",
            "name": "outage_events_streaming",
            "description": "Real-time outage event processing and alerting",
            "flink_config": FlinkTemplates.get_flink_config_template(),
            "sources": [
                {
                    "type": "kafka",
                    "name": "outage_source",
                    "bootstrap_servers": "redpanda:9092",
                    "topics": ["OUTAGES"],
                    "group_id": "flink-outage-consumer",
                    "starting_offsets": "latest",
                    "format": "json"
                }
            ],
            "transforms": [
                {
                    "type": "filter",
                    "name": "filter_critical_outages",
                    "condition": "capacity_mw > 500"
                },
                {
                    "type": "enrich",
                    "name": "add_severity",
                    "enrichment_logic": "capacity_mw > 1000 ? 'critical' : 'major'"
                }
            ],
            "sinks": [
                {
                    "type": "iceberg",
                    "name": "outage_iceberg_sink",
                    "catalog_name": "hive_prod",
                    "database": "power_markets",
                    "table": "outages",
                    "write_mode": "append"
                },
                {
                    "type": "kafka",
                    "name": "critical_alerts_sink",
                    "bootstrap_servers": "redpanda:9092",
                    "topic": "OUTAGE_ALERTS",
                    "format": "json"
                }
            ]
        }

    @staticmethod
    def get_hardened_iso_streaming_template() -> Dict[str, Any]:
        """
        Get hardened ISO streaming template with:
        - Event-time watermarks with bounded out-of-order
        - Keyed deduplication for exactly-once
        - EOS sinks to Iceberg/ClickHouse
        - Late data side-output to quarantine
        """
        return {
            "job_type": "streaming",
            "name": "hardened_iso_rt_streaming",
            "description": "Production-grade ISO RT data with watermarks, dedup, and EOS",
            "flink_config": FlinkTemplates.get_flink_config_template(),
            "sources": [
                {
                    "type": "kafka",
                    "name": "iso_rt_source",
                    "bootstrap_servers": "redpanda:9092",
                    "topics": ["ISO_RT_LMP"],
                    "group_id": "flink-iso-rt-hardened",
                    "starting_offsets": "latest",
                    "format": "json",
                    "properties": {
                        "isolation.level": "read_committed"  # For exactly-once
                    }
                }
            ],
            "transforms": [
                {
                    "type": "watermark",
                    "name": "add_event_time_watermark",
                    "timestamp_column": "event_timestamp",
                    "max_out_of_orderness_ms": 600000,  # 10 min for ISO feeds
                    "idle_source_timeout_ms": 60000,
                    "strategy": "bounded_out_of_orderness"
                },
                {
                    "type": "keyed_dedup",
                    "name": "dedup_by_message_id",
                    "key_fields": ["iso", "node", "interval_start", "message_id"],
                    "time_window_ms": 600000,  # 10-minute dedup window
                    "strategy": "first",
                    "state_ttl_ms": 3600000
                },
                {
                    "type": "late_data_side_output",
                    "name": "quarantine_late_data",
                    "output_tag": "late-iso-data",
                    "destination": {
                        "type": "iceberg",
                        "table": "quarantine.late_iso_lmp",
                        "partition_by": ["processing_date", "iso"]
                    }
                },
                {
                    "type": "tumbling_window",
                    "name": "aggregate_5min",
                    "window_size_ms": 300000,
                    "aggregations": [
                        {"field": "lmp", "function": "avg"},
                        {"field": "congestion", "function": "avg"},
                        {"field": "loss", "function": "avg"}
                    ],
                    "group_by": ["iso", "node"]
                }
            ],
            "sinks": [
                {
                    "type": "iceberg",
                    "name": "curated_iceberg_sink",
                    "catalog_name": "hive_prod",
                    "database": "curated",
                    "table": "rt_lmp_5m",
                    "format": "parquet",
                    "write_mode": "append",
                    "exactly_once": True,
                    "checkpoint_enabled": True
                },
                {
                    "type": "clickhouse",
                    "name": "rt_clickhouse_sink",
                    "host": "clickhouse",
                    "port": 8123,
                    "database": "default",
                    "table": "rt_lmp_5m",
                    "batch_size": 5000,
                    "flush_interval_ms": 10000,
                    "exactly_once": True,
                    "idempotency": {
                        "enabled": True,
                        "key_columns": ["iso", "node", "interval_start"],
                        "engine": "ReplacingMergeTree",
                        "version_column": "event_version"
                    }
                }
            ]
        }
    
    @staticmethod
    def get_weather_rt_streaming_template() -> Dict[str, Any]:
        """Get real-time weather streaming pipeline template."""
        return {
            "job_type": "streaming",
            "name": "weather_rt_streaming",
            "description": "Real-time weather processing with H3 spatial aggregation",
            "flink_config": FlinkTemplates.get_flink_config_template(),
            "sources": [
                {
                    "type": "kafka",
                    "name": "weather_rt_source",
                    "bootstrap_servers": "redpanda:9092",
                    "topics": ["WEATHER_RT"],
                    "group_id": "flink-weather-consumer",
                    "starting_offsets": "latest",
                    "format": "json"
                }
            ],
            "transforms": [
                {
                    "type": "watermark",
                    "name": "add_event_time_watermark",
                    "timestamp_column": "timestamp",
                    "max_out_of_orderness_ms": 60000
                },
                {
                    "type": "sliding_window",
                    "name": "rolling_1hr_avg",
                    "window_size_ms": 3600000,
                    "slide_ms": 300000,
                    "aggregations": [
                        {"field": "temperature_f", "function": "avg"},
                        {"field": "humidity_pct", "function": "avg"},
                        {"field": "wind_speed_mph", "function": "avg"}
                    ],
                    "group_by": ["station_id"]
                }
            ],
            "sinks": [
                {
                    "type": "clickhouse",
                    "name": "weather_rt_clickhouse_sink",
                    "host": "clickhouse",
                    "port": 8123,
                    "database": "default",
                    "table": "weather_rt",
                    "batch_size": 10000,
                    "flush_interval_ms": 5000
                }
            ]
        }
