"""
Spark configuration templates for micro-batch processing.
"""

from typing import Dict, Any


class SparkTemplates:
    """Common Spark configuration templates for micro-batch processing."""

    @staticmethod
    def get_base_microbatch_config() -> Dict[str, Any]:
        """Get base micro-batch configuration template."""
        return {
            "job_type": "micro_batch",
            "spark_config": {
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "spark.sql.adaptive.skewJoin.enabled": "true",
                "spark.sql.streaming.trigger.processingTime": "5 minutes",
                "spark.sql.streaming.checkpointLocation": "s3://spark-checkpoints/",
                "spark.sql.streaming.minBatchesToRetain": "10"
            },
            "sources": [],
            "transforms": [],
            "sinks": [],
            "schedule": {
                "trigger_type": "processing_time",
                "trigger_interval": "5 minutes"
            },
            "monitoring": {
                "metrics_enabled": True,
                "metrics_prefix": "uis"
            }
        }

    @staticmethod
    def get_rest_api_source_template() -> Dict[str, Any]:
        """Get REST API source template."""
        return {
            "type": "rest_api",
            "name": "rest_source",
            "url": "",
            "method": "GET",
            "headers": {},
            "params": {},
            "format": "json",
            "response_path": "$",
            "rate_limit": 10,
            "retry_config": {
                "max_retries": 3,
                "retry_delay_seconds": 60
            }
        }

    @staticmethod
    def get_file_source_template() -> Dict[str, Any]:
        """Get file source template."""
        return {
            "type": "file",
            "name": "file_source",
            "path": "",
            "format": "csv",
            "options": {
                "header": "true",
                "inferSchema": "false"
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
            "starting_offsets": "latest",
            "format": "json",
            "options": {
                "kafka.bootstrap.servers": "kafka:9092",
                "failOnDataLoss": "false"
            }
        }

    @staticmethod
    def get_iceberg_sink_template() -> Dict[str, Any]:
        """Get Iceberg sink template."""
        return {
            "type": "iceberg",
            "name": "iceberg_sink",
            "table": "",
            "namespace": "default",
            "catalog": "spark_catalog",
            "mode": "append",
            "options": {
                "format": "parquet",
                "write.wap.enabled": "true"
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
            "mode": "append",
            "options": {
                "batch_size": 10000,
                "flush_interval": 5000,
                "retry_count": 3
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
            "format": "json",
            "options": {
                "kafka.bootstrap.servers": "kafka:9092"
            }
        }

    @staticmethod
    def get_parquet_sink_template() -> Dict[str, Any]:
        """Get Parquet sink template targeting MinIO/S3."""
        return {
            "type": "parquet",
            "name": "parquet_sink",
            "format": "parquet",
            "path": "s3://landing-zone/spec-name",
            "mode": "append",
            "options": {
                "compression": "snappy",
                "partitionBy": "ingestion_date"
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
            "mode": "FAILFAST"
        }

    @staticmethod
    def get_spark_config_template() -> Dict[str, Any]:
        """Get comprehensive Spark configuration template."""
        return {
            # Adaptive query execution
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.adaptive.localShuffleReader.enabled": "true",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes": "64MB",

            # Streaming configuration
            "spark.sql.streaming.trigger.processingTime": "5 minutes",
            "spark.sql.streaming.checkpointLocation": "s3://spark-checkpoints/",
            "spark.sql.streaming.minBatchesToRetain": "10",
            "spark.sql.streaming.statefulOperator.checkCorrectness.enabled": "false",

            # Memory management
            "spark.executor.memory": "4g",
            "spark.executor.cores": "2",
            "spark.driver.memory": "2g",
            "spark.driver.cores": "2",
            "spark.executor.memoryOverhead": "1g",
            "spark.driver.memoryOverhead": "512m",

            # Storage configuration
            "spark.sql.parquet.compression.codec": "snappy",
            "spark.sql.parquet.mergeSchema": "false",
            "spark.sql.broadcastTimeout": "300",
            "spark.sql.autoBroadcastJoinThreshold": "10485760",

            # Iceberg integration
            "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            "spark.sql.catalog.spark_catalog": "org.apache.iceberg.spark.SparkSessionCatalog",
            "spark.sql.catalog.spark_catalog.type": "hive",

            # S3/MinIO configuration
            "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
            "spark.hadoop.fs.s3a.access.key": "{{minio_access_key}}",
            "spark.hadoop.fs.s3a.secret.key": "{{minio_secret_key}}",
            "spark.hadoop.fs.s3a.path.style.access": "true",
            "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",

            # Hive metastore
            "spark.sql.warehouse.dir": "s3://warehouse/",
            "hive.metastore.uris": "thrift://hive-metastore:9083",

            # Monitoring and logging
            "spark.ui.enabled": "true",
            "spark.eventLog.enabled": "true",
            "spark.eventLog.dir": "s3://spark-logs/",
            "spark.sql.streaming.metricsEnabled": "true"
        }
