"""
SeaTunnel configuration templates.
"""

from typing import Dict, Any


class SeaTunnelTemplates:
    """Common SeaTunnel configuration templates."""

    @staticmethod
    def get_base_job_template() -> Dict[str, Any]:
        """Get base SeaTunnel job template."""
        return {
            "env": {
                "parallelism": 1,
                "checkpoint.interval": 10000,
                "checkpoint.timeout": 60000,
                "checkpoint.storage": "hdfs://namenode:9000/seatunnel/checkpoint"
            },
            "source": [],
            "transform": [],
            "sink": []
        }

    @staticmethod
    def get_http_source_template() -> Dict[str, Any]:
        """Get HTTP source template."""
        return {
            "plugin_name": "Http",
            "result_table_name": "http_source",
            "url": "",
            "method": "GET",
            "headers": {},
            "params": {},
            "format": "json",
            "json_field": "$",
            "rate_limit": {
                "read_per_second": 10
            }
        }

    @staticmethod
    def get_iceberg_sink_template() -> Dict[str, Any]:
        """Get Iceberg sink template."""
        return {
            "plugin_name": "Iceberg",
            "source_table_name": "transformed_data",
            "result_table_name": "iceberg_output",
            "catalog_name": "hive_prod",
            "namespace": "default",
            "table": "ingested_data",
            "warehouse": "s3://warehouse/",
            "hadoop_conf_path": "/etc/hadoop/conf",
            "save_mode": "append"
        }

    @staticmethod
    def get_clickhouse_sink_template() -> Dict[str, Any]:
        """Get ClickHouse sink template."""
        return {
            "plugin_name": "Clickhouse",
            "source_table_name": "transformed_data",
            "result_table_name": "clickhouse_output",
            "host": "clickhouse",
            "port": 8123,
            "database": "default",
            "table": "ingested_data",
            "username": "default",
            "password": "",
            "bulk_size": 20000
        }

    @staticmethod
    def get_kafka_sink_template() -> Dict[str, Any]:
        """Get Kafka sink template."""
        return {
            "plugin_name": "Kafka",
            "source_table_name": "transformed_data",
            "result_table_name": "kafka_output",
            "bootstrap_servers": "kafka:9092",
            "topic": "ingested_data",
            "key_field": None,
            "format": "json"
        }

    @staticmethod
    def get_field_mapper_template() -> Dict[str, Any]:
        """Get field mapper transform template."""
        return {
            "plugin_name": "FieldMapper",
            "source_table_name": "source_data",
            "result_table_name": "mapped_data",
            "field_mapper": {}
        }

    @staticmethod
    def get_schema_validator_template() -> Dict[str, Any]:
        """Get schema validator transform template."""
        return {
            "plugin_name": "SchemaValidator",
            "source_table_name": "mapped_data",
            "result_table_name": "validated_data",
            "schema": {}
        }


