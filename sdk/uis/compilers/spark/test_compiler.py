#!/usr/bin/env python3
"""
Tests for Spark compiler.
"""

import json
import sys
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from spec import UnifiedIngestionSpec, ProviderConfig, EndpointConfig, ProviderType, SinkType, IngestionMode, AuthType

sys.path.insert(0, str(Path(__file__).parent))
from compiler import SparkCompiler, SparkCompileError
from spark_job import SparkMicroBatchJob


def test_compile_simple_microbatch():
    """Test compiling a simple micro-batch UIS spec."""
    print("Testing simple micro-batch compilation...")

    # Create a simple micro-batch UIS spec
    spec = UnifiedIngestionSpec(
        version="1.1",
        name="test-microbatch",
        provider=ProviderConfig(
            name="rest_api_provider",
            display_name="REST API Provider",
            provider_type=ProviderType.REST_API,
            base_url="https://api.example.com",
            config={"executor_memory": "8g"},
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.MICRO_BATCH,
            parallelism=4,
            endpoints=[
                EndpointConfig(
                    name="users",
                    path="/users",
                    method="GET",
                    auth=AuthType.API_KEY,
                    auth_config={"header_name": "X-API-Key"},
                    query_params={"limit": "1000"},
                    response_path="$.data",
                    field_mapping={"id": "user_id", "name": "full_name"},
                    rate_limit_per_second=20
                )
            ],
            sinks=[
                {
                    "type": SinkType.PARQUET,
                    "config": {
                        "bucket": "uis-micro-batch",
                        "path_prefix": "test-microbatch",
                        "compression": "zstd"
                    }
                },
                {
                    "type": SinkType.ICEBERG,
                    "table_name": "analytics.users",
                    "partition_by": ["date"],
                    "config": {
                        "catalog": "hive_prod",
                        "namespace": "analytics",
                        "warehouse": "s3://prod-warehouse/"
                    }
                }
            ]
        ),
        created_by="test-user"
    )

    # Compile to Spark configuration
    compiler = SparkCompiler()
    config = compiler.compile(spec)

    # Validate structure
    assert "job_type" in config
    assert config["job_type"] == "micro_batch"
    assert "spark_config" in config
    assert "sources" in config
    assert "sinks" in config
    assert "schedule" in config

    # Check Spark configuration
    spark_config = config["spark_config"]
    assert spark_config["spark.executor.memory"] == "8g"
    assert "spark.sql.adaptive.enabled" in spark_config
    assert spark_config["spark.sql.streaming.trigger.processingTime"] == "5 minutes"

    # Check source configuration
    assert len(config["sources"]) == 1
    source = config["sources"][0]
    assert source["type"] == "rest_api"
    assert source["name"] == "rest_users"
    assert source["url"] == "https://api.example.com/users"
    assert "X-API-Key" in source["headers"]
    assert source["rate_limit"] == 20

    # Check transforms
    assert len(config["transforms"]) >= 1  # At least field mapping
    field_transform = config["transforms"][0]
    assert field_transform["type"] == "field_mapping"
    assert field_transform["mapping"]["id"] == "user_id"

    # Check sink configuration
    assert len(config["sinks"]) == 2
    parquet_sink = next(s for s in config["sinks"] if s["type"] == "parquet")
    assert parquet_sink["path"] == "s3://uis-micro-batch/test-microbatch"
    assert parquet_sink["options"]["compression"] == "zstd"

    iceberg_sink = next(s for s in config["sinks"] if s["type"] == "iceberg")
    assert iceberg_sink["table"] == "analytics.users"
    assert iceberg_sink["options"]["partitionBy"] == "date"

    print("✓ Simple micro-batch compilation successful")
    return config


def test_compile_file_source():
    """Test compiling a file-based UIS spec."""
    print("Testing file source compilation...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="csv-ingestion",
        provider=ProviderConfig(
            name="csv_provider",
            display_name="CSV File Provider",
            provider_type=ProviderType.FILE_FTP,
            config={"file_format": "csv", "delimiter": "|"},
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.MICRO_BATCH,
            endpoints=[
                EndpointConfig(
                    name="daily_data",
                    path="/data/input/*.csv",
                    field_mapping={"date": "date", "value": "amount"}
                )
            ],
            sinks=[{
                "type": SinkType.CLICKHOUSE,
                "clickhouse_table": "daily_analytics",
                "config": {"host": "clickhouse", "database": "analytics"}
            }]
        ),
        created_by="test-user"
    )

    compiler = SparkCompiler()
    config = compiler.compile(spec)

    # Check source configuration
    assert len(config["sources"]) == 1
    source = config["sources"][0]
    assert source["type"] == "file"
    assert source["name"] == "file_daily_data"
    assert source["path"] == "/data/input/*.csv"
    assert source["format"] == "csv"
    assert source["options"]["delimiter"] == "|"

    # Check sink configuration
    assert len(config["sinks"]) == 1
    sink = config["sinks"][0]
    assert sink["type"] == "clickhouse"
    assert sink["table"] == "daily_analytics"

    print("✓ File source compilation successful")
    return config


def test_compile_kafka_source():
    """Test compiling a Kafka-based UIS spec."""
    print("Testing Kafka source compilation...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="kafka-streaming",
        provider=ProviderConfig(
            name="kafka_provider",
            display_name="Kafka Provider",
            provider_type=ProviderType.KAFKA,
            config={
                "kafka_bootstrap_servers": "kafka:9092",
                "kafka_topics": ["user-events", "page-views"]
            },
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.MICRO_BATCH,
            endpoints=[
                EndpointConfig(
                    name="events",
                    path="user-events",
                    field_mapping={"timestamp": "event_time", "user_id": "user_id"}
                )
            ],
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "events.user_events",
                "config": {"warehouse": "s3://warehouse/"}
            }]
        ),
        created_by="test-user"
    )

    compiler = SparkCompiler()
    config = compiler.compile(spec)

    # Check source configuration
    assert len(config["sources"]) == 1
    source = config["sources"][0]
    assert source["type"] == "kafka"
    assert source["name"] == "kafka_events"
    assert source["bootstrap_servers"] == "kafka:9092"
    assert "user-events" in source["topics"]

    print("✓ Kafka source compilation successful")
    return config


def test_spark_args_generation():
    """Test Spark job arguments generation."""
    print("Testing Spark args generation...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="args-test",
        provider=ProviderConfig(
            name="test_provider",
            display_name="Test Provider",
            provider_type=ProviderType.REST_API,
            base_url="https://api.test.com",
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.MICRO_BATCH,
            endpoints=[
                EndpointConfig(name="data", path="/data")
            ],
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "test_table",
                "config": {}
            }]
        ),
        created_by="test-user"
    )

    compiler = SparkCompiler()
    artifacts = compiler.compile_to_job_artifacts(spec, job_config_path="/mnt/configs/args-test.json")
    args = artifacts.spark_submit_args

    # Validate Spark arguments
    assert "--class" in args
    assert "--master" in args
    assert "--deploy-mode" in args
    assert "k8s://https://kubernetes.default.svc.cluster.local:443" in args
    assert f"spark.kubernetes.namespace={spec.provider.tenant_id}" in args
    assert f"spark.app.name=uis-{spec.name}" in args

    # Check for job config
    job_config_idx = args.index("--job-config")
    assert args[job_config_idx + 1] == "/mnt/configs/args-test.json"

    # Session config returned
    assert artifacts.spark_session_conf["spark.sql.adaptive.enabled"] == "true"
    assert artifacts.job_config_path == "/mnt/configs/args-test.json"
    assert artifacts.job_config["spark_config"]["spark.sql.streaming.trigger.processingTime"] == "5 minutes"

    print(f"✓ Generated {len(args)} Spark arguments")
    return args


def test_validation():
    """Test Spark configuration validation."""
    print("Testing configuration validation...")

    # Test valid config
    compiler = SparkCompiler()
    valid_config = {
        "job_type": "micro_batch",
        "spark_config": {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.warehouse.dir": "s3://warehouse/",
            "spark.hadoop.fs.s3a.endpoint": "http://minio:9000"
        },
        "sources": [{"type": "rest_api", "name": "test"}],
        "sinks": [{"type": "iceberg", "name": "test"}]
    }

    errors = compiler.validate_config(valid_config)
    assert len(errors) == 0

    # Test invalid config
    invalid_config = {
        "job_type": "micro_batch",
        "sources": [],  # Missing sources
        "sinks": []  # Missing sinks
    }

    errors = compiler.validate_config(invalid_config)
    assert len(errors) > 0
    assert any("No source configuration" in error for error in errors)
    assert any("No sink configuration" in error for error in errors)

    invalid_parquet_config = {
        "job_type": "micro_batch",
        "spark_config": valid_config["spark_config"],
        "sources": [{"type": "rest_api", "name": "test"}],
        "sinks": [{"type": "parquet", "name": "parquet_sink"}]
    }

    parquet_errors = compiler.validate_config(invalid_parquet_config)
    assert "Parquet sink requires an output path" in parquet_errors

    print(f"✓ Validation found {len(errors)} errors as expected")


def test_unsupported_mode():
    """Test compilation with unsupported mode."""
    print("Testing unsupported mode...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="streaming-test",
        provider=ProviderConfig(
            name="streaming_provider",
            display_name="Streaming Provider",
            provider_type=ProviderType.REST_API,
            mode=IngestionMode.STREAMING,  # Not supported by Spark compiler
            tenant_id="test-tenant",
            owner="test@example.com",
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "test",
                "config": {}
            }]
        ),
        created_by="test-user"
    )

    compiler = SparkCompiler()

    try:
        compiler.compile(spec)
        assert False, "Should have raised SparkCompileError"
    except SparkCompileError as e:
        print(f"✓ Unsupported mode correctly rejected: {e}")


def test_json_output():
    """Test JSON output generation."""
    print("Testing JSON output...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="json-test",
        provider=ProviderConfig(
            name="json_provider",
            display_name="JSON Provider",
            provider_type=ProviderType.REST_API,
            base_url="https://api.json.com",
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.MICRO_BATCH,
            endpoints=[
                EndpointConfig(name="data", path="/data")
            ],
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "json_data",
                "config": {}
            }]
        ),
        created_by="test-user"
    )

    compiler = SparkCompiler()
    json_output = compiler.compile_to_json(spec)

    # Should be valid JSON
    config = json.loads(json_output)
    assert "spark_config" in config
    assert "sources" in config
    assert "sinks" in config

    print("✓ JSON output generation successful")


def test_schema_contract():
    """Test schema contract integration."""
    print("Testing schema contract...")

    schema = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "name": {"type": "string"},
            "created_at": {"type": "string", "format": "date-time"}
        },
        "required": ["user_id", "name"]
    }

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="schema-test",
        provider=ProviderConfig(
            name="schema_provider",
            display_name="Schema Provider",
            provider_type=ProviderType.REST_API,
            base_url="https://api.schema.com",
            schema_contract=schema,
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.MICRO_BATCH,
            endpoints=[
                EndpointConfig(name="users", path="/users")
            ],
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "users",
                "config": {}
            }]
        ),
        created_by="test-user"
    )

    compiler = SparkCompiler()
    config = compiler.compile(spec)

    # Should have schema validation transform
    schema_transforms = [t for t in config["transforms"] if t["type"] == "schema_validation"]
    assert len(schema_transforms) == 1
    assert schema_transforms[0]["schema"] == schema

    print("✓ Schema contract integration successful")


def test_parquet_sink_writer():
    """Test Spark job Parquet sink writing behavior."""
    print("Testing Parquet sink writer...")

    sink_config = {
        "type": "parquet",
        "name": "parquet_output",
        "path": "s3://landing/output",
        "mode": "overwrite",
        "options": {
            "compression": "zstd",
            "partitionBy": "ingestion_date"
        }
    }

    job = SparkMicroBatchJob({
        "job_type": "micro_batch",
        "spark_config": {},
        "sources": [],
        "transforms": [],
        "sinks": [sink_config]
    })

    df = MagicMock()
    df_write = MagicMock()
    writer = MagicMock()

    df.write = df_write
    df_write.mode.return_value = writer
    writer.format.return_value = writer
    writer.partitionBy.return_value = writer
    writer.option.return_value = writer

    job.write_parquet_sink(df, sink_config)

    df_write.mode.assert_called_once_with("overwrite")
    writer.format.assert_called_once_with("parquet")
    writer.partitionBy.assert_called_once_with("ingestion_date")
    writer.option.assert_called_once_with("compression", "zstd")
    writer.save.assert_called_once_with("s3://landing/output")

    print("✓ Parquet sink writer behavior validated")


def run_all_tests():
    """Run all Spark compiler tests."""
    print("Running Spark Compiler Tests\n" + "="*40)

    try:
        # Basic compilation tests
        test_compile_simple_microbatch()
        test_compile_file_source()
        test_compile_kafka_source()

        # Output tests
        test_spark_args_generation()
        test_json_output()

        # Validation tests
        test_validation()

        # Schema tests
        test_schema_contract()
        test_parquet_sink_writer()

        # Error handling tests
        test_unsupported_mode()

        print("\n" + "="*40)
        print("✓ All Spark compiler tests passed!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
