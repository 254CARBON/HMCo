#!/usr/bin/env python3
"""
Tests for Flink compiler.
"""

import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from spec import UnifiedIngestionSpec, ProviderConfig, EndpointConfig, ProviderType, SinkType, IngestionMode, AuthType

sys.path.insert(0, str(Path(__file__).parent))
from compiler import FlinkCompiler, FlinkCompileError


def test_compile_websocket_streaming():
    """Test compiling a WebSocket streaming UIS spec."""
    print("Testing WebSocket streaming compilation...")

    # Create a WebSocket streaming UIS spec
    spec = UnifiedIngestionSpec(
        version="1.1",
        name="websocket-streaming",
        provider=ProviderConfig(
            name="websocket_provider",
            display_name="WebSocket Provider",
            provider_type=ProviderType.WEBSOCKET,
            base_url="wss://api.example.com",
            config={"taskmanager_memory": "2g"},
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.WEBSOCKET,
            parallelism=2,
            endpoints=[
                EndpointConfig(
                    name="realtime_data",
                    path="/ws/realtime",
                    query_params={"protocol": "wss"},
                    headers={"Authorization": "Bearer {{token}}"},
                    field_mapping={"timestamp": "event_time", "data": "payload"},
                    rate_limit_per_second=100
                )
            ],
            sinks=[{
                "type": SinkType.KAFKA,
                "kafka_topic": "realtime-events",
                "kafka_key_field": "event_id",
                "config": {"bootstrap_servers": "kafka:9092"}
            }]
        ),
        created_by="test-user"
    )

    # Compile to Flink configuration
    compiler = FlinkCompiler()
    config = compiler.compile(spec)

    # Validate structure
    assert "job_type" in config
    assert config["job_type"] == "streaming"
    assert "flink_config" in config
    assert "sources" in config
    assert "sinks" in config

    # Check Flink configuration
    flink_config = config["flink_config"]
    assert flink_config["taskmanager.memory.process.size"] == "2g"
    assert "state.checkpointing.mode" in flink_config
    assert flink_config["state.checkpointing.mode"] == "EXACTLY_ONCE"

    # Check source configuration
    assert len(config["sources"]) == 1
    source = config["sources"][0]
    assert source["type"] == "websocket"
    assert source["name"] == "websocket_realtime_data"
    assert source["url"] == "wss://api.example.com/ws/realtime"
    assert source["protocol"] == "wss"
    assert source["endpoint_name"] == "realtime_data"
    assert "reconnect_config" in source

    # Check transforms metadata
    field_transform = next(t for t in config["transforms"] if t["type"] == "field_mapping")
    assert field_transform["applies_to_endpoints"] == ["realtime_data"]

    # Check sink configuration
    assert len(config["sinks"]) == 1
    sink = config["sinks"][0]
    assert sink["type"] == "kafka"
    assert sink["topic"] == "realtime-events"
    assert sink["transactional"] == True

    # Pipeline wiring should point WebSocket → Kafka topic
    assert len(config["pipelines"]) == 1
    pipeline = config["pipelines"][0]
    assert pipeline["source"]["name"] == "websocket_realtime_data"
    assert pipeline["sink"]["topic"] == "realtime-events"
    assert pipeline["delivery_guarantee"] == "exactly_once"
    assert "map_realtime_data" in pipeline["transforms"]

    print("✓ WebSocket streaming compilation successful")
    return config


def test_compile_webhook_streaming():
    """Test compiling a webhook streaming UIS spec."""
    print("Testing webhook streaming compilation...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="webhook-streaming",
        provider=ProviderConfig(
            name="webhook_provider",
            display_name="Webhook Provider",
            provider_type=ProviderType.WEBHOOK,
            base_url="https://api.example.com",
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.WEBHOOK,
            endpoints=[
                EndpointConfig(
                    name="webhook_events",
                    path="/webhook/events",
                    method="POST",
                    auth=AuthType.HMAC,
                    auth_config={
                        "signature_header": "X-Signature",
                        "signature_algorithm": "HMAC-SHA256"
                    },
                    field_mapping={"event": "data", "timestamp": "received_at"},
                    rate_limit_per_second=50
                )
            ],
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "events.webhook_events",
                "config": {"catalog": "hive_prod"}
            }]
        ),
        created_by="test-user"
    )

    compiler = FlinkCompiler()
    config = compiler.compile(spec)

    # Check source configuration
    assert len(config["sources"]) == 1
    source = config["sources"][0]
    assert source["type"] == "webhook"
    assert source["name"] == "webhook_webhook_events"
    assert source["path"] == "/webhook/events"
    assert source["validation"]["signature_required"] == True

    # Check sink configuration
    assert len(config["sinks"]) == 1
    sink = config["sinks"][0]
    assert sink["type"] == "iceberg"
    assert sink["table"] == "events.webhook_events"

    print("✓ Webhook streaming compilation successful")
    return config


def test_compile_kafka_streaming():
    """Test compiling a Kafka streaming UIS spec."""
    print("Testing Kafka streaming compilation...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="kafka-streaming",
        provider=ProviderConfig(
            name="kafka_provider",
            display_name="Kafka Provider",
            provider_type=ProviderType.KAFKA,
            config={
                "kafka_bootstrap_servers": "kafka:9092",
                "kafka_topics": ["user-events", "system-logs"]
            },
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.STREAMING,
            endpoints=[
                EndpointConfig(
                    name="events",
                    path="user-events",
                    field_mapping={"ts": "event_time", "uid": "user_id"}
                )
            ],
            sinks=[{
                "type": SinkType.CLICKHOUSE,
                "clickhouse_table": "user_events",
                "config": {"host": "clickhouse", "database": "analytics"}
            }]
        ),
        created_by="test-user"
    )

    compiler = FlinkCompiler()
    config = compiler.compile(spec)

    # Check source configuration
    assert len(config["sources"]) == 1
    source = config["sources"][0]
    assert source["type"] == "kafka"
    assert source["name"] == "kafka_events"
    assert source["bootstrap_servers"] == "kafka:9092"
    assert "user-events" in source["topics"]

    # Check sink configuration
    assert len(config["sinks"]) == 1
    sink = config["sinks"][0]
    assert sink["type"] == "clickhouse"
    assert sink["table"] == "user_events"

    print("✓ Kafka streaming compilation successful")
    return config


def test_compile_cdc_streaming():
    """Test compiling a CDC streaming UIS spec."""
    print("Testing CDC streaming compilation...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="cdc-streaming",
        provider=ProviderConfig(
            name="postgres_cdc",
            display_name="PostgreSQL CDC",
            provider_type=ProviderType.DATABASE,
            config={
                "cdc_connector": "debezium",
                "database_type": "postgres",
                "db_host": "postgres",
                "db_port": 5432,
                "db_name": "analytics",
                "schema_name": "public",
                "table_names": ["users", "orders"]
            },
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.STREAMING,
            sinks=[{
                "type": SinkType.KAFKA,
                "kafka_topic": "cdc-changes",
                "config": {"bootstrap_servers": "kafka:9092"}
            }]
        ),
        created_by="test-user"
    )

    compiler = FlinkCompiler()
    config = compiler.compile(spec)

    # Check source configuration
    assert len(config["sources"]) == 1
    source = config["sources"][0]
    assert source["type"] == "cdc"
    assert source["name"] == "cdc_postgres_cdc"
    assert source["connector"] == "debezium"
    assert source["database"] == "postgres"

    # Check sink configuration
    assert len(config["sinks"]) == 1
    sink = config["sinks"][0]
    assert sink["type"] == "kafka"
    assert sink["topic"] == "cdc-changes"

    # CDC pipeline should route into Kafka topic
    assert len(config["pipelines"]) == 1
    pipeline = config["pipelines"][0]
    assert pipeline["source"]["type"] == "cdc"
    assert pipeline["sink"]["topic"] == "cdc-changes"
    assert pipeline["delivery_guarantee"] == "exactly_once"

    print("✓ CDC streaming compilation successful")
    return config


def test_flink_sql_generation():
    """Test Flink SQL generation."""
    print("Testing Flink SQL generation...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="sql-test",
        provider=ProviderConfig(
            name="kafka_provider",
            display_name="Kafka Provider",
            provider_type=ProviderType.KAFKA,
            config={"kafka_bootstrap_servers": "kafka:9092"},
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.STREAMING,
            endpoints=[
                EndpointConfig(name="events", path="user-events")
            ],
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "events.user_events",
                "config": {"catalog": "hive_prod"}
            }]
        ),
        created_by="test-user"
    )

    compiler = FlinkCompiler()
    sql = compiler.compile_to_flink_sql(spec)

    # Validate SQL structure
    assert "CREATE TABLE" in sql
    assert "kafka_events" in sql
    assert "iceberg_events.user_events" in sql
    assert "INSERT INTO" in sql

    print("✓ Flink SQL generation successful")
    return sql


def test_flink_args_generation():
    """Test Flink job arguments generation."""
    print("Testing Flink args generation...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="args-test",
        provider=ProviderConfig(
            name="test_provider",
            display_name="Test Provider",
            provider_type=ProviderType.WEBSOCKET,
            base_url="wss://api.test.com",
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.WEBSOCKET,
            parallelism=4,
            endpoints=[
                EndpointConfig(name="data", path="/ws/data")
            ],
            sinks=[{
                "type": SinkType.KAFKA,
                "kafka_topic": "test-events",
                "config": {}
            }]
        ),
        created_by="test-user"
    )

    compiler = FlinkCompiler()
    args = compiler.compile_to_flink_args(spec)

    # Validate Flink arguments
    assert "--jobmanager" in args
    assert "--taskmanager" in args
    assert "--parallelism" in args
    assert "4" in args  # parallelism value
    assert f"uis-{spec.name}" in args  # job name

    print(f"✓ Generated {len(args)} Flink arguments")
    return args


def test_validation():
    """Test Flink configuration validation."""
    print("Testing configuration validation...")

    # Test valid config
    compiler = FlinkCompiler()
    valid_config = {
        "job_type": "streaming",
        "flink_config": {
            "state.backend": "rocksdb",
            "state.checkpoints.dir": "s3://checkpoints/",
            "execution.checkpointing.mode": "EXACTLY_ONCE"
        },
        "sources": [{"type": "websocket", "name": "test", "url": "ws://test.com"}],
        "sinks": [{"type": "kafka", "name": "test", "bootstrap_servers": "kafka:9092"}]
    }

    errors = compiler.validate_config(valid_config)
    assert len(errors) == 0

    # Test invalid config
    invalid_config = {
        "job_type": "streaming",
        "sources": [],  # Missing sources
        "sinks": []  # Missing sinks
    }

    errors = compiler.validate_config(invalid_config)
    assert len(errors) > 0
    print(f"Validation errors: {errors}")
    # The validation should check for missing required keys first
    assert any("Missing required configuration: flink_config" in error for error in errors)
    # Since flink_config is missing, it should also check for missing flink config keys
    assert any("Missing required Flink configuration" in error for error in errors)

    print(f"✓ Validation found {len(errors)} errors as expected")


def test_unsupported_mode():
    """Test compilation with unsupported mode."""
    print("Testing unsupported mode...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="batch-test",
        provider=ProviderConfig(
            name="batch_provider",
            display_name="Batch Provider",
            provider_type=ProviderType.REST_API,
            mode=IngestionMode.BATCH,  # Not supported by Flink compiler
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

    compiler = FlinkCompiler()

    try:
        compiler.compile(spec)
        assert False, "Should have raised FlinkCompileError"
    except FlinkCompileError as e:
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
            provider_type=ProviderType.WEBSOCKET,
            base_url="wss://api.json.com",
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.WEBSOCKET,
            endpoints=[
                EndpointConfig(name="data", path="/ws/data")
            ],
            sinks=[{
                "type": SinkType.KAFKA,
                "kafka_topic": "json-data",
                "config": {}
            }]
        ),
        created_by="test-user"
    )

    compiler = FlinkCompiler()
    json_output = compiler.compile_to_json(spec)

    # Should be valid JSON
    config = json.loads(json_output)
    assert "flink_config" in config
    assert "sources" in config
    assert "sinks" in config

    print("✓ JSON output generation successful")


def test_state_management():
    """Test state management configuration."""
    print("Testing state management...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="state-test",
        provider=ProviderConfig(
            name="stateful_provider",
            display_name="Stateful Provider",
            provider_type=ProviderType.KAFKA,
            config={"kafka_bootstrap_servers": "kafka:9092"},
            tenant_id="test-tenant",
            owner="test@example.com",
            mode=IngestionMode.STREAMING,
            endpoints=[
                EndpointConfig(name="events", path="user-events")
            ],
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "stateful_events",
                "config": {}
            }]
        ),
        created_by="test-user"
    )

    compiler = FlinkCompiler()
    config = compiler.compile(spec)

    # Should have state management configuration
    assert "state_management" in config
    state_config = config["state_management"]
    assert state_config["backend"] == "rocksdb"
    assert state_config["checkpoint_config"]["mode"] == "EXACTLY_ONCE"

    print("✓ State management configuration successful")


def run_all_tests():
    """Run all Flink compiler tests."""
    print("Running Flink Compiler Tests\n" + "="*40)

    try:
        # Basic compilation tests
        test_compile_websocket_streaming()
        test_compile_webhook_streaming()
        test_compile_kafka_streaming()
        test_compile_cdc_streaming()

        # Output tests
        test_flink_sql_generation()
        test_flink_args_generation()
        test_json_output()

        # Validation tests
        test_validation()

        # State management tests
        test_state_management()

        # Error handling tests
        test_unsupported_mode()

        print("\n" + "="*40)
        print("✓ All Flink compiler tests passed!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
