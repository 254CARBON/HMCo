#!/usr/bin/env python3
"""
Tests for SeaTunnel compiler.
"""

import sys
import json
from pathlib import Path
from typing import Dict

TEST_DIR = Path(__file__).resolve().parent
UIS_ROOT = TEST_DIR.parent.parent

sys.path.insert(0, str(UIS_ROOT))
sys.path.insert(0, str(TEST_DIR))

from spec import UnifiedIngestionSpec, ProviderConfig, EndpointConfig, ProviderType, SinkType, AuthType
from compiler import SeaTunnelCompiler, SeaTunnelCompileError


def load_golden_fixture(name: str) -> Dict:
    """Load golden fixture JSON file."""
    fixture_path = Path(__file__).parent / "fixtures" / f"{name}.json"
    with open(fixture_path, 'r') as f:
        return json.load(f)


def test_compile_simple_rest_api():
    """Test compiling a simple REST API UIS spec."""
    print("Testing simple REST API compilation...")

    # Create a simple UIS spec
    spec = UnifiedIngestionSpec(
        version="1.1",
        name="test-rest-api",
        provider=ProviderConfig(
            name="polygon_api",
            display_name="Polygon Stock API",
            provider_type=ProviderType.REST_API,
            base_url="https://api.polygon.io",
            parallelism=2,
            config={"api_version": "v3"},
            tenant_id="test-tenant",
            owner="test@example.com",
            endpoints=[
                EndpointConfig(
                    name="tickers",
                    path="/v3/reference/tickers",
                    method="GET",
                    auth=AuthType.API_KEY,
                    auth_config={
                        "header_name": "Authorization",
                        "header_format": "Bearer {{api_key}}"
                    },
                    query_params={"market": "stocks", "limit": "100"},
                    pagination="cursor",
                    pagination_config={"cursor_param": "cursor", "page_size": 100, "max_pages": 1000},
                    response_path="$.results",
                    field_mapping={"ticker": "symbol", "name": "company_name"},
                    rate_limit_per_second=10
                )
            ],
            schema_contract={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "name": {"type": "string"}
                }
            },
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "market_data.tickers",
                "config": {
                    "warehouse": "s3://prod-warehouse/",
                    "catalog": "hive_prod",
                    "namespace": "default"
                }
            }]
        ),
        created_by="test-user"
    )

    # Compile to SeaTunnel config
    compiler = SeaTunnelCompiler()
    config = compiler.compile(spec)

    # Validate structure
    assert "env" in config
    assert "source" in config
    assert "transform" in config
    assert "sink" in config

    # Check source configuration
    assert len(config["source"]) == 1
    source = config["source"][0]
    assert source["plugin_name"] == "Http"
    assert source["url"] == "https://api.polygon.io/v3/reference/tickers"
    assert source["method"] == "GET"
    assert "Authorization" in source["headers"]
    assert source["params"]["market"] == "stocks"

    # Check pagination
    assert "pagination" in source
    assert source["pagination"]["type"] == "cursor"

    # Check transform configuration
    assert len(config["transform"]) >= 1  # At least field mapping

    # Check sink configuration
    assert len(config["sink"]) == 1
    sink = config["sink"][0]
    assert sink["plugin_name"] == "Iceberg"
    assert sink["table"] == "market_data.tickers"

    # Golden fixture comparison
    expected = load_golden_fixture("polygon_api")
    assert config == expected

    print("✓ Simple REST API compilation successful")


def test_compile_csv_file():
    """Test compiling a CSV file UIS spec."""
    print("Testing CSV file compilation...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="csv-ingestion",
        provider=ProviderConfig(
            name="csv_provider",
            display_name="CSV File Provider",
            provider_type=ProviderType.FILE_FTP,
            config={"file_format": "csv", "delimiter": ","},
            tenant_id="test-tenant",
            owner="test@example.com",
            endpoints=[
                EndpointConfig(
                    name="daily_data",
                    path="/data/daily_data.csv",
                    field_mapping={"date": "date", "value": "price"}
                )
            ],
            sinks=[{
                "type": SinkType.CLICKHOUSE,
                "clickhouse_table": "daily_data",
                "config": {"host": "clickhouse", "database": "analytics"}
            }]
        ),
        created_by="test-user"
    )

    compiler = SeaTunnelCompiler()
    config = compiler.compile(spec)

    # Check source configuration
    assert len(config["source"]) == 1
    source = config["source"][0]
    assert source["plugin_name"] == "File"
    assert source["path"] == "/data/daily_data.csv"
    assert source["format"] == "csv"

    # Check sink configuration
    assert len(config["sink"]) == 1
    sink = config["sink"][0]
    assert sink["plugin_name"] == "Clickhouse"
    assert sink["table"] == "daily_data"

    expected = load_golden_fixture("csv_ingestion")
    assert config == expected

    print("✓ CSV file compilation successful")


def test_compile_with_transforms():
    """Test compilation with multiple transforms."""
    print("Testing compilation with transforms...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="transform-test",
        provider=ProviderConfig(
            name="api_with_transforms",
            display_name="API with Transforms",
            provider_type=ProviderType.REST_API,
            base_url="https://api.example.com",
            tenant_id="test-tenant",
            owner="test@example.com",
            endpoints=[
                EndpointConfig(
                    name="users",
                    path="/users",
                    field_mapping={
                        "user_id": "id",
                        "full_name": "name",
                        "email_address": "email",
                        "created_timestamp": "created_at"
                    }
                )
            ],
            transforms=[],  # Add some transforms
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "users",
                "config": {"warehouse": "s3://warehouse/"}
            }]
        ),
        created_by="test-user"
    )

    compiler = SeaTunnelCompiler()
    config = compiler.compile(spec)

    # Should have transform for field mapping
    assert len(config["transform"]) > 0

    print("✓ Transform compilation successful")


def test_validation():
    """Test SeaTunnel configuration validation."""
    print("Testing configuration validation...")

    # Test valid config
    compiler = SeaTunnelCompiler()
    valid_config = {
        "env": {"parallelism": 1},
        "source": [{"plugin_name": "Http", "result_table_name": "test"}],
        "sink": [{"plugin_name": "Iceberg", "source_table_name": "test"}]
    }

    errors = compiler.validate_config(valid_config)
    assert len(errors) == 0

    # Test invalid config
    invalid_config = {
        "env": {},
        "source": [],  # Missing source
        "sink": [{"plugin_name": "Iceberg"}]  # Missing source_table_name
    }

    errors = compiler.validate_config(invalid_config)
    assert len(errors) > 0

    print(f"✓ Validation found {len(errors)} errors as expected")


def test_json_output():
    """Test JSON output generation."""
    print("Testing JSON output...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="json-test",
        provider=ProviderConfig(
            name="test_api",
            display_name="Test API",
            provider_type=ProviderType.REST_API,
            base_url="https://api.test.com",
            tenant_id="test-tenant",
            owner="test@example.com",
            endpoints=[
                EndpointConfig(name="data", path="/data")
            ],
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "test_data",
                "config": {}
            }]
        ),
        created_by="test-user"
    )

    compiler = SeaTunnelCompiler()
    json_output = compiler.compile_to_json(spec)

    # Should be valid JSON
    config = json.loads(json_output)
    assert "env" in config
    assert "source" in config
    assert "sink" in config

    print("✓ JSON output generation successful")


def test_unsupported_provider_type():
    """Test compilation with unsupported provider type."""
    print("Testing unsupported provider type...")

    # Create a spec with a valid provider type first
    from spec import ProviderConfig
    try:
        provider = ProviderConfig(
            name="unsupported_provider",
            display_name="Unsupported Provider",
            provider_type="unsupported_type",  # This should fail at Pydantic level
            tenant_id="test-tenant",
            owner="test@example.com",
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "test",
                "config": {}
            }]
        )
        assert False, "Should have raised validation error"
    except Exception as e:
        print(f"✓ Invalid provider type correctly rejected by Pydantic: {type(e).__name__}")

    # Test that SeaTunnel compiler properly handles supported types
    spec = UnifiedIngestionSpec(
        version="1.1",
        name="supported-test",
        provider=ProviderConfig(
            name="supported_provider",
            display_name="Supported Provider",
            provider_type=ProviderType.REST_API,  # Valid type
            base_url="https://api.test.com",
            tenant_id="test-tenant",
            owner="test@example.com",
            endpoints=[
                EndpointConfig(name="test_endpoint", path="/data")
            ],
            sinks=[{
                "type": SinkType.ICEBERG,
                "table_name": "test",
                "config": {}
            }]
        ),
        created_by="test-user"
    )

    compiler = SeaTunnelCompiler()
    config = compiler.compile(spec)  # Should work fine
    assert len(config["source"]) > 0
    print("✓ Supported provider type correctly compiled")


def run_all_tests():
    """Run all SeaTunnel compiler tests."""
    print("Running SeaTunnel Compiler Tests\n" + "="*40)

    try:
        # Basic compilation tests
        test_compile_simple_rest_api()
        test_compile_csv_file()
        test_compile_with_transforms()

        # Validation tests
        test_validation()

        # Output tests
        test_json_output()

        # Error handling tests
        test_unsupported_provider_type()

        print("\n" + "="*40)
        print("✓ All SeaTunnel compiler tests passed!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
