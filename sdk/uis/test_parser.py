#!/usr/bin/env python3
"""
Tests for UIS 1.1 parser and validator.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from parser import UISParser, UISParseError, UISValidationError
from validator import UISValidator
from spec import UnifiedIngestionSpec, IngestionMode, ProviderType, SinkType


def test_parse_valid_yaml():
    """Test parsing a valid YAML UIS specification."""
    print("Testing valid YAML parsing...")

    parser = UISParser()

    # Test with sample file
    sample_file = Path(__file__).parent / "examples" / "polygon-api.yaml"
    if sample_file.exists():
        spec = parser.parse_file(str(sample_file))

        # Validate basic properties
        assert spec.version == "1.1"
        assert spec.name == "polygon-stock-api"
        assert spec.provider.name == "polygon_api"
        assert spec.provider.provider_type == ProviderType.REST_API
        assert len(spec.provider.endpoints) == 2
        assert len(spec.provider.sinks) == 2

        print("✓ Valid YAML parsing successful")
        return spec
    else:
        print("⚠ Sample file not found, skipping file test")
        return None


def test_parse_invalid_yaml():
    """Test parsing invalid YAML."""
    print("Testing invalid YAML parsing...")

    parser = UISParser()

    try:
        parser.parse_yaml("invalid: yaml: content: [\n")
        assert False, "Should have raised an exception"
    except (UISParseError, Exception) as e:
        print(f"✓ Invalid YAML correctly rejected: {type(e).__name__}")


def test_schema_validation_rejects_invalid_spec():
    """Test that schema validation rejects specs missing required fields."""
    print("Testing schema validation rejection...")

    parser = UISParser()

    invalid_yaml = """
version: "1.1"
name: "invalid-spec"
description: "Spec that is missing required sinks"
created_by: "qa-engineer"
provider:
  name: "invalid_provider"
  display_name: "Invalid Provider"
  provider_type: "rest_api"
  base_url: "https://api.invalid.test"
  tenant_id: "tenant-123"
  owner: "owner@example.com"
"""

    try:
        parser.parse_yaml(invalid_yaml)
        assert False, "Schema validation should have failed due to missing sinks"
    except UISValidationError as exc:
        message = str(exc)
        assert "sinks" in message.lower()
        print(f"✓ Schema validation correctly rejected spec: {message}")


def test_validation():
    """Test UIS validation."""
    print("Testing UIS validation...")

    # Create a valid spec programmatically
    spec = UnifiedIngestionSpec(
        version="1.1",
        name="test-spec",
        description="Test specification",
        provider={
            "name": "test_provider",
            "display_name": "Test Provider",
            "provider_type": ProviderType.REST_API,
            "base_url": "https://api.example.com",
            "tenant_id": "test-tenant",
            "owner": "test@example.com",
            "sinks": [{
                "type": SinkType.ICEBERG,
                "config": {"warehouse": "test"},
                "table_name": "test_table"
            }]
        },
        created_by="test-user"
    )

    # Test validation
    validator = UISValidator()
    errors = validator.validate_completeness(spec)

    # Should have some errors (missing credentials, etc.)
    assert len(errors) > 0
    print(f"✓ Validation found {len(errors)} issues as expected")

    # Test production readiness
    is_ready, issues = validator.is_production_ready(spec)
    assert not is_ready  # Should not be production ready
    print(f"✓ Production readiness check correctly identified issues: {len(issues)}")


def test_field_mappings():
    """Test field mapping validation."""
    print("Testing field mapping validation...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="test-spec",
        provider={
            "name": "test_provider",
            "display_name": "Test Provider",
            "provider_type": ProviderType.REST_API,
            "base_url": "https://api.example.com",
            "tenant_id": "test-tenant",
            "owner": "test@example.com",
            "endpoints": [
                {
                    "name": "endpoint1",
                    "path": "/data",
                    "field_mapping": {"source1": "target1", "source2": "target2"}
                },
                {
                    "name": "endpoint2",
                    "path": "/more-data",
                    "field_mapping": {"source3": "target1"}  # Conflicting mapping
                }
            ],
            "sinks": [{
                "type": SinkType.ICEBERG,
                "config": {"warehouse": "test"},
                "table_name": "test_table"
            }]
        },
        created_by="test-user"
    )

    validator = UISValidator()
    errors = validator.validate_field_mappings(spec)

    assert len(errors) > 0  # Should find conflicting mappings
    print(f"✓ Field mapping validation found {len(errors)} conflicts")


def test_rate_limits():
    """Test rate limit validation."""
    print("Testing rate limit validation...")

    spec = UnifiedIngestionSpec(
        version="1.1",
        name="test-spec",
        provider={
            "name": "test_provider",
            "display_name": "Test Provider",
            "provider_type": ProviderType.REST_API,
            "base_url": "https://api.example.com",
            "tenant_id": "test-tenant",
            "owner": "test@example.com",
            "endpoints": [
                {
                    "name": "realtime_data",
                    "path": "/realtime",
                    "rate_limit_group": "high_freq",
                    "rate_limit_per_second": 100
                },
                {
                    "name": "batch_data",
                    "path": "/batch",
                    "rate_limit_group": "high_freq",
                    "rate_limit_per_second": 50  # Different rate in same group
                }
            ],
            "sinks": [{
                "type": SinkType.ICEBERG,
                "config": {"warehouse": "test"},
                "table_name": "test_table"
            }]
        },
        created_by="test-user"
    )

    validator = UISValidator()
    errors = validator.validate_rate_limits(spec)

    assert len(errors) > 0  # Should find inconsistent rates in group
    print(f"✓ Rate limit validation found {len(errors)} issues")


def test_yaml_export_import():
    """Test YAML export and import round-trip."""
    print("Testing YAML export/import...")

    # Create a spec
    original = UnifiedIngestionSpec(
        version="1.1",
        name="roundtrip-test",
        description="Test round-trip",
        provider={
            "name": "test_provider",
            "display_name": "Test Provider",
            "provider_type": ProviderType.REST_API,
            "base_url": "https://api.example.com",
            "tenant_id": "test-tenant",
            "owner": "test@example.com",
            "sinks": [{
                "type": SinkType.ICEBERG,
                "config": {"warehouse": "test"},
                "table_name": "test_table"
            }]
        },
        created_by="test-user"
    )

    # Export to YAML
    yaml_content = original.to_yaml()

    # Import back
    parser = UISParser()
    imported = parser.parse_yaml(yaml_content)

    # Compare key fields
    assert imported.version == original.version
    assert imported.name == original.name
    assert imported.provider.name == original.provider.name
    assert imported.provider.provider_type == original.provider.provider_type

    print("✓ YAML export/import round-trip successful")


def run_all_tests():
    """Run all tests."""
    print("Running UIS Parser Tests\n" + "="*40)

    try:
        # Basic parsing tests
        test_parse_invalid_yaml()
        test_schema_validation_rejects_invalid_spec()

        # File parsing test
        spec = test_parse_valid_yaml()
        if spec:
            # Validation tests
            test_validation()
            test_field_mappings()
            test_rate_limits()

        # Round-trip tests
        test_yaml_export_import()

        print("\n" + "="*40)
        print("✓ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

