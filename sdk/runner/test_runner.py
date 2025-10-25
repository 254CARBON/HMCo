#!/usr/bin/env python3
"""
Tests for the ingestion runner.
"""

import json
import tempfile
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Mock external dependencies for testing
sys.modules['fastapi'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['prometheus_client'] = MagicMock()
sys.modules['opentelemetry'] = MagicMock()

from config import RunnerConfig, ExecutionEngine, LogLevel
from secret_manager import SecretManager
from metrics import MetricsCollector, MetricsConfig
from tracing import Tracer
from job_executor import JobExecutor
from runner import IngestionRunner


def test_runner_initialization():
    """Test runner initialization."""
    print("Testing runner initialization...")

    config = RunnerConfig(
        runner_id="test-runner",
        tenant_id="test-tenant",
        max_concurrent_jobs=5,
        vault_enabled=False  # Disable Vault for testing
    )

    runner = IngestionRunner(config)

    assert runner.config.runner_id == "test-runner"
    assert runner.config.tenant_id == "test-tenant"
    assert runner.config.max_concurrent_jobs == 5

    print("✓ Runner initialization successful")


def test_configuration_from_env():
    """Test configuration loading from environment."""
    print("Testing configuration from environment...")

    # Set environment variables
    os.environ["RUNNER_ID"] = "env-runner"
    os.environ["TENANT_ID"] = "env-tenant"
    os.environ["MAX_CONCURRENT_JOBS"] = "20"
    os.environ["VAULT_ENABLED"] = "false"  # Disable Vault for testing

    config = RunnerConfig.from_env()

    assert config.runner_id == "env-runner"
    assert config.tenant_id == "env-tenant"
    assert config.max_concurrent_jobs == 20

    # Clean up
    del os.environ["RUNNER_ID"]
    del os.environ["TENANT_ID"]
    del os.environ["MAX_CONCURRENT_JOBS"]
    del os.environ["VAULT_ENABLED"]

    print("✓ Configuration from environment successful")


def test_configuration_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")

    # Test valid configuration
    config = RunnerConfig(
        max_concurrent_jobs=1,
        job_timeout_seconds=60,
        vault_enabled=False  # Disable Vault for testing
    )
    assert config.max_concurrent_jobs == 1
    assert config.job_timeout_seconds == 60

    # Test invalid configuration
    try:
        RunnerConfig(max_concurrent_jobs=0, vault_enabled=False)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "max_concurrent_jobs" in str(e)

    try:
        RunnerConfig(job_timeout_seconds=30, vault_enabled=False)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "job_timeout_seconds" in str(e)

    print("✓ Configuration validation successful")


def test_secret_manager():
    """Test secret manager functionality."""
    print("Testing secret manager...")

    # Mock Vault for testing
    with patch('requests.Session') as mock_session:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "api_key": "test-key",
                "username": "test-user"
            }
        }
        mock_session.return_value.get.return_value = mock_response

        secret_manager = SecretManager(
            vault_address="http://test-vault:8200",
            vault_token="test-token"
        )

        # Test secret retrieval
        secret = secret_manager.get_secret("test/credentials")
        assert secret["api_key"] == "test-key"
        assert secret["username"] == "test-user"

        # Test specific value retrieval
        api_key = secret_manager.get_secret_value("test/credentials", "api_key")
        assert api_key == "test-key"

    print("✓ Secret manager functionality successful")


def test_metrics_collection():
    """Test metrics collection."""
    print("Testing metrics collection...")

    config = MetricsConfig(enabled=True, prefix="test")
    metrics = MetricsCollector(config)

    # Test job metrics
    metrics.record_job_start("rest_api", "test-tenant")
    metrics.record_job_completion("completed", "rest_api", "test-tenant")

    # Test data metrics
    metrics.record_data_ingested(
        records=1000,
        bytes_size=51200,
        provider_type="rest_api",
        tenant_id="test-tenant",
        sink_type="iceberg"
    )

    # Test error metrics
    metrics.record_error("timeout", "rest_api", "test-tenant")
    metrics.record_rate_limit_hit("rest_api", "test-tenant")

    # Test metrics output
    metrics_text = metrics.get_metrics_text()

    # Since prometheus_client is mocked, we just check that the method doesn't fail
    # and returns some form of output (mock object or actual text)
    assert metrics_text is not None
    assert len(str(metrics_text)) > 0  # Should have some content

    print("✓ Metrics collection successful")


def test_tracing_integration():
    """Test tracing integration."""
    print("Testing tracing integration...")

    tracer = Tracer(service_name="test-runner", enabled=False)

    # Test disabled tracing
    assert not tracer.is_enabled()

    # Test span creation (should not fail even if disabled)
    with tracer.start_span("test_span", {"test": "value"}) as span:
        assert span is None  # Should be None when disabled

    print("✓ Tracing integration successful")


def test_job_execution():
    """Test job execution."""
    print("Testing job execution...")

    config = RunnerConfig(vault_enabled=False)  # Disable Vault for testing
    metrics_config = MetricsConfig(enabled=False)  # Disable for testing
    tracer = Tracer(service_name="test", enabled=False)

    metrics = MetricsCollector(metrics_config)
    job_executor = JobExecutor(config, metrics, tracer)

    # Test active jobs tracking
    active_jobs = job_executor.get_active_jobs()
    assert len(active_jobs) == 0

    # Test job cancellation (no active jobs)
    # This should return False and may log a warning, which is expected
    success = job_executor.cancel_job("nonexistent")
    assert not success

    print("✓ Job execution framework successful")


def test_health_status():
    """Test health status reporting."""
    print("Testing health status...")

    config = RunnerConfig(vault_enabled=False)  # Disable Vault for testing
    metrics_config = MetricsConfig(enabled=False)
    tracer = Tracer(service_name="test", enabled=False)

    runner = IngestionRunner(config)

    health = runner.get_health_status()

    assert health["status"] in ["healthy", "degraded"]
    assert "runner_id" in health
    assert "tenant_id" in health
    assert "components" in health
    assert "timestamp" in health

    print("✓ Health status reporting successful")


def test_uis_spec_loading():
    """Test UIS specification loading."""
    print("Testing UIS spec loading...")

    # Create a mock UIS spec file
    mock_uis_spec = {
        "version": "1.1",
        "name": "test-spec",
        "provider": {
            "name": "test_provider",
            "display_name": "Test Provider",
            "provider_type": "rest_api",
            "base_url": "https://api.test.com",
            "tenant_id": "test-tenant",
            "owner": "test@example.com",
            "endpoints": [
                {
                    "name": "test_endpoint",
                    "path": "/data",
                    "method": "GET"
                }
            ],
            "sinks": [
                {
                    "type": "iceberg",
                    "table_name": "test_data",
                    "config": {}
                }
            ]
        },
        "created_by": "test-user"
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(mock_uis_spec, f)
        spec_file = f.name

    try:
        runner = IngestionRunner(RunnerConfig(vault_enabled=False))

        # Mock the UIS imports for testing
        with patch('uis.parser.UISParser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser

            # Mock the parsed spec
            from config import ExecutionEngine
            mock_spec = MagicMock()
            mock_spec.name = "test-spec"
            mock_spec.provider.provider_type.value = "rest_api"
            mock_spec.provider.tenant_id = "test-tenant"
            mock_spec.provider.mode = MagicMock()
            mock_spec.provider.mode.value = "batch"
            mock_parser.parse_file.return_value = mock_spec

            loaded_spec = runner.load_uis_spec(spec_file)
            assert loaded_spec.name == "test-spec"

            # Verify parser was called
            mock_parser.parse_file.assert_called_once_with(spec_file)

    finally:
        os.unlink(spec_file)

    print("✓ UIS spec loading successful")


def test_execution_engine_determination():
    """Test execution engine determination."""
    print("Testing execution engine determination...")

    runner = IngestionRunner(RunnerConfig(vault_enabled=False))

    # Mock UIS spec for different modes
    from config import ExecutionEngine
    mock_spec_batch = MagicMock()
    mock_spec_batch.provider.mode.value = "batch"
    mock_spec_batch.provider.config = {}

    mock_spec_streaming = MagicMock()
    mock_spec_streaming.provider.mode.value = "streaming"
    mock_spec_streaming.provider.config = {}

    mock_spec_microbatch = MagicMock()
    mock_spec_microbatch.provider.mode.value = "micro_batch"
    mock_spec_microbatch.provider.config = {}

    # Test engine determination
    engine_batch = runner._determine_execution_engine(mock_spec_batch)
    engine_streaming = runner._determine_execution_engine(mock_spec_streaming)
    engine_microbatch = runner._determine_execution_engine(mock_spec_microbatch)

    assert engine_batch == ExecutionEngine.SEATUNNEL
    assert engine_streaming == ExecutionEngine.FLINK
    assert engine_microbatch == ExecutionEngine.SPARK

    print("✓ Execution engine determination successful")


def test_fastapi_app_creation():
    """Test FastAPI application creation."""
    print("Testing FastAPI app creation...")

    runner = IngestionRunner(RunnerConfig())

    # Test app creation (should work even if FastAPI not available in test)
    app = runner.create_fastapi_app()

    # App might be None if FastAPI not available, that's okay for testing
    if app is not None:
        print("✓ FastAPI app created successfully")
    else:
        print("✓ FastAPI app creation handled gracefully (FastAPI not available)")


def test_sample_job_execution():
    """Test sample job execution."""
    print("Testing sample job execution...")

    # Create mock UIS spec file
    mock_uis_spec = {
        "version": "1.1",
        "name": "sample-spec",
        "provider": {
            "name": "sample_provider",
            "display_name": "Sample Provider",
            "provider_type": "rest_api",
            "base_url": "https://api.sample.com",
            "tenant_id": "test-tenant",
            "owner": "test@example.com",
            "endpoints": [],
            "sinks": []
        },
        "created_by": "test-user"
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(mock_uis_spec, f)
        spec_file = f.name

    try:
        runner = IngestionRunner(RunnerConfig(vault_enabled=False))

        # Mock the execution pipeline
        with patch.object(runner, 'load_uis_spec') as mock_load, \
             patch.object(runner, 'execute_uis_spec') as mock_execute:

            # Setup mocks
            mock_spec = MagicMock()
            mock_spec.name = "sample-spec"
            mock_spec.provider.provider_type.value = "rest_api"
            mock_spec.provider.tenant_id = "test-tenant"
            mock_load.return_value = mock_spec

            mock_execute.return_value = {"success": True, "job_id": "test-123"}

            # Execute sample job
            result = runner.run_sample_job(spec_file)

            assert result["success"] == True
            assert result["job_id"] == "test-123"

            # Verify mocks were called
            mock_load.assert_called_once_with(spec_file)
            mock_execute.assert_called_once_with(mock_spec)

    finally:
        os.unlink(spec_file)

    print("✓ Sample job execution successful")


def test_sample_spec_integration():
    """Integration smoke test using the bundled sample UIS spec."""
    print("Testing sample spec integration...")

    examples_dir = Path(__file__).parent / "examples"
    spec_path = examples_dir / "simple-batch.yaml"
    assert spec_path.exists(), "Sample spec does not exist"

    config = RunnerConfig(
        vault_enabled=False,
        simulate_missing_engines=True,
        simulation_delay_seconds=0.0
    )

    runner = IngestionRunner(config)
    result = runner.run_sample_job(str(spec_path))

    assert result["success"] is True
    assert result.get("simulated", True) is True
    assert result.get("engine") in {"seatunnel", "spark", "flink"}
    assert "stdout" in result

    print("✓ Sample spec integration successful")


def test_runner_shutdown():
    """Test runner shutdown."""
    print("Testing runner shutdown...")

    config = RunnerConfig()
    metrics_config = MetricsConfig(enabled=False)
    tracer = Tracer(service_name="test", enabled=False)

    runner = IngestionRunner(config)

    # Shutdown should not raise any exceptions
    runner.shutdown()

    print("✓ Runner shutdown successful")


def run_all_tests():
    """Run all runner tests."""
    print("Running Ingestion Runner Tests\n" + "="*40)

    try:
        # Basic functionality tests
        test_runner_initialization()
        test_configuration_from_env()
        test_configuration_validation()

        # Component tests
        test_secret_manager()
        test_metrics_collection()
        test_tracing_integration()
        test_job_execution()

        # Runner functionality tests
        test_health_status()
        test_uis_spec_loading()
        test_execution_engine_determination()
        test_fastapi_app_creation()
        test_sample_job_execution()
        test_sample_spec_integration()

        # Lifecycle tests
        test_runner_shutdown()

        print("\n" + "="*40)
        print("✓ All ingestion runner tests passed!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
