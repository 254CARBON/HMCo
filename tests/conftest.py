"""
Pytest configuration and shared fixtures
"""
import pytest
import os
from typing import Generator
from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

@pytest.fixture(scope="session")
def k8s_client() -> Generator:
    """Kubernetes client for integration tests"""
    try:
        config.load_incluster_config()
    except ConfigException:
        try:
            config.load_kube_config()
        except ConfigException:
            pytest.skip("Kubernetes not accessible in test environment")
    
    yield client.CoreV1Api()

@pytest.fixture(scope="session")
def namespace():
    """Default namespace for tests"""
    return os.getenv("TEST_NAMESPACE", "data-platform")

@pytest.fixture
def datahub_url(namespace):
    """DataHub GMS URL"""
    return f"http://datahub-gms.{namespace}.svc.cluster.local:8080"

@pytest.fixture
def trino_url(namespace):
    """Trino coordinator URL"""
    return f"http://trino-coordinator.{namespace}.svc.cluster.local:8080"

@pytest.fixture
def dolphinscheduler_url(namespace):
    """DolphinScheduler API URL"""
    return f"http://dolphinscheduler-api.{namespace}.svc.cluster.local:12345"

@pytest.fixture
def mlflow_url(namespace):
    """MLflow tracking server URL"""
    return f"http://mlflow.{namespace}.svc.cluster.local:5000"

@pytest.fixture
def feast_url(namespace):
    """Feast feature server URL"""
    return f"http://feast-server.{namespace}.svc.cluster.local:6566"

@pytest.fixture
def postgres_connection(namespace):
    """PostgreSQL connection string"""
    host = f"postgres-shared-service.{namespace}.svc.cluster.local"
    return {
        "host": host,
        "port": 5432,
        "database": "datahub",
        "user": "postgres"
    }

@pytest.fixture
def kafka_bootstrap_servers(namespace):
    """Kafka bootstrap servers"""
    return f"kafka-service.{namespace}.svc.cluster.local:9093"

@pytest.fixture(autouse=True)
def reset_test_state():
    """Reset state before each test"""
    yield
    # Cleanup code here if needed

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests"
    )


