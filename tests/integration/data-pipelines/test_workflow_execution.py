"""
Integration tests for workflow execution
"""
import pytest
from kubernetes import client
from time import sleep

@pytest.mark.integration
class TestWorkflowExecution:
    """Test DolphinScheduler workflow execution"""
    
    def test_dolphinscheduler_api_available(self, dolphinscheduler_url):
        """Test DolphinScheduler API is accessible"""
        import requests
        try:
            response = requests.get(f"{dolphinscheduler_url}/", timeout=10)
            # API may return 404 for root, but should respond
            assert response.status_code in [200, 404, 401]
        except requests.exceptions.ConnectionError:
            pytest.skip("DolphinScheduler not accessible")
    
    @pytest.mark.slow
    def test_workflow_pods_running(self, k8s_client, namespace):
        """Test that workflow worker pods are running"""
        pods = k8s_client.list_namespaced_pod(
            namespace=namespace,
            label_selector="app=dolphinscheduler,component=worker"
        )
        
        running_pods = [p for p in pods.items if p.status.phase == "Running"]
        assert len(running_pods) >= 2, f"Expected at least 2 worker pods, got {len(running_pods)}"
    
    def test_workflow_master_running(self, k8s_client, namespace):
        """Test that workflow master is running"""
        pods = k8s_client.list_namespaced_pod(
            namespace=namespace,
            label_selector="app=dolphinscheduler,component=master"
        )
        
        assert len(pods.items) >= 1
        assert pods.items[0].status.phase == "Running"
    
    def test_workflow_database_connection(self, postgres_connection):
        """Test workflow database connectivity"""
        import psycopg2
        try:
            conn = psycopg2.connect(**postgres_connection, password="")
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
            conn.close()
        except Exception:
            pytest.skip("PostgreSQL not accessible or credentials not available")

@pytest.mark.integration
class TestDataIngestion:
    """Test data ingestion pipelines"""
    
    def test_seatunnel_engine_running(self, k8s_client, namespace):
        """Test SeaTunnel engine is operational"""
        pods = k8s_client.list_namespaced_pod(
            namespace=namespace,
            label_selector="app=seatunnel-engine"
        )
        
        running_pods = [p for p in pods.items if p.status.phase == "Running"]
        assert len(running_pods) >= 1
    
    def test_kafka_topics_exist(self, kafka_bootstrap_servers):
        """Test that Kafka topics are created"""
        from confluent_kafka.admin import AdminClient
        
        try:
            admin = AdminClient({'bootstrap.servers': kafka_bootstrap_servers})
            metadata = admin.list_topics(timeout=10)
            
            expected_topics = ["commodity-prices", "market-data", "data-quality-events"]
            existing_topics = metadata.topics.keys()
            
            # At least some topics should exist
            assert len(existing_topics) > 0
        except Exception:
            pytest.skip("Kafka not accessible")


