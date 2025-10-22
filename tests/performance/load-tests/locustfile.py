"""
Load testing for 254Carbon platform APIs
Uses Locust for distributed load testing
"""
from locust import HttpUser, task, between
import random

class DataHubUser(HttpUser):
    """Simulate DataHub API user"""
    wait_time = between(1, 3)
    host = "http://datahub-gms.data-platform.svc.cluster.local:8080"
    
    @task(3)
    def search_datasets(self):
        """Search for datasets"""
        self.client.post("/graphql", json={
            "query": """
            query search($input: SearchInput!) {
              search(input: $input) {
                total
                searchResults {
                  entity {
                    ... on Dataset {
                      urn
                      name
                    }
                  }
                }
              }
            }
            """,
            "variables": {
                "input": {
                    "type": "DATASET",
                    "query": "*",
                    "start": 0,
                    "count": 10
                }
            }
        })
    
    @task(1)
    def get_health(self):
        """Check health endpoint"""
        self.client.get("/health")

class TrinoUser(HttpUser):
    """Simulate Trino query user"""
    wait_time = between(2, 5)
    host = "http://trino-coordinator.data-platform.svc.cluster.local:8080"
    
    @task
    def execute_query(self):
        """Execute sample query"""
        queries = [
            "SELECT COUNT(*) FROM iceberg_catalog.commodity_data.energy_prices",
            "SELECT commodity, AVG(price) FROM iceberg_catalog.commodity_data.energy_prices GROUP BY commodity",
            "SELECT * FROM iceberg_catalog.commodity_data.energy_prices LIMIT 100"
        ]
        
        query = random.choice(queries)
        self.client.post("/v1/statement", 
                        data=query,
                        headers={"X-Trino-User": "load-test"})

class MLFlowUser(HttpUser):
    """Simulate MLflow user"""
    wait_time = between(1, 2)
    host = "http://mlflow.data-platform.svc.cluster.local:5000"
    
    @task(2)
    def list_experiments(self):
        """List experiments"""
        self.client.get("/api/2.0/mlflow/experiments/search")
    
    @task(1)
    def get_run(self):
        """Get run details"""
        self.client.get("/api/2.0/mlflow/runs/search")

class FeastUser(HttpUser):
    """Simulate Feast feature serving user"""
    wait_time = between(0.5, 1.5)
    host = "http://feast-server.data-platform.svc.cluster.local:6566"
    
    @task
    def get_online_features(self):
        """Get online features"""
        self.client.post("/get-online-features", json={
            "features": ["feature1", "feature2"],
            "entities": {"user_id": "123"}
        })


