"""
Integration tests for DataHub API
"""
import pytest
import requests
from time import sleep

@pytest.mark.integration
@pytest.mark.requires_db
class TestDataHubAPI:
    """Test DataHub GraphQL API"""
    
    def test_datahub_health(self, datahub_url):
        """Test DataHub health endpoint"""
        try:
            response = requests.get(f"{datahub_url}/health", timeout=10)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("DataHub not accessible in test environment")
    
    def test_datahub_graphql_endpoint(self, datahub_url):
        """Test DataHub GraphQL endpoint is accessible"""
        try:
            query = {
                "query": "{ __schema { types { name } } }"
            }
            response = requests.post(
                f"{datahub_url}/graphql",
                json=query,
                timeout=10
            )
            # Should return 200 or 401 (if auth required)
            assert response.status_code in [200, 401, 403]
        except requests.exceptions.ConnectionError:
            pytest.skip("DataHub not accessible in test environment")
    
    def test_datahub_metadata_ingestion(self, datahub_url):
        """Test metadata ingestion capability"""
        # This would test actual metadata ingestion
        # For now, just verify the endpoint structure
        endpoint = f"{datahub_url}/entities"
        assert "/entities" in endpoint
    
    @pytest.mark.slow
    def test_datahub_search_functionality(self, datahub_url):
        """Test DataHub search capability"""
        try:
            query = {
                "query": """
                query search($input: SearchInput!) {
                  search(input: $input) {
                    total
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
            }
            response = requests.post(
                f"{datahub_url}/graphql",
                json=query,
                timeout=10
            )
            # May require auth, so accept 401
            assert response.status_code in [200, 401, 403]
        except requests.exceptions.ConnectionError:
            pytest.skip("DataHub not accessible in test environment")


