#!/usr/bin/env python3
"""
Database CRUD test script.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from models.database import get_db
from models import ExternalProvider, ProviderEndpoint, ProviderRun
from models.provider import ProviderType, ProviderStatus
from models.run import RunStatus


def test_crud_operations():
    """Test basic CRUD operations."""
    print("Testing CRUD operations...")

    # Create a database session
    db = get_db()

    try:
        # Create a test provider
        provider = ExternalProvider(
            name="test_polygon_api",
            display_name="Polygon Stock API",
            description="Test provider for Polygon stock market data",
            provider_type=ProviderType.REST_API,
            status=ProviderStatus.ACTIVE,
            base_url="https://api.polygon.io",
            tenant_id="test-tenant",
            owner="test-user",
            tags=["stocks", "market-data"],
            sink_type="iceberg",
            created_by="test-user",
            updated_by="test-user"
        )

        db.add(provider)
        db.commit()
        db.refresh(provider)

        print(f"Created provider: {provider.id} - {provider.name}")

        # Create a test endpoint
        endpoint = ProviderEndpoint(
            provider_id=provider.id,
            name="stocks_endpoint",
            path="/v3/reference/tickers",
            method="GET",
            headers={"Authorization": "Bearer {{api_key}}"},
            query_params={"market": "stocks", "limit": "100"},
            pagination_type="cursor",
            pagination_config={"cursor_param": "cursor", "page_size": 100},
            response_path="$.results",
            is_active=True
        )

        db.add(endpoint)
        db.commit()
        db.refresh(endpoint)

        print(f"Created endpoint: {endpoint.id} - {endpoint.name}")

        # Create a test run
        run = ProviderRun(
            provider_id=provider.id,
            run_id="test-run-001",
            run_mode="batch",
            triggered_by="manual",
            status=RunStatus.COMPLETED,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            records_ingested=1000,
            bytes_ingested=51200,
            duration_ms=30000,
            throughput_records_sec=33,
            data_quality_score=95,
            created_by="test-user"
        )

        db.add(run)
        db.commit()
        db.refresh(run)

        print(f"Created run: {run.id} - {run.run_id}")

        # Test queries
        print("\n--- Query Tests ---")

        # Get provider with relationships
        provider_with_relations = db.query(ExternalProvider).filter(
            ExternalProvider.id == provider.id
        ).first()

        if provider_with_relations:
            print(f"Provider found: {provider_with_relations.name}")
            print(f"Endpoints count: {len(provider_with_relations.endpoints)}")
            print(f"Runs count: {len(provider_with_relations.runs)}")

        # Get runs by status
        completed_runs = db.query(ProviderRun).filter(
            ProviderRun.status == RunStatus.COMPLETED
        ).all()

        print(f"Completed runs: {len(completed_runs)}")

        # Get providers by tenant
        tenant_providers = db.query(ExternalProvider).filter(
            ExternalProvider.tenant_id == "test-tenant"
        ).all()

        print(f"Providers for tenant 'test-tenant': {len(tenant_providers)}")

        print("\n--- CRUD Tests Passed! ---")

    except Exception as e:
        print(f"Error during testing: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    test_crud_operations()


