"""
Tests for new data platform capabilities
Validates structure and basic functionality of new services
"""
import os
import json
import yaml
import pytest
from pathlib import Path


class TestHelmCharts:
    """Test helm chart structure and validity"""
    
    @pytest.fixture
    def repo_root(self):
        return Path(__file__).parent.parent.parent
    
    def test_lakefs_chart_exists(self, repo_root):
        """Test lakeFS chart structure"""
        chart_path = repo_root / "helm/charts/data-platform/charts/lakefs"
        assert chart_path.exists(), "lakeFS chart directory should exist"
        
        # Check required files
        assert (chart_path / "Chart.yaml").exists()
        assert (chart_path / "values.yaml").exists()
        assert (chart_path / "templates").exists()
        
        # Validate Chart.yaml
        with open(chart_path / "Chart.yaml") as f:
            chart = yaml.safe_load(f)
        assert chart['name'] == 'lakefs'
        assert chart['version'] == '1.0.0'
    
    def test_schema_registry_chart_exists(self, repo_root):
        """Test Schema Registry chart structure"""
        chart_path = repo_root / "helm/charts/streaming/schema-registry"
        assert chart_path.exists(), "Schema Registry chart should exist"
        
        with open(chart_path / "Chart.yaml") as f:
            chart = yaml.safe_load(f)
        assert chart['name'] == 'schema-registry'
    
    def test_marquez_chart_exists(self, repo_root):
        """Test Marquez/OpenLineage chart structure"""
        chart_path = repo_root / "helm/charts/data-platform/charts/marquez"
        assert chart_path.exists(), "Marquez chart should exist"
        
        with open(chart_path / "values.yaml") as f:
            values = yaml.safe_load(f)
        assert 'marquez' in values
        assert 'openlineage' in values
    
    def test_debezium_chart_exists(self, repo_root):
        """Test Debezium CDC chart structure"""
        chart_path = repo_root / "helm/charts/streaming/debezium"
        assert chart_path.exists(), "Debezium chart should exist"
        
        with open(chart_path / "values.yaml") as f:
            values = yaml.safe_load(f)
        assert 'connectors' in values
        assert 'outputTargets' in values
    
    def test_vault_transform_chart_exists(self, repo_root):
        """Test Vault Transform chart for column security"""
        chart_path = repo_root / "helm/charts/security/vault-transform"
        assert chart_path.exists(), "Vault Transform chart should exist"
        
        with open(chart_path / "values.yaml") as f:
            values = yaml.safe_load(f)
        assert 'transform' in values
        assert 'clickhouse' in values
        assert 'trino' in values


class TestDBT:
    """Test dbt analytics project"""
    
    @pytest.fixture
    def dbt_root(self):
        return Path(__file__).parent.parent.parent / "analytics/dbt"
    
    def test_dbt_project_exists(self, dbt_root):
        """Test dbt project structure"""
        assert dbt_root.exists(), "dbt directory should exist"
        assert (dbt_root / "dbt_project.yml").exists()
        assert (dbt_root / "profiles.yml").exists()
        assert (dbt_root / "models").exists()
    
    def test_dbt_project_config(self, dbt_root):
        """Test dbt project configuration"""
        with open(dbt_root / "dbt_project.yml") as f:
            config = yaml.safe_load(f)
        
        assert config['name'] == 'hmco_analytics'
        assert 'models' in config
        assert 'hmco_analytics' in config['models']
    
    def test_dbt_models_exist(self, dbt_root):
        """Test dbt models exist"""
        models_path = dbt_root / "models"
        
        # Check staging models
        assert (models_path / "staging/stg_lmp_data.sql").exists()
        
        # Check mart models
        assert (models_path / "marts/lmp/lmp_hourly_summary.sql").exists()
        assert (models_path / "marts/weather/weather_lmp_join.sql").exists()
    
    def test_dbt_schema_yml_exists(self, dbt_root):
        """Test dbt schema.yml with tests"""
        schema_path = dbt_root / "models/schema.yml"
        assert schema_path.exists()
        
        with open(schema_path) as f:
            schema = yaml.safe_load(f)
        
        assert 'sources' in schema
        assert 'models' in schema


class TestServices:
    """Test microservices structure"""
    
    @pytest.fixture
    def services_root(self):
        return Path(__file__).parent.parent.parent / "services"
    
    def test_data_sharing_service_exists(self, services_root):
        """Test data sharing service structure"""
        service_path = services_root / "data-sharing"
        assert service_path.exists()
        assert (service_path / "requirements.txt").exists()
        assert (service_path / "app/main.py").exists()
        
        # Check FastAPI app exists
        main_content = (service_path / "app/main.py").read_text()
        assert "FastAPI" in main_content
        assert "Partner" in main_content
        assert "DatasetEntitlement" in main_content
    
    def test_ch_mv_optimizer_exists(self, services_root):
        """Test ClickHouse MV optimizer service"""
        service_path = services_root / "ch-mv-optimizer"
        assert service_path.exists()
        assert (service_path / "requirements.txt").exists()
        assert (service_path / "app/optimizer.py").exists()
        assert (service_path / "config/policy.yaml").exists()
        
        # Check policy configuration
        with open(service_path / "config/policy.yaml") as f:
            policy = yaml.safe_load(f)
        assert 'optimizer' in policy
        assert 'guardrails' in policy
        assert 'patterns' in policy
    
    def test_cost_attribution_exists(self, services_root):
        """Test cost attribution service"""
        service_path = services_root / "cost-attribution"
        assert service_path.exists()
        assert (service_path / "requirements.txt").exists()
        assert (service_path / "app/collector.py").exists()
        assert (service_path / "dashboards/cost-dashboard.json").exists()
        
        # Check dashboard configuration
        with open(service_path / "dashboards/cost-dashboard.json") as f:
            dashboard = json.load(f)
        assert 'dashboard' in dashboard


class TestSchemas:
    """Test schema definitions"""
    
    @pytest.fixture
    def schema_root(self):
        return Path(__file__).parent.parent.parent / "sdk/uis/schema"
    
    def test_uis_12_schema_exists(self, schema_root):
        """Test UIS 1.2 schema with schema registry support"""
        schema_path = schema_root / "uis-1.2.json"
        assert schema_path.exists()
        
        with open(schema_path) as f:
            schema = json.load(f)
        
        # Check for schema registry fields
        assert 'properties' in schema
        assert 'schemaRef' in schema['properties']
        assert 'compatMode' in schema['properties']
    
    def test_uis_12_backward_compatible(self, schema_root):
        """Test UIS 1.2 is backward compatible with 1.1"""
        v11_path = schema_root / "uis-1.1.json"
        v12_path = schema_root / "uis-1.2.json"
        
        with open(v11_path) as f:
            v11 = json.load(f)
        
        with open(v12_path) as f:
            v12 = json.load(f)
        
        # All required fields in v1.1 should be in v1.2
        v11_required = set(v11.get('required', []))
        v12_required = set(v12.get('required', []))
        
        assert v11_required.issubset(v12_required), \
            f"Missing required fields: {v11_required - v12_required}"


class TestCIWorkflows:
    """Test CI/CD workflow configurations"""
    
    @pytest.fixture
    def workflows_root(self):
        return Path(__file__).parent.parent.parent / ".github/workflows"
    
    def test_schema_compatibility_workflow_exists(self, workflows_root):
        """Test schema compatibility workflow"""
        workflow_path = workflows_root / "schema-compatibility.yml"
        assert workflow_path.exists()
        
        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)
        
        assert 'jobs' in workflow
        assert 'schema-compatibility' in workflow['jobs']
    
    def test_dbt_test_workflow_exists(self, workflows_root):
        """Test dbt test workflow"""
        workflow_path = workflows_root / "dbt-test.yml"
        assert workflow_path.exists()
        
        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)
        
        assert 'jobs' in workflow
        assert 'dbt-test' in workflow['jobs']


class TestDocumentation:
    """Test documentation exists and is comprehensive"""
    
    @pytest.fixture
    def docs_root(self):
        return Path(__file__).parent.parent.parent / "docs"
    
    def test_new_capabilities_doc_exists(self, docs_root):
        """Test NEW_CAPABILITIES.md documentation"""
        doc_path = docs_root / "NEW_CAPABILITIES.md"
        assert doc_path.exists()
        
        content = doc_path.read_text()
        
        # Check all 10 capabilities are documented
        assert "lakeFS" in content
        assert "Schema Registry" in content
        assert "OpenLineage" in content
        assert "Debezium" in content
        assert "dbt" in content
        assert "Data Sharing" in content
        assert "Adaptive Materialization" in content
        assert "Column-Level Security" in content
        assert "Autoscaling" in content
        assert "Cost Attribution" in content
    
    def test_lakefs_readme_exists(self):
        """Test lakeFS README"""
        readme_path = Path(__file__).parent.parent.parent / \
                     "helm/charts/data-platform/charts/lakefs/README.md"
        assert readme_path.exists()
        
        content = readme_path.read_text()
        assert "Branch/Merge/Rollback" in content
    
    def test_dbt_readme_exists(self):
        """Test dbt README"""
        readme_path = Path(__file__).parent.parent.parent / \
                     "analytics/dbt/README.md"
        assert readme_path.exists()
        
        content = readme_path.read_text()
        assert "dbt run" in content


class TestTrinoAutoscaling:
    """Test Trino autoscaling configuration"""
    
    @pytest.fixture
    def trino_root(self):
        return Path(__file__).parent.parent.parent / \
               "helm/charts/data-platform/charts/trino/templates"
    
    def test_keda_scaledobject_exists(self, trino_root):
        """Test KEDA ScaledObject for Trino"""
        template_path = trino_root / "keda-scaledobject.yaml"
        assert template_path.exists()
        
        content = template_path.read_text()
        assert "ScaledObject" in content
        assert "trino-worker" in content
    
    def test_resource_groups_config_exists(self, trino_root):
        """Test resource groups configuration"""
        template_path = trino_root / "resource-groups-config.yaml"
        assert template_path.exists()
        
        content = template_path.read_text()
        assert "interactive" in content
        assert "etl" in content
        assert "adhoc" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
