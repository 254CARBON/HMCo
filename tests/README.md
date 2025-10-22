# 254Carbon Platform - Testing Framework

Comprehensive testing framework for the 254Carbon data platform.

## Test Structure

```
tests/
├── unit/               # Unit tests (fast, isolated)
│   └── services/      # Service logic tests
├── integration/       # Integration tests (requires services)
│   ├── api/          # API integration tests
│   └── data-pipelines/ # Pipeline integration tests
├── e2e/              # End-to-end tests (full workflows)
│   └── user-journeys/ # User workflow tests
├── performance/       # Performance and load tests
│   └── load-tests/   # Locust load testing
├── conftest.py       # Shared fixtures
├── pytest.ini        # Pytest configuration
└── requirements.txt  # Testing dependencies
```

## Running Tests

### All Tests
```bash
pytest tests/
```

### By Category
```bash
# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v -m integration

# E2E tests
pytest tests/e2e/ -v -m e2e

# Performance tests
pytest tests/performance/ -v -m performance
```

### With Coverage
```bash
pytest tests/ \
  --cov=services \
  --cov-report=html \
  --cov-report=term-missing
```

### Parallel Execution
```bash
pytest tests/ -n auto  # Uses all CPU cores
pytest tests/ -n 4     # Uses 4 workers
```

## Load Testing

### Using Locust
```bash
# Install locust
pip install locust

# Run load test
cd tests/performance/load-tests
locust -f locustfile.py --host=http://datahub-gms.data-platform.svc.cluster.local:8080

# Headless mode (for CI)
locust -f locustfile.py \
  --headless \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m \
  --host=http://datahub-gms.data-platform.svc.cluster.local:8080
```

## CI/CD Integration

Tests run automatically on:
- Every push to main/develop
- Every pull request
- GitHub Actions workflow: `.github/workflows/ci.yml`

### CI Pipeline Stages
1. **Unit Tests**: Fast, isolated tests
2. **Integration Tests**: Service integration
3. **Helm Validation**: Chart linting and templating
4. **Security Scan**: Trivy vulnerability scanning
5. **Kubernetes Validation**: Manifest validation

## Test Markers

Use markers to categorize and filter tests:

```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.e2e           # End-to-end test
@pytest.mark.performance   # Performance test
@pytest.mark.slow          # Test takes >1 second
@pytest.mark.requires_gpu  # Requires GPU
@pytest.mark.requires_db   # Requires database
@pytest.mark.requires_kafka # Requires Kafka
```

### Run Specific Markers
```bash
# Only fast tests
pytest -m "not slow"

# Only tests that don't require external services
pytest -m "unit and not requires_db and not requires_kafka"

# GPU tests only
pytest -m "requires_gpu"
```

## Coverage Goals

- **Unit Tests**: 80% code coverage
- **Integration Tests**: All critical paths
- **E2E Tests**: Main user workflows
- **Performance Tests**: Baseline benchmarks

### Check Coverage
```bash
pytest tests/ --cov --cov-report=term-missing
# View HTML report
open test-reports/coverage-html/index.html
```

## Writing Tests

### Unit Test Example
```python
import pytest

@pytest.mark.unit
class TestMyService:
    def test_function(self):
        result = my_function(input)
        assert result == expected
```

### Integration Test Example
```python
import pytest

@pytest.mark.integration
@pytest.mark.requires_db
class TestDatabaseIntegration:
    def test_query(self, postgres_connection):
        # Use fixture from conftest.py
        result = query_database(postgres_connection)
        assert result is not None
```

### E2E Test Example
```python
import pytest

@pytest.mark.e2e
@pytest.mark.slow
class TestUserJourney:
    def test_full_workflow(self, datahub_url, trino_url):
        # Test complete user workflow
        pass
```

## Performance Testing

### Benchmark Tests
```bash
pytest tests/performance/ --benchmark-only
```

### Load Tests
```bash
locust -f tests/performance/load-tests/locustfile.py
```

## Continuous Integration

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### GitHub Actions
Automatic testing on:
- Push to main/develop
- Pull requests
- Manual workflow dispatch

View results:
- GitHub Actions tab
- Coverage reports in artifacts
- Security scan results in Security tab

## Best Practices

1. **Fast Unit Tests**: Keep under 1 second
2. **Isolated**: No external dependencies in unit tests
3. **Clear Names**: Descriptive test function names
4. **Single Assert**: One assertion per test when possible
5. **Use Fixtures**: Leverage pytest fixtures for setup
6. **Mark Appropriately**: Use markers for categorization
7. **Document**: Add docstrings to test classes and functions

## Troubleshooting

### Tests Fail Locally
```bash
# Run in verbose mode
pytest tests/ -vv -s

# Run single test
pytest tests/unit/services/test_event_producer.py::TestEventProducer::test_initialization -vv

# Debug mode
pytest tests/ --pdb  # Drops into debugger on failure
```

### CI Failures
- Check GitHub Actions logs
- Verify dependencies in requirements.txt
- Ensure services are mocked properly
- Check for timing issues (add sleeps if needed)

## Metrics

Track testing metrics:
- Coverage percentage
- Test execution time
- Flaky test rate
- CI success rate

Target:
- 80% code coverage
- <5 minutes for unit tests
- <15 minutes for all tests
- <1% flaky test rate

---

**Last Updated**: October 22, 2025  
**Framework Version**: 1.0.0  
**Test Count**: 10+ (growing)


