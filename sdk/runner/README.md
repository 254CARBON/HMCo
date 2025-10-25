# HMCo Data Platform Ingestion Runner

The Ingestion Runner is the core data plane component that executes Unified Ingestion Spec (UIS) jobs using various execution engines (SeaTunnel, Spark, Flink). It provides comprehensive monitoring, metrics collection, distributed tracing, and secret management integration.

## Features

- **Multi-Engine Support**: Execute jobs using SeaTunnel, Spark, or Flink
- **Secret Management**: HashiCorp Vault integration for secure credential handling
- **Metrics & Monitoring**: Prometheus metrics and OpenTelemetry tracing
- **Health Monitoring**: Built-in health checks and status reporting
- **HTTP API**: REST API for job management and monitoring
- **Kubernetes Integration**: Native Kubernetes deployment support
- **Configuration Management**: Environment-based and file-based configuration

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   UIS Parser    │───▶│   Compilers      │───▶│  Job Executor   │
│   (YAML/JSON)   │    │ (SeaTunnel/Spark/│    │ (Engine Runner) │
└─────────────────┘    │      Flink)      │    └─────────────────┘
         │              └──────────────────┘             │
         ▼                       │                     ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Secret Manager  │    │   Metrics        │    │   HTTP Server   │
│   (Vault)       │    │ (Prometheus)     │    │   (FastAPI)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- **FastAPI**: HTTP server and API endpoints
- **Uvicorn**: ASGI server for FastAPI
- **Prometheus Client**: Metrics collection and export
- **OpenTelemetry**: Distributed tracing
- **PyYAML**: YAML configuration parsing
- **Requests**: HTTP client for Vault integration

## Configuration

### Environment Variables

```bash
# Basic Configuration
export RUNNER_ID="production-runner"
export TENANT_ID="trading-platform"
export MAX_CONCURRENT_JOBS="20"
export JOB_TIMEOUT_SECONDS="3600"

# Vault Integration
export VAULT_ENABLED="true"
export VAULT_ADDRESS="http://vault:8200"
export VAULT_TOKEN="your-vault-token"
# OR
export VAULT_ROLE_ID="your-role-id"
export VAULT_SECRET_ID="your-secret-id"

# Kubernetes Integration
export KUBERNETES_ENABLED="true"
export KUBERNETES_NAMESPACE="data-platform"

# Metrics and Monitoring
export METRICS_ENABLED="true"
export METRICS_PORT="9090"
export METRICS_PATH="/metrics"

# Tracing
export TRACING_ENABLED="true"
export TRACING_ENDPOINT="http://jaeger:14268/api/traces"
export TRACING_SERVICE_NAME="ingestion-runner"

# Logging
export LOG_LEVEL="info"
export LOG_FORMAT="json"
```

### Configuration File

Create a configuration file for advanced settings:

```json
{
  "runner_id": "production-runner",
  "tenant_id": "trading-platform",
  "max_concurrent_jobs": 20,
  "job_timeout_seconds": 3600,
  "vault_enabled": true,
  "vault_address": "http://vault:8200",
  "vault_token": "your-vault-token",
  "metrics_enabled": true,
  "metrics_port": 9090,
  "tracing_enabled": true,
  "tracing_endpoint": "http://jaeger:14268/api/traces",
  "log_level": "info",
  "supported_engines": ["seatunnel", "spark", "flink"],
  "engine_configs": {
    "spark": {
      "executor_memory": "8g",
      "executor_cores": "4"
    },
    "flink": {
      "taskmanager_memory": "4g",
      "task_slots": "4"
    }
  }
}
```

## Usage

### Command Line Interface

#### Run Sample Job
```bash
# Execute a UIS specification
python main.py sample examples/polygon-api.yaml

# With custom configuration
python main.py sample spec.yaml --config custom-config.json

# With Vault token
python main.py sample spec.yaml --vault-token your-token
```

#### Start HTTP Server
```bash
# Start the HTTP server for API access
python main.py server --host 0.0.0.0 --port 8080

# With custom configuration
python main.py server --config runner-config.json
```

#### Health Check
```bash
# Check runner health
python main.py health

# Returns JSON health status
```

#### Information
```bash
# Show runner information and capabilities
python main.py info
```

### Programmatic Usage

```python
from runner import IngestionRunner, RunnerConfig

# Create runner with default configuration
runner = IngestionRunner()

# Load and execute UIS specification
result = runner.run_sample_job("examples/polygon-api.yaml")

if result["success"]:
    print(f"Job completed successfully: {result['job_id']}")
else:
    print(f"Job failed: {result['error']}")

# Check health
health = runner.get_health_status()
print(f"Runner status: {health['status']}")

# Get metrics
metrics = runner.get_metrics()
print(metrics)
```

## HTTP API

The runner provides a REST API for job management and monitoring:

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "runner_id": "production-runner",
  "tenant_id": "trading-platform",
  "timestamp": 1640995200.0,
  "components": {
    "vault": {
      "status": "healthy",
      "initialized": true,
      "sealed": false
    },
    "metrics": {
      "status": "enabled",
      "registry_type": "prometheus"
    },
    "jobs": {
      "active_count": 2,
      "active_jobs": ["job-123", "job-456"]
    }
  }
}
```

### Execute Job
```bash
POST /jobs/execute?spec_path=/path/to/spec.yaml
```

Response:
```json
{
  "status": "queued",
  "job_id": "job-uuid-123",
  "spec_name": "polygon-stock-api"
}
```

### Active Jobs
```bash
GET /jobs/active
```

Response:
```json
{
  "active_jobs": ["job-123", "job-456"]
}
```

### Cancel Job
```bash
DELETE /jobs/{job_id}
```

Response:
```json
{
  "status": "cancelled",
  "job_id": "job-123"
}
```

### Metrics
```bash
GET /metrics
```

Returns Prometheus metrics in text format.

## Vault Integration

The runner integrates with HashiCorp Vault for secure secret management:

### Authentication Methods

#### Token Authentication
```bash
export VAULT_TOKEN="your-vault-token"
python main.py sample spec.yaml
```

#### AppRole Authentication
```bash
export VAULT_ROLE_ID="your-role-id"
export VAULT_SECRET_ID="your-secret-id"
python main.py sample spec.yaml
```

### Secret References in UIS Specs

UIS specifications can reference Vault secrets:

```yaml
provider:
  credentials_ref: "vault://polygon-api/credentials"
  endpoints:
    - name: "api_endpoint"
      headers:
        Authorization: "Bearer {{api_key}}"
```

The runner automatically resolves `{{variable}}` placeholders with values from Vault.

## Metrics and Monitoring

### Prometheus Metrics

The runner exports comprehensive metrics:

```
# Job metrics
uis_runner_jobs_total{status="completed",provider_type="rest_api",tenant_id="trading"} 150
uis_runner_job_duration_seconds{provider_type="rest_api",tenant_id="trading"} 45.2

# Data metrics
uis_runner_records_ingested_total{provider_type="rest_api",tenant_id="trading",sink_type="iceberg"} 1000000
uis_runner_bytes_ingested_total{provider_type="rest_api",tenant_id="trading",sink_type="iceberg"} 52428800

# Error metrics
uis_runner_errors_total{error_type="timeout",provider_type="rest_api",tenant_id="trading"} 3
uis_runner_rate_limit_hits_total{provider_type="rest_api",tenant_id="trading"} 12
```

### Custom Metrics

- **Job Performance**: Execution time, throughput, success/failure rates
- **Data Quality**: Schema drift detection, validation errors
- **Resource Usage**: CPU, memory, network utilization
- **Cost Tracking**: Estimated costs per provider and tenant

## Distributed Tracing

### OpenTelemetry Integration

The runner supports distributed tracing with OpenTelemetry:

```bash
# Jaeger tracing
export TRACING_ENDPOINT="http://jaeger:14268/api/traces"
export TRACING_SERVICE_NAME="ingestion-runner"

# OTLP tracing
export TRACING_ENDPOINT="http://otel-collector:4318/v1/traces"
```

### Trace Structure

```
Job Execution Span
├── Secret Resolution
├── Compilation
├── Job Submission
└── Completion
```

## Kubernetes Deployment

### Deployment Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ingestion-runner
  namespace: data-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ingestion-runner
  template:
    metadata:
      labels:
        app: ingestion-runner
    spec:
      serviceAccountName: ingestion-runner
      containers:
      - name: runner
        image: hmco/ingestion-runner:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: RUNNER_ID
          value: "k8s-runner"
        - name: TENANT_ID
          value: "trading-platform"
        - name: VAULT_ADDRESS
          value: "http://vault.data-platform.svc.cluster.local:8200"
        - name: METRICS_ENABLED
          value: "true"
        - name: TRACING_ENABLED
          value: "true"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service Configuration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ingestion-runner
  namespace: data-platform
spec:
  selector:
    app: ingestion-runner
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

## Job Execution Engines

### SeaTunnel (Default)
- **Best for**: Batch data processing, simple transformations
- **Features**: REST APIs, file processing, basic data quality
- **Performance**: Good for moderate data volumes

### Apache Spark
- **Best for**: Micro-batch processing, complex transformations
- **Features**: Advanced analytics, machine learning, large datasets
- **Performance**: Excellent for analytical workloads

### Apache Flink
- **Best for**: Real-time streaming, event processing
- **Features**: Exactly-once processing, state management, windowing
- **Performance**: Low-latency, high-throughput streaming

## Development

### Running Tests

```bash
cd sdk/runner
python test_runner.py
```

### Adding New Execution Engines

1. Add engine to `ExecutionEngine` enum
2. Implement compiler in `uis.compilers`
3. Add execution logic to `JobExecutor`
4. Update configuration and validation

### Custom Metrics

```python
# Add custom metrics
metrics.record_custom_metric(
    name="custom_processing_time",
    value=processing_duration,
    labels={"provider": "custom_api", "tenant": "custom-tenant"}
)
```

## Integration with DolphinScheduler

The runner integrates with DolphinScheduler for workflow orchestration:

1. **Task Definition**: Create "External Feed" task type
2. **Configuration**: Set UIS spec path and runner endpoint
3. **Scheduling**: Configure cron schedules and dependencies
4. **Monitoring**: Track execution through DolphinScheduler UI

## Troubleshooting

### Common Issues

1. **Vault Connection**: Verify Vault address and authentication
2. **Missing Dependencies**: Install required packages for execution engines
3. **Resource Limits**: Increase memory/CPU limits for large jobs
4. **Network Access**: Ensure proper network policies for external APIs

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=debug
python main.py sample spec.yaml
```

### Health Check Issues

```bash
# Check component status
python main.py health

# Verify Vault connectivity
python -c "
from runner.secret_manager import SecretManager
sm = SecretManager('http://vault:8200', 'your-token')
print(sm.health_check())
"
```

## Performance Tuning

### Memory Configuration
```json
{
  "engine_configs": {
    "spark": {
      "executor_memory": "16g",
      "executor_cores": "8"
    },
    "flink": {
      "taskmanager_memory": "8g",
      "task_slots": "8"
    }
  }
}
```

### Concurrency Settings
```bash
export MAX_CONCURRENT_JOBS=50
export JOB_TIMEOUT_SECONDS=7200
```

## Security Considerations

- **Secret Management**: All credentials stored in Vault, never in configuration
- **Network Security**: Istio egress policies restrict external access
- **RBAC**: Tenant isolation enforced at all levels
- **Audit Logging**: All job executions logged with full traceability

## Monitoring Dashboard

Access the monitoring dashboard at:
```
http://localhost:8080/docs  # FastAPI interactive docs
http://localhost:9090/metrics  # Prometheus metrics
```

## License

This runner is part of the HMCo data platform SDK.
