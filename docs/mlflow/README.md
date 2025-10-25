# MLFlow Documentation

Complete documentation for MLFlow integration in the 254Carbon data platform.

## Quick Start

MLFlow is deployed as a tracking server and model registry for the 254Carbon platform. Access it at: **https://mlflow.254carbon.com**

### What is MLFlow?

MLFlow is an open-source ML platform that provides:
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version and manage trained models
- **Model Serving**: Deploy models as REST endpoints
- **Project Management**: Package ML code reproducibly

### Why MLFlow?

For 254Carbon:
- Centralized tracking for all ML experiments
- Integrated with DolphinScheduler workflows
- Metadata governance via DataHub
- Full audit trail and reproducibility

---

## Documentation Files

### 1. Integration Guide: `integration-guide.md`

**Purpose**: How to use MLFlow in your workflows

**Contains**:
- DolphinScheduler integration setup
- Python client usage examples
- DataHub metadata integration
- Best practices for experiment tracking
- End-to-end ML pipeline example

**Who should read**: Data Scientists, ML Engineers

### 2. Troubleshooting: `troubleshooting.md`

**Purpose**: Diagnose and fix problems

**Contains**:
- Common issues and solutions
- Health check procedures
- Connection testing commands
- Performance optimization tips
- Recovery procedures

**Who should read**: DevOps, Platform Engineers, On-Call Support

### 3. Operations Runbook: `operations-runbook.md`

**Purpose**: Day-to-day operational tasks

**Contains**:
- Daily health checks
- Common tasks (experiments, runs, models)
- Backup and recovery procedures
- Scaling and performance tuning
- Maintenance procedures
- Disaster recovery

**Who should read**: DevOps, Operations Engineers

---

## Key Components

### MLFlow Server
- **Location**: Kubernetes deployment in `data-platform` namespace
- **Access**: `https://mlflow.254carbon.com` (via Cloudflare Access SSO)
- **Health Check**: `curl http://mlflow.data-platform.svc.cluster.local:5000/health`

### Backend Store
- **Database**: PostgreSQL (`mlflow` database)
- **Location**: Shared PostgreSQL instance
- **Purpose**: Store experiment metadata, runs, parameters, metrics

### Artifact Store
- **Storage**: MinIO S3-compatible storage
- **Bucket**: `mlflow-artifacts`
- **Purpose**: Store models and training artifacts

### Integration Points
- **DolphinScheduler**: Track ML training workflows
- **DataHub**: Model metadata governance
- **Portal**: Service discovery (`mlflow.254carbon.com`)
- **Prometheus**: Metrics collection
- **Grafana**: Dashboard visualization

---

## Common Tasks

### I want to track an ML experiment

See: [Integration Guide - Part 1: DolphinScheduler Integration](integration-guide.md#part-1-dolphinscheduler-integration)

### I want to use MLFlow in a DolphinScheduler job

See: [Integration Guide - Part 1: Create DolphinScheduler Python Task](integration-guide.md#step-3-create-dolphinscheduler-python-task)

### MLFlow is not responding

See: [Troubleshooting - Issue 1](troubleshooting.md#issue-1-mlflow-pods-in-crashloopbackoff)

### I cannot access MLFlow UI

See: [Troubleshooting - Issue 2](troubleshooting.md#issue-2-cannot-access-mlflow-ui-401-unauthorized)

### My DolphinScheduler task can't log to MLFlow

See: [Troubleshooting - Issue 3](troubleshooting.md#issue-3-experiment-tracking-fails-in-dolphinscheduler)

### Model artifacts won't upload

See: [Troubleshooting - Issue 4](troubleshooting.md#issue-4-s3-minio-artifact-upload-fails)

### I need to back up MLFlow data

See: [Operations Runbook - Backup Procedures](operations-runbook.md#backup-procedures)

### I need to recover from a disaster

See: [Operations Runbook - Disaster Recovery](operations-runbook.md#disaster-recovery)

---

## MLFlow Client Library

The `services/mlflow-orchestration/` package provides a Python wrapper for MLFlow:

```python
from mlflow_client import setup_mlflow_for_dolphinscheduler

# Initialize
client = setup_mlflow_for_dolphinscheduler(
    experiment_name="my_experiment",
    tags={"task": "training"}
)

# Start tracking
client.start_run("run_v1")

# Log data
client.log_params({"lr": 0.01})
client.log_metrics({"accuracy": 0.95})
client.log_model(model, "model", flavor="sklearn")

# Finish
client.end_run()
```

See: [services/mlflow-orchestration/README.md](../../services/mlflow-orchestration/README.md)

---

## Kubernetes Deployment

MLFlow manifests are in `k8s/compute/mlflow/`:

| File | Purpose |
|------|---------|
| `mlflow-secrets.yaml` | PostgreSQL & MinIO credentials |
| `mlflow-configmap.yaml` | Configuration settings |
| `mlflow-service.yaml` | Kubernetes service |
| `mlflow-deployment.yaml` | Main deployment (2 replicas) |
| `mlflow-backend-db.sql` | PostgreSQL schema |
| `README.md` | Deployment guide |

Deploy all:
```bash
kubectl apply -f k8s/compute/mlflow/
```

---

## Monitoring & Observability

### Prometheus Metrics

MLFlow exposes Prometheus metrics at `/metrics` on port 5000.

Scrape config: `k8s/monitoring/mlflow-servicemonitor.yaml`

### Key Metrics

- `mlflow_requests_total`: Total requests
- `mlflow_request_duration_seconds`: Request latency
- `mlflow_active_experiments`: Number of active experiments
- `mlflow_artifacts_stored`: Total artifacts stored

### Grafana Dashboards

Dashboard template: `k8s/monitoring/mlflow-dashboard.json`

View available dashboards at: https://grafana.254carbon.com

---

## Architecture Diagram

```
┌─────────────────────────────────────┐
│   Data Scientists / ML Engineers    │
│   (DolphinScheduler Tasks)          │
└────────────────┬────────────────────┘
                 │
                 ↓ (MLFlow Python Client)
┌─────────────────────────────────────┐
│     MLFlow Tracking Server          │
│     (2 replicas, HA)                │
├─────────────────────────────────────┤
│  Backend: PostgreSQL                │
│  Artifacts: MinIO S3                │
├─────────────────────────────────────┤
│  UI: https://mlflow.254carbon.com   │
│  API: Port 5000                     │
└─────────────────────────────────────┘
         ↓              ↓
    ┌────────────────────────────┐
    │  Metadata Integration      │
    ├────────────────────────────┤
    │  DataHub (Ingestion)       │
    │  Prometheus (Metrics)      │
    │  Grafana (Dashboards)      │
    └────────────────────────────┘
```

---

## Security Considerations

### Authentication & Authorization

- **User Access**: Via Cloudflare Access SSO
- **Session Duration**: 8 hours (configurable)
- **Audit Logging**: All access logged to Cloudflare

### Data Protection

- **In Transit**: HTTPS/TLS (end-to-end)
- **At Rest**: MinIO encryption (optional)
- **Database**: PostgreSQL credentials in Kubernetes secrets

### Network Security

- **Pod Security**: Non-root user, resource limits
- **Network Policies**: Restrict pod-to-pod communication (optional)
- **Ingress**: NGINX with authentication headers

---

## Performance Targets

| Metric | Target |
|--------|--------|
| API Response Time | < 500ms (p95) |
| Artifact Upload | > 10 MB/s |
| UI Load Time | < 3s |
| Database Query | < 100ms (p95) |

---

## Support & Escalation

### Getting Help

1. **Check documentation first**:
   - [Troubleshooting Guide](troubleshooting.md)
   - [Integration Guide](integration-guide.md)

2. **Check Kubernetes status**:
   ```bash
   kubectl get all -n data-platform -l app=mlflow
   ```

3. **Review logs**:
   ```bash
   kubectl logs -n data-platform -l app=mlflow
   ```

4. **Contact support**:
   - Data Platform Team (business hours)
   - DevOps On-Call (emergencies)

---

## Related Resources

### In This Repository

- [MLFlow Orchestration Client](../../services/mlflow-orchestration/README.md)
- [DolphinScheduler Integration](integration-guide.md)
- [DataHub Integration](integration-guide.md#part-2-datahub-integration)

### External Resources

- [MLFlow Official Documentation](https://mlflow.org/docs/latest/)
- [MLFlow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLFlow API Reference](https://mlflow.org/docs/latest/rest-api.html)
- [DolphinScheduler Documentation](https://dolphinscheduler.apache.org/docs/)
- [DataHub Documentation](https://datahubproject.io/docs/)

---

## Frequently Asked Questions

### Q: What's the difference between experiments and runs?

**A**: An experiment is a collection of related runs. A run is a single execution of training code with specific parameters and metrics.

### Q: Can I use MLFlow with non-Python models?

**A**: Yes, MLFlow supports multiple flavors: Scikit-learn, TensorFlow, PyTorch, Spark, and generic Python functions.

### Q: How do I deploy a model from MLFlow?

**A**: MLFlow provides several deployment options:
- MLFlow Model Serving (REST API)
- Kubernetes deployment (via MLFlow Docker image)
- AWS SageMaker, Azure ML, GCP Vertex AI

### Q: How long are experiments stored?

**A**: Indefinitely, unless archived or manually deleted. Old runs can be archived for performance.

### Q: Can I export experiments from MLFlow?

**A**: Yes, you can:
- Export via API: `/api/2.0/experiments/list`
- Export artifacts: S3 access directly
- Export database: PostgreSQL backup

---

## Contributing

To update this documentation:

1. Make changes to relevant guide
2. Update this README if needed
3. Test procedures in a staging environment
4. Submit for review

---

**Last Updated**: October 2025
**Version**: 1.0
**Status**: Production Ready
