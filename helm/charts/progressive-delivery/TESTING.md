# Progressive Delivery Testing Guide

This document provides test scenarios to verify canary deployments with Argo Rollouts.

## Prerequisites

1. Install the progressive-delivery chart:
   ```bash
   helm install progressive-delivery ./helm/charts/progressive-delivery \
     --namespace argo-rollouts \
     --create-namespace
   ```

2. Install kubectl argo rollouts plugin:
   ```bash
   curl -LO https://github.com/argoproj/argo-rollouts/releases/latest/download/kubectl-argo-rollouts-linux-amd64
   chmod +x kubectl-argo-rollouts-linux-amd64
   sudo mv kubectl-argo-rollouts-linux-amd64 /usr/local/bin/kubectl-argo-rollouts
   ```

3. Ensure Prometheus is running and accessible at the configured address.

## Test Scenario 1: Successful Canary with Automatic Promotion

This test verifies that a canary deployment with healthy metrics automatically promotes to 100%.

### Steps:

1. Deploy portal-services with Rollout enabled:
   ```bash
   helm upgrade --install portal-services ./helm/charts/portal-services \
     --namespace data-platform \
     --set rollout.enabled=true
   ```

2. Watch the rollout status:
   ```bash
   kubectl argo rollouts get rollout portal-services -n data-platform --watch
   ```

3. Trigger a canary by updating the image tag:
   ```bash
   helm upgrade portal-services ./helm/charts/portal-services \
     --namespace data-platform \
     --set image.tag=1.0.1 \
     --set rollout.enabled=true
   ```

4. Monitor the canary progression:
   ```bash
   kubectl argo rollouts get rollout portal-services -n data-platform --watch
   ```

### Expected Result:
- Canary starts at 20% traffic
- Analysis runs every 60 seconds checking success rate, error rate, and latency
- If metrics pass thresholds:
  - Success rate >= 95%
  - Error rate <= 5%
  - P95 latency <= 1000ms
- Canary progresses through 20% → 40% → 60% → 80% → 100%
- Full promotion occurs automatically

### Verification Commands:
```bash
# Check rollout status
kubectl argo rollouts status portal-services -n data-platform

# View analysis runs
kubectl get analysisrun -n data-platform

# Check analysis results
kubectl describe analysisrun <analysis-run-name> -n data-platform
```

## Test Scenario 2: Failed Canary with Automatic Rollback

This test verifies that a canary deployment with failing metrics automatically rolls back.

### Steps:

1. Deploy a version that will fail metrics (simulate by artificially increasing error rate):
   ```bash
   # Option 1: Use a buggy version if available
   helm upgrade portal-services ./helm/charts/portal-services \
     --namespace data-platform \
     --set image.tag=1.0.2-buggy \
     --set rollout.enabled=true

   # Option 2: Manually fail the analysis by injecting errors
   # (requires access to the application to trigger 5xx errors)
   ```

2. Watch the rollout status:
   ```bash
   kubectl argo rollouts get rollout portal-services -n data-platform --watch
   ```

### Expected Result:
- Canary starts at 20% traffic
- Analysis detects failing metrics (e.g., error rate > 5%)
- After 3 consecutive failures, rollout is aborted
- Traffic automatically reverts to stable version (previous version)
- No manual intervention required

### Verification Commands:
```bash
# Check rollout status (should show "Degraded")
kubectl argo rollouts status portal-services -n data-platform

# View analysis runs (should show "Failed")
kubectl get analysisrun -n data-platform

# Check analysis details
kubectl describe analysisrun <analysis-run-name> -n data-platform

# Verify old ReplicaSet is scaled up
kubectl get rs -n data-platform -l app.kubernetes.io/name=portal-services
```

## Test Scenario 3: Manual Promotion

This test verifies manual promotion capabilities for cases where automatic analysis is not needed.

### Steps:

1. Deploy with manual analysis pauses:
   ```bash
   helm upgrade portal-services ./helm/charts/portal-services \
     --namespace data-platform \
     --set image.tag=1.0.3 \
     --set rollout.enabled=true
   ```

2. Monitor rollout (it will pause at each step):
   ```bash
   kubectl argo rollouts get rollout portal-services -n data-platform --watch
   ```

3. Manually promote when ready:
   ```bash
   kubectl argo rollouts promote portal-services -n data-platform
   ```

### Expected Result:
- Canary waits at each percentage step
- Manual promotion moves to next step
- Repeat promotion until 100%

## Test Scenario 4: Manual Abort

This test verifies manual abort capabilities.

### Steps:

1. Start a canary deployment:
   ```bash
   helm upgrade portal-services ./helm/charts/portal-services \
     --namespace data-platform \
     --set image.tag=1.0.4 \
     --set rollout.enabled=true
   ```

2. Abort the rollout manually:
   ```bash
   kubectl argo rollouts abort portal-services -n data-platform
   ```

### Expected Result:
- Rollout immediately aborts
- Traffic reverts to stable version
- New ReplicaSet is scaled down

### Verification:
```bash
# Check rollout status (should show "Degraded")
kubectl argo rollouts status portal-services -n data-platform

# View rollout history
kubectl argo rollouts history portal-services -n data-platform
```

## Test Scenario 5: MLflow Canary Deployment

This test verifies canary deployment for MLflow with longer pause durations.

### Steps:

1. Deploy MLflow with Rollout enabled:
   ```bash
   helm upgrade --install mlflow ./helm/charts/ml-platform/charts/mlflow \
     --namespace data-platform \
     --set rollout.enabled=true \
     --set global.vault.enabled=false
   ```

2. Trigger a canary by updating configuration:
   ```bash
   helm upgrade mlflow ./helm/charts/ml-platform/charts/mlflow \
     --namespace data-platform \
     --set rollout.enabled=true \
     --set global.vault.enabled=false \
     --set image.tag=v2.11.0
   ```

3. Monitor the rollout:
   ```bash
   kubectl argo rollouts get rollout mlflow -n data-platform --watch
   ```

### Expected Result:
- Canary progresses with 3-minute pauses (longer than portal-services)
- Analysis validates MLflow tracking server health
- Automatic promotion on success

## Monitoring and Debugging

### View All Rollouts:
```bash
kubectl argo rollouts list rollouts -n data-platform
```

### Get Detailed Rollout Information:
```bash
kubectl argo rollouts get rollout <rollout-name> -n data-platform
```

### View Analysis Templates:
```bash
kubectl get analysistemplate -n data-platform
kubectl describe analysistemplate success-rate -n data-platform
```

### Check Prometheus Queries:
Access Prometheus and run the queries from the AnalysisTemplates to verify metrics are available:

```promql
# Success Rate
sum(rate(http_requests_total{job="portal-services",namespace="data-platform",status=~"2.."}[5m])) /
sum(rate(http_requests_total{job="portal-services",namespace="data-platform"}[5m])) * 100

# Error Rate
sum(rate(http_requests_total{job="portal-services",namespace="data-platform",status=~"5.."}[5m])) /
sum(rate(http_requests_total{job="portal-services",namespace="data-platform"}[5m])) * 100

# Latency P95
histogram_quantile(0.95,
  sum(rate(http_request_duration_milliseconds_bucket{job="portal-services",namespace="data-platform"}[5m])) by (le)
)
```

### Access Argo Rollouts Dashboard:
```bash
kubectl argo rollouts dashboard -n data-platform
```
Then open http://localhost:3100 in your browser.

## Troubleshooting

### Rollout Stuck in Progressing State:
- Check if AnalysisRuns are being created: `kubectl get analysisrun -n data-platform`
- Verify Prometheus is accessible: `kubectl get svc -n monitoring`
- Check analysis template configuration: `kubectl describe analysistemplate -n data-platform`

### Metrics Not Available:
- Ensure your application exposes Prometheus metrics
- Verify ServiceMonitor is configured correctly
- Check Prometheus targets: Access Prometheus UI and check Status → Targets

### Automatic Rollback Not Working:
- Verify failureLimit in AnalysisTemplate is set (default: 3)
- Check if metrics are returning values
- Review analysis run logs: `kubectl logs <analysis-run-pod> -n data-platform`

## Cleanup

Remove test deployments:
```bash
kubectl delete rollout portal-services -n data-platform
kubectl delete rollout mlflow -n data-platform
kubectl delete analysisrun --all -n data-platform
```
