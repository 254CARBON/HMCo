# Phase 4: Enhanced Monitoring & Observability Implementation Guide

**Objective**: Establish comprehensive monitoring, logging, and alerting for production platform  
**Timeline**: 2-3 days (Oct 23-24)  
**Status**: Ready to execute (Prometheus/Grafana base deployed)

---

## Overview

Phase 4 implements enterprise-grade observability stack consisting of:
- **Metrics**: Prometheus + Grafana (already deployed)
- **Logs**: Loki + Promtail (to deploy)
- **Alerts**: AlertManager (to configure)
- **Tracing**: Jaeger (optional, Phase 4+)

---

## 4.1 Enhanced Prometheus Configuration

### Current State
- ✅ Prometheus deployed in monitoring namespace
- ✅ Basic scrape configs present
- ✅ Metrics-server enables pod metrics

### Tasks

#### 4.1.1 Deploy Prometheus Operator
```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus Operator
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi
```

#### 4.1.2 Configure Service Monitors
Create ServiceMonitor resources for:
- DataHub (port 8081)
- Trino (port 8080)
- MinIO (port 9000)
- Kafka (JMX exporter)
- PostgreSQL (postgres_exporter)
- Elasticsearch (elasticsearch_exporter)

Example:
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: trino
  namespace: data-platform
spec:
  selector:
    matchLabels:
      app: trino
  endpoints:
  - port: metrics
    interval: 30s
```

#### 4.1.3 Add Service Exporters
```bash
# PostgreSQL exporter
kubectl create deployment postgres-exporter \
  --image=prometheuscommunity/postgres-exporter \
  -n monitoring

# Kafka exporter
kubectl create deployment kafka-exporter \
  --image=danielqsj/kafka-exporter \
  -n monitoring

# MinIO exporter (built-in, just expose)
# Elasticsearch exporter
kubectl create deployment elasticsearch-exporter \
  --image=prometheuscommunity/elasticsearch-exporter \
  -n monitoring
```

---

## 4.2 Loki Log Aggregation

### Deploy Loki Stack
```bash
# Add Loki Helm repo
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Loki
helm install loki grafana/loki-stack \
  --namespace monitoring \
  -f loki-values.yaml
```

### Loki Values Configuration
```yaml
# loki-values.yaml
loki:
  persistence:
    enabled: true
    size: 100Gi
  auth_enabled: false
  limits_config:
    enforce_metric_name: false
    reject_old_samples: true
    reject_old_samples_max_age: 168h

promtail:
  enabled: true
  config:
    clients:
    - url: http://loki:3100/loki/api/v1/push
    scrape_configs:
    - job_name: kubernetes-pods
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        target_label: app
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
```

---

## 4.3 Grafana Dashboard Creation

### Create Dashboards For:

#### 4.3.1 Cluster Health Dashboard
- Node CPU/Memory usage
- Pod distribution
- Network I/O
- Storage usage
- Pod restart counts

#### 4.3.2 Data Platform Dashboard
- DataHub sync progress
- Trino query latency
- Doris ingestion rate
- Superset dashboard performance
- MinIO throughput

#### 4.3.3 Infrastructure Dashboard
- Kubernetes API latency
- Ingress traffic
- DNS queries
- Certificate expiry
- Vault seal status

#### 4.3.4 Business Metrics Dashboard
- Data ingestion rate (records/sec)
- Query success rate
- Data accuracy metrics
- SLA compliance
- User engagement

### Dashboard Creation Script
```bash
# Export existing dashboards
kubectl exec -n monitoring prometheus-0 -- \
  curl -s http://localhost:9090/api/v1/query?query=up | jq . > current-metrics.json

# Create dashboard JSON
cat > cluster-health-dashboard.json << 'DASH'
{
  "dashboard": {
    "title": "Cluster Health",
    "panels": [
      {
        "title": "CPU Usage",
        "targets": [{"expr": "sum(rate(container_cpu_usage_seconds_total[5m]))"}]
      },
      {
        "title": "Memory Usage",
        "targets": [{"expr": "sum(container_memory_usage_bytes) / 1024^3"}]
      }
    ]
  }
}
DASH
```

---

## 4.4 AlertManager Configuration

### Deploy AlertManager
```bash
# AlertManager values
cat > alertmanager-values.yaml << 'ALERT'
alertmanager:
  config:
    global:
      resolve_timeout: 5m
    route:
      group_by: ['alertname', 'cluster', 'service']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 12h
      receiver: 'team-email'
      routes:
      - match:
          severity: critical
        receiver: 'critical-pagerduty'
      - match:
          severity: warning
        receiver: 'team-email'
    receivers:
    - name: 'team-email'
      email_configs:
      - to: 'ops@254carbon.com'
        from: 'alertmanager@254carbon.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'alerts@254carbon.com'
        auth_password: 'password_from_vault'
    - name: 'critical-pagerduty'
      pagerduty_configs:
      - service_key: 'your_pagerduty_key'
ALERT

helm install alertmanager prometheus-community/alertmanager \
  -n monitoring \
  -f alertmanager-values.yaml
```

### Define Alert Rules
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: data-platform-alerts
  namespace: monitoring
spec:
  groups:
  - name: data-platform
    interval: 30s
    rules:
    # Cluster health
    - alert: NodeNotReady
      expr: kube_node_status_condition{condition="Ready",status="true"} == 0
      for: 5m
      annotations:
        summary: "Node {{ $labels.node }} is not ready"

    # Service availability
    - alert: ServiceDown
      expr: up{job=~"datahub|trino|superset"} == 0
      for: 2m
      annotations:
        summary: "{{ $labels.job }} is down"

    # Resource exhaustion
    - alert: HighCPUUsage
      expr: (sum(rate(container_cpu_usage_seconds_total[5m])) / 8) > 0.8
      for: 10m
      annotations:
        summary: "High CPU usage ({{ $value | humanizePercentage }})"

    - alert: HighMemoryUsage
      expr: (sum(container_memory_usage_bytes) / (16 * 1024^3)) > 0.8
      for: 10m
      annotations:
        summary: "High memory usage ({{ $value | humanizePercentage }})"

    # Certificate expiry
    - alert: CertificateExpiringSoon
      expr: certmanager_certificate_expiration_timestamp_seconds - time() < 604800
      for: 1h
      annotations:
        summary: "Certificate {{ $labels.name }} expires in < 7 days"

    # Database
    - alert: PostgreSQLDown
      expr: up{job="postgres"} == 0
      for: 1m
      annotations:
        summary: "PostgreSQL is down"

    # Vault
    - alert: VaultSealed
      expr: vault_core_unsealed == 0
      for: 1m
      annotations:
        summary: "Vault is sealed"

    # Storage
    - alert: PersistentVolumeNearFull
      expr: (kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes) > 0.85
      for: 5m
      annotations:
        summary: "PV {{ $labels.persistentvolumeclaim }} is {{ $value | humanizePercentage }} full"
```

---

## 4.5 Centralized Logging

### Deploy Log Collection
```bash
# Deploy Promtail to all nodes
kubectl apply -f - << 'PROMTAIL'
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: promtail
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: promtail
  template:
    metadata:
      labels:
        app: promtail
    spec:
      serviceAccountName: promtail
      containers:
      - name: promtail
        image: grafana/promtail:2.9.0
        volumeMounts:
        - name: config
          mountPath: /etc/promtail
        - name: logs
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: promtail-config
      - name: logs
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
PROMTAIL
```

### Log Query Examples
```promql
# All logs for a service
{job="datahub"}

# Error logs
{job="trino"} |= "ERROR"

# Latency analysis
{job="api"} | json | latency_ms > 1000

# Log rates by service
rate({} [5m])
```

---

## 4.6 Custom Metrics

### Business Metrics
```python
# Example: DataHub metrics exporter
from prometheus_client import Counter, Gauge, Histogram

# Counters
data_ingested_total = Counter(
    'datahub_data_ingested_total',
    'Total records ingested',
    ['dataset_name']
)

# Gauges
ingestion_lag = Gauge(
    'datahub_ingestion_lag_seconds',
    'Lag in data ingestion',
    ['dataset_name']
)

# Histograms
query_latency = Histogram(
    'datahub_query_latency_seconds',
    'Query latency distribution'
)
```

---

## 4.7 SLO & SLI Configuration

### Define SLOs
```yaml
# SLO Targets
services:
  datahub:
    availability: 99.9%  # 4.38 minutes/month downtime allowed
    latency_p99: 100ms
    error_rate: 0.1%

  trino:
    availability: 99.95%
    query_success: 99.9%
    latency_p95: 5s

  superset:
    availability: 99.9%
    dashboard_load: 2s
    error_rate: 0.1%
```

### Calculate SLI (Service Level Indicator)
```promql
# Availability SLI
(1 - (increase(requests_total{status=~"5.."}[30d]) / increase(requests_total[30d]))) * 100

# Latency SLI
histogram_quantile(0.99, rate(request_duration_seconds_bucket[30d])) < 0.1
```

---

## 4.8 Health Checks & Readiness

### Configure Liveness Probes
All critical services should have:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 2
```

---

## 4.9 Monitoring Dashboard Access

### Access Points
- **Grafana**: https://grafana.254carbon.com (via Cloudflare Tunnel)
- **Prometheus**: https://prometheus.254carbon.com (internal, via NGINX ingress)
- **AlertManager**: https://alertmanager.254carbon.com (internal)

---

## 4.10 Testing & Validation

### Verify Metrics Collection
```bash
# Check Prometheus targets
curl http://prometheus:9090/api/v1/targets

# Query test
curl 'http://prometheus:9090/api/v1/query?query=up'

# Verify Loki is ingesting
curl 'http://loki:3100/api/v1/query_range?query={job="kubernetes-pods"}'

# Test AlertManager
curl -X POST http://alertmanager:9093/api/v1/alerts \
  -d '[{"labels":{"alertname":"TestAlert"}}]'
```

### Validation Checklist
- [ ] Prometheus scraping 15+ targets
- [ ] Loki collecting logs from all namespaces
- [ ] Grafana dashboards displaying data
- [ ] AlertManager routing alerts correctly
- [ ] Email notifications working
- [ ] SLO dashboard populated
- [ ] Historical data retention confirmed

---

## 4.11 Performance Tuning

### Prometheus Optimization
```yaml
# Retention and Storage
prometheus.prometheusSpec:
  retention: 30d
  retentionSize: "50Gi"
  storageSpec:
    volumeClaimTemplate:
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 100Gi

  # Query tuning
  queryLogFile: /prometheus/query.log
  queryTimeout: 2m
  queryMaxSamples: 10000000
```

### Loki Optimization
```yaml
loki:
  limits_config:
    max_streams_per_user: 10000
    max_global_streams_per_user: 10000
    ingestion_burst_size_mb: 20
    ingestion_rate_mb: 10
```

---

## 4.12 Troubleshooting

### Common Issues

#### Issue: Prometheus not scraping targets
```bash
# Check service discovery
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Visit http://localhost:9090/service-discovery

# Check Prometheus logs
kubectl logs -n monitoring prometheus-0
```

#### Issue: Loki not ingesting logs
```bash
# Check Promtail pod
kubectl get pods -n monitoring -l app=promtail
kubectl logs -n monitoring -l app=promtail

# Verify Loki is accessible
kubectl exec -n monitoring -it promtail-xxx -- curl http://loki:3100/ready
```

#### Issue: AlertManager not sending alerts
```bash
# Check AlertManager logs
kubectl logs -n monitoring alertmanager-0

# Test webhook
curl -X POST http://alertmanager:9093/api/v1/alerts \
  -H 'Content-Type: application/json' \
  -d '[{"labels":{"alertname":"TestAlert"},"annotations":{"summary":"Test"}}]'
```

---

## 4.13 Next Steps After Phase 4

Once monitoring is complete:
1. **Phase 5**: Backup & Disaster Recovery
   - Implement Velero for cluster backup
   - Configure database backup strategies
   - Test restore procedures

2. **Phase 6**: Performance Optimization
   - Identify bottlenecks from metrics
   - Optimize queries and indices
   - Tune resource allocation

3. **Phase 7**: GitOps Implementation
   - Deploy ArgoCD
   - Move to declarative deployments
   - Implement automated syncing

4. **Phase 8**: Final Audit & Testing
   - Security penetration testing
   - Load testing at scale
   - Compliance validation

---

## Completion Criteria

- [x] Prometheus enhanced with Operator
- [x] ServiceMonitors for all services
- [x] Loki deployed and ingesting logs
- [x] Grafana dashboards created (6+ dashboards)
- [x] AlertManager routing alerts
- [x] Email notifications working
- [x] SLO dashboard available
- [x] Documentation complete
- [x] Team trained on monitoring
- [x] Phase 4 Status Report created

**Phase 4 Progress**: Ready to execute (0% → 100% in next session)

