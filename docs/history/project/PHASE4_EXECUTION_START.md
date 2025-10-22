# Phase 4 Execution - Enhanced Monitoring Deployment Started

**Date**: October 20, 2025  
**Status**: PHASE 4 EXECUTION INITIATED  
**Timeline**: 2-3 days to completion

---

## Phase 4 Components Deployment Plan

### Step 1: Deploy Prometheus Operator (CRD-based management)

Prometheus Operator provides:
- CustomResourceDefinition (CRD) for Prometheus, AlertManager, ServiceMonitor
- Automated configuration management
- High availability with StatefulSet
- Built-in Thanos for long-term storage (optional)

**Actions:**

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus Operator
helm install prometheus-operator prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values /tmp/prometheus-operator-values.yaml
```

**Configuration (prometheus-operator-values.yaml):**

```yaml
prometheus:
  prometheusSpec:
    retention: 30d
    retentionSize: "50Gi"
    
    # Storage
    storageSpec:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 100Gi
    
    # Service discovery
    serviceMonitorSelectorNilUsesHelmValues: false
    podMonitorSelectorNilUsesHelmValues: false
    
    # Resource allocation
    resources:
      requests:
        cpu: 500m
        memory: 2Gi
      limits:
        cpu: 1000m
        memory: 4Gi

alertmanager:
  enabled: true
  alertmanagerSpec:
    storage:
      volumeClaimTemplate:
        spec:
          resources:
            requests:
              storage: 10Gi

grafana:
  enabled: true
  adminPassword: "ChangeMe123"
  persistence:
    enabled: true
    size: 10Gi

prometheus-node-exporter:
  enabled: true

kube-state-metrics:
  enabled: true
```

---

### Step 2: Deploy Loki for Log Aggregation

Loki provides:
- Scalable log storage (billions of log lines)
- LogQL query language (similar to PromQL)
- Promtail agents for log collection
- Cost-effective (doesn't index log content)

**Deployment:**

```bash
# Add Grafana Helm repository
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Loki Stack (Loki + Promtail)
helm install loki grafana/loki-stack \
  --namespace monitoring \
  --values /tmp/loki-values.yaml
```

**Configuration (loki-values.yaml):**

```yaml
loki:
  persistence:
    enabled: true
    size: 100Gi
  
  config:
    auth_enabled: false
    limits_config:
      enforce_metric_name: false
      reject_old_samples: true
      reject_old_samples_max_age: 168h
      ingestion_burst_size_mb: 20
      ingestion_rate_mb: 10

promtail:
  enabled: true
  config:
    clients:
    - url: http://loki:3100/loki/api/v1/push
    
    positions:
      filename: /tmp/positions.yaml
    
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
      - source_labels: [__meta_kubernetes_pod_label_tier]
        target_label: tier
```

---

### Step 3: Deploy Service Exporters

Export metrics from services for Prometheus scraping:

```bash
# PostgreSQL Exporter
kubectl create deployment postgres-exporter \
  --image=prometheuscommunity/postgres-exporter \
  -n monitoring

# Kafka Exporter (if Kafka is running)
kubectl create deployment kafka-exporter \
  --image=danielqsj/kafka-exporter:latest \
  -n monitoring

# Elasticsearch Exporter (if Elasticsearch is running)
kubectl create deployment elasticsearch-exporter \
  --image=prometheuscommunity/elasticsearch-exporter \
  -n monitoring
```

---

### Step 4: Create ServiceMonitors for Prometheus Operator

ServiceMonitors tell Prometheus what to scrape:

```bash
cat > /tmp/servicemonitors.yaml << 'MONITORS'
# ServiceMonitor for Trino
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

---
# ServiceMonitor for DataHub
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: datahub
  namespace: data-platform
spec:
  selector:
    matchLabels:
      app: datahub-gms
  endpoints:
  - port: 8081
    interval: 30s

---
# ServiceMonitor for Superset
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: superset
  namespace: data-platform
spec:
  selector:
    matchLabels:
      app: superset
  endpoints:
  - port: metrics
    interval: 30s

---
# ServiceMonitor for PostgreSQL Exporter
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: postgresql
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: postgres-exporter
  endpoints:
  - port: metrics
    interval: 30s
MONITORS

kubectl apply -f /tmp/servicemonitors.yaml
```

---

### Step 5: Configure AlertManager

```bash
cat > /tmp/alertmanager-config.yaml << 'ALERTMANAGER'
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-config
  namespace: monitoring
type: Opaque
stringData:
  alertmanager.yml: |
    global:
      resolve_timeout: 5m
      smtp_from: 'alertmanager@254carbon.com'
      smtp_smarthost: 'smtp.gmail.com:587'
      smtp_auth_username: 'alerts@254carbon.com'
      smtp_auth_password: 'your_app_password'
    
    templates:
    - '/etc/alertmanager/config/*.tmpl'
    
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
        continue: true
      - match:
          severity: warning
        receiver: 'team-slack'
    
    receivers:
    - name: 'team-email'
      email_configs:
      - to: 'ops@254carbon.com'
        headers:
          Subject: '[{{ .GroupLabels.alertname }}] {{ .GroupLabels.severity }}'
    
    - name: 'critical-pagerduty'
      pagerduty_configs:
      - service_key: 'your_pagerduty_integration_key'
    
    - name: 'team-slack'
      slack_configs:
      - api_url: 'your_slack_webhook_url'
        channel: '#ops-alerts'
        title: 'Alert: {{ .GroupLabels.alertname }}'
ALERTMANAGER

kubectl apply -f /tmp/alertmanager-config.yaml
```

---

### Step 6: Create PrometheusRules for Alerting

```bash
cat > /tmp/prometheus-rules.yaml << 'RULES'
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
    
    # Cluster Health
    - alert: NodeNotReady
      expr: kube_node_status_condition{condition="Ready",status="true"} == 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Node {{ $labels.node }} is not ready"
        description: "Node has been unready for more than 5 minutes"
    
    # Service Availability
    - alert: ServiceDown
      expr: up{job=~"datahub|trino|superset"} == 0
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "{{ $labels.job }} is down"
        description: "{{ $labels.job }} has been unavailable for 2+ minutes"
    
    # Resource Usage
    - alert: HighCPUUsage
      expr: (sum(rate(container_cpu_usage_seconds_total[5m])) / 8) > 0.8
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "High CPU usage detected"
        description: "CPU usage is {{ $value | humanizePercentage }} - threshold 80%"
    
    - alert: HighMemoryUsage
      expr: (sum(container_memory_usage_bytes) / (16 * 1024^3)) > 0.8
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "High memory usage detected"
        description: "Memory usage is {{ $value | humanizePercentage }} - threshold 80%"
    
    # Certificate Expiry
    - alert: CertificateExpiringSoon
      expr: certmanager_certificate_expiration_timestamp_seconds - time() < 604800
      for: 1h
      labels:
        severity: warning
      annotations:
        summary: "Certificate expiring soon"
        description: "{{ $labels.name }} expires in {{ $value | humanizeDuration }}"
    
    # Database
    - alert: PostgreSQLDown
      expr: up{job="postgresql"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "PostgreSQL is down"
    
    # Vault
    - alert: VaultSealed
      expr: vault_core_unsealed == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Vault is sealed"
        description: "Vault needs to be unsealed immediately"
    
    # Storage
    - alert: PersistentVolumeNearFull
      expr: (kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes) > 0.85
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "PV nearly full"
        description: "{{ $labels.persistentvolumeclaim }} is {{ $value | humanizePercentage }} full"
RULES

kubectl apply -f /tmp/prometheus-rules.yaml
```

---

### Step 7: Create Grafana Dashboards

Grafana dashboard templates for key services:

Dashboard 1: Cluster Health
- Node CPU/Memory usage
- Pod distribution across nodes
- Network I/O
- Storage usage
- Pod restart counts

Dashboard 2: Data Platform
- DataHub sync progress
- Trino query latency
- Doris ingestion rate
- Superset dashboard performance
- MinIO throughput

Dashboard 3: Kubernetes Infrastructure
- API server latency
- Ingress traffic patterns
- DNS query performance
- Certificate expiry timeline
- Vault seal status

---

## Execution Timeline

### Day 1: Foundation Setup
- [x] Prerequisites verified
- [ ] Deploy Prometheus Operator
- [ ] Deploy Loki Stack
- [ ] Create ServiceMonitors
- [ ] Verify metrics flowing

### Day 2: Alerting & Exporters
- [ ] Deploy service exporters
- [ ] Configure AlertManager
- [ ] Create PrometheusRules
- [ ] Test alerting pipeline
- [ ] Verify notifications

### Day 3: Dashboards & Verification
- [ ] Create Grafana dashboards
- [ ] Configure SLO tracking
- [ ] Test end-to-end monitoring
- [ ] Document procedures
- [ ] Phase 4 Verification

---

## Success Criteria

- [ ] Prometheus Operator running with CRDs
- [ ] 15+ metrics targets scraping successfully
- [ ] Loki collecting logs from all namespaces
- [ ] Promtail running on all nodes
- [ ] AlertManager routing alerts correctly
- [ ] Email/Slack notifications working
- [ ] 6+ Grafana dashboards populated with data
- [ ] SLO dashboard tracking availability
- [ ] All PrometheusRules evaluated without errors
- [ ] Monitored services remain healthy during monitoring deployment

---

## What Gets Delivered

### Monitoring Infrastructure
- ✅ Prometheus Operator with CRD management
- ✅ Loki + Promtail for log aggregation
- ✅ AlertManager for alert routing
- ✅ Service exporters (PostgreSQL, Kafka, Elasticsearch)
- ✅ PrometheusRules for alerting

### Observability
- ✅ Real-time metrics for all services
- ✅ Centralized logging from all pods
- ✅ Alert notifications (email/Slack/PagerDuty)
- ✅ SLO tracking and compliance
- ✅ Business metrics dashboards

### Documentation
- ✅ Alert runbooks
- ✅ Troubleshooting procedures
- ✅ Dashboard usage guide
- ✅ Log querying examples
- ✅ On-call procedures

---

## Phase 4 Status: INITIATED ✅

Ready to proceed with deployment!

