# Iceberg Monitoring and Alerting Guide

## Overview

This guide covers monitoring and alerting setup for the Iceberg REST Catalog integration using Prometheus, Grafana, and other tools.

## Architecture

```
Iceberg REST Catalog
        ↓
   /metrics endpoint
        ↓
Prometheus (scraping)
        ↓
    Recording Rules
    & Alerting Rules
        ↓
  ┌─────┴─────┐
  ↓           ↓
Grafana    Alert Manager
          (PagerDuty, Slack, etc.)
```

## Deployment

### 1. Deploy Prometheus Monitoring

```bash
# Apply monitoring configuration
kubectl apply -f k8s/monitoring/iceberg-monitoring.yaml

# Verify ServiceMonitor creation
kubectl get servicemonitor -n data-platform | grep iceberg

# Verify PrometheusRule creation
kubectl get prometheusrule -n data-platform | grep iceberg
```

### 2. Verify Metrics Collection

```bash
# Port-forward to Prometheus
kubectl port-forward -n monitoring svc/prometheus 9090:9090 &

# Check Iceberg targets
# Navigate to http://localhost:9090/targets

# Look for: iceberg-rest-catalog (should show "UP")
```

### 3. Import Grafana Dashboard

```bash
# The monitoring configuration includes a dashboard definition
# It will be available in Grafana automatically if using Prometheus datasource

# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000 &

# Navigate to http://localhost:3000
# Default credentials: admin/admin

# Import dashboard:
# 1. Create → Import
# 2. Paste dashboard JSON from iceberg-monitoring.yaml
# 3. Select Prometheus as datasource
# 4. Import
```

## Key Metrics

### Availability Metrics

```prometheus
# Catalog availability (0 = down, 1 = up)
up{job="iceberg-rest-catalog"}

# Uptime percentage
(
  count(up{job="iceberg-rest-catalog"} == 1)
  /
  count(up{job="iceberg-rest-catalog"})
) * 100
```

### Performance Metrics

```prometheus
# Request rate (requests per second)
rate(http_requests_total{job="iceberg-rest-catalog"}[5m])

# P95 latency (seconds)
histogram_quantile(0.95, 
  rate(http_request_duration_seconds_bucket{job="iceberg-rest-catalog"}[5m])
)

# P99 latency (seconds)
histogram_quantile(0.99, 
  rate(http_request_duration_seconds_bucket{job="iceberg-rest-catalog"}[5m])
)
```

### Error Metrics

```prometheus
# Error rate
sum(rate(http_requests_total{job="iceberg-rest-catalog",status=~"5.."}[5m]))
/
sum(rate(http_requests_total{job="iceberg-rest-catalog"}[5m]))

# Errors per second
sum(rate(http_requests_total{job="iceberg-rest-catalog",status=~"5.."}[5m]))

# 4xx errors (client errors)
sum(rate(http_requests_total{job="iceberg-rest-catalog",status=~"4.."}[5m]))
```

### Resource Metrics

```prometheus
# Memory usage
container_memory_working_set_bytes{pod=~"iceberg-rest-catalog.*"}

# Memory percentage
(
  container_memory_working_set_bytes{pod=~"iceberg-rest-catalog.*"}
  /
  container_spec_memory_limit_bytes{pod=~"iceberg-rest-catalog.*"}
) * 100

# CPU usage
rate(container_cpu_usage_seconds_total{pod=~"iceberg-rest-catalog.*"}[5m])

# CPU percentage
(
  rate(container_cpu_usage_seconds_total{pod=~"iceberg-rest-catalog.*"}[5m])
  /
  container_spec_cpu_quota{pod=~"iceberg-rest-catalog.*"}
) * 100
```

### Database Connectivity

```prometheus
# Database connection errors
rate(iceberg_database_connection_errors_total[5m])

# Active database connections
iceberg_database_connections_active

# Database connection pool utilization
(
  iceberg_database_connections_active
  /
  iceberg_database_connections_max
) * 100
```

### S3/MinIO Metrics

```prometheus
# S3 request errors
rate(iceberg_s3_errors_total[5m])

# S3 request rate
rate(iceberg_s3_requests_total[5m])

# S3 latency
histogram_quantile(0.95, rate(iceberg_s3_request_duration_seconds_bucket[5m]))
```

## Alert Rules

### Critical Alerts

| Alert | Condition | Action |
|-------|-----------|--------|
| CatalogDown | `up == 0` for 5m | Page on-call engineer |
| DatabaseDown | DB connection errors > 0.1/sec for 5m | Investigate DB connectivity |
| S3Down | S3 errors > 0.1/sec for 5m | Check MinIO/S3 access |
| HighErrorRate | Error rate > 5% for 5m | Review error logs |

### Warning Alerts

| Alert | Condition | Action |
|-------|-----------|--------|
| HighLatency | P95 latency > 1 second for 5m | Monitor and investigate |
| HighMemory | Memory > 85% for 5m | Increase memory or optimize |
| HighCPU | CPU > 80% for 5m | Check query complexity |
| SlowQueries | Avg query time > 500ms | Investigate slow queries |

## Grafana Dashboard

### Panels Included

1. **Status Gauge**: Current availability status
2. **Request Rate**: Requests per second over time
3. **Status Distribution**: Pie chart of response statuses
4. **Latency Percentiles**: P50/P95/P99 latency trends

### Custom Dashboards

Create additional dashboards for:

**Dashboard 1: Database Performance**
```
- Connection pool utilization
- Query execution times
- Connection errors
- Active connections
```

**Dashboard 2: S3/MinIO Performance**
```
- Request rate by operation
- Request latency distribution
- Error rate
- Bucket size trends
```

**Dashboard 3: Application Health**
```
- Memory usage trend
- CPU usage trend
- Garbage collection time
- Thread count
```

## Log Aggregation

### Centralize Logs

```yaml
# Configure Iceberg pods to send logs to Loki/ELK
env:
- name: LOG_FORMAT
  value: "json"
- name: LOG_LEVEL
  value: "INFO"

# All logs will be picked up by Promtail/Filebeat
```

### Key Log Queries

```logql
# All Iceberg REST Catalog logs
{pod="iceberg-rest-catalog-*"}

# Error logs only
{pod="iceberg-rest-catalog-*"} | json | level="ERROR"

# Database connection issues
{pod="iceberg-rest-catalog-*"} | json | message=~".*connection.*"

# S3 errors
{pod="iceberg-rest-catalog-*"} | json | message=~".*s3.*"
```

## Alerting Integration

### Configure Alert Destinations

#### 1. Slack Integration

```yaml
# AlertManager configuration for Slack
global:
  slack_api_url: "${SLACK_WEBHOOK_URL}"

route:
  receiver: "slack"

receivers:
- name: "slack"
  slack_configs:
  - channel: "#alerts"
    title: "{{ .GroupLabels.alertname }}"
    text: "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
```

#### 2. PagerDuty Integration

```yaml
# PagerDuty configuration
receivers:
- name: "pagerduty"
  pagerduty_configs:
  - service_key: "${PAGERDUTY_KEY}"
    description: "{{ .GroupLabels.alertname }}"
```

#### 3. Email Integration

```yaml
# Email configuration
receivers:
- name: "email"
  email_configs:
  - to: "oncall@company.com"
    from: "alerts@company.com"
    smarthost: "mail.company.com:587"
    auth_username: "${MAIL_USER}"
    auth_password: "${MAIL_PASSWORD}"
```

## Health Checks

### Manual Health Check

```bash
# Check all components
curl -s http://iceberg-rest-catalog:8181/v1/config && echo "✓ Iceberg OK"
curl -s http://minio-service:9000/minio/health/live && echo "✓ MinIO OK"
curl -s http://postgres-shared-service:5432 && echo "✓ PostgreSQL OK" || true
curl -s http://trino-coordinator:8080/v1/info && echo "✓ Trino OK"
```

### Kubernetes Probe Configuration

Probes are already configured in deployments:
- **Liveness Probe**: Checks if pod should be restarted
- **Readiness Probe**: Checks if pod should receive traffic
- **Startup Probe**: Allows time for slow-starting applications

## Dashboarding Best Practices

1. **One Dashboard Per Component**
   - Iceberg REST Catalog
   - Database Performance
   - S3/MinIO Storage
   - Data Pipeline Health

2. **Use Variables for Flexibility**
   - Environment: prod, staging, dev
   - Component: catalog, database, storage
   - Time range: last hour, last day, last week

3. **Set Meaningful Thresholds**
   - Critical: Red
   - Warning: Yellow
   - Healthy: Green

4. **Update Regularly**
   - Review dashboard quarterly
   - Add new metrics as needed
   - Remove obsolete metrics

## SLI/SLO Definition

### Service Level Indicators (SLIs)

```prometheus
# Availability SLI
sum(increase(http_requests_total{job="iceberg-rest-catalog",status!~"5.."}[1m]))
/
sum(increase(http_requests_total{job="iceberg-rest-catalog"}[1m]))

# Latency SLI (% of requests < 1 second)
sum(increase(http_request_duration_seconds_bucket{job="iceberg-rest-catalog",le="1"}[1m]))
/
sum(increase(http_requests_total{job="iceberg-rest-catalog"}[1m]))

# Error Rate SLI (% of errors)
1 -
(
  sum(increase(http_requests_total{job="iceberg-rest-catalog",status!~"5.."}[1m]))
  /
  sum(increase(http_requests_total{job="iceberg-rest-catalog"}[1m]))
)
```

### Service Level Objectives (SLOs)

| SLO | Target | Window |
|-----|--------|--------|
| Availability | 99.9% | 30 days |
| Latency P95 | < 500ms | 30 days |
| Error Rate | < 0.1% | 30 days |

## Troubleshooting Alerts

### CatalogDown Alert

```bash
# 1. Check pod status
kubectl get pod -n data-platform iceberg-rest-catalog-xxx

# 2. Check pod logs
kubectl logs -n data-platform iceberg-rest-catalog-xxx | tail -50

# 3. Check resource limits
kubectl describe pod -n data-platform iceberg-rest-catalog-xxx

# 4. Test connectivity
kubectl exec -it postgres-shared-xxx -- \
  psql -U iceberg_user -d iceberg_rest -c "SELECT 1"
```

### HighErrorRate Alert

```bash
# 1. Check error logs
kubectl logs -f -n data-platform iceberg-rest-catalog-xxx | grep -i error

# 2. Query recent errors in Prometheus
# error_rate in Prometheus UI

# 3. Check dependent services
kubectl get pod -n data-platform | grep -E "postgres|minio"
```

### HighLatency Alert

```bash
# 1. Check current latency in Prometheus
# histogram_quantile(0.95, ...) in Prometheus UI

# 2. Check resource usage
kubectl top pod -n data-platform iceberg-rest-catalog-xxx

# 3. Check database performance
kubectl exec -it postgres-shared-xxx -- \
  psql -U postgres -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 5;"
```

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
- [Kubernetes Monitoring Best Practices](https://kubernetes.io/docs/tasks/debug-application-cluster/resource-metrics-pipeline/)

## Support

For monitoring issues:
1. Check Prometheus targets status
2. Review AlertManager configuration
3. Verify webhook/integration endpoints
4. Check Grafana datasources
5. Review logs in Elasticsearch/Loki
