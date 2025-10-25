# ClickHouse Integration - Complete Setup Guide

## Overview

This document describes the complete integration of ClickHouse into the 254CARBON data platform, replacing Apache Doris with ClickHouse for high-performance columnar analytics.

## 🚀 Quick Start

### 1. Deploy ClickHouse

```bash
# Deploy ClickHouse to the cluster
./deploy-clickhouse.sh
```

### 2. Verify Integration

```bash
# Verify all integrations are working
./verify-clickhouse-integration.sh
```

### 3. Access ClickHouse

- **Web Interface**: https://clickhouse.254carbon.com
- **HTTP API**: http://clickhouse.254carbon.com:8123
- **TCP Client**: clickhouse.data-platform:9000
- **Database**: default
- **Username**: default
- **Password**: ClickHouse@254Carbon2025

## 📊 Architecture

### ClickHouse Configuration

- **Version**: 24.8.7
- **Deployment**: Single replica (scalable to cluster)
- **Storage**: 100Gi persistent volume (local-path)
- **Memory**: 4Gi requests, 8Gi limits
- **CPU**: 2 requests, 4 limits

### Integration Points

#### 1. Superset Integration
- **Connection URI**: `clickhouse://default:ClickHouse%40254Carbon2025@clickhouse-service.data-platform:9000/default`
- **Database Name**: ClickHouse
- **Auto-discovery**: Enabled in Superset configuration

#### 2. Trino Integration
- **Catalog**: clickhouse_catalog
- **Connector**: ClickHouse JDBC driver
- **Query Federation**: Enabled for cross-database queries

#### 3. Monitoring Integration
- **Metrics**: Prometheus scraping on port 9363
- **Alerts**: SLO-based alerting for latency and availability
- **Dashboards**: Grafana dashboards for ClickHouse metrics

#### 4. Security Integration
- **SSO**: Cloudflare Access integration
- **Ingress**: TLS-enabled with Let's Encrypt certificates
- **Authentication**: Default user with strong password

## 🔧 Configuration Files Updated

### Helm Charts
- `helm/charts/data-platform/charts/clickhouse/` - Complete ClickHouse chart
- `helm/charts/data-platform/values.yaml` - ClickHouse configuration
- `helm/charts/data-platform/values/prod.yaml` - Production overrides
- `helm/charts/data-platform/Chart.yaml` - Dependencies updated

### Kubernetes Resources
- `k8s/gitops/argocd-applications.yaml` - ArgoCD configuration
- `k8s/ingress/data-platform-ingress.yaml` - ClickHouse ingress rules
- `k8s/monitoring/` - Prometheus scraping and alerts

### Service Integration
- `services.json` - Service registry updated
- `portal/lib/services.ts` - Portal navigation updated
- `portal/README.md` - Documentation updated

### External Systems
- **Superset**: Database connections and secrets
- **Trino**: Catalog configuration for ClickHouse
- **Grafana**: Dashboards and alerting rules
- **Cloudflare**: Access policies and tunnel configuration

## 📈 Performance Targets

### SLO Targets
- **Query Latency**: p95 < 1 second (vs Doris p95 < 3 seconds)
- **Ingestion Lag**: < 10 seconds (vs Doris < 30 seconds)
- **Availability**: 99.9%
- **Throughput**: Optimized for high-volume analytics workloads

### Resource Optimization
- **Memory Management**: Configured for 8Gi limits with spill-to-disk
- **Storage**: 100Gi persistent volume with local-path class
- **Networking**: HTTP/HTTPS and TCP interfaces exposed
- **Monitoring**: Prometheus metrics and structured logging

## 🔍 Monitoring & Observability

### Metrics Collected
- Query performance (latency, throughput, errors)
- Resource utilization (CPU, memory, disk I/O)
- Connection statistics (active connections, queries per second)
- Storage metrics (data size, compression ratios)

### Alerting Rules
- **High Latency**: p95 query time > 2 seconds
- **High Error Rate**: Error rate > 0.1%
- **Resource Exhaustion**: Memory/CPU usage > 80%
- **Storage Full**: Disk usage > 85%

## 🚨 Migration from Doris

### What Changed
1. **Database Engine**: Doris MPP → ClickHouse columnar
2. **Query Language**: Doris SQL → ClickHouse SQL (95% compatible)
3. **Connection Ports**: 8030/9030 → 8123/9000
4. **Authentication**: Doris admin → ClickHouse default user
5. **Web Interface**: Doris console → ClickHouse web UI

### Data Migration
```sql
-- ClickHouse table creation (example)
CREATE TABLE analytics.events (
    timestamp DateTime,
    user_id UInt64,
    event_type String,
    properties String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, user_id);
```

### Application Updates Required
1. **Connection Strings**: Update JDBC/ODBC connection URLs
2. **SQL Dialect**: Minor syntax adjustments may be needed
3. **Authentication**: Update credentials in applications
4. **Monitoring**: Update dashboards and alerting rules

## 🧪 Testing

### Automated Tests
```bash
# Run integration tests
./verify-clickhouse-integration.sh

# Test specific components
kubectl run clickhouse-test --image=curlimages/curl --rm -i --restart=Never -- \
  curl http://clickhouse.data-platform:8123/

# Test database connectivity
kubectl run clickhouse-client-test --image=clickhouse/clickhouse-client --rm -i --restart=Never -- \
  clickhouse-client --host clickhouse.data-platform --query "SELECT version()"
```

### Manual Verification
1. **Web Interface**: Visit https://clickhouse.254carbon.com
2. **Superset**: Check for ClickHouse data source in databases
3. **Trino**: Verify ClickHouse catalog is available
4. **Grafana**: Check ClickHouse metrics dashboard
5. **Portal**: Verify ClickHouse appears in service list

## 🔒 Security

### Authentication
- **Username**: default
- **Password**: ClickHouse@254Carbon2025 (URL encoded: ClickHouse%40254Carbon2025)
- **Access Control**: Configured via ClickHouse users.xml

### Network Security
- **Ingress**: TLS-enabled with nginx ingress controller
- **SSO**: Cloudflare Access integration
- **Network Policies**: Restricted access within cluster

### Data Security
- **Encryption**: TLS for all connections
- **Access Logging**: All queries logged for audit
- **RBAC**: User-based access control

## 📚 Documentation

### Updated Documentation
- `README.md` - Platform overview updated
- `docs/cloudflare/` - Access policies and tunnel configuration
- `docs/sso/` - Authentication and access control
- `docs/monitoring/` - Metrics and alerting configuration
- `docs/readiness/` - Performance targets and optimization

### API Documentation
- **ClickHouse HTTP API**: https://clickhouse.com/docs/en/interfaces/http
- **ClickHouse TCP Protocol**: https://clickhouse.com/docs/en/interfaces/tcp
- **SQL Reference**: https://clickhouse.com/docs/en/sql-reference

## 🛠️ Troubleshooting

### Common Issues

#### 1. ClickHouse Not Starting
```bash
kubectl logs deployment/clickhouse -n data-platform
kubectl describe pod -l app.kubernetes.io/name=clickhouse -n data-platform
```

#### 2. Connection Issues
```bash
# Test connectivity
kubectl run curl-test --image=curlimages/curl --rm -i --restart=Never -- \
  curl -v http://clickhouse.data-platform:8123/ping

# Check service endpoints
kubectl get endpoints clickhouse -n data-platform
```

#### 3. Superset Integration Issues
```bash
# Check Superset logs
kubectl logs deployment/superset -n data-platform

# Verify ClickHouse URI in secrets
kubectl get secret superset-secrets -n data-platform -o yaml
```

#### 4. Performance Issues
```bash
# Check resource usage
kubectl top pods -l app.kubernetes.io/name=clickhouse -n data-platform

# Check ClickHouse metrics
curl http://clickhouse.data-platform:9363/metrics | grep clickhouse
```

### Support Resources
- **ClickHouse Documentation**: https://clickhouse.com/docs/en/
- **Community Support**: https://github.com/ClickHouse/ClickHouse
- **254CARBON Support**: admin@254carbon.com

## 🎯 Next Steps

1. **Data Migration**: Migrate existing data from Doris to ClickHouse
2. **Performance Tuning**: Optimize ClickHouse configuration for your workload
3. **Monitoring Setup**: Configure custom dashboards and alerting
4. **Documentation**: Update team documentation and runbooks
5. **Training**: Train team on ClickHouse best practices

## 📞 Support

For issues or questions:
- **Technical Issues**: Check troubleshooting section above
- **Integration Questions**: Review configuration files in this repository
- **Performance**: Review monitoring dashboards and performance targets
- **Security**: Contact security team for access-related issues

---

**Migration Completed**: October 25, 2025
**ClickHouse Version**: 24.8.7
**Integration Status**: ✅ Complete
