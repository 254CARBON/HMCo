#!/bin/bash

# Verify ClickHouse integration with all services

echo "🔍 Verifying ClickHouse integration..."

# Check ClickHouse deployment
echo "📊 Checking ClickHouse deployment status..."
kubectl get pods -n data-platform -l app.kubernetes.io/name=clickhouse
kubectl get svc -n data-platform -l app.kubernetes.io/name=clickhouse

# Check ingress
echo "🌐 Checking ClickHouse ingress..."
kubectl get ingress clickhouse -n data-platform
echo "ClickHouse URL: http://clickhouse.254carbon.com"

# Check persistent volumes
echo "💾 Checking ClickHouse persistent storage..."
kubectl get pvc -n data-platform -l app.kubernetes.io/name=clickhouse

# Check monitoring integration
echo "📈 Checking ClickHouse monitoring..."
kubectl get servicemonitor -n monitoring -l app.kubernetes.io/name=clickhouse

# Test ClickHouse connectivity
echo "🔗 Testing ClickHouse HTTP endpoint..."
kubectl run clickhouse-test --image=curlimages/curl --rm -i --restart=Never -- \
  curl -s -o /dev/null -w "%{http_code}" http://clickhouse.data-platform:8123/ || echo "HTTP test failed"

# Test ClickHouse TCP endpoint
echo "🔗 Testing ClickHouse TCP endpoint..."
kubectl run clickhouse-tcp-test --image=clickhouse/clickhouse-client --rm -i --restart=Never -- \
  clickhouse-client --host clickhouse.data-platform --port 9000 --user default --password "ClickHouse@254Carbon2025" --query "SELECT version()" || echo "TCP test failed"

# Check Superset integration
echo "📋 Checking Superset ClickHouse configuration..."
kubectl get secret superset-secrets -n data-platform -o yaml | grep -q SUPERSET_CLICKHOUSE_URI && echo "✅ ClickHouse URI configured in Superset" || echo "❌ ClickHouse URI missing in Superset"

# Check Trino integration
echo "📋 Checking Trino ClickHouse catalog..."
kubectl get configmap -n data-platform -l app.kubernetes.io/name=trino | grep -q clickhouse && echo "✅ ClickHouse catalog configured in Trino" || echo "❌ ClickHouse catalog missing in Trino"

# Check Cloudflare Access
echo "🔐 Checking Cloudflare Access policies..."
# This would need to be checked manually or via API

# Check monitoring alerts
echo "🚨 Checking ClickHouse monitoring alerts..."
kubectl get prometheusrules -n monitoring | grep -q clickhouse && echo "✅ ClickHouse alerts configured" || echo "❌ ClickHouse alerts missing"

echo ""
echo "🎯 Integration Status Summary:"
echo "1. ClickHouse Deployment: $(kubectl get deployment clickhouse -n data-platform -o jsonpath='{.status.readyReplicas}')/$(kubectl get deployment clickhouse -n data-platform -o jsonpath='{.spec.replicas}') ready"
echo "2. ClickHouse Service: $(kubectl get svc clickhouse -n data-platform > /dev/null && echo '✅' || echo '❌')"
echo "3. ClickHouse Ingress: $(kubectl get ingress clickhouse -n data-platform > /dev/null && echo '✅' || echo '❌')"
echo "4. ClickHouse Storage: $(kubectl get pvc -n data-platform -l app.kubernetes.io/name=clickhouse --no-headers | wc -l) PVC(s)"
echo "5. ClickHouse Monitoring: $(kubectl get servicemonitor -n monitoring -l app.kubernetes.io/name=clickhouse > /dev/null && echo '✅' || echo '❌')"
echo "6. Superset Integration: $(kubectl get secret superset-secrets -n data-platform -o yaml | grep -q SUPERSET_CLICKHOUSE_URI && echo '✅' || echo '❌')"

echo ""
echo "🔧 Manual Verification Steps:"
echo "1. Visit https://clickhouse.254carbon.com (should show ClickHouse interface)"
echo "2. Check Superset dashboards for ClickHouse data source"
echo "3. Verify Trino can query ClickHouse tables"
echo "4. Check Grafana for ClickHouse metrics"
echo "5. Test data ingestion and querying workflows"

