#!/bin/bash

# Deploy ClickHouse to replace Doris in the data platform

echo "🚀 Deploying ClickHouse to replace Doris..."

# Create namespace if it doesn't exist
kubectl create namespace data-platform --dry-run=client -o yaml | kubectl apply -f -

# Deploy ClickHouse using Helm
echo "📦 Installing ClickHouse via Helm..."
helm upgrade --install clickhouse helm/charts/data-platform \
  --namespace data-platform \
  --values helm/charts/data-platform/values.yaml \
  --values helm/charts/data-platform/values/prod.yaml \
  --set clickhouse.enabled=true \
  --set superset.enabled=true \
  --set trino.enabled=true \
  --set datahub.enabled=false \
  --set dolphinscheduler.enabled=true \
  --set spark-operator.enabled=true \
  --set data-lake.enabled=false

# Wait for deployment to be ready
echo "⏳ Waiting for ClickHouse deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/clickhouse -n data-platform

# Verify ClickHouse is running
echo "✅ Verifying ClickHouse deployment..."
kubectl get pods -n data-platform -l app.kubernetes.io/name=clickhouse

# Test ClickHouse connectivity
echo "🔍 Testing ClickHouse connectivity..."
kubectl run clickhouse-test --image=curlimages/curl --rm -i --restart=Never -- \
  curl -f http://clickhouse.data-platform:8123/ || echo "ClickHouse HTTP endpoint test failed"

# Update ArgoCD to sync the new configuration
echo "🔄 Updating ArgoCD configuration..."
kubectl apply -f k8s/gitops/argocd-applications.yaml

echo "🎉 ClickHouse deployment completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. Update your applications to connect to ClickHouse instead of Doris"
echo "2. Test ClickHouse integration with Superset and Trino"
echo "3. Update monitoring and alerting configurations"
echo "4. Verify Cloudflare Access policies are updated"
echo ""
echo "🔗 ClickHouse Access:"
echo "  - HTTP Interface: http://clickhouse.254carbon.com"
echo "  - TCP Interface: clickhouse.data-platform:9000"
echo "  - Database: default"
echo "  - Username: default"
echo "  - Password: ClickHouse@254Carbon2025"

