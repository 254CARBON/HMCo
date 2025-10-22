#!/bin/bash
# Kong API Gateway Service Configuration Script
# Registers all 254Carbon services with Kong via Admin API

set -e

KONG_ADMIN_URL="http://kong-admin.kong.svc.cluster.local:8001"

echo "═══════════════════════════════════════════════════"
echo "Kong API Gateway - Service Registration"
echo "═══════════════════════════════════════════════════"
echo ""

# Function to register a service
register_service() {
    local name=$1
    local host=$2
    local port=$3
    local path=${4:-/}
    local retries=${5:-3}
    local connect_timeout=${6:-60000}
    local write_timeout=${7:-60000}
    local read_timeout=${8:-60000}
    
    echo -n "Registering service: $name... "
    
    # Create or update service
    curl -s -X PUT "${KONG_ADMIN_URL}/services/${name}" \
        -d "host=${host}" \
        -d "port=${port}" \
        -d "protocol=http" \
        -d "path=${path}" \
        -d "retries=${retries}" \
        -d "connect_timeout=${connect_timeout}" \
        -d "write_timeout=${write_timeout}" \
        -d "read_timeout=${read_timeout}" > /dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅"
    else
        echo "❌"
        return 1
    fi
}

# Function to register a route
register_route() {
    local service_name=$1
    local route_name=$2
    local path=$3
    local strip_path=${4:-true}
    
    echo -n "  → Creating route: ${route_name}... "
    
    curl -s -X POST "${KONG_ADMIN_URL}/services/${service_name}/routes" \
        -d "name=${route_name}" \
        -d "paths[]=${path}" \
        -d "strip_path=${strip_path}" \
        -d "protocols[]=http" \
        -d "protocols[]=https" > /dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅"
    else
        echo "❌"
    fi
}

# Function to enable plugin
enable_plugin() {
    local service_name=$1
    local plugin_name=$2
    shift 2
    local config_params="$@"
    
    echo -n "  → Enabling plugin: ${plugin_name}... "
    
    curl -s -X POST "${KONG_ADMIN_URL}/services/${service_name}/plugins" \
        -d "name=${plugin_name}" \
        ${config_params} > /dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅"
    else
        echo "❌"
    fi
}

echo "Step 1: Registering Services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Register all services
register_service "datahub-gms" "datahub-gms.data-platform.svc.cluster.local" "8080"
register_service "datahub-frontend" "datahub-frontend.data-platform.svc.cluster.local" "9002"
register_service "trino" "trino-coordinator.data-platform.svc.cluster.local" "8080" "/" 0 60000 3600000 3600000
register_service "superset" "superset-web.data-platform.svc.cluster.local" "8088"
register_service "dolphinscheduler" "dolphinscheduler-api-service.data-platform.svc.cluster.local" "12345" "/" 2 60000 120000 120000
register_service "minio-api" "minio.data-platform.svc.cluster.local" "9000" "/" 3 60000 300000 300000
register_service "mlflow" "mlflow-service.data-platform.svc.cluster.local" "5000"
register_service "iceberg-rest" "iceberg-rest-catalog.data-platform.svc.cluster.local" "8181" "/v1"
register_service "portal-services" "portal-services.data-platform.svc.cluster.local" "8080"
register_service "grafana" "kube-prometheus-stack-grafana.monitoring.svc.cluster.local" "80"
register_service "prometheus" "kube-prometheus-stack-prometheus.monitoring.svc.cluster.local" "9090"
register_service "schema-registry" "schema-registry.data-platform.svc.cluster.local" "8081"

echo ""
echo "Step 2: Creating Routes"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create routes
register_route "datahub-gms" "datahub-api" "/api/datahub"
register_route "datahub-frontend" "datahub-ui" "/datahub"
register_route "trino" "trino-api" "/api/trino"
register_route "trino" "trino-ui" "/trino-ui" false
register_route "superset" "superset-api" "/api/superset"
register_route "superset" "superset-ui" "/superset" false
register_route "dolphinscheduler" "dolphinscheduler-api" "/api/dolphinscheduler"
register_route "minio-api" "minio-s3" "/api/s3"
register_route "mlflow" "mlflow-api" "/api/mlflow"
register_route "iceberg-rest" "iceberg-api" "/api/iceberg"
register_route "portal-services" "portal-api" "/api/services" false
register_route "grafana" "grafana-api" "/api/grafana"
register_route "prometheus" "prometheus-api" "/api/prometheus"
register_route "schema-registry" "schema-registry-api" "/api/schema"

echo ""
echo "Step 3: Enabling Plugins"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Enable CORS for web services
enable_plugin "superset" "cors" \
    -d "config.origins=*" \
    -d "config.credentials=true" \
    -d "config.max_age=3600"

enable_plugin "grafana" "cors" \
    -d "config.origins=*" \
    -d "config.credentials=true"

enable_plugin "portal-services" "cors" \
    -d "config.origins=*" \
    -d "config.credentials=true"

# Enable rate limiting
enable_plugin "datahub-gms" "rate-limiting" \
    -d "config.minute=200" \
    -d "config.hour=10000" \
    -d "config.policy=local"

enable_plugin "trino" "rate-limiting" \
    -d "config.minute=50" \
    -d "config.hour=1000" \
    -d "config.policy=local"

enable_plugin "superset" "rate-limiting" \
    -d "config.minute=100" \
    -d "config.hour=5000" \
    -d "config.policy=local"

# Enable prometheus metrics globally
curl -s -X POST "${KONG_ADMIN_URL}/plugins" \
    -d "name=prometheus" > /dev/null && echo "  → Prometheus metrics enabled globally ✅"

echo ""
echo "Step 4: Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# List all services
SERVICE_COUNT=$(curl -s "${KONG_ADMIN_URL}/services" | grep -o '"id"' | wc -l)
ROUTE_COUNT=$(curl -s "${KONG_ADMIN_URL}/routes" | grep -o '"id"' | wc -l)
PLUGIN_COUNT=$(curl -s "${KONG_ADMIN_URL}/plugins" | grep -o '"id"' | wc -l)

echo "Services registered: ${SERVICE_COUNT}"
echo "Routes created: ${ROUTE_COUNT}"
echo "Plugins enabled: ${PLUGIN_COUNT}"

echo ""
echo "═══════════════════════════════════════════════════"
echo "✅ Kong Configuration Complete!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Test API endpoints:"
echo "  curl http://kong-proxy.kong.svc.cluster.local/api/services"
echo "  curl http://kong-proxy.kong.svc.cluster.local/api/datahub"
echo ""
echo "Access Admin UI:"
echo "  kubectl port-forward -n kong svc/kong-admin 8001:8001"
echo "  Open: http://localhost:8001"



