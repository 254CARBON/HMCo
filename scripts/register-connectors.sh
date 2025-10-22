#!/bin/bash
# Register Kafka Connect Connectors
# Submits connector configurations to Kafka Connect REST API

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

function log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Port-forward to Kafka Connect (in background)
log_info "Setting up port-forward to Kafka Connect..."
kubectl port-forward -n data-platform svc/kafka-connect-service 8083:8083 &
PF_PID=$!
trap "kill $PF_PID 2>/dev/null" EXIT

sleep 5

CONNECT_URL="http://localhost:8083"

# Check Kafka Connect availability
log_info "Checking Kafka Connect availability..."
if ! curl -s "${CONNECT_URL}/" > /dev/null; then
    log_error "Cannot connect to Kafka Connect at ${CONNECT_URL}"
    exit 1
fi

log_info "Kafka Connect is available!"
echo ""

# Function to register connector
register_connector() {
    local name=$1
    local config=$2
    
    log_info "Registering connector: $name"
    
    response=$(curl -s -w "%{http_code}" -o /tmp/connector_response.json \
        -X POST "${CONNECT_URL}/connectors" \
        -H "Content-Type: application/json" \
        -d "$config")
    
    if [ "$response" = "201" ] || [ "$response" = "409" ]; then
        log_info "✓ Connector $name registered successfully"
        return 0
    else
        log_error "✗ Failed to register connector $name (HTTP $response)"
        cat /tmp/connector_response.json
        return 1
    fi
}

# Register Source Connectors
echo "============================================"
echo "Registering Source Connectors"
echo "============================================"
echo ""

# Note: These are example configurations. Adjust based on actual connector availability
log_info "Source connectors require specific connector plugins to be installed."
log_info "Skipping automated registration. Please configure manually via REST API."
echo ""

# Register Sink Connectors
echo "============================================"
echo "Registering Sink Connectors"
echo "============================================"
echo ""

# Elasticsearch Sink
log_info "Registering Elasticsearch sink connector..."
register_connector "elasticsearch-commodity-alerts-sink" '{
  "name": "elasticsearch-commodity-alerts-sink",
  "config": {
    "connector.class": "io.confluent.connect.elasticsearch.ElasticsearchSinkConnector",
    "tasks.max": "1",
    "topics": "market-alerts",
    "connection.url": "http://elasticsearch-service:9200",
    "type.name": "_doc",
    "key.ignore": "false",
    "schema.ignore": "true",
    "key.converter": "org.apache.kafka.connect.storage.StringConverter",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter.schemas.enable": "false"
  }
}'

# S3/MinIO Sink
log_info "Registering S3/MinIO archive sink connector..."
register_connector "s3-raw-data-archive-sink" '{
  "name": "s3-raw-data-archive-sink",
  "config": {
    "connector.class": "io.confluent.connect.s3.S3SinkConnector",
    "tasks.max": "2",
    "topics.regex": ".*-raw",
    "s3.bucket.name": "kafka-archive",
    "s3.region": "us-east-1",
    "store.url": "http://minio-service:9000",
    "aws.access.key.id": "minioadmin",
    "aws.secret.access.key": "minioadmin123",
    "format.class": "io.confluent.connect.s3.format.json.JsonFormat",
    "partitioner.class": "io.confluent.connect.storage.partitioner.TimeBasedPartitioner",
    "path.format": "YYYY/MM/dd/HH",
    "flush.size": "10000",
    "key.converter": "org.apache.kafka.connect.storage.StringConverter",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter.schemas.enable": "false"
  }
}'

echo ""
echo "============================================"
echo "Connector Status"
echo "============================================"
echo ""

# List all connectors
log_info "Current connectors:"
curl -s "${CONNECT_URL}/connectors" | jq .

echo ""
log_info "To view connector status:"
echo "  curl ${CONNECT_URL}/connectors/<connector-name>/status | jq ."
echo ""
log_info "To delete a connector:"
echo "  curl -X DELETE ${CONNECT_URL}/connectors/<connector-name>"
echo ""

log_info "Connector registration complete!"


