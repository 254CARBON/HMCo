#!/bin/bash
#
# Mirror Container Images to Private Registry
# Resolves Docker Hub rate limiting for 254Carbon platform
#
# Usage: ./mirror-images.sh [REGISTRY_URL] [REGISTRY_TYPE] [--only key1,key2,...]
#
# Examples:
#   ./mirror-images.sh harbor-registry.254carbon.local harbor
#   ./mirror-images.sh 123456789.dkr.ecr.us-east-1.amazonaws.com ecr
#   ./mirror-images.sh gcr.io/myproject gcr
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REGISTRY_URL="${1:-}"
REGISTRY_TYPE="${2:-harbor}"
ONLY_ARG="${3:-}"

if [[ -z "$REGISTRY_URL" ]]; then
    echo -e "${RED}Error: Registry URL required${NC}"
    echo ""
    echo "Usage: $0 REGISTRY_URL [REGISTRY_TYPE] [--only key1,key2,...]"
    echo ""
    echo "Examples:"
    echo "  $0 harbor.254carbon.local harbor"
    echo "  $0 123456789.dkr.ecr.us-east-1.amazonaws.com ecr"
    echo "  $0 gcr.io/myproject gcr"
    echo "  $0 localhost:5000/254carbon harbor --only grafana,prometheus,nginx-ingress"
    exit 1
fi

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}254Carbon Image Mirroring Script${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "Registry: $REGISTRY_URL"
echo "Type: $REGISTRY_TYPE"
if [[ -n "$ONLY_ARG" ]]; then
    echo "Filter: $ONLY_ARG"
fi
echo ""

# All images used by 254Carbon platform
# Organized by component
declare -A IMAGES=(
    # Core infrastructure
    ["nginx-ingress"]="registry.k8s.io/ingress-nginx/controller:v1.8.1"
    ["cert-manager"]="quay.io/jetstack/cert-manager-controller:v1.13.0"
    ["cert-manager-webhook"]="quay.io/jetstack/cert-manager-webhook:v1.13.0"
    ["cert-manager-cainjector"]="quay.io/jetstack/cert-manager-cainjector:v1.13.0"
    
    # Monitoring
    ["prometheus"]="prom/prometheus:v2.48.0"
    ["grafana"]="grafana/grafana:10.2.0"
    ["loki"]="grafana/loki:2.9.3"
    ["promtail"]="grafana/promtail:2.9.3"
    ["alertmanager"]="prom/alertmanager:v0.26.0"
    
    # Data Platform
    ["doris"]="amd64/doris:2.0.0"
    ["trino"]="trinodb/trino:424"
    ["superset"]="apache/superset:4.0.1"
    ["datahub"]="linkedin/datahub-gms:v0.12.0"
    
    # Storage & Secrets
    ["minio"]="minio/minio:RELEASE.2024-01-11T08-13-15Z"
    ["minio-mc"]="minio/mc:RELEASE.2024-01-11T03-23-48Z"
    ["vault"]="hashicorp/vault:1.15.0"
    ["postgres"]="postgres:15.5"
    ["redis"]="redis:7.2"
    
    # Messaging & Coordination
    ["kafka"]="confluentinc/cp-kafka:7.5.0"
    ["zookeeper"]="confluentinc/cp-zookeeper:7.5.0"
    ["schema-registry"]="confluentinc/cp-schema-registry:7.5.0"
    
    # Elasticsearch Stack
    ["elasticsearch"]="docker.elastic.co/elasticsearch/elasticsearch:8.10.2"
    ["kibana"]="docker.elastic.co/kibana/kibana:8.10.2"
    
    # Workflow & Compute
    ["dolphinscheduler-api"]="apache/dolphinscheduler:3.1.5"
    ["dolphinscheduler-master"]="apache/dolphinscheduler:3.1.5"
    ["dolphinscheduler-worker"]="apache/dolphinscheduler:3.1.5"
    ["spark"]="bitnami/spark:3.5.0"
    ["flink"]="flink:1.17.1"
    ["seatunnel"]="apache/seatunnel:2.3.4"
    
    # Data Lake
    ["iceberg-rest"]="ghcr.io/tabulario/iceberg-rest:0.6.0"
    ["lakefs"]="treeverse/lakefs:v1.26.0"
    ["hudi"]="ghcr.io/apache/hudi:hudi-0.13.0"
    
    # Utilities
    ["busybox"]="busybox:latest"
    ["curl"]="curlimages/curl:latest"
    ["alpine"]="alpine:latest"
)

echo -e "${YELLOW}Total images defined: ${#IMAGES[@]}${NC}"
echo ""

# Counter
SUCCESS=0
FAILED=0
SKIPPED=0

# Function to mirror image
mirror_image() {
    local name="$1"
    local image="$2"
    local target_image="$REGISTRY_URL/$name:latest"
    
    echo -n "Mirroring $image ... "
    
    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}SKIPPED${NC} (Docker not available)"
        ((SKIPPED++))
        return
    fi
    
    # Pull image
    if ! docker pull "$image" 2>/dev/null; then
        echo -e "${RED}FAILED${NC} (Pull failed)"
        ((FAILED++))
        return
    fi
    
    # Tag image
    if ! docker tag "$image" "$target_image" 2>/dev/null; then
        echo -e "${RED}FAILED${NC} (Tag failed)"
        ((FAILED++))
        return
    fi
    
    # Push image
    if ! docker push "$target_image" 2>/dev/null; then
        echo -e "${YELLOW}SKIPPED${NC} (Push failed - may need authentication)"
        ((SKIPPED++))
        return
    fi
    
    echo -e "${GREEN}OK${NC}"
    ((SUCCESS++))
}

# Build selection if --only passed
declare -A SELECTED_IMAGES

if [[ -n "$ONLY_ARG" ]]; then
    # Expect format: --only key1,key2,...
    if [[ "$ONLY_ARG" == --only* ]]; then
        ONLY_LIST="${ONLY_ARG#--only}"
        ONLY_LIST="${ONLY_LIST#,}"
    else
        ONLY_LIST="$ONLY_ARG"
    fi
    IFS=',' read -r -a KEYS <<< "$ONLY_LIST"
    for key in "${KEYS[@]}"; do
        key_trimmed="${key// /}"
        if [[ -n "$key_trimmed" && -n "${IMAGES[$key_trimmed]:-}" ]]; then
            SELECTED_IMAGES["$key_trimmed"]="${IMAGES[$key_trimmed]}"
        else
            echo -e "${YELLOW}Warning:${NC} image key '$key_trimmed' not found; skipping."
        fi
    done
else
    # No filter; use all
    for k in "${!IMAGES[@]}"; do SELECTED_IMAGES["$k"]="${IMAGES[$k]}"; done
fi

echo -e "${YELLOW}Images to mirror: ${#SELECTED_IMAGES[@]}${NC}"

# Mirror each selected image
for name in "${!SELECTED_IMAGES[@]}"; do
    mirror_image "$name" "${SELECTED_IMAGES[$name]}"
done

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Mirroring Summary${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "Successful: ${GREEN}$SUCCESS${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo -e "${YELLOW}Note: Some images failed to mirror.${NC}"
    echo -e "${YELLOW}Ensure you are logged into the registry:${NC}"
    echo ""
    case "$REGISTRY_TYPE" in
        harbor)
            echo "  docker login $REGISTRY_URL"
            ;;
        ecr)
            echo "  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $REGISTRY_URL"
            ;;
        gcr)
            echo "  gcloud auth configure-docker"
            ;;
        acr)
            echo "  az acr login --name <your-registry-name>"
            ;;
    esac
    echo ""
fi

echo -e "${BLUE}Next steps:${NC}"
echo "1. Update image references in deployments:"
echo "   kubectl set image deployment/DEPLOYMENT -n NAMESPACE IMAGE_NAME=$REGISTRY_URL/IMAGE_NAME:latest"
echo ""
echo "2. Or update the deployment YAML files with new registry URLs"
echo ""
echo "3. Restart affected deployments:"
echo "   kubectl rollout restart deployment/DEPLOYMENT -n NAMESPACE"
echo ""
