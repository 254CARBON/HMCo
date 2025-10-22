#!/bin/bash

###############################################################################
# 254Carbon Advanced Analytics Platform - Deployment Script
# 
# This script deploys the complete advanced analytics platform including:
# - Real-time ML serving (Ray Serve + Feast)
# - ML/AI orchestration (Kubeflow + Katib)
# - Advanced model serving (Seldon Core)
# - Data quality & governance (Great Expectations)
# - Real-time data pipelines (Debezium CDC + WebSocket)
#
# Usage: ./deploy-advanced-analytics-platform.sh [--phase=<1-5>] [--component=<name>]
###############################################################################

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE_DATA="data-platform"
NAMESPACE_KUBEFLOW="kubeflow"
NAMESPACE_SELDON="seldon-system"
NAMESPACE_RAY="ray-system"

# Parse arguments
DEPLOY_PHASE="all"
DEPLOY_COMPONENT=""

for arg in "$@"; do
    case $arg in
        --phase=*)
            DEPLOY_PHASE="${arg#*=}"
            shift
            ;;
        --component=*)
            DEPLOY_COMPONENT="${arg#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--phase=<1-5|all>] [--component=<name>]"
            echo ""
            echo "Phases:"
            echo "  1 - Real-time ML Pipeline Integration"
            echo "  2 - ML/AI Platform Expansion"
            echo "  3 - Data Quality & Governance"
            echo "  4 - Advanced Observability & AIOps"
            echo "  5 - User Experience & Integration"
            echo "  all - Deploy everything (default)"
            echo ""
            echo "Components:"
            echo "  ray-serve, feast, debezium, websocket, kubeflow, katib,"
            echo "  seldon, great-expectations, pytorch, tensorflow"
            exit 0
            ;;
    esac
done

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check existing services
    log_info "Verifying existing services..."
    kubectl get pods -n $NAMESPACE_DATA -l app=mlflow &> /dev/null || log_warning "MLflow not found"
    kubectl get pods -n $NAMESPACE_DATA -l app=minio &> /dev/null || log_warning "MinIO not found"
    kubectl get pods -n $NAMESPACE_DATA -l app=kafka &> /dev/null || log_warning "Kafka not found"
    kubectl get pods -n $NAMESPACE_DATA -l app=redis &> /dev/null || log_warning "Redis not found"
    kubectl get pods -n $NAMESPACE_DATA -l app=postgres &> /dev/null || log_warning "PostgreSQL not found"
    
    log_success "Prerequisites check completed"
}

deploy_phase1() {
    log_info "========================================="
    log_info "Phase 1: Real-time ML Pipeline Integration"
    log_info "========================================="
    
    # Ray Serve
    if [[ -z "$DEPLOY_COMPONENT" || "$DEPLOY_COMPONENT" == "ray-serve" ]]; then
        log_info "Deploying Ray Serve..."
        kubectl apply -f k8s/ml-platform/ray-serve/namespace.yaml
        kubectl apply -f k8s/ml-platform/ray-serve/ray-operator.yaml
        
        log_info "Waiting for Ray operator..."
        kubectl wait --for=condition=ready pod -l app=ray-operator -n $NAMESPACE_RAY --timeout=300s
        
        kubectl apply -f k8s/ml-platform/ray-serve/ray-serve-cluster.yaml
        log_success "Ray Serve deployed"
    fi
    
    # Feast Feature Store
    if [[ -z "$DEPLOY_COMPONENT" || "$DEPLOY_COMPONENT" == "feast" ]]; then
        log_info "Deploying Feast Feature Store..."
        kubectl apply -f k8s/ml-platform/feast-feature-store/feast-deployment.yaml
        kubectl apply -f k8s/ml-platform/feast-feature-store/feast-init-job.yaml
        
        log_info "Waiting for Feast initialization..."
        kubectl wait --for=condition=complete job/feast-init -n $NAMESPACE_DATA --timeout=300s || true
        
        kubectl apply -f k8s/ml-platform/feast-feature-store/feast-materialization-job.yaml
        log_success "Feast Feature Store deployed"
    fi
    
    # Debezium CDC
    if [[ -z "$DEPLOY_COMPONENT" || "$DEPLOY_COMPONENT" == "debezium" ]]; then
        log_info "Deploying Debezium CDC connectors..."
        kubectl apply -f k8s/streaming/connectors/debezium-postgres-connector.yaml
        
        log_info "Setting up PostgreSQL for CDC..."
        kubectl wait --for=condition=complete job/debezium-postgres-setup -n $NAMESPACE_DATA --timeout=300s || true
        
        kubectl apply -f k8s/streaming/connectors/debezium-connector-deployment.yaml
        log_success "Debezium CDC deployed"
    fi
    
    # WebSocket Gateway
    if [[ -z "$DEPLOY_COMPONENT" || "$DEPLOY_COMPONENT" == "websocket" ]]; then
        log_info "Deploying WebSocket Gateway..."
        kubectl apply -f k8s/streaming/connectors/websocket-gateway.yaml
        log_success "WebSocket Gateway deployed"
    fi
    
    log_success "Phase 1 deployment completed!"
}

deploy_phase2() {
    log_info "========================================="
    log_info "Phase 2: ML/AI Platform Expansion"
    log_info "========================================="
    
    # Kubeflow Pipelines
    if [[ -z "$DEPLOY_COMPONENT" || "$DEPLOY_COMPONENT" == "kubeflow" ]]; then
        log_info "Deploying Kubeflow Pipelines..."
        kubectl apply -f k8s/ml-platform/kubeflow/namespace.yaml
        kubectl apply -f k8s/ml-platform/kubeflow/kubeflow-pipelines.yaml
        
        log_info "Waiting for Kubeflow Pipelines..."
        kubectl wait --for=condition=ready pod -l app=ml-pipeline -n $NAMESPACE_KUBEFLOW --timeout=600s || true
        log_success "Kubeflow Pipelines deployed"
    fi
    
    # Katib
    if [[ -z "$DEPLOY_COMPONENT" || "$DEPLOY_COMPONENT" == "katib" ]]; then
        log_info "Deploying Katib..."
        kubectl apply -f k8s/ml-platform/kubeflow/katib.yaml
        log_success "Katib deployed"
    fi
    
    # Training Operators
    if [[ -z "$DEPLOY_COMPONENT" || "$DEPLOY_COMPONENT" == "pytorch" ]]; then
        log_info "Deploying PyTorch Operator..."
        kubectl apply -f k8s/ml-platform/training-operators/pytorch-operator.yaml
        log_success "PyTorch Operator deployed"
    fi
    
    if [[ -z "$DEPLOY_COMPONENT" || "$DEPLOY_COMPONENT" == "tensorflow" ]]; then
        log_info "Deploying TensorFlow Operator..."
        kubectl apply -f k8s/ml-platform/training-operators/tensorflow-operator.yaml
        log_success "TensorFlow Operator deployed"
    fi
    
    # Seldon Core
    if [[ -z "$DEPLOY_COMPONENT" || "$DEPLOY_COMPONENT" == "seldon" ]]; then
        log_info "Deploying Seldon Core..."
        kubectl apply -f k8s/ml-platform/seldon-core/seldon-operator.yaml
        
        log_info "Waiting for Seldon operator..."
        kubectl wait --for=condition=ready pod -l app=seldon-controller-manager -n $NAMESPACE_SELDON --timeout=300s || true
        
        log_info "Deploying example Seldon deployments..."
        kubectl apply -f k8s/ml-platform/seldon-core/seldon-deployment-example.yaml || true
        log_success "Seldon Core deployed"
    fi
    
    log_success "Phase 2 deployment completed!"
}

deploy_phase3() {
    log_info "========================================="
    log_info "Phase 3: Data Quality & Governance"
    log_info "========================================="
    
    # Great Expectations
    if [[ -z "$DEPLOY_COMPONENT" || "$DEPLOY_COMPONENT" == "great-expectations" ]]; then
        log_info "Deploying Great Expectations..."
        kubectl apply -f k8s/governance/great-expectations/great-expectations-deployment.yaml
        log_success "Great Expectations deployed"
    fi
    
    log_info "Note: Apache Atlas, OPA policies, and PII scanner are referenced in plan."
    log_info "Deploy them using their official Helm charts with our configurations."
    
    log_success "Phase 3 deployment completed!"
}

deploy_phase4() {
    log_info "========================================="
    log_info "Phase 4: Advanced Observability & AIOps"
    log_info "========================================="
    
    log_info "Note: VictoriaMetrics, Thanos, and Chaos Mesh deployments are referenced in plan."
    log_info "Deploy them using their official Helm charts:"
    log_info "  - helm install victoria-metrics vm/victoria-metrics-k8s-stack"
    log_info "  - helm install thanos bitnami/thanos"
    log_info "  - helm install chaos-mesh chaos-mesh/chaos-mesh"
    
    log_success "Phase 4 deployment guide provided!"
}

deploy_phase5() {
    log_info "========================================="
    log_info "Phase 5: User Experience & Integration"
    log_info "========================================="
    
    log_info "Note: Portal enhancements, SDK libraries, and CLI tools are referenced in plan."
    log_info "See ADVANCED_ANALYTICS_PLATFORM_SUMMARY.md for details."
    
    log_success "Phase 5 deployment guide provided!"
}

verify_deployment() {
    log_info "========================================="
    log_info "Verifying Deployment"
    log_info "========================================="
    
    # Check Ray Serve
    log_info "Checking Ray Serve..."
    kubectl get pods -n $NAMESPACE_DATA -l app=ray-serve 2>/dev/null || log_warning "Ray Serve not found"
    
    # Check Feast
    log_info "Checking Feast..."
    kubectl get pods -n $NAMESPACE_DATA -l app=feast-server 2>/dev/null || log_warning "Feast not found"
    
    # Check Kubeflow
    log_info "Checking Kubeflow..."
    kubectl get pods -n $NAMESPACE_KUBEFLOW -l app=ml-pipeline 2>/dev/null || log_warning "Kubeflow not found"
    
    # Check Seldon
    log_info "Checking Seldon..."
    kubectl get pods -n $NAMESPACE_SELDON -l app=seldon-controller-manager 2>/dev/null || log_warning "Seldon not found"
    
    log_info "========================================="
    log_info "Deployment Summary"
    log_info "========================================="
    log_info "Ray Serve UI: kubectl port-forward -n $NAMESPACE_DATA svc/ray-serve-service 8265:8265"
    log_info "Kubeflow UI: kubectl port-forward -n $NAMESPACE_KUBEFLOW svc/ml-pipeline-ui 8080:80"
    log_info "Feast: kubectl port-forward -n $NAMESPACE_DATA svc/feast-server 8080:8080"
    log_info ""
    log_info "For complete documentation, see:"
    log_info "  - ADVANCED_ANALYTICS_PLATFORM_SUMMARY.md"
    log_info "  - k8s/ml-platform/*/README.md"
    log_success "Verification completed!"
}

# Main execution
main() {
    log_info "Starting Advanced Analytics Platform Deployment"
    log_info "Phase: $DEPLOY_PHASE"
    if [[ -n "$DEPLOY_COMPONENT" ]]; then
        log_info "Component: $DEPLOY_COMPONENT"
    fi
    echo ""
    
    check_prerequisites
    
    case $DEPLOY_PHASE in
        1)
            deploy_phase1
            ;;
        2)
            deploy_phase2
            ;;
        3)
            deploy_phase3
            ;;
        4)
            deploy_phase4
            ;;
        5)
            deploy_phase5
            ;;
        all)
            deploy_phase1
            deploy_phase2
            deploy_phase3
            deploy_phase4
            deploy_phase5
            ;;
        *)
            log_error "Invalid phase: $DEPLOY_PHASE"
            log_info "Valid phases: 1, 2, 3, 4, 5, all"
            exit 1
            ;;
    esac
    
    echo ""
    verify_deployment
    
    echo ""
    log_success "========================================="
    log_success "Advanced Analytics Platform Deployment Complete!"
    log_success "========================================="
    log_info "Next steps:"
    log_info "1. Review logs for any warnings"
    log_info "2. Access service UIs (see port-forward commands above)"
    log_info "3. Run integration tests"
    log_info "4. Configure monitoring and alerts"
    log_info "5. Review ADVANCED_ANALYTICS_PLATFORM_SUMMARY.md for usage examples"
}

# Run main
main



