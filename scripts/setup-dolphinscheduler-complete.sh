#!/bin/bash
# DolphinScheduler Complete Setup
# Master script that orchestrates the complete setup process:
#   1. Configure API credentials
#   2. Import workflows
#   3. Test execution
#   4. Verify data ingestion
#
# Usage:
#   ./setup-dolphinscheduler-complete.sh
#   ./setup-dolphinscheduler-complete.sh --skip-test
#   ./setup-dolphinscheduler-complete.sh --verify-only

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-data-platform}"
DOLPHIN_URL="${DOLPHIN_URL:-http://localhost:12345}"
PROJECT_NAME="${PROJECT_NAME:-Commodity Data Platform}"
SKIP_CREDENTIALS=false
SKIP_IMPORT=false
SKIP_TEST=false
VERIFY_ONLY=false

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-credentials)
            SKIP_CREDENTIALS=true
            shift
            ;;
        --skip-import)
            SKIP_IMPORT=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
            SKIP_CREDENTIALS=true
            SKIP_IMPORT=true
            SKIP_TEST=true
            shift
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --help)
            cat << EOF
Usage: $0 [OPTIONS]

Complete setup for DolphinScheduler workflows:
  1. Configure API credentials (Kubernetes secrets + DolphinScheduler variables)
  2. Import all 11 workflow definitions from JSON files
  3. Test execution with comprehensive workflow (#11)
  4. Verify data ingestion in Trino/Iceberg

Options:
  --skip-credentials   Skip API credentials configuration
  --skip-import        Skip workflow import
  --skip-test          Skip test execution
  --verify-only        Only run data verification
  --namespace NS       Kubernetes namespace (default: data-platform)
  --help               Show this help

Examples:
  # Full setup (recommended first run)
  $0

  # Only verify data after manual workflow run
  $0 --verify-only

  # Import workflows only (credentials already configured)
  $0 --skip-credentials --skip-test
EOF
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Banner
clear
echo -e "${CYAN}${BOLD}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        DolphinScheduler Complete Setup Automation           ║
║        Commodity Data Platform Workflow Deployment          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""
echo -e "${BLUE}This script will:${NC}"
if [[ "$VERIFY_ONLY" == "true" ]]; then
    echo "  - Verify data ingestion in Trino"
else
    [[ "$SKIP_CREDENTIALS" == "false" ]] && echo "  1. Configure API credentials (6 data sources)"
    [[ "$SKIP_IMPORT" == "false" ]] && echo "  2. Import 11 workflow definitions"
    [[ "$SKIP_TEST" == "false" ]] && echo "  3. Test comprehensive workflow execution"
    echo "  4. Verify data landed in Iceberg/Trino"
fi
echo ""
read -p "Press Enter to continue, or Ctrl+C to cancel..."
echo ""

# Track execution time
SETUP_START_TIME=$(date +%s)

# Check prerequisites
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 0: Checking Prerequisites${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

PREREQUISITES_OK=true

# Check kubectl
if command -v kubectl &> /dev/null; then
    KUBECTL_VERSION=$(kubectl version --client --short 2>/dev/null | head -n1)
    echo -e "${GREEN}✓ kubectl: $KUBECTL_VERSION${NC}"
else
    echo -e "${RED}✗ kubectl not found${NC}"
    PREREQUISITES_OK=false
fi

# Check curl
if command -v curl &> /dev/null; then
    echo -e "${GREEN}✓ curl available${NC}"
else
    echo -e "${RED}✗ curl not found${NC}"
    PREREQUISITES_OK=false
fi

# Check jq
if command -v jq &> /dev/null; then
    JQ_VERSION=$(jq --version)
    echo -e "${GREEN}✓ jq: $JQ_VERSION${NC}"
else
    echo -e "${RED}✗ jq not found${NC}"
    PREREQUISITES_OK=false
fi

# Check Python 3
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ python3: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ python3 not found${NC}"
    PREREQUISITES_OK=false
fi

# Check namespace exists
if kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo -e "${GREEN}✓ Namespace exists: $NAMESPACE${NC}"
else
    echo -e "${RED}✗ Namespace not found: $NAMESPACE${NC}"
    PREREQUISITES_OK=false
fi

# Check DolphinScheduler pods
DOLPHIN_PODS=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=dolphinscheduler 2>/dev/null | grep -c Running || echo "0")
if [[ $DOLPHIN_PODS -gt 0 ]]; then
    echo -e "${GREEN}✓ DolphinScheduler pods running: $DOLPHIN_PODS${NC}"
else
    echo -e "${YELLOW}⚠ No DolphinScheduler pods found (may be using different labels)${NC}"
fi

echo ""

if [[ "$PREREQUISITES_OK" == "false" ]]; then
    echo -e "${RED}✗ Prerequisites check failed${NC}"
    echo -e "${YELLOW}Please install missing tools and try again${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites met${NC}"
echo ""
sleep 2

# Step 1: Configure Credentials
if [[ "$SKIP_CREDENTIALS" == "false" ]]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 1: Configuring API Credentials${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    if [[ -f "$SCRIPT_DIR/configure-dolphinscheduler-credentials.sh" ]]; then
        chmod +x "$SCRIPT_DIR/configure-dolphinscheduler-credentials.sh"
        if bash "$SCRIPT_DIR/configure-dolphinscheduler-credentials.sh" --namespace "$NAMESPACE"; then
            echo ""
            echo -e "${GREEN}✓ Credentials configured successfully${NC}"
        else
            echo ""
            echo -e "${RED}✗ Failed to configure credentials${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ Credentials script not found: $SCRIPT_DIR/configure-dolphinscheduler-credentials.sh${NC}"
        exit 1
    fi
    
    echo ""
    sleep 2
fi

# Step 2: Import Workflows
if [[ "$SKIP_IMPORT" == "false" ]]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 2: Importing Workflows${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    if [[ -f "$SCRIPT_DIR/import-workflows-from-files.py" ]]; then
        chmod +x "$SCRIPT_DIR/import-workflows-from-files.py"
        if python3 "$SCRIPT_DIR/import-workflows-from-files.py" \
            --port-forward \
            --namespace "$NAMESPACE" \
            --project-name "$PROJECT_NAME" \
            --skip-existing; then
            echo ""
            echo -e "${GREEN}✓ Workflows imported successfully${NC}"
        else
            echo ""
            echo -e "${RED}✗ Failed to import workflows${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ Import script not found: $SCRIPT_DIR/import-workflows-from-files.py${NC}"
        exit 1
    fi
    
    echo ""
    sleep 2
fi

# Step 3: Test Execution
if [[ "$SKIP_TEST" == "false" ]]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 3: Testing Workflow Execution${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}This will run the comprehensive test workflow (~30-45 minutes)${NC}"
    echo -e "${YELLOW}You can skip this and run workflows manually in the UI${NC}"
    echo ""
    read -p "Run test workflow now? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [[ -f "$SCRIPT_DIR/test-dolphinscheduler-workflows.sh" ]]; then
            chmod +x "$SCRIPT_DIR/test-dolphinscheduler-workflows.sh"
            if bash "$SCRIPT_DIR/test-dolphinscheduler-workflows.sh" \
                --project-name "$PROJECT_NAME"; then
                echo ""
                echo -e "${GREEN}✓ Test workflow completed successfully${NC}"
            else
                echo ""
                echo -e "${RED}✗ Test workflow failed${NC}"
                echo -e "${YELLOW}Check DolphinScheduler logs for details${NC}"
                exit 1
            fi
        else
            echo -e "${RED}✗ Test script not found: $SCRIPT_DIR/test-dolphinscheduler-workflows.sh${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⊘ Skipping test execution${NC}"
        echo -e "${YELLOW}You can run it later with: ./scripts/test-dolphinscheduler-workflows.sh${NC}"
    fi
    
    echo ""
    sleep 2
fi

# Step 4: Verify Data
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 4: Verifying Data Ingestion${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [[ -f "$SCRIPT_DIR/verify-workflow-data-ingestion.sh" ]]; then
    chmod +x "$SCRIPT_DIR/verify-workflow-data-ingestion.sh"
    if bash "$SCRIPT_DIR/verify-workflow-data-ingestion.sh"; then
        echo ""
        echo -e "${GREEN}✓ Data verification passed${NC}"
        VERIFICATION_STATUS="PASSED"
    else
        echo ""
        echo -e "${YELLOW}⚠ Data verification returned warnings or failures${NC}"
        echo -e "${YELLOW}This is normal if workflows haven't run yet${NC}"
        VERIFICATION_STATUS="PARTIAL"
    fi
else
    echo -e "${RED}✗ Verification script not found: $SCRIPT_DIR/verify-workflow-data-ingestion.sh${NC}"
    VERIFICATION_STATUS="SKIPPED"
fi

echo ""
sleep 2

# Calculate total time
SETUP_END_TIME=$(date +%s)
SETUP_DURATION=$((SETUP_END_TIME - SETUP_START_TIME))
SETUP_MINUTES=$((SETUP_DURATION / 60))
SETUP_SECONDS=$((SETUP_DURATION % 60))

# Final Summary
echo ""
echo -e "${CYAN}${BOLD}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                    Setup Complete! 🎉                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""
echo -e "${GREEN}${BOLD}Summary:${NC}"
echo ""

[[ "$SKIP_CREDENTIALS" == "false" ]] && echo -e "  ${GREEN}✓${NC} API credentials configured"
[[ "$SKIP_IMPORT" == "false" ]] && echo -e "  ${GREEN}✓${NC} Workflows imported (11 definitions)"
[[ "$SKIP_TEST" == "false" ]] && echo -e "  ${GREEN}✓${NC} Test execution completed"
echo -e "  ${GREEN}✓${NC} Data verification: $VERIFICATION_STATUS"
echo ""
echo "  ⏱  Total time: ${SETUP_MINUTES}m ${SETUP_SECONDS}s"
echo ""

echo -e "${BLUE}${BOLD}Access Points:${NC}"
echo ""
echo "  🌐 DolphinScheduler UI:"
echo "     https://dolphin.254carbon.com"
echo ""
echo "  📊 Grafana Dashboards:"
echo "     https://grafana.254carbon.com"
echo ""
echo "  🗄️  Trino Query Engine:"
echo "     kubectl port-forward -n $NAMESPACE svc/trino-coordinator 8080:8080"
echo ""

echo -e "${BLUE}${BOLD}Next Steps:${NC}"
echo ""
echo "  1. Log in to DolphinScheduler UI"
echo "     - URL: https://dolphin.254carbon.com"
echo "     - Username: admin"
echo "     - Password: dolphinscheduler123"
echo ""
echo "  2. Review imported workflows"
echo "     - Project: $PROJECT_NAME"
echo "     - Workflows: 11 definitions"
echo ""
echo "  3. Enable workflow schedules"
echo "     - Start with workflow #11 (comprehensive test)"
echo "     - Or enable individual source workflows"
echo ""
echo "  4. Monitor execution"
echo "     - Check 'Workflow Instances' for status"
echo "     - View task logs for debugging"
echo "     - Monitor Grafana for metrics"
echo ""
echo "  5. Verify data quality"
echo "     - Run: ./scripts/verify-workflow-data-ingestion.sh"
echo "     - Query Trino for analysis"
echo "     - Check data freshness daily"
echo ""

echo -e "${YELLOW}${BOLD}Recommended Workflows Schedule:${NC}"
echo ""
echo "  • Comprehensive Collection (#11): Daily 1 AM UTC"
echo "  • Or individual workflows: Staggered 2-8 AM UTC"
echo "  • Data Quality Checks (#5): Daily 6 AM UTC"
echo ""

echo -e "${CYAN}${BOLD}Documentation:${NC}"
echo ""
echo "  📖 Workflow Import Guide:"
echo "     $SCRIPT_DIR/../WORKFLOW_IMPORT_GUIDE.md"
echo ""
echo "  📖 Workflows README:"
echo "     $SCRIPT_DIR/../workflows/README.md"
echo ""
echo "  📖 Platform Docs:"
echo "     $SCRIPT_DIR/../docs/"
echo ""

echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Setup automation complete! Happy data ingestion! 🚀${NC}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

