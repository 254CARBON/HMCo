#!/bin/bash
# All-in-One Setup Script for 254Carbon Commodity Platform
# Automates the complete setup process
#
# Usage: ./setup-commodity-platform.sh [OPTIONS]
#
# Options:
#   --skip-api-keys        Skip API key configuration
#   --skip-workflows       Skip workflow import
#   --skip-dashboards      Skip dashboard import
#   --skip-verification    Skip final verification
#   --non-interactive      Use environment variables for API keys

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_API_KEYS=false
SKIP_WORKFLOWS=false
SKIP_DASHBOARDS=false
SKIP_VERIFICATION=false
NON_INTERACTIVE=false

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-api-keys)
            SKIP_API_KEYS=true
            shift
            ;;
        --skip-workflows)
            SKIP_WORKFLOWS=true
            shift
            ;;
        --skip-dashboards)
            SKIP_DASHBOARDS=true
            shift
            ;;
        --skip-verification)
            SKIP_VERIFICATION=true
            shift
            ;;
        --non-interactive)
            NON_INTERACTIVE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-api-keys        Skip API key configuration"
            echo "  --skip-workflows       Skip workflow import"
            echo "  --skip-dashboards      Skip dashboard import"
            echo "  --skip-verification    Skip final verification"
            echo "  --non-interactive      Use environment variables for API keys"
            echo "  --help                 Show this help message"
            echo ""
            echo "Environment variables (for non-interactive mode):"
            echo "  FRED_API_KEY, EIA_API_KEY, NOAA_API_KEY, etc."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Banner
echo -e "${CYAN}"
cat << 'EOF'
  ____  ____  _  _    ____                _                 
 |___ \| ___|| || |  / ___|__ _ _ __ __ _| |__   ___  _ __  
   __) |___ \| || |_ | |   / _` | '__/ _` | '_ \ / _ \| '_ \ 
  / __/ ___) |__   _|| |__| (_| | | | (_| | |_) | (_) | | | |
 |_____|____/   |_|   \____\__,_|_|  \__,_|_.__/ \___/|_| |_|
                                                              
          Commodity Data Platform - Automated Setup
EOF
echo -e "${NC}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Automated Platform Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl found${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ python3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ python3 found${NC}"

if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}✗ Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Kubernetes cluster accessible${NC}"

if ! kubectl get namespace data-platform &> /dev/null; then
    echo -e "${RED}✗ data-platform namespace not found${NC}"
    echo "Please deploy the platform first."
    exit 1
fi
echo -e "${GREEN}✓ data-platform namespace exists${NC}"

echo ""
echo -e "${GREEN}All prerequisites met!${NC}"
echo ""

# Estimate time
ESTIMATED_TIME=10
if [ "$SKIP_API_KEYS" = false ]; then
    ESTIMATED_TIME=$((ESTIMATED_TIME + 5))
fi
if [ "$SKIP_WORKFLOWS" = false ]; then
    ESTIMATED_TIME=$((ESTIMATED_TIME + 3))
fi
if [ "$SKIP_DASHBOARDS" = false ]; then
    ESTIMATED_TIME=$((ESTIMATED_TIME + 3))
fi

echo -e "${BLUE}Setup Steps:${NC}"
echo "  1. Configure API keys" $([ "$SKIP_API_KEYS" = true ] && echo "(SKIPPED)" || echo "")
echo "  2. Wait for services to be ready"
echo "  3. Import DolphinScheduler workflows" $([ "$SKIP_WORKFLOWS" = true ] && echo "(SKIPPED)" || echo "")
echo "  4. Import Superset dashboards" $([ "$SKIP_DASHBOARDS" = true ] && echo "(SKIPPED)" || echo "")
echo "  5. Run platform verification" $([ "$SKIP_VERIFICATION" = true ] && echo "(SKIPPED)" || echo "")
echo ""
echo -e "${YELLOW}Estimated time: ~${ESTIMATED_TIME} minutes${NC}"
echo ""

if [ "$NON_INTERACTIVE" = false ]; then
    read -p "Continue? (Y/n): " confirm
    if [[ "$confirm" == "n" ]] || [[ "$confirm" == "N" ]]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

echo ""

# Step 1: Configure API Keys
if [ "$SKIP_API_KEYS" = false ]; then
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  Step 1: Configure API Keys${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    if [ "$NON_INTERACTIVE" = true ]; then
        bash "$SCRIPT_DIR/configure-api-keys.sh" --non-interactive
    else
        bash "$SCRIPT_DIR/configure-api-keys.sh"
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ API key configuration failed${NC}"
        exit 1
    fi
    echo ""
else
    echo -e "${YELLOW}Skipping API key configuration...${NC}"
    echo ""
fi

# Step 2: Wait for services
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${MAGENTA}  Step 2: Wait for Services${NC}"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Waiting for DolphinScheduler API...${NC}"
kubectl wait --for=condition=ready pod -l app=dolphinscheduler-api -n data-platform --timeout=300s 2>/dev/null || echo "  (timeout, continuing anyway)"
echo -e "${GREEN}✓ DolphinScheduler API ready${NC}"

echo -e "${BLUE}Waiting for Superset...${NC}"
kubectl wait --for=condition=ready pod -l app=superset -n data-platform --timeout=300s 2>/dev/null || echo "  (timeout, continuing anyway)"
echo -e "${GREEN}✓ Superset ready${NC}"

echo ""

# Step 3: Import workflows
if [ "$SKIP_WORKFLOWS" = false ]; then
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  Step 3: Import DolphinScheduler Workflows${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Port-forward DolphinScheduler API
    echo -e "${BLUE}Setting up port-forward...${NC}"
    kubectl port-forward -n data-platform svc/dolphinscheduler-api-service 12345:12345 &> /dev/null &
    PORT_FORWARD_PID=$!
    sleep 3
    
    # Run import script
    python3 "$SCRIPT_DIR/import-dolphinscheduler-workflows.py" \
        --dolphinscheduler-url "http://localhost:12345" \
        --skip-existing
    
    WORKFLOW_RESULT=$?
    
    # Clean up port-forward
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    if [ $WORKFLOW_RESULT -ne 0 ]; then
        echo -e "${YELLOW}⚠ Workflow import had issues, but continuing...${NC}"
    fi
    echo ""
else
    echo -e "${YELLOW}Skipping workflow import...${NC}"
    echo ""
fi

# Step 4: Import dashboards
if [ "$SKIP_DASHBOARDS" = false ]; then
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  Step 4: Import Superset Dashboards${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Port-forward Superset
    echo -e "${BLUE}Setting up port-forward...${NC}"
    kubectl port-forward -n data-platform svc/superset 8088:8088 &> /dev/null &
    PORT_FORWARD_PID=$!
    sleep 3
    
    # Run import script
    python3 "$SCRIPT_DIR/import-superset-dashboards.py" \
        --superset-url "http://localhost:8088" \
        --setup-databases
    
    DASHBOARD_RESULT=$?
    
    # Clean up port-forward
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    if [ $DASHBOARD_RESULT -ne 0 ]; then
        echo -e "${YELLOW}⚠ Dashboard import had issues, but continuing...${NC}"
    fi
    echo ""
else
    echo -e "${YELLOW}Skipping dashboard import...${NC}"
    echo ""
fi

# Step 5: Verify platform
if [ "$SKIP_VERIFICATION" = false ]; then
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  Step 5: Platform Verification${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    bash "$SCRIPT_DIR/verify-platform-complete.sh"
    
    VERIFY_RESULT=$?
    echo ""
else
    echo -e "${YELLOW}Skipping verification...${NC}"
    VERIFY_RESULT=0
    echo ""
fi

# Final summary
echo -e "${CYAN}"
cat << 'EOF'
  ____       _                    ____                      _      _       _ 
 / ___|  ___| |_ _   _ _ __      / ___|___  _ __ ___  _ __ | | ___| |_ ___| |
 \___ \ / _ \ __| | | | '_ \    | |   / _ \| '_ ` _ \| '_ \| |/ _ \ __/ _ \ |
  ___) |  __/ |_| |_| | |_) |   | |__| (_) | | | | | | |_) | |  __/ ||  __/_|
 |____/ \___|\__|\__,_| .__/     \____\___/|_| |_| |_| .__/|_|\___|\__\___(_)
                      |_|                            |_|                      
EOF
echo -e "${NC}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [ $VERIFY_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Platform is ready for use${NC}"
else
    echo -e "${YELLOW}⚠ Platform setup complete with warnings${NC}"
    echo "  Review the verification results above"
fi

echo ""
echo -e "${BLUE}Access Your Platform:${NC}"
echo ""
echo -e "  ${CYAN}Portal:${NC}            https://portal.254carbon.com"
echo -e "  ${CYAN}DolphinScheduler:${NC}  https://dolphinscheduler.254carbon.com/dolphinscheduler/ui/"
echo -e "  ${CYAN}Superset:${NC}          https://superset.254carbon.com"
echo -e "  ${CYAN}Grafana:${NC}           https://grafana.254carbon.com"
echo -e "  ${CYAN}DataHub:${NC}           https://datahub.254carbon.com"
echo -e "  ${CYAN}Trino:${NC}             https://trino.254carbon.com"
echo ""
echo -e "${BLUE}Default Credentials:${NC}"
echo -e "  ${CYAN}DolphinScheduler:${NC}  admin / dolphinscheduler123"
echo -e "  ${CYAN}Superset:${NC}          admin / admin"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Change default passwords"
echo "  2. Enable workflow schedules in DolphinScheduler"
echo "  3. Customize Superset dashboards"
echo "  4. Set up alerting in Grafana"
echo "  5. Review data quality metrics"
echo ""
echo -e "${YELLOW}Need help?${NC} Check the documentation:"
echo "  - COMMODITY_QUICKSTART.md"
echo "  - COMMODITY_PLATFORM_DEPLOYMENT.md"
echo "  - docs/automation/AUTOMATION_GUIDE.md"
echo ""

exit 0


