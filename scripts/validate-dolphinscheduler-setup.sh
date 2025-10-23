#!/bin/bash
# Validate DolphinScheduler Setup
# Quick health check to verify all components are configured correctly
#
# Usage:
#   ./validate-dolphinscheduler-setup.sh

set -e

# Configuration
NAMESPACE="${NAMESPACE:-data-platform}"
SECRET_NAME="dolphinscheduler-api-keys"
WORKFLOWS_DIR="/home/m/tff/254CARBON/HMCo/workflows"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}"
cat << "EOF"
╔══════════════════════════════════════════════════╗
║   DolphinScheduler Setup Validation             ║
╚══════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""

ISSUES=0
WARNINGS=0
PASSED=0

# Function to check and report
check() {
    local name="$1"
    local status="$2"
    local message="$3"
    
    if [[ "$status" == "pass" ]]; then
        echo -e "${GREEN}✓${NC} $name"
        [[ -n "$message" ]] && echo -e "  ${message}"
        ((PASSED++))
    elif [[ "$status" == "warn" ]]; then
        echo -e "${YELLOW}⚠${NC} $name"
        [[ -n "$message" ]] && echo -e "  ${YELLOW}${message}${NC}"
        ((WARNINGS++))
    else
        echo -e "${RED}✗${NC} $name"
        [[ -n "$message" ]] && echo -e "  ${RED}${message}${NC}"
        ((ISSUES++))
    fi
}

# Check 1: Prerequisites
echo -e "${BLUE}${BOLD}Prerequisites:${NC}"
echo ""

if command -v kubectl &> /dev/null; then
    check "kubectl installed" "pass" "$(kubectl version --client --short 2>&1 | head -n1)"
else
    check "kubectl installed" "fail" "Not found - install kubectl"
fi

if command -v curl &> /dev/null; then
    check "curl installed" "pass"
else
    check "curl installed" "fail" "Not found - install curl"
fi

if command -v jq &> /dev/null; then
    check "jq installed" "pass" "$(jq --version)"
else
    check "jq installed" "fail" "Not found - install jq"
fi

if command -v python3 &> /dev/null; then
    check "python3 installed" "pass" "$(python3 --version)"
else
    check "python3 installed" "fail" "Not found - install python3"
fi

echo ""

# Check 2: Kubernetes Resources
echo -e "${BLUE}${BOLD}Kubernetes Resources:${NC}"
echo ""

if kubectl get namespace "$NAMESPACE" &> /dev/null; then
    check "Namespace: $NAMESPACE" "pass"
else
    check "Namespace: $NAMESPACE" "fail" "Namespace not found"
fi

if kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" &> /dev/null; then
    check "Secret: $SECRET_NAME" "pass"
    
    # Check API keys in secret
    KEYS=$(kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" -o jsonpath='{.data}' | jq -r 'keys[]' 2>/dev/null || echo "")
    EXPECTED_KEYS=("ALPHAVANTAGE_API_KEY" "POLYGON_API_KEY" "EIA_API_KEY" "GIE_API_KEY" "CENSUS_API_KEY" "NOAA_API_KEY")
    
    MISSING_KEYS=()
    for key in "${EXPECTED_KEYS[@]}"; do
        if ! echo "$KEYS" | grep -q "$key"; then
            MISSING_KEYS+=("$key")
        fi
    done
    
    if [[ ${#MISSING_KEYS[@]} -eq 0 ]]; then
        check "All API keys present" "pass" "6/6 keys found"
    else
        check "All API keys present" "warn" "Missing: ${MISSING_KEYS[*]}"
    fi
else
    check "Secret: $SECRET_NAME" "fail" "Secret not found - run configure-dolphinscheduler-credentials.sh"
fi

# Check DolphinScheduler pods
DOLPHIN_PODS=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=dolphinscheduler --no-headers 2>/dev/null | wc -l || echo "0")
RUNNING_PODS=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=dolphinscheduler --no-headers 2>/dev/null | grep -c Running || echo "0")

if [[ $DOLPHIN_PODS -gt 0 ]]; then
    if [[ $RUNNING_PODS -eq $DOLPHIN_PODS ]]; then
        check "DolphinScheduler pods" "pass" "$RUNNING_PODS/$DOLPHIN_PODS running"
    else
        check "DolphinScheduler pods" "warn" "Only $RUNNING_PODS/$DOLPHIN_PODS running"
    fi
else
    check "DolphinScheduler pods" "fail" "No pods found"
fi

# Check Trino pods
TRINO_PODS=$(kubectl get pods -n "$NAMESPACE" -l app=trino --no-headers 2>/dev/null | wc -l || echo "0")
TRINO_RUNNING=$(kubectl get pods -n "$NAMESPACE" -l app=trino --no-headers 2>/dev/null | grep -c Running || echo "0")

if [[ $TRINO_PODS -gt 0 ]]; then
    if [[ $TRINO_RUNNING -eq $TRINO_PODS ]]; then
        check "Trino pods" "pass" "$TRINO_RUNNING/$TRINO_PODS running"
    else
        check "Trino pods" "warn" "Only $TRINO_RUNNING/$TRINO_PODS running"
    fi
else
    check "Trino pods" "warn" "No pods found (not critical)"
fi

echo ""

# Check 3: Automation Scripts
echo -e "${BLUE}${BOLD}Automation Scripts:${NC}"
echo ""

SCRIPTS=(
    "configure-dolphinscheduler-credentials.sh"
    "import-workflows-from-files.py"
    "test-dolphinscheduler-workflows.sh"
    "verify-workflow-data-ingestion.sh"
    "setup-dolphinscheduler-complete.sh"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "${SCRIPTS[@]}"; do
    SCRIPT_PATH="$SCRIPT_DIR/$script"
    if [[ -f "$SCRIPT_PATH" ]]; then
        if [[ -x "$SCRIPT_PATH" ]]; then
            check "$script" "pass" "Executable"
        else
            check "$script" "warn" "Not executable - run: chmod +x $SCRIPT_PATH"
        fi
    else
        check "$script" "fail" "Not found at: $SCRIPT_PATH"
    fi
done

echo ""

# Check 4: Workflow Files
echo -e "${BLUE}${BOLD}Workflow Files:${NC}"
echo ""

if [[ -d "$WORKFLOWS_DIR" ]]; then
    check "Workflows directory" "pass" "$WORKFLOWS_DIR"
    
    WORKFLOW_COUNT=$(ls -1 "$WORKFLOWS_DIR"/*.json 2>/dev/null | wc -l)
    if [[ $WORKFLOW_COUNT -eq 11 ]]; then
        check "Workflow JSON files" "pass" "11/11 files found"
    elif [[ $WORKFLOW_COUNT -gt 0 ]]; then
        check "Workflow JSON files" "warn" "Only $WORKFLOW_COUNT/11 files found"
    else
        check "Workflow JSON files" "fail" "No workflow files found"
    fi
else
    check "Workflows directory" "fail" "Directory not found: $WORKFLOWS_DIR"
fi

echo ""

# Check 5: DolphinScheduler API Connectivity
echo -e "${BLUE}${BOLD}DolphinScheduler API:${NC}"
echo ""

# Find API pod
API_POD=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api --no-headers 2>/dev/null | head -n1 | awk '{print $1}' || echo "")

if [[ -n "$API_POD" ]]; then
    check "API pod found" "pass" "$API_POD"
    
    # Test port-forward (brief)
    kubectl port-forward -n "$NAMESPACE" "$API_POD" 12345:12345 &> /dev/null &
    PF_PID=$!
    sleep 2
    
    # Test API health
    API_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:12345/dolphinscheduler/ui/" 2>/dev/null || echo "000")
    
    kill $PF_PID 2>/dev/null || true
    
    if [[ "$API_RESPONSE" == "200" ]] || [[ "$API_RESPONSE" == "302" ]]; then
        check "API connectivity" "pass" "HTTP $API_RESPONSE"
    else
        check "API connectivity" "warn" "HTTP $API_RESPONSE (may need authentication)"
    fi
else
    check "API pod found" "fail" "No API pod found"
fi

echo ""

# Summary
echo -e "${CYAN}${BOLD}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                  Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"

echo -e "  ${GREEN}✓ Passed:${NC}   $PASSED"
[[ $WARNINGS -gt 0 ]] && echo -e "  ${YELLOW}⚠ Warnings:${NC} $WARNINGS"
[[ $ISSUES -gt 0 ]] && echo -e "  ${RED}✗ Failed:${NC}   $ISSUES"

echo ""

# Overall status
if [[ $ISSUES -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}🎉 Setup validation: PERFECT!${NC}"
    echo ""
    echo "All components are properly configured. You're ready to:"
    echo "  1. Run: ./scripts/setup-dolphinscheduler-complete.sh"
    echo "  2. Or manually import workflows via DolphinScheduler UI"
    EXIT_CODE=0
elif [[ $ISSUES -eq 0 ]]; then
    echo -e "${YELLOW}${BOLD}⚠ Setup validation: GOOD (with warnings)${NC}"
    echo ""
    echo "Most components are configured, but there are some warnings."
    echo "Review warnings above and address if needed."
    EXIT_CODE=0
else
    echo -e "${RED}${BOLD}✗ Setup validation: ISSUES FOUND${NC}"
    echo ""
    echo "Please address the failed checks above before proceeding."
    echo ""
    echo "Common fixes:"
    echo "  - Install missing tools: kubectl, curl, jq, python3"
    echo "  - Run: ./scripts/configure-dolphinscheduler-credentials.sh"
    echo "  - Check DolphinScheduler deployment in Kubernetes"
    EXIT_CODE=1
fi

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

exit $EXIT_CODE

