#!/bin/bash
# Test DolphinScheduler Workflow Execution
# Runs workflow #11 (comprehensive test) and monitors execution
#
# Usage:
#   ./test-dolphinscheduler-workflows.sh
#   ./test-dolphinscheduler-workflows.sh --workflow-name "Comprehensive Commodity Data Collection"

set -e

# Configuration
DOLPHIN_URL="${DOLPHIN_URL:-http://localhost:12345}"
DOLPHIN_USER="${DOLPHIN_USER:-admin}"
DOLPHIN_PASS="${DOLPHIN_PASS:-dolphinscheduler123}"
PROJECT_NAME="${PROJECT_NAME:-Commodity Data Platform}"
WORKFLOW_NAME="${WORKFLOW_NAME:-Comprehensive Commodity Data Collection}"
NAMESPACE="${NAMESPACE:-data-platform}"
MAX_WAIT_TIME=3600  # 1 hour max wait

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workflow-name)
            WORKFLOW_NAME="$2"
            shift 2
            ;;
        --dolphin-url)
            DOLPHIN_URL="$2"
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --workflow-name NAME    Workflow to test (default: Comprehensive Commodity Data Collection)"
            echo "  --dolphin-url URL       DolphinScheduler API URL"
            echo "  --project-name NAME     Project name"
            echo "  --help                  Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DolphinScheduler Workflow Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command -v curl &> /dev/null; then
    echo -e "${RED}✗ curl not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ curl available${NC}"

if ! command -v jq &> /dev/null; then
    echo -e "${RED}✗ jq not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ jq available${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl available${NC}"

# Setup port-forward if needed
PORT_FORWARD_PID=""
if [[ "$DOLPHIN_URL" == "http://localhost:12345" ]]; then
    echo ""
    echo -e "${YELLOW}Setting up port-forward...${NC}"
    
    API_POD=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$API_POD" ]]; then
        kubectl port-forward -n "$NAMESPACE" "$API_POD" 12345:12345 &> /dev/null &
        PORT_FORWARD_PID=$!
        sleep 3
        echo -e "${GREEN}✓ Port-forward established (PID: $PORT_FORWARD_PID)${NC}"
    fi
fi

# Cleanup function
cleanup() {
    if [[ -n "$PORT_FORWARD_PID" ]]; then
        kill "$PORT_FORWARD_PID" 2>/dev/null || true
        echo ""
        echo -e "${YELLOW}Port-forward terminated${NC}"
    fi
}
trap cleanup EXIT

# Login
echo ""
echo -e "${BLUE}Authenticating...${NC}"
LOGIN_RESPONSE=$(curl -s -X POST "$DOLPHIN_URL/dolphinscheduler/login" \
    -d "userName=$DOLPHIN_USER&userPassword=$DOLPHIN_PASS" \
    -H "Content-Type: application/x-www-form-urlencoded")

TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.data.token // empty')

if [[ -z "$TOKEN" ]] || [[ "$TOKEN" == "null" ]]; then
    echo -e "${RED}✗ Authentication failed${NC}"
    echo "$LOGIN_RESPONSE" | jq '.'
    exit 1
fi
echo -e "${GREEN}✓ Authenticated${NC}"

# Get project
echo ""
echo -e "${BLUE}Finding project...${NC}"
PROJECTS_RESPONSE=$(curl -s "$DOLPHIN_URL/dolphinscheduler/projects/list?token=$TOKEN")
PROJECT_CODE=$(echo "$PROJECTS_RESPONSE" | jq -r ".data[] | select(.name == \"$PROJECT_NAME\") | .code")

if [[ -z "$PROJECT_CODE" ]] || [[ "$PROJECT_CODE" == "null" ]]; then
    echo -e "${RED}✗ Project not found: $PROJECT_NAME${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Project found (code: $PROJECT_CODE)${NC}"

# Get workflow definition
echo ""
echo -e "${BLUE}Finding workflow...${NC}"
WORKFLOWS_RESPONSE=$(curl -s "$DOLPHIN_URL/dolphinscheduler/projects/$PROJECT_CODE/process-definition?token=$TOKEN")
WORKFLOW_CODE=$(echo "$WORKFLOWS_RESPONSE" | jq -r ".data.totalList[]? | select(.name == \"$WORKFLOW_NAME\") | .code")

if [[ -z "$WORKFLOW_CODE" ]] || [[ "$WORKFLOW_CODE" == "null" ]]; then
    echo -e "${RED}✗ Workflow not found: $WORKFLOW_NAME${NC}"
    echo -e "${YELLOW}Available workflows:${NC}"
    echo "$WORKFLOWS_RESPONSE" | jq -r '.data.totalList[]?.name' | sed 's/^/  - /'
    exit 1
fi
echo -e "${GREEN}✓ Workflow found (code: $WORKFLOW_CODE)${NC}"

# Start workflow execution
echo ""
echo -e "${BLUE}Starting workflow execution...${NC}"
echo "Workflow: $WORKFLOW_NAME"
echo "Project: $PROJECT_NAME"
echo ""

START_RESPONSE=$(curl -s -X POST "$DOLPHIN_URL/dolphinscheduler/projects/$PROJECT_CODE/executors/start-process-instance" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "token=$TOKEN&processDefinitionCode=$WORKFLOW_CODE&scheduleTime=&failureStrategy=CONTINUE&warningType=NONE&warningGroupId=0&execType=&startNodeList=&taskDependType=TASK_POST&runMode=RUN_MODE_SERIAL&processInstancePriority=MEDIUM&workerGroup=default&environmentCode=-1&startParams=&expectedParallelismNumber=&dryRun=0&testFlag=NO&complementDependentMode=OFF_MODE")

INSTANCE_ID=$(echo "$START_RESPONSE" | jq -r '.data // empty')

if [[ -z "$INSTANCE_ID" ]] || [[ "$INSTANCE_ID" == "null" ]]; then
    echo -e "${RED}✗ Failed to start workflow${NC}"
    echo "$START_RESPONSE" | jq '.'
    exit 1
fi

echo -e "${GREEN}✓ Workflow started (instance ID: $INSTANCE_ID)${NC}"
echo ""

# Monitor execution
echo -e "${BLUE}Monitoring execution...${NC}"
echo "(This may take 30-45 minutes for comprehensive test)"
echo ""

START_TIME=$(date +%s)
LAST_STATE=""
TASK_COUNTS=""

while true; do
    # Get instance status
    STATUS_RESPONSE=$(curl -s "$DOLPHIN_URL/dolphinscheduler/projects/$PROJECT_CODE/instance/list?token=$TOKEN&searchVal=&stateType=&host=&startDate=&endDate=&pageNo=1&pageSize=20")
    
    INSTANCE=$(echo "$STATUS_RESPONSE" | jq -r ".data.totalList[]? | select(.id == $INSTANCE_ID)")
    
    if [[ -z "$INSTANCE" ]]; then
        echo -e "${RED}✗ Instance not found${NC}"
        break
    fi
    
    STATE=$(echo "$INSTANCE" | jq -r '.state')
    DURATION=$(echo "$INSTANCE" | jq -r '.duration // "0"')
    
    # Get task status counts
    TASKS=$(curl -s "$DOLPHIN_URL/dolphinscheduler/projects/$PROJECT_CODE/instance/$INSTANCE_ID/task-list-by-process-id?token=$TOKEN")
    TOTAL_TASKS=$(echo "$TASKS" | jq -r '.data | length')
    SUCCESS_TASKS=$(echo "$TASKS" | jq -r '[.data[]? | select(.state == "SUCCESS")] | length')
    RUNNING_TASKS=$(echo "$TASKS" | jq -r '[.data[]? | select(.state == "RUNNING_EXECUTION")] | length')
    FAILED_TASKS=$(echo "$TASKS" | jq -r '[.data[]? | select(.state == "FAILURE")] | length')
    
    CURRENT_COUNTS="$SUCCESS_TASKS/$TOTAL_TASKS tasks"
    
    # Print status if changed
    if [[ "$STATE" != "$LAST_STATE" ]] || [[ "$TASK_COUNTS" != "$CURRENT_COUNTS" ]]; then
        TIMESTAMP=$(date '+%H:%M:%S')
        echo "[$TIMESTAMP] State: $STATE | Progress: $SUCCESS_TASKS/$TOTAL_TASKS tasks | Running: $RUNNING_TASKS | Failed: $FAILED_TASKS | Duration: ${DURATION}s"
        LAST_STATE="$STATE"
        TASK_COUNTS="$CURRENT_COUNTS"
    fi
    
    # Check terminal states
    if [[ "$STATE" == "SUCCESS" ]]; then
        echo ""
        echo -e "${GREEN}✓ Workflow completed successfully!${NC}"
        echo "Total duration: ${DURATION}s ($(($DURATION / 60)) minutes)"
        break
    elif [[ "$STATE" == "FAILURE" ]]; then
        echo ""
        echo -e "${RED}✗ Workflow failed${NC}"
        echo ""
        echo -e "${YELLOW}Failed tasks:${NC}"
        echo "$TASKS" | jq -r '.data[]? | select(.state == "FAILURE") | "  - " + .name + ": " + (.state // "unknown")'
        exit 1
    elif [[ "$STATE" == "STOP" ]]; then
        echo ""
        echo -e "${YELLOW}⚠ Workflow stopped${NC}"
        exit 1
    fi
    
    # Check timeout
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [[ $ELAPSED -gt $MAX_WAIT_TIME ]]; then
        echo ""
        echo -e "${RED}✗ Timeout waiting for workflow completion${NC}"
        exit 1
    fi
    
    sleep 10
done

# Get final task details
echo ""
echo -e "${BLUE}Task Summary:${NC}"
echo "$TASKS" | jq -r '.data[]? | "  " + .name + ": " + .state'

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Workflow Test Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "  ✓ Workflow: $WORKFLOW_NAME"
echo "  ✓ Status: SUCCESS"
echo "  ✓ Tasks: $SUCCESS_TASKS/$TOTAL_TASKS completed"
echo "  ✓ Duration: $(($DURATION / 60)) minutes"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Verify data ingestion: ./scripts/verify-workflow-data-ingestion.sh"
echo "  2. Check Grafana dashboards for metrics"
echo "  3. Enable workflow schedules in DolphinScheduler UI"
echo ""

