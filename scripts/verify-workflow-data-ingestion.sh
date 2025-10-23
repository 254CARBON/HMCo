#!/bin/bash
# Verify Workflow Data Ingestion
# Connects to Trino and verifies data from DolphinScheduler workflows landed correctly
#
# Usage:
#   ./verify-workflow-data-ingestion.sh
#   ./verify-workflow-data-ingestion.sh --trino-host trino-coordinator.data-platform

set -e

# Configuration
TRINO_HOST="${TRINO_HOST:-trino-coordinator.data-platform}"
TRINO_PORT="${TRINO_PORT:-8080}"
TRINO_USER="${TRINO_USER:-admin}"
CATALOG="${CATALOG:-iceberg}"
SCHEMA="${SCHEMA:-commodity_data}"
NAMESPACE="${NAMESPACE:-data-platform}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --trino-host)
            TRINO_HOST="$2"
            shift 2
            ;;
        --catalog)
            CATALOG="$2"
            shift 2
            ;;
        --schema)
            SCHEMA="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --trino-host HOST    Trino coordinator host (default: trino-coordinator.data-platform)"
            echo "  --catalog NAME       Catalog name (default: iceberg)"
            echo "  --schema NAME        Schema name (default: commodity_data)"
            echo "  --help               Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Data Ingestion Verification${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl available${NC}"

# Setup port-forward to Trino
echo ""
echo -e "${YELLOW}Setting up port-forward to Trino...${NC}"

TRINO_POD=$(kubectl get pods -n "$NAMESPACE" -l app=trino,component=coordinator -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [[ -z "$TRINO_POD" ]]; then
    echo -e "${RED}✗ Trino coordinator pod not found${NC}"
    exit 1
fi

kubectl port-forward -n "$NAMESPACE" "$TRINO_POD" 8080:8080 &> /dev/null &
PORT_FORWARD_PID=$!
sleep 3
echo -e "${GREEN}✓ Port-forward established (PID: $PORT_FORWARD_PID)${NC}"

# Cleanup function
cleanup() {
    if [[ -n "$PORT_FORWARD_PID" ]]; then
        kill "$PORT_FORWARD_PID" 2>/dev/null || true
        echo ""
        echo -e "${YELLOW}Port-forward terminated${NC}"
    fi
}
trap cleanup EXIT

# Function to run Trino query
run_query() {
    local query="$1"
    local format="${2:-json}"
    
    curl -s -X POST "http://localhost:8080/v1/statement" \
        -H "X-Trino-User: $TRINO_USER" \
        -H "X-Trino-Catalog: $CATALOG" \
        -H "X-Trino-Schema: $SCHEMA" \
        -H "Content-Type: text/plain" \
        -d "$query" | jq -r '.data[][]? // empty' 2>/dev/null
}

# Test connection
echo ""
echo -e "${BLUE}Testing Trino connection...${NC}"

TEST_QUERY="SELECT 1"
RESULT=$(curl -s -X POST "http://localhost:8080/v1/statement" \
    -H "X-Trino-User: $TRINO_USER" \
    -d "$TEST_QUERY" | jq -r '.stats.state' 2>/dev/null || echo "FAILED")

if [[ "$RESULT" == "FINISHED" ]] || [[ "$RESULT" == "RUNNING" ]]; then
    echo -e "${GREEN}✓ Trino connection successful${NC}"
else
    echo -e "${RED}✗ Failed to connect to Trino${NC}"
    exit 1
fi

# Check schema exists
echo ""
echo -e "${BLUE}Checking schema: $CATALOG.$SCHEMA${NC}"

SCHEMA_CHECK=$(curl -s -X POST "http://localhost:8080/v1/statement" \
    -H "X-Trino-User: $TRINO_USER" \
    -H "X-Trino-Catalog: $CATALOG" \
    -d "SHOW SCHEMAS LIKE '$SCHEMA'" | jq -r '.data[][]? // empty' 2>/dev/null || echo "")

if [[ -z "$SCHEMA_CHECK" ]]; then
    echo -e "${YELLOW}⚠ Schema $SCHEMA does not exist (may be created by workflows)${NC}"
else
    echo -e "${GREEN}✓ Schema exists: $SCHEMA${NC}"
fi

# List tables
echo ""
echo -e "${BLUE}Listing tables in $SCHEMA...${NC}"

TABLES=$(curl -s -X POST "http://localhost:8080/v1/statement" \
    -H "X-Trino-User: $TRINO_USER" \
    -H "X-Trino-Catalog: $CATALOG" \
    -H "X-Trino-Schema: $SCHEMA" \
    -d "SHOW TABLES" | jq -r '.data[][]? // empty' 2>/dev/null || echo "")

if [[ -z "$TABLES" ]]; then
    echo -e "${YELLOW}⚠ No tables found (workflows may not have run yet)${NC}"
    echo ""
    echo -e "${BLUE}Expected tables from workflows:${NC}"
    echo "  - energy_prices (EIA)"
    echo "  - economic_indicators (FRED)"
    echo "  - commodity_futures (AlphaVantage)"
    echo "  - market_data (Polygon.io)"
    echo "  - gas_storage (GIE)"
    echo "  - census_data (US Census)"
    echo ""
    echo -e "${YELLOW}Run workflows first: ./scripts/test-dolphinscheduler-workflows.sh${NC}"
    exit 0
fi

echo -e "${GREEN}✓ Found tables:${NC}"
echo "$TABLES" | sed 's/^/  - /'

# Verify data in each table
echo ""
echo -e "${BLUE}Verifying data in tables...${NC}"
echo ""

REPORT=""
TOTAL_TABLES=0
TABLES_WITH_DATA=0
EMPTY_TABLES=0
TOTAL_RECORDS=0

for TABLE in $TABLES; do
    ((TOTAL_TABLES++))
    
    # Get record count
    COUNT_RESULT=$(curl -s -X POST "http://localhost:8080/v1/statement" \
        -H "X-Trino-User: $TRINO_USER" \
        -H "X-Trino-Catalog: $CATALOG" \
        -H "X-Trino-Schema: $SCHEMA" \
        -d "SELECT COUNT(*) FROM $TABLE" | jq -r '.data[][]? // 0' 2>/dev/null || echo "0")
    
    # Try to get date range (if date column exists)
    DATE_RANGE=$(curl -s -X POST "http://localhost:8080/v1/statement" \
        -H "X-Trino-User: $TRINO_USER" \
        -H "X-Trino-Catalog: $CATALOG" \
        -H "X-Trino-Schema: $SCHEMA" \
        -d "SELECT MIN(date) as min_date, MAX(date) as max_date FROM $TABLE" 2>/dev/null | jq -r '.data[]? | "\(.[0]) to \(.[1])"' 2>/dev/null || echo "N/A")
    
    if [[ "$COUNT_RESULT" -gt 0 ]]; then
        ((TABLES_WITH_DATA++))
        TOTAL_RECORDS=$((TOTAL_RECORDS + COUNT_RESULT))
        echo -e "${GREEN}✓ $TABLE${NC}"
        echo "    Records: $COUNT_RESULT"
        if [[ "$DATE_RANGE" != "N/A" ]] && [[ "$DATE_RANGE" != "null to null" ]]; then
            echo "    Date range: $DATE_RANGE"
        fi
        REPORT="$REPORT\n  ✓ $TABLE: $COUNT_RESULT records"
    else
        ((EMPTY_TABLES++))
        echo -e "${YELLOW}⚠ $TABLE (empty)${NC}"
        REPORT="$REPORT\n  ⚠ $TABLE: EMPTY"
    fi
    echo ""
done

# Data freshness check
echo -e "${BLUE}Checking data freshness...${NC}"
echo ""

STALE_TABLES=0
FRESH_TABLES=0

for TABLE in $TABLES; do
    # Try to find the latest timestamp
    LATEST=$(curl -s -X POST "http://localhost:8080/v1/statement" \
        -H "X-Trino-User: $TRINO_USER" \
        -H "X-Trino-Catalog: $CATALOG" \
        -H "X-Trino-Schema: $SCHEMA" \
        -d "SELECT MAX(date) FROM $TABLE" 2>/dev/null | jq -r '.data[][]? // empty' 2>/dev/null || echo "")
    
    if [[ -n "$LATEST" ]] && [[ "$LATEST" != "null" ]]; then
        LATEST_TS=$(date -d "$LATEST" +%s 2>/dev/null || echo "0")
        NOW=$(date +%s)
        AGE_HOURS=$(( (NOW - LATEST_TS) / 3600 ))
        
        if [[ $AGE_HOURS -lt 48 ]]; then
            echo -e "${GREEN}✓ $TABLE: Latest data is ${AGE_HOURS}h old${NC}"
            ((FRESH_TABLES++))
        else
            echo -e "${YELLOW}⚠ $TABLE: Latest data is ${AGE_HOURS}h old (>48h)${NC}"
            ((STALE_TABLES++))
        fi
    else
        echo -e "${YELLOW}⚠ $TABLE: Cannot determine data age${NC}"
    fi
done

# Final summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Verification Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Tables:"
echo "  - Total: $TOTAL_TABLES"
echo "  - With data: $TABLES_WITH_DATA"
echo "  - Empty: $EMPTY_TABLES"
echo ""
echo "Data:"
echo "  - Total records: $TOTAL_RECORDS"
echo "  - Fresh tables (<48h): $FRESH_TABLES"
if [[ $STALE_TABLES -gt 0 ]]; then
    echo "  - Stale tables (>48h): $STALE_TABLES"
fi
echo ""

if [[ $TABLES_WITH_DATA -gt 0 ]] && [[ $EMPTY_TABLES -eq 0 ]]; then
    echo -e "${GREEN}✓ All tables have data${NC}"
    STATUS="SUCCESS"
elif [[ $TABLES_WITH_DATA -gt 0 ]]; then
    echo -e "${YELLOW}⚠ Some tables are empty${NC}"
    STATUS="PARTIAL"
else
    echo -e "${RED}✗ No data found in any table${NC}"
    STATUS="FAILED"
fi

echo ""
echo "Catalog: $CATALOG"
echo "Schema: $SCHEMA"
echo ""

if [[ "$STATUS" == "SUCCESS" ]]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Data Verification: PASSED ✓${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "  1. Enable workflow schedules in DolphinScheduler"
    echo "  2. Monitor data quality in Grafana dashboards"
    echo "  3. Set up alerts for data freshness"
    exit 0
elif [[ "$STATUS" == "PARTIAL" ]]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}  Data Verification: PARTIAL ⚠${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Some workflows may need to run again${NC}"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Data Verification: FAILED ✗${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${RED}No data found. Check workflow execution logs.${NC}"
    exit 1
fi

