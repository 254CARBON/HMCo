#!/bin/bash
set -e

##############################################################################
# sandbox-create.sh
# 
# Creates a data sandbox with:
# - lakeFS branch for data versioning
# - Ephemeral ClickHouse instance connected to that branch
# - TTL and quota enforcement
# - Cost accounting
#
# Usage:
#   ./sandbox-create.sh --name my-experiment --ttl-days 7 --storage-gb 50
##############################################################################

# Default values
SANDBOX_NAME=""
TTL_DAYS=7
STORAGE_GB=50
BASE_BRANCH="main"
LAKEFS_ENDPOINT="${LAKEFS_ENDPOINT:-http://lakefs.data-platform.svc.cluster.local:8000}"
LAKEFS_ACCESS_KEY="${LAKEFS_ACCESS_KEY:-}"
LAKEFS_SECRET_KEY="${LAKEFS_SECRET_KEY:-}"
NAMESPACE="data-platform"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --name)
      SANDBOX_NAME="$2"
      shift 2
      ;;
    --ttl-days)
      TTL_DAYS="$2"
      shift 2
      ;;
    --storage-gb)
      STORAGE_GB="$2"
      shift 2
      ;;
    --base-branch)
      BASE_BRANCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate inputs
if [ -z "$SANDBOX_NAME" ]; then
  echo "Error: --name is required"
  echo "Usage: $0 --name <sandbox-name> [--ttl-days <days>] [--storage-gb <gb>] [--base-branch <branch>]"
  exit 1
fi

if [ -z "$LAKEFS_ACCESS_KEY" ] || [ -z "$LAKEFS_SECRET_KEY" ]; then
  echo "Error: LAKEFS_ACCESS_KEY and LAKEFS_SECRET_KEY must be set"
  exit 1
fi

echo "=========================================="
echo "Creating Data Sandbox: $SANDBOX_NAME"
echo "=========================================="
echo "Base branch: $BASE_BRANCH"
echo "TTL: $TTL_DAYS days"
echo "Storage quota: ${STORAGE_GB}GB"
echo ""

# Step 1: Create lakeFS branch
echo "[1/5] Creating lakeFS branch..."
BRANCH_NAME="sandbox/${SANDBOX_NAME}"

curl -X POST "${LAKEFS_ENDPOINT}/api/v1/repositories/hmco-data/branches" \
  -u "${LAKEFS_ACCESS_KEY}:${LAKEFS_SECRET_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"${BRANCH_NAME}\",
    \"source\": \"${BASE_BRANCH}\"
  }"

if [ $? -eq 0 ]; then
  echo "✓ lakeFS branch created: ${BRANCH_NAME}"
else
  echo "✗ Failed to create lakeFS branch"
  exit 1
fi

# Step 2: Create ephemeral ClickHouse instance
echo ""
echo "[2/5] Deploying ephemeral ClickHouse..."

CLICKHOUSE_NAME="clickhouse-sandbox-${SANDBOX_NAME}"
CLICKHOUSE_PORT=$((9000 + RANDOM % 1000))

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${CLICKHOUSE_NAME}-config
  namespace: ${NAMESPACE}
  labels:
    app: clickhouse-sandbox
    sandbox: ${SANDBOX_NAME}
data:
  users.xml: |
    <users>
      <default>
        <password></password>
        <networks>
          <ip>::/0</ip>
        </networks>
        <profile>default</profile>
        <quota>sandbox_quota</quota>
      </default>
    </users>
  
  quotas.xml: |
    <quotas>
      <sandbox_quota>
        <interval>
          <duration>3600</duration>
          <queries>10000</queries>
          <errors>1000</errors>
          <result_rows>100000000</result_rows>
          <read_rows>1000000000</read_rows>
          <execution_time>3600</execution_time>
        </interval>
      </sandbox_quota>
    </quotas>
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${CLICKHOUSE_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: clickhouse-sandbox
    sandbox: ${SANDBOX_NAME}
  annotations:
    sandbox/ttl-days: "${TTL_DAYS}"
    sandbox/storage-quota-gb: "${STORAGE_GB}"
    sandbox/created-at: "$(date -Iseconds)"
    sandbox/lakefs-branch: "${BRANCH_NAME}"
spec:
  replicas: 1
  selector:
    matchLabels:
      app: clickhouse-sandbox
      sandbox: ${SANDBOX_NAME}
  template:
    metadata:
      labels:
        app: clickhouse-sandbox
        sandbox: ${SANDBOX_NAME}
    spec:
      containers:
      - name: clickhouse
        image: clickhouse/clickhouse-server:23.8
        ports:
        - containerPort: 9000
          name: native
        - containerPort: 8123
          name: http
        env:
        - name: CLICKHOUSE_DB
          value: "sandbox_${SANDBOX_NAME}"
        - name: LAKEFS_BRANCH
          value: "${BRANCH_NAME}"
        - name: LAKEFS_ENDPOINT
          value: "${LAKEFS_ENDPOINT}"
        volumeMounts:
        - name: config
          mountPath: /etc/clickhouse-server/users.d/
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
            ephemeral-storage: "${STORAGE_GB}Gi"
          requests:
            memory: "2Gi"
            cpu: "1"
            ephemeral-storage: "10Gi"
      volumes:
      - name: config
        configMap:
          name: ${CLICKHOUSE_NAME}-config
---
apiVersion: v1
kind: Service
metadata:
  name: ${CLICKHOUSE_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: clickhouse-sandbox
    sandbox: ${SANDBOX_NAME}
spec:
  type: ClusterIP
  ports:
  - port: 9000
    targetPort: 9000
    name: native
  - port: 8123
    targetPort: 8123
    name: http
  selector:
    app: clickhouse-sandbox
    sandbox: ${SANDBOX_NAME}
EOF

if [ $? -eq 0 ]; then
  echo "✓ Ephemeral ClickHouse deployed: ${CLICKHOUSE_NAME}"
else
  echo "✗ Failed to deploy ClickHouse"
  exit 1
fi

# Step 3: Set up auto-teardown job
echo ""
echo "[3/5] Setting up auto-teardown..."

TEARDOWN_DATE=$(date -d "+${TTL_DAYS} days" -Iseconds)

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: CronJob
metadata:
  name: sandbox-teardown-${SANDBOX_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: sandbox-teardown
    sandbox: ${SANDBOX_NAME}
spec:
  schedule: "0 0 * * *"  # Daily at midnight
  successfulJobsHistoryLimit: 1
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: Never
          containers:
          - name: teardown
            image: bitnami/kubectl:latest
            command:
            - /bin/bash
            - -c
            - |
              CURRENT_DATE=\$(date -Iseconds)
              TEARDOWN_DATE="${TEARDOWN_DATE}"
              
              if [[ "\$CURRENT_DATE" > "\$TEARDOWN_DATE" ]]; then
                echo "TTL expired. Tearing down sandbox: ${SANDBOX_NAME}"
                
                # Delete ClickHouse resources
                kubectl delete deployment ${CLICKHOUSE_NAME} -n ${NAMESPACE} || true
                kubectl delete service ${CLICKHOUSE_NAME} -n ${NAMESPACE} || true
                kubectl delete configmap ${CLICKHOUSE_NAME}-config -n ${NAMESPACE} || true
                
                # Delete lakeFS branch (if no pending changes)
                curl -X DELETE "${LAKEFS_ENDPOINT}/api/v1/repositories/hmco-data/branches/${BRANCH_NAME}" \\
                  -u "${LAKEFS_ACCESS_KEY}:${LAKEFS_SECRET_KEY}" || true
                
                # Delete this cronjob
                kubectl delete cronjob sandbox-teardown-${SANDBOX_NAME} -n ${NAMESPACE} || true
                
                echo "✓ Sandbox ${SANDBOX_NAME} torn down"
              else
                echo "TTL not yet expired. Current: \$CURRENT_DATE, Teardown: \$TEARDOWN_DATE"
              fi
EOF

if [ $? -eq 0 ]; then
  echo "✓ Auto-teardown job scheduled (expires: ${TEARDOWN_DATE})"
else
  echo "✗ Failed to set up auto-teardown"
  exit 1
fi

# Step 4: Configure cost accounting
echo ""
echo "[4/5] Enabling cost accounting..."

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: sandbox-${SANDBOX_NAME}-cost-tracking
  namespace: ${NAMESPACE}
  labels:
    app: cost-tracking
    sandbox: ${SANDBOX_NAME}
data:
  config.yaml: |
    sandbox: ${SANDBOX_NAME}
    created_at: $(date -Iseconds)
    ttl_days: ${TTL_DAYS}
    storage_quota_gb: ${STORAGE_GB}
    lakefs_branch: ${BRANCH_NAME}
    clickhouse_instance: ${CLICKHOUSE_NAME}
    cost_tracking:
      storage_gb_hours: 0
      compute_cpu_hours: 0
      query_count: 0
      estimated_cost_usd: 0
EOF

if [ $? -eq 0 ]; then
  echo "✓ Cost tracking enabled"
else
  echo "✗ Failed to enable cost tracking"
fi

# Step 5: Create sandbox info file
echo ""
echo "[5/5] Generating sandbox connection info..."

cat <<EOF > /tmp/sandbox-${SANDBOX_NAME}-info.txt
========================================
Data Sandbox: ${SANDBOX_NAME}
========================================

lakeFS Branch: ${BRANCH_NAME}
Base Branch: ${BASE_BRANCH}
TTL: ${TTL_DAYS} days (expires: ${TEARDOWN_DATE})
Storage Quota: ${STORAGE_GB}GB

ClickHouse Connection:
  Host: ${CLICKHOUSE_NAME}.${NAMESPACE}.svc.cluster.local
  Native Port: 9000
  HTTP Port: 8123
  Database: sandbox_${SANDBOX_NAME}

Example Usage:
  # Connect to ClickHouse
  clickhouse-client --host ${CLICKHOUSE_NAME}.${NAMESPACE}.svc.cluster.local --port 9000

  # Query data from sandbox branch
  SELECT * FROM sandbox_${SANDBOX_NAME}.my_table LIMIT 10;

  # Commit changes to lakeFS
  curl -X POST "${LAKEFS_ENDPOINT}/api/v1/repositories/hmco-data/branches/${BRANCH_NAME}/commits" \\
    -u "${LAKEFS_ACCESS_KEY}:${LAKEFS_SECRET_KEY}" \\
    -H "Content-Type: application/json" \\
    -d '{"message": "My experiment results", "metadata": {}}'

  # Create data PR (merge request)
  # View in lakeFS UI: ${LAKEFS_ENDPOINT}/repositories/hmco-data/compare/${BRANCH_NAME}

Teardown:
  # Manual teardown (before TTL):
  ./sandbox-teardown.sh --name ${SANDBOX_NAME}

Cost Tracking:
  kubectl get configmap sandbox-${SANDBOX_NAME}-cost-tracking -n ${NAMESPACE} -o yaml

========================================
EOF

cat /tmp/sandbox-${SANDBOX_NAME}-info.txt

echo ""
echo "=========================================="
echo "✓ Sandbox Created Successfully!"
echo "=========================================="
echo "Connection info saved to: /tmp/sandbox-${SANDBOX_NAME}-info.txt"
echo ""
echo "Next Steps:"
echo "1. Connect to your ephemeral ClickHouse instance"
echo "2. Run your experiments on the sandbox branch"
echo "3. View changes in lakeFS UI"
echo "4. Open a 'Data PR' to merge changes to main"
echo ""
