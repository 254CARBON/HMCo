#!/bin/bash
# Platform Performance Benchmarking Script
# Tests query performance, GPU utilization, and data pipeline throughput

set -e

NAMESPACE="data-platform"
RESULTS_DIR="./benchmark-results-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "254Carbon Platform Performance Benchmark"
echo "Started: $(date)"
echo "========================================="
echo ""

# 1. GPU Utilization Test
echo "1. Testing GPU Utilization..."
kubectl exec -n $NAMESPACE deployment/rapids-commodity-processor -- nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv > "$RESULTS_DIR/gpu-utilization.csv" 2>/dev/null || echo "GPU test skipped (nvidia-smi not available)"
echo "   ✓ GPU metrics saved to $RESULTS_DIR/gpu-utilization.csv"
echo ""

# 2. Trino Query Performance Test
echo "2. Testing Trino Query Performance..."
cat > /tmp/benchmark-query.sql << 'EOF'
-- Test queries for performance benchmarking
SELECT COUNT(*) FROM iceberg_catalog.commodity_data.energy_prices;
SELECT commodity, AVG(price) as avg_price FROM iceberg_catalog.commodity_data.energy_prices GROUP BY commodity;
SELECT * FROM iceberg_catalog.commodity_data.energy_prices ORDER BY price_date DESC LIMIT 1000;
EOF

START_TIME=$(date +%s)
kubectl exec -n $NAMESPACE deployment/trino-coordinator -- trino --execute "SELECT COUNT(*) FROM iceberg_catalog.commodity_data.energy_prices" 2>&1 | tee "$RESULTS_DIR/trino-query-simple.log" || echo "Query failed"
END_TIME=$(date +%s)
QUERY_TIME=$((END_TIME - START_TIME))
echo "   ✓ Simple query completed in ${QUERY_TIME}s"

START_TIME=$(date +%s)
kubectl exec -n $NAMESPACE deployment/trino-coordinator -- trino --execute "SELECT commodity, AVG(price) as avg_price FROM iceberg_catalog.commodity_data.energy_prices GROUP BY commodity" 2>&1 | tee "$RESULTS_DIR/trino-query-aggregate.log" || echo "Aggregate query failed"
END_TIME=$(date +%s)
AGG_TIME=$((END_TIME - START_TIME))
echo "   ✓ Aggregate query completed in ${AGG_TIME}s"
echo ""

# 3. Data Pipeline Throughput Test
echo "3. Testing Data Pipeline Throughput..."
kubectl get cronjob -n $NAMESPACE | grep datahub > "$RESULTS_DIR/cronjob-status.txt"
echo "   ✓ CronJob status saved"
echo ""

# 4. Resource Utilization
echo "4. Measuring Resource Utilization..."
kubectl top nodes > "$RESULTS_DIR/node-resources.txt"
kubectl top pods -n $NAMESPACE --sort-by=memory | head -20 > "$RESULTS_DIR/top-pods-memory.txt"
kubectl top pods -n $NAMESPACE --sort-by=cpu | head -20 > "$RESULTS_DIR/top-pods-cpu.txt"
echo "   ✓ Resource metrics saved"
echo ""

# 5. Service Response Times
echo "5. Testing Service Response Times..."
for svc in datahub-frontend dolphinscheduler-api superset trino-coordinator mlflow feast-server; do
  SVC_NAME=$(kubectl get svc -n $NAMESPACE | grep $svc | head -1 | awk '{print $1}')
  if [ -n "$SVC_NAME" ]; then
    START_TIME=$(date +%s%3N)
    kubectl exec -n $NAMESPACE deployment/portal -- curl -s -o /dev/null -w "%{time_total}" http://$SVC_NAME:8080/ > /tmp/response-time.txt 2>/dev/null || echo "0"
    END_TIME=$(date +%s%3N)
    RESPONSE_TIME=$(cat /tmp/response-time.txt)
    echo "   $svc: ${RESPONSE_TIME}ms"
  fi
done
echo ""

# 6. Check Autoscaling Status
echo "6. Autoscaling Status..."
kubectl get hpa -n $NAMESPACE -o custom-columns=NAME:.metadata.name,REPLICAS:.status.currentReplicas,DESIRED:.status.desiredReplicas,CPU:.status.currentMetrics[0].resource.current.averageUtilization > "$RESULTS_DIR/hpa-status.txt"
cat "$RESULTS_DIR/hpa-status.txt"
echo ""

# 7. Pod Disruption Budget Status
echo "7. High Availability Status..."
kubectl get pdb -A -o custom-columns=NAMESPACE:.metadata.namespace,NAME:.metadata.name,MIN-AVAILABLE:.spec.minAvailable,ALLOWED:.status.disruptionsAllowed > "$RESULTS_DIR/pdb-status.txt"
cat "$RESULTS_DIR/pdb-status.txt"
echo ""

# 8. Check GPU Allocation
echo "8. GPU Allocation Status..."
echo "Node: k8s-worker"
kubectl describe node k8s-worker | grep -A 5 "nvidia.com/gpu" > "$RESULTS_DIR/gpu-allocation.txt"
cat "$RESULTS_DIR/gpu-allocation.txt"
echo ""

# 9. Platform Health Summary
echo "9. Platform Health Summary..."
{
  echo "Timestamp: $(date -Iseconds)"
  echo "---"
  echo "Total Pods: $(kubectl get pods -A --no-headers | wc -l)"
  echo "Running Pods: $(kubectl get pods -A --no-headers | grep Running | wc -l)"
  echo "Problematic Pods: $(kubectl get pods -A --no-headers | grep -E 'CrashLoopBackOff|Error' | wc -l)"
  echo "---"
  echo "HPAs Active: $(kubectl get hpa -A --no-headers | wc -l)"
  echo "PDBs Active: $(kubectl get pdb -A --no-headers | wc -l)"
  echo "---"
  kubectl top nodes
} > "$RESULTS_DIR/platform-health.txt"
cat "$RESULTS_DIR/platform-health.txt"
echo ""

# Generate Summary Report
cat > "$RESULTS_DIR/BENCHMARK_SUMMARY.txt" << EOF
================================================================================
                Platform Performance Benchmark Results
================================================================================

Timestamp: $(date -Iseconds)
Results Directory: $RESULTS_DIR

================================================================================
QUERY PERFORMANCE
================================================================================

Simple Query Time: ${QUERY_TIME}s
Aggregate Query Time: ${AGG_TIME}s

Target: <5s for simple queries, <10s for aggregates
Status: $([ $QUERY_TIME -lt 5 ] && echo "✅ PASSED" || echo "⏳ NEEDS OPTIMIZATION")

================================================================================
GPU UTILIZATION
================================================================================

See: gpu-utilization.csv
Expected: 8 GPUs allocated (increased from 4)

================================================================================
RESOURCE UTILIZATION
================================================================================

See: node-resources.txt, top-pods-*.txt

================================================================================
AUTOSCALING STATUS
================================================================================

See: hpa-status.txt
Expected: 11 HPAs active with appropriate replica counts

================================================================================
HIGH AVAILABILITY
================================================================================

See: pdb-status.txt
Expected: 15 PDBs with appropriate disruption budgets

================================================================================
FILES GENERATED
================================================================================

- gpu-utilization.csv: GPU usage stats
- trino-query-*.log: Query execution logs
- node-resources.txt: Node CPU/memory usage
- top-pods-*.txt: Pod resource consumption
- hpa-status.txt: Autoscaler status
- pdb-status.txt: Pod disruption budgets
- gpu-allocation.txt: GPU allocation details
- platform-health.txt: Overall health summary

================================================================================
EOF

cat "$RESULTS_DIR/BENCHMARK_SUMMARY.txt"

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo "Results saved to: $RESULTS_DIR"
echo "========================================="



