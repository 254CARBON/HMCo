#!/bin/bash
# Fix ImagePullBackOff issues for Iceberg deployment

set -e

NAMESPACE="data-platform"
DEPLOYMENT="iceberg-rest-catalog"

echo "==================================="
echo "Fixing ImagePullBackOff Issues"
echo "==================================="
echo ""

# Function to try different image registries
try_image_registry() {
  local registry=$1
  local image=$2
  
  echo "Attempting to use image from: $registry"
  
  if kubectl set image deployment/$DEPLOYMENT \
    -n $NAMESPACE \
    iceberg-rest-catalog=$registry/$image:0.6.0 \
    --record 2>/dev/null; then
    
    echo "✓ Image updated to $registry/$image"
    
    # Restart deployment
    kubectl rollout restart deployment/$DEPLOYMENT -n $NAMESPACE
    
    # Wait for rollout
    echo "Waiting for deployment to be ready..."
    if kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=5m; then
      echo "✓ Deployment successful!"
      return 0
    fi
  fi
  
  return 1
}

# Check current pod status
echo "Current pod status:"
kubectl get pod -n $NAMESPACE -l app=$DEPLOYMENT
echo ""

# Try different registries in order of preference
echo "Attempting to fix image pull..."
echo ""

# Option 1: Try Quay.io (often has better availability)
if try_image_registry "quay.io" "tabulario/iceberg-rest"; then
  exit 0
fi

# Option 2: Try original registry with wait
echo ""
echo "Trying original registry (Docker Hub)..."
kubectl set image deployment/$DEPLOYMENT \
  -n $NAMESPACE \
  iceberg-rest-catalog=tabulario/iceberg-rest:0.6.0 \
  --record

echo ""
echo "Pod will retry image pull automatically."
echo "Monitoring pod status..."
echo ""

# Watch pod status
kubectl get pod -n $NAMESPACE -l app=$DEPLOYMENT --watch &
WATCH_PID=$!

# Wait up to 10 minutes for pod to become ready
echo "Waiting up to 10 minutes for pod to become Running..."
timeout=0
max_timeout=600

while [ $timeout -lt $max_timeout ]; do
  status=$(kubectl get pod -n $NAMESPACE -l app=$DEPLOYMENT -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")
  
  if [ "$status" = "Running" ]; then
    echo ""
    echo "✓ Pod is now Running!"
    kill $WATCH_PID 2>/dev/null || true
    break
  fi
  
  sleep 10
  timeout=$((timeout + 10))
done

if [ $timeout -ge $max_timeout ]; then
  echo ""
  echo "⚠ Pod did not start within timeout. Checking pod details..."
  kill $WATCH_PID 2>/dev/null || true
  echo ""
  kubectl describe pod -n $NAMESPACE -l app=$DEPLOYMENT
  exit 1
fi

echo ""
echo "Verifying API connectivity..."
echo ""

# Port-forward and test
kubectl port-forward -n $NAMESPACE svc/$DEPLOYMENT 8181:8181 &
PF_PID=$!
sleep 2

if curl -s http://localhost:8181/v1/config > /dev/null; then
  echo "✓ Iceberg REST Catalog API is responding!"
  kill $PF_PID 2>/dev/null || true
  exit 0
else
  echo "✗ API is not responding. Checking logs..."
  kill $PF_PID 2>/dev/null || true
  kubectl logs -n $NAMESPACE -l app=$DEPLOYMENT --tail=20
  exit 1
fi
