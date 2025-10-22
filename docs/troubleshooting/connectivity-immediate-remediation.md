# Immediate Remediation Steps - Connectivity Timeout Issue

## Quick Start (Choose ONE option below)

### Option 1: Quick Fix - Restart Kube-Proxy (30 seconds)
```bash
cd /home/m/tff/254CARBON/HMCo

# This may resolve transient networking issues
kubectl rollout restart ds/kube-proxy -n kube-system

# Wait for it to restart
kubectl wait --for=condition=ready pod -l component=kube-proxy -n kube-system --timeout=60s

# Test connectivity
kubectl run -it --rm test-pod \
  --image=curlimages/curl \
  --restart=Never \
  -n data-platform \
  -- curl -v http://iceberg-rest-catalog:8181/v1/config 2>&1 | head -20
```

### Option 2: Medium Fix - Restart Kubelet (2 minutes)
```bash
cd /home/m/tff/254CARBON/HMCo

# Restart kubelet on the control plane
docker exec dev-cluster-control-plane systemctl restart kubelet

# Wait for node to be ready
kubectl wait --for=condition=Ready node/dev-cluster-control-plane --timeout=120s

# Test connectivity
kubectl run -it --rm test-pod \
  --image=curlimages/curl \
  --restart=Never \
  -n data-platform \
  -- curl -v http://iceberg-rest-catalog:8181/v1/config 2>&1 | head -20
```

### Option 3: Full Remediation - Recreate Kind Cluster (10 minutes)
**⚠️ WARNING: This will delete the cluster. Export any important data first!**

```bash
cd /home/m/tff/254CARBON/HMCo

# Step 1: Backup current configuration (OPTIONAL)
mkdir -p /tmp/254carbon-backup
kubectl get all --all-namespaces -o yaml > /tmp/254carbon-backup/all-resources.yaml
kubectl get pvc --all-namespaces -o yaml > /tmp/254carbon-backup/pvcs.yaml
kubectl get secrets --all-namespaces -o yaml > /tmp/254carbon-backup/secrets.yaml

# Step 2: Delete the problematic cluster
kind delete cluster --name dev-cluster

# Step 3: Verify cluster is deleted
docker ps | grep dev-cluster || echo "Cluster successfully deleted"

# Step 4: Create new cluster with optimized networking
kind create cluster --name dev-cluster \
  --image kindest/node:v1.31.0 \
  --config - <<'EOF'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
metadata:
  name: dev-cluster
networking:
  apiServerPort: 6443
  podSubnet: "10.244.0.0/16"
  serviceSubnet: "10.96.0.0/12"
  disableDefaultCNI: false
  kubeProxyMode: "iptables"
EOF

# Step 5: Verify new cluster
kubectl get nodes
kubectl get pods --all-namespaces

# Step 6: Re-deploy 254Carbon platform
# Run your deployment scripts here
```

## Verification Tests

After applying any of the above solutions, run these tests:

```bash
# Test 1: Service DNS Resolution
echo "Test 1: DNS Resolution"
kubectl run -it --rm test-dns \
  --image=busybox \
  --restart=Never \
  -n data-platform \
  -- nslookup iceberg-rest-catalog

# Test 2: Service Connectivity (cluster IP)
echo "Test 2: Service Cluster IP Connectivity"
kubectl run -it --rm test-svc \
  --image=curlimages/curl \
  --restart=Never \
  -n data-platform \
  -- timeout 10 curl -v http://iceberg-rest-catalog:8181/v1/config

# Test 3: Direct Pod Connectivity  
echo "Test 3: Direct Pod IP Connectivity"
POD_IP=$(kubectl get pod -n data-platform -l app=iceberg-rest-catalog -o jsonpath='{.items[0].status.podIP}')
kubectl run -it --rm test-pod \
  --image=curlimages/curl \
  --restart=Never \
  -n data-platform \
  -- timeout 10 curl -v http://$POD_IP:8181/v1/config

# Test 4: Cross-namespace Communication
echo "Test 4: Cross-namespace Communication"
kubectl run -it --rm test-cross \
  --image=curlimages/curl \
  --restart=Never \
  -n data-platform \
  -- timeout 10 curl -v http://prometheus-operator-kube-p-prometheus.monitoring:9090/api/v1/query?query=up
```

## Troubleshooting Script

Run the comprehensive troubleshooting script:

```bash
cd /home/m/tff/254CARBON/HMCo
bash scripts/troubleshoot-connectivity.sh
```

This will:
- Check cluster status
- Verify DNS resolution
- Check network policies
- Test pod-to-pod connectivity
- Provide recommendations

## If None of the Above Works

The issue may be environmental or infrastructure-related:

1. **Check Docker Resources**:
   ```bash
   docker stats --no-stream
   docker system df
   ```

2. **Check Host Network**:
   ```bash
   ifconfig | grep -A 10 "kind"
   docker network inspect kind
   ```

3. **Check for Known Issues**:
   - Docker Bridge MTU mismatch
   - Windows Hyper-V network configuration issues
   - macOS Docker Desktop resource limits
   - Linux firewall rules (ufw, firewalld)

4. **Nuclear Option - Full Clean**:
   ```bash
   # Clean up everything
   kind delete cluster --name dev-cluster
   docker system prune -a --volumes
   
   # Reinstall
   kind create cluster --name dev-cluster
   ```

## Success Criteria

You'll know the issue is fixed when:

✅ `kubectl run ... -- curl http://service-name:port` responds within 1-2 seconds
✅ Service DNS resolution returns quickly
✅ Pod logs show successful service connections
✅ No "connection timed out" errors in curl output

## Related Documentation

- **Detailed Diagnosis**: See `CONNECTIVITY_TIMEOUT_DIAGNOSIS.md`
- **Network Policies**: Check `k8s/network/network-policies.yaml`
- **Platform Overview**: See main `README.md`

---

**Need help?** Review the comprehensive diagnostic document or contact your infrastructure team with the output of `troubleshoot-connectivity.sh`.
