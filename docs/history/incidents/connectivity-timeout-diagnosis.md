# 254Carbon Connectivity Timeout Diagnosis & Resolution

## Problem Summary

**Symptoms:**
- Services appear to be running (pods are in Running state)
- Service DNS resolution works correctly
- Direct connectivity to service cluster IPs times out after ~130 seconds
- Direct pod-to-pod connectivity times out
- TCP connections hang before establishing

**Root Cause:**
This is a Kind cluster (Kubernetes in Docker) networking issue where:
1. The cluster is running inside Docker containers on the `kind` bridge network
2. There's a blocking network interface or routing issue preventing TCP packets from reaching pods
3. The network connectivity appears to be established at Layer 3 (DNS works) but fails at Layer 4 (TCP)

## Diagnostic Findings

### ✅ What Works:
- Service DNS resolution (FQDN to cluster IP works)
- Service discovery mechanism
- Service endpoints are properly registered
- Pods are running and healthy
- Container startup and initialization

### ❌ What Doesn't Work:
- TCP connections to service cluster IPs (10.96.x.x range)
- TCP connections to pod IPs (10.244.x.x range)
- HTTP requests to internal services
- Service-to-service communication

### Network Information:
- Control Plane Node IP: `172.19.0.2` (on `kind` Docker network)
- Service CIDR: `10.96.0.0/12`
- Pod CIDR: `10.244.0.0/16`
- Kube-proxy mode: `iptables`
- CNI Plugin: (not visible, likely built-in Kind networking)

## Root Causes Analysis

### 1. Kind Network Bridge Configuration
The Kind cluster is running inside Docker, which has inherent networking constraints:
- Docker bridge networking doesn't properly support overlay networks
- veth interface configuration may be preventing packet forwarding
- iptables rules may not be properly configured for inter-pod communication

### 2. Possible Kernel/Network Stack Issues
- Docker resource constraints limiting network buffers
- MTU mismatches between layers
- UDP checksum offloading disabled on veth interfaces
- Missing kernel modules for proper bridging

### 3. Container Runtime Configuration
- containerd (version 1.7.18) may have connection pooling limits
- CNI networking plugin may not be properly initialized
- Network namespace isolation issues

## Solutions

### Solution 1: Restart Kind Cluster (Recommended)
```bash
# Stop the current cluster
kind delete cluster --name dev-cluster

# Recreate with optimized networking
kind create cluster --name dev-cluster \
  --image kindest/node:v1.31.0 \
  --config - <<EOF
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
containerd:
  configOverride: |
    [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
    runtime_engine = ""
    runtime_root = ""
    runtime_type = "io.containerd.runc.v2"
    [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
    BinaryName = "runc"
EOF
```

### Solution 2: Troubleshoot Without Restarting

#### Step 1: Check CNI Plugin Status
```bash
kubectl get daemonsets --all-namespaces -o wide
kubectl get nodes -o wide
kubectl describe node dev-cluster-control-plane
```

#### Step 2: Verify iptables Rules
```bash
# From host machine
docker exec dev-cluster-control-plane iptables-save | grep -E "(FORWARD|INPUT|OUTPUT)" | head -20
```

#### Step 3: Check coredns Service
```bash
kubectl get svc -n kube-system
kubectl get pods -n kube-system -l k8s-app=kube-dns
kubectl logs -n kube-system -l k8s-app=kube-dns
```

#### Step 4: Restart Kind Networking Components
```bash
# Restart kubelet
docker exec dev-cluster-control-plane systemctl restart kubelet

# Restart kube-proxy
kubectl rollout restart ds/kube-proxy -n kube-system

# Wait for pods to restart
kubectl wait --for=condition=ready pod -l component=kube-proxy -n kube-system --timeout=300s
```

### Solution 3: Docker Desktop Networking Optimization
If using Docker Desktop, check these settings:
1. **Memory**: Ensure sufficient RAM allocated (minimum 8GB recommended)
2. **CPUs**: Allocate at least 4 cores
3. **Disk**: Sufficient disk space for image layers
4. **Network**: Use "VPN-Friendly" or "Host" networking mode if available

### Solution 4: Network Policy Remediation
```bash
# The default-deny-ingress policy is enforced, but allow rules exist
# However, they may not be properly functioning. Verify:
kubectl get networkpolicies -n data-platform
kubectl describe networkpolicy default-deny-ingress -n data-platform
kubectl describe networkpolicy allow-data-platform-internal -n data-platform

# If rules are correct but not working, try:
# 1. Temporarily disable network policies to test
kubectl delete networkpolicy default-deny-ingress -n data-platform

# 2. Apply permissive rules
kubectl apply -f - <<'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-all-internal
  namespace: data-platform
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector: {}
EOF

# 3. Test connectivity
# 4. If it works, the issue was network policies
# 5. Re-apply with specific rules
```

## Recommended Action Plan

### For Production/Important Environments:
1. **Backup**: Export all configurations and data
2. **Recreate Cluster**: Use `kind delete` and `kind create` with optimized config
3. **Restore**: Re-apply all configurations
4. **Verify**: Run full connectivity test suite

### For Development/Lab Environments:
1. Try Solution 2 (Troubleshooting steps) first
2. If that doesn't work, proceed with Solution 1 (Restart)
3. If restarting doesn't help, consider a clean Kind install

## Testing Connectivity After Fix

```bash
# Create a test pod with curl
kubectl run -it --rm debug-pod \
  --image=curlimages/curl \
  --restart=Never \
  -n data-platform \
  -- curl -v http://iceberg-rest-catalog:8181/v1/config

# Test service DNS
kubectl run -it --rm debug-pod \
  --image=busybox \
  --restart=Never \
  -n data-platform \
  -- nslookup kubernetes.default

# Test external connectivity
kubectl run -it --rm debug-pod \
  --image=curlimages/curl \
  --restart=Never \
  -n data-platform \
  -- curl -v http://www.google.com
```

## Prevention & Best Practices

1. **Use Production-Grade Kubernetes**:
   - Consider EKS, GKE, or AKS for production workloads
   - Kind is designed for local development/testing only

2. **Network Monitoring**:
   - Deploy network policies explicitly
   - Monitor pod-to-pod communication
   - Set up network performance baselines

3. **Infrastructure Health Checks**:
   - Regular connectivity verification
   - Network plugin health monitoring
   - Docker resource utilization checks

4. **Documentation**:
   - Keep network topology diagrams current
   - Document cluster creation procedures
   - Maintain runbooks for common issues

## Next Steps

Choose one of the solutions above based on your environment. The **Recommended Action** is to use **Solution 1** (restart Kind cluster) with optimized networking configuration, which has the highest success rate for resolving deep networking issues.

If you need help implementing any of these solutions, please ask!
