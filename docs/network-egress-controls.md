# Network Egress Controls (T13.1)

## Overview

This document describes the network egress control implementation for the HMCo platform, implementing a default-deny egress policy with explicit allowlists across all critical namespaces.

## Architecture

### Default-Deny Egress Policy

All pods in the following namespaces are subject to default-deny egress policies:

- `data-platform`
- `vault-prod`
- `monitoring`
- `istio-system`
- `cloudflare-tunnel`

Without explicit allow rules, pods cannot initiate outbound connections, providing strong security against:
- Data exfiltration
- Command and control (C2) communications
- Lateral movement attacks
- Supply chain attacks via compromised dependencies

### Explicit Allowlists

#### 1. DNS Resolution (All Namespaces)

**Policy Name:** `allow-dns-egress`

All pods across all 5 namespaces can perform DNS lookups via kube-system DNS service:

- Protocol: UDP/TCP
- Port: 53
- Target: kube-system namespace

```yaml
egress:
- to:
  - namespaceSelector:
      matchLabels:
        kubernetes.io/metadata.name: kube-system
  ports:
  - protocol: UDP
    port: 53
  - protocol: TCP
    port: 53
```

#### 2. Cloudflare Tunnel Egress

**Policy Name:** `allow-cloudflare-egress`
**Namespace:** `cloudflare-tunnel`

Allows Cloudflare tunnel pods to:
- Connect to Cloudflare API endpoints (port 443)
- Maintain tunnel connections (port 7844)
- Route traffic to ingress-nginx

```yaml
podSelector:
  matchLabels:
    app.kubernetes.io/name: cloudflare-tunnel
egress:
- to:
  - namespaceSelector: {}
  ports:
  - protocol: TCP
    port: 443
  - protocol: TCP
    port: 7844
- to:
  - namespaceSelector:
      matchLabels:
        kubernetes.io/metadata.name: ingress-nginx
  ports:
  - protocol: TCP
    port: 80
  - protocol: TCP
    port: 443
```

#### 3. Package Mirror Access (Build Pods Only)

**Policy Names:** 
- `allow-package-mirrors-spark-driver`
- `allow-package-mirrors-spark-executor`

**Namespace:** `data-platform`

Spark driver and executor pods can download packages from public mirrors:

- PyPI (pypi.org)
- NPM (npmjs.com)
- Maven Central (repo1.maven.org)

```yaml
podSelector:
  matchLabels:
    spark-role: driver  # or executor
egress:
- to:
  - namespaceSelector: {}
  ports:
  - protocol: TCP
    port: 443  # HTTPS
  - protocol: TCP
    port: 80   # HTTP fallback
```

### Removed Overly Permissive Policies

The following policies were removed as they bypassed the default-deny egress model:

1. **allow-envoy-interception** - Allowed unrestricted egress to all namespaces
2. **allow-istio-external-egress** - Allowed unrestricted external egress

Individual service policies were also cleaned up to remove redundant DNS rules.

## Deployment

### Prerequisites

- Kubernetes cluster with NetworkPolicy support
- Helm 3.x installed

### Installation

```bash
# Lint the chart
helm lint helm/charts/networking/

# Dry-run to preview changes
helm upgrade --install networking ./helm/charts/networking/ \
  --namespace data-platform \
  --dry-run

# Deploy to cluster
helm upgrade --install networking ./helm/charts/networking/ \
  --namespace data-platform
```

### Verification

Run the validation test script:

```bash
./test-network-policies.sh
```

Or manually verify in a live cluster:

```bash
# 1. Verify default-deny blocks unauthorized egress
kubectl run test-pod -n data-platform --rm -it --image=alpine -- wget https://google.com
# Expected: Connection should timeout/fail

# 2. Verify DNS works
kubectl run test-pod -n data-platform --rm -it --image=alpine -- nslookup kubernetes.default
# Expected: Success

# 3. Verify Cloudflare tunnel can egress
kubectl exec -n cloudflare-tunnel deployment/cloudflared -- curl -I https://api.cloudflare.com
# Expected: Success

# 4. Verify Spark pods can access package mirrors
kubectl run spark-test -n data-platform --rm -it --image=alpine \
  --labels=spark-role=driver -- wget -O- https://pypi.org
# Expected: Success
```

## Security Benefits

### Defense in Depth

1. **Prevent Data Exfiltration:** Compromised pods cannot send data to external endpoints
2. **Block C2 Communications:** Malware cannot establish command and control channels
3. **Limit Blast Radius:** Lateral movement between namespaces is restricted
4. **Supply Chain Protection:** Only build pods can access package repositories

### Compliance

This implementation helps meet compliance requirements for:

- **NIST 800-53:** SC-7 (Boundary Protection)
- **CIS Kubernetes Benchmark:** 5.3.2 (Minimize egress traffic)
- **PCI-DSS:** Requirement 1.3 (Network Segmentation)

## Troubleshooting

### Pod Cannot Resolve DNS

**Symptom:** DNS lookups fail in pods

**Solution:** Verify the `allow-dns-egress` policy is applied:

```bash
kubectl get networkpolicy -n data-platform allow-dns-egress
```

### Cloudflare Tunnel Connection Issues

**Symptom:** Cloudflare tunnel pods cannot establish connections

**Solution:** Check the `allow-cloudflare-egress` policy:

```bash
kubectl get networkpolicy -n cloudflare-tunnel allow-cloudflare-egress
kubectl logs -n cloudflare-tunnel deployment/cloudflared
```

### Spark Jobs Fail to Download Dependencies

**Symptom:** Spark jobs fail during package installation

**Solution:** Verify pod has correct labels:

```bash
kubectl get pods -n data-platform -l spark-role=driver
kubectl get networkpolicy -n data-platform allow-package-mirrors-spark-driver
```

### Service Cannot Reach Another Service

**Symptom:** Inter-service communication fails

**Solution:** Review existing service-specific policies and add explicit rules if needed:

```bash
kubectl get networkpolicy -n data-platform
```

## Maintenance

### Adding New Namespaces

To apply default-deny egress to a new namespace:

1. Add default-deny-egress policy
2. Add DNS egress allowlist
3. Add any namespace-specific egress rules

### Allowing New Egress Endpoints

To allow a new egress destination:

1. Identify the minimum required ports
2. Create a targeted NetworkPolicy with specific podSelector
3. Document the business justification
4. Test in non-production first

### Audit and Review

Review network policies quarterly:

```bash
# List all network policies
kubectl get networkpolicy -A

# Review egress rules
kubectl get networkpolicy -A -o yaml | grep -A 10 "policyTypes:\s*- Egress"
```

## References

- [Kubernetes Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Network Policy Recipes](https://github.com/ahmetb/kubernetes-network-policy-recipes)
- [NIST SP 800-53](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final)
