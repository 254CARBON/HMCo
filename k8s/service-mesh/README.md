# Service Mesh Architecture

**Platform**: 254Carbon Service Integration  
**Technology**: Istio Service Mesh  
**Status**: Implementation in Progress  
**Updated**: October 22, 2025

---

## Overview

This directory contains the service mesh implementation for the 254Carbon platform, providing:

- **mTLS**: Automatic mutual TLS for service-to-service communication
- **Traffic Management**: Advanced routing, load balancing, and circuit breaking
- **Observability**: Distributed tracing, metrics, and service graphs
- **Security**: Authorization policies and secure communication
- **Resilience**: Retries, timeouts, and circuit breakers

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Istio Control Plane                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Pilot   │  │  Citadel │  │  Galley  │  │  Mixer   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Plane (Envoy Sidecars)               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ Service A  │  │ Service B  │  │ Service C  │           │
│  │ + Sidecar  │  │ + Sidecar  │  │ + Sidecar  │  ...      │
│  └────────────┘  └────────────┘  └────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Core Infrastructure
- **istio-base.yaml** - Base resources and CRDs
- **istio-operator.yaml** - Istio Operator for lifecycle management
- **istio-config.yaml** - Core Istio configuration (IstioOperator CR)

### Traffic Management
- **virtual-services/** - Route rules and traffic splitting
- **destination-rules/** - Load balancing and circuit breaking
- **gateways/** - Ingress and egress gateways
- **service-entries/** - External service definitions

### Security
- **peer-authentication.yaml** - mTLS configuration
- **authorization-policies/** - Service-level access control
- **request-authentication.yaml** - JWT validation

### Observability
- **telemetry/** - Metrics, logs, and traces configuration
- **kiali.yaml** - Service mesh visualization
- **jaeger.yaml** - Distributed tracing

## Deployment

### Prerequisites

1. Kubernetes cluster with sufficient resources:
   - Control plane: 2 CPU, 4GB RAM
   - Sidecars: ~200MB RAM per service

2. Existing services in data-platform namespace

### Installation Steps

```bash
# 1. Install Istio CRDs and base
kubectl apply -f k8s/service-mesh/istio-base.yaml

# 2. Deploy Istio Operator
kubectl apply -f k8s/service-mesh/istio-operator.yaml

# 3. Wait for operator to be ready
kubectl wait --for=condition=available --timeout=300s \
  deployment/istio-operator -n istio-operator

# 4. Install Istio control plane
kubectl apply -f k8s/service-mesh/istio-config.yaml

# 5. Verify installation
kubectl get pods -n istio-system

# 6. Enable sidecar injection for data-platform namespace
kubectl label namespace data-platform istio-injection=enabled

# 7. Deploy observability tools
kubectl apply -f k8s/service-mesh/observability/

# 8. Deploy security policies
kubectl apply -f k8s/service-mesh/security/

# 9. Deploy traffic management rules
kubectl apply -f k8s/service-mesh/traffic-management/
```

### Service Migration

Services will automatically get Envoy sidecars when pods are restarted:

```bash
# Restart services to inject sidecars
kubectl rollout restart deployment -n data-platform

# Verify sidecar injection
kubectl get pods -n data-platform -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].name}{"\n"}{end}'
```

## Traffic Management

### VirtualServices

Control routing behavior:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: service-routes
spec:
  hosts:
  - service-name
  http:
  - match:
    - headers:
        version:
          exact: v2
    route:
    - destination:
        host: service-name
        subset: v2
  - route:
    - destination:
        host: service-name
        subset: v1
```

### DestinationRules

Configure load balancing and circuit breaking:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: service-circuit-breaker
spec:
  host: service-name
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 10
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
```

## Security

### Mutual TLS

Enforce mTLS across all services:

```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: data-platform
spec:
  mtls:
    mode: STRICT
```

### Authorization Policies

Fine-grained access control:

```yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: service-policy
spec:
  selector:
    matchLabels:
      app: protected-service
  action: ALLOW
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/data-platform/sa/client-service"]
    to:
    - operation:
        methods: ["GET", "POST"]
```

## Observability

### Kiali Dashboard

Access service mesh visualization:

```bash
kubectl port-forward -n istio-system svc/kiali 20001:20001
# Open http://localhost:20001
```

### Jaeger Tracing

View distributed traces:

```bash
kubectl port-forward -n istio-system svc/tracing 16686:16686
# Open http://localhost:16686
```

### Grafana Dashboards

Istio metrics are automatically exported to Prometheus and available in Grafana:
- Istio Service Dashboard
- Istio Workload Dashboard
- Istio Performance Dashboard
- Istio Control Plane Dashboard

## Monitoring

### Key Metrics

- **Request Rate**: `istio_requests_total`
- **Request Duration**: `istio_request_duration_milliseconds`
- **Request Size**: `istio_request_bytes`
- **Response Size**: `istio_response_bytes`
- **TCP Connections**: `istio_tcp_connections_opened_total`

### Health Checks

```bash
# Check control plane health
kubectl get pods -n istio-system

# Check sidecar injection
kubectl get namespace -L istio-injection

# Verify mTLS status
istioctl authn tls-check <pod-name>.<namespace>

# Check configuration
istioctl proxy-status
```

## Troubleshooting

### Sidecar Not Injected

```bash
# Check namespace label
kubectl get namespace data-platform --show-labels

# Check pod annotations
kubectl get pod <pod-name> -n data-platform -o yaml | grep sidecar

# Manual injection
istioctl kube-inject -f deployment.yaml | kubectl apply -f -
```

### mTLS Issues

```bash
# Check peer authentication
kubectl get peerauthentication -n data-platform

# Verify certificates
istioctl proxy-config secret <pod-name>.<namespace>

# Check TLS mode
kubectl get destinationrule -n data-platform -o yaml | grep -A 5 trafficPolicy
```

### Performance Issues

```bash
# Check sidecar resource usage
kubectl top pods -n data-platform --containers

# Adjust sidecar resources
kubectl annotate deployment <deployment-name> \
  sidecar.istio.io/proxyCPU=200m \
  sidecar.istio.io/proxyMemory=256Mi
```

## Configuration Reference

### Sidecar Resource Annotations

```yaml
metadata:
  annotations:
    sidecar.istio.io/proxyCPU: "100m"
    sidecar.istio.io/proxyCPULimit: "500m"
    sidecar.istio.io/proxyMemory: "128Mi"
    sidecar.istio.io/proxyMemoryLimit: "256Mi"
```

### Traffic Management Annotations

```yaml
metadata:
  annotations:
    traffic.sidecar.istio.io/includeInboundPorts: "8080,9090"
    traffic.sidecar.istio.io/excludeOutboundPorts: "3306"
    traffic.sidecar.istio.io/excludeOutboundIPRanges: "10.0.0.0/8"
```

## Best Practices

1. **Gradual Rollout**: Enable mesh for non-critical services first
2. **Resource Limits**: Set appropriate CPU/memory limits for sidecars
3. **Health Checks**: Use readiness probes to prevent traffic to unhealthy pods
4. **Monitoring**: Watch for increased latency after mesh enablement
5. **mTLS**: Start with PERMISSIVE mode, move to STRICT gradually
6. **Circuit Breaking**: Configure based on service capacity
7. **Retries**: Use exponential backoff with jitter
8. **Timeouts**: Set realistic timeouts based on SLAs

## Documentation

- **Istio Official Docs**: https://istio.io/latest/docs/
- **Performance Best Practices**: https://istio.io/latest/docs/ops/best-practices/performance/
- **Security Best Practices**: https://istio.io/latest/docs/ops/best-practices/security/

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Istio logs: `kubectl logs -n istio-system -l app=istiod`
3. Use istioctl analyze: `istioctl analyze -n data-platform`



