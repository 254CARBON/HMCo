# API Gateway Architecture

**Platform**: 254Carbon Service Integration  
**Technology**: Kong API Gateway  
**Status**: Implementation in Progress  
**Updated**: October 22, 2025

---

## Overview

Kong API Gateway provides a centralized entry point for all service APIs with:

- **Unified API Management**: Single point for all service APIs
- **Authentication & Authorization**: JWT, OAuth2, API keys, and mTLS
- **Rate Limiting**: Per-consumer, per-route rate limits
- **Traffic Control**: Request/response transformation, routing
- **Observability**: Logging, metrics, and tracing
- **Developer Portal**: API documentation and testing

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   External Clients                        │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│              Cloudflare Access (SSO)                      │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│         NGINX Ingress Controller                          │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│         Kong API Gateway (Control Plane)                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Kong Admin │  │   Kong     │  │  Kong Dev  │        │
│  │    API     │  │  Ingress   │  │   Portal   │        │
│  └────────────┘  └────────────┘  └────────────┘        │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│         Kong Data Plane (Proxy Nodes)                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │  Proxy 1   │  │  Proxy 2   │  │  Proxy 3   │        │
│  └────────────┘  └────────────┘  └────────────┘        │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│               Microservices                               │
│  DataHub | Trino | Superset | DolphinScheduler | ...    │
└──────────────────────────────────────────────────────────┘
```

## Components

### Core Infrastructure
- **kong-namespace.yaml** - Namespace and RBAC
- **kong-postgres.yaml** - PostgreSQL database for Kong
- **kong-migrations.yaml** - Database migration jobs
- **kong-deployment.yaml** - Kong control plane and data plane

### Configuration
- **kong-services/** - Service definitions for each microservice
- **kong-routes/** - Route configurations and path mappings
- **kong-plugins/** - Authentication, rate limiting, transformations
- **kong-consumers/** - API consumers and credentials

### Ingress
- **kong-ingress-controller.yaml** - Kubernetes ingress controller
- **kong-ingress-rules.yaml** - Ingress resources

### Developer Portal
- **kong-dev-portal.yaml** - Developer documentation portal

## Installation

### Prerequisites

1. PostgreSQL for Kong database (included in deployment)
2. Kubernetes cluster with Istio service mesh
3. Existing services deployed in data-platform namespace

### Deployment Steps

```bash
# 1. Create namespace and RBAC
kubectl apply -f k8s/api-gateway/kong-namespace.yaml

# 2. Deploy PostgreSQL for Kong
kubectl apply -f k8s/api-gateway/kong-postgres.yaml

# 3. Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=kong-postgres -n kong --timeout=300s

# 4. Run database migrations
kubectl apply -f k8s/api-gateway/kong-migrations.yaml

# 5. Wait for migrations to complete
kubectl wait --for=condition=complete job/kong-migrations -n kong --timeout=300s

# 6. Deploy Kong control plane and data plane
kubectl apply -f k8s/api-gateway/kong-deployment.yaml

# 7. Verify Kong is running
kubectl get pods -n kong

# 8. Deploy Kong Ingress Controller
kubectl apply -f k8s/api-gateway/kong-ingress-controller.yaml

# 9. Configure services and routes
kubectl apply -f k8s/api-gateway/kong-services/
kubectl apply -f k8s/api-gateway/kong-routes/

# 10. Enable plugins
kubectl apply -f k8s/api-gateway/kong-plugins/

# 11. Create API consumers
kubectl apply -f k8s/api-gateway/kong-consumers/
```

## Configuration

### Service Registration

Register a service with Kong:

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongService
metadata:
  name: datahub-api
  namespace: kong
spec:
  host: datahub-gms.data-platform.svc.cluster.local
  port: 8080
  protocol: http
  path: /
  retries: 3
  connect_timeout: 60000
  write_timeout: 60000
  read_timeout: 60000
```

### Route Configuration

Create a route to the service:

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongRoute
metadata:
  name: datahub-api-route
  namespace: kong
spec:
  service: datahub-api
  paths:
  - /api/datahub
  strip_path: true
  preserve_host: false
  protocols:
  - http
  - https
  methods:
  - GET
  - POST
  - PUT
  - DELETE
```

### Plugin Configuration

Enable rate limiting:

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: rate-limiting
  namespace: kong
config:
  minute: 100
  hour: 10000
  policy: local
plugin: rate-limiting
```

Apply to a service:

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongService
metadata:
  name: datahub-api
  annotations:
    konghq.com/plugins: rate-limiting
spec:
  # ... service config
```

## Authentication

### API Key Authentication

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: api-key-auth
  namespace: kong
config:
  key_names:
  - apikey
  hide_credentials: true
plugin: key-auth
```

### JWT Authentication

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: jwt-auth
  namespace: kong
config:
  uri_param_names:
  - jwt
  key_claim_name: iss
  secret_is_base64: false
plugin: jwt
```

### OAuth2

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: oauth2
  namespace: kong
config:
  scopes:
  - read
  - write
  - admin
  mandatory_scope: true
  enable_client_credentials: true
  enable_authorization_code: true
plugin: oauth2
```

## Consumer Management

Create an API consumer:

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongConsumer
metadata:
  name: external-client
  namespace: kong
  annotations:
    kubernetes.io/ingress.class: kong
username: external-client
custom_id: "ext-client-001"
```

Add API key credential:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: external-client-apikey
  namespace: kong
  labels:
    konghq.com/credential: key-auth
stringData:
  key: "super-secret-api-key-12345"
  kongCredType: key-auth
  kongConsumer: external-client
```

## Rate Limiting

### Per-Consumer Limits

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: consumer-rate-limit
  namespace: kong
config:
  minute: 100
  hour: 5000
  policy: redis
  redis_host: redis-service.data-platform.svc.cluster.local
  redis_port: 6379
  redis_database: 1
plugin: rate-limiting
```

### Per-Route Limits

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongRoute
metadata:
  name: expensive-operation
  annotations:
    konghq.com/plugins: strict-rate-limit
spec:
  # ... route config
---
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: strict-rate-limit
config:
  minute: 10
  hour: 100
plugin: rate-limiting
```

## Request/Response Transformation

### Add Headers

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: add-headers
config:
  add:
    headers:
    - "X-Platform:254carbon"
    - "X-Environment:production"
plugin: request-transformer
```

### Response Transformation

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: response-transformer
config:
  add:
    headers:
    - "X-Kong-Response:true"
  remove:
    headers:
    - "X-Internal-Header"
plugin: response-transformer
```

## Monitoring

### Prometheus Plugin

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongClusterPlugin
metadata:
  name: prometheus
  annotations:
    kubernetes.io/ingress.class: kong
plugin: prometheus
```

### Access Logs

```yaml
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: file-log
config:
  path: /dev/stdout
plugin: file-log
```

## Admin API Access

```bash
# Port-forward to Kong admin API
kubectl port-forward -n kong svc/kong-admin 8001:8001

# List services
curl http://localhost:8001/services

# List routes
curl http://localhost:8001/routes

# List consumers
curl http://localhost:8001/consumers

# List plugins
curl http://localhost:8001/plugins
```

## Troubleshooting

### Service Not Reachable

```bash
# Check Kong proxy logs
kubectl logs -n kong -l app=kong -c proxy

# Check service connectivity
kubectl exec -n kong -it <kong-pod> -- curl http://service.namespace.svc.cluster.local:port

# Verify service registration
curl http://localhost:8001/services/<service-name>
```

### Authentication Issues

```bash
# Check plugin configuration
curl http://localhost:8001/plugins/<plugin-id>

# Test with credentials
curl -H "apikey: your-api-key" https://api.254carbon.com/endpoint
```

### Performance Issues

```bash
# Check resource usage
kubectl top pods -n kong

# Review metrics
kubectl port-forward -n kong svc/kong-admin 8001:8001
curl http://localhost:8001/metrics
```

## Best Practices

1. **Service Registration**: Register all services through Kong for unified access
2. **Authentication**: Use JWT or OAuth2 for production, API keys for internal services
3. **Rate Limiting**: Apply per-consumer and per-route limits based on SLAs
4. **Monitoring**: Enable Prometheus plugin and integrate with Grafana
5. **Versioning**: Use path prefixes (/v1/, /v2/) for API versioning
6. **Security**: Always use HTTPS, hide credentials, validate inputs
7. **Caching**: Enable proxy-cache plugin for GET requests
8. **Documentation**: Keep developer portal updated with API changes

## Resources

- **Kong Official Docs**: https://docs.konghq.com/
- **Kong Ingress Controller**: https://docs.konghq.com/kubernetes-ingress-controller/
- **Plugin Hub**: https://docs.konghq.com/hub/



