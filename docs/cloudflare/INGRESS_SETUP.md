# Cloudflare Tunnel & Ingress Configuration for 254Carbon

## Overview

This document describes the DNS, tunnel, and ingress configuration for exposing all 254Carbon platform services through the `*.254carbon.com` domain using Cloudflare Tunnel and Kubernetes Ingress.

## Architecture

```
Internet (cloudflare.com)
    ↓ (DNS CNAME)
Cloudflare Tunnel (291bc289-e3c3-4446-a9ad-8e327660ecd5)
    ↓ (cloudflared agent)
Kubernetes Cluster
    ↓
NGINX Ingress Controller (ingress-nginx)
    ↓
Service-specific Ingress Resources
    ↓
Kubernetes Services → Pods
```

## Components

### 1. Cloudflare Tunnel
- **Tunnel ID**: `291bc289-e3c3-4446-a9ad-8e327660ecd5`
- **Account ID**: `0c93c74d5269a228e91d4bf91c547f56`
- **Location**: `k8s/cloudflare-tunnel-ingress.yaml`
- **Namespace**: `cloudflare-tunnel`
- **Replicas**: 2 (for high availability)

The tunnel is configured as a ConfigMap that maps hostnames to services:

```yaml
ingress:
  - hostname: <service>.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
```

### 2. NGINX Ingress Controller
- **Namespace**: `ingress-nginx`
- **Port**: 80 (HTTP), 443 (HTTPS)
- **Class**: nginx

All traffic from Cloudflare tunnel routes through this controller, which then dispatches to service-specific ingress resources.

### 3. Service Ingress Resources
Individual ingress resources in `/k8s/ingress/`:

| Service | Namespace | Ingress File | URL |
|---------|-----------|--------------|-----|
| Prometheus | monitoring | prometheus-ingress.yaml | prometheus.254carbon.com |
| AlertManager | monitoring | alertmanager-ingress.yaml | alertmanager.254carbon.com |
| Victoria Metrics | victoria-metrics | victoria-metrics-ingress.yaml | victoria.254carbon.com |
| Loki | victoria-metrics | loki-ingress.yaml | loki.254carbon.com |
| ClickHouse | data-platform | clickhouse-ingress.yaml | clickhouse.254carbon.com |
| Katib UI | kubeflow | katib-ingress.yaml | katib.254carbon.com |
| Kong Admin | kong | kong-admin-ingress.yaml | kong.254carbon.com |

Plus all existing services (DataHub, Grafana, Superset, etc.)

## DNS Configuration (Cloudflare)

All service URLs should be configured as CNAME records pointing to the tunnel:

```
*.254carbon.com  CNAME  <tunnel-id>.cfargotunnel.com
```

Or individual A/AAAA records if using specific IPs provided by Cloudflare.

## SSL/TLS Configuration

- **Issuer**: Let's Encrypt (letsencrypt-prod)
- **Certificate Manager**: cert-manager
- **Automated Renewal**: Enabled
- **HSTS**: Enabled via ingress annotations

Each ingress resource includes:
```yaml
annotations:
  cert-manager.io/cluster-issuer: letsencrypt-prod
  nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
  nginx.ingress.kubernetes.io/ssl-redirect: "true"
```

Certificates are stored as Kubernetes Secrets:
```
<service>-tls  (in each service's namespace)
```

## Service Routing Rules

### Basic Routing (No Authentication)
Services accessible without authentication route directly:
- Portal, DataHub, Grafana, Superset, etc.

### Authenticated Routing (Cloudflare Access)
Services with sensitive data can use Cloudflare Access:
- Vault, Kong Admin, Prometheus, etc.

Annotation for authentication:
```yaml
nginx.ingress.kubernetes.io/auth-url: https://qagi.cloudflareaccess.com/cdn-cgi/access/authorize
nginx.ingress.kubernetes.io/auth-signin: https://qagi.cloudflareaccess.com/cdn-cgi/access/login?redirect_url=$escaped_request_uri
```

## Deployment

### Prerequisites
1. Cloudflare account with tunnel configured
2. Kubernetes cluster with cert-manager and nginx-ingress-controller
3. kubectl access to the cluster

### Deployment Steps

```bash
# 1. Apply the deployment script
chmod +x scripts/deploy-cloudflare-ingress.sh
./scripts/deploy-cloudflare-ingress.sh

# OR manually:

# 2. Apply Cloudflare tunnel configuration
kubectl apply -f k8s/cloudflare-tunnel-ingress.yaml

# 3. Apply all ingress resources
kubectl apply -f k8s/ingress/prometheus-ingress.yaml
kubectl apply -f k8s/ingress/alertmanager-ingress.yaml
kubectl apply -f k8s/ingress/victoria-metrics-ingress.yaml
kubectl apply -f k8s/ingress/loki-ingress.yaml
kubectl apply -f k8s/ingress/clickhouse-ingress.yaml
kubectl apply -f k8s/ingress/katib-ingress.yaml
kubectl apply -f k8s/ingress/kong-admin-ingress.yaml

# 4. Restart tunnel pods to pick up changes
kubectl rollout restart deployment cloudflared -n cloudflare-tunnel
```

## Verification

### Check Ingress Resources
```bash
# List all ingress resources
kubectl get ingress -A

# Get details of specific ingress
kubectl describe ingress prometheus-ingress -n monitoring
```

### Check Tunnel Status
```bash
# View tunnel pods
kubectl get pods -n cloudflare-tunnel

# View tunnel logs
kubectl logs -f -n cloudflare-tunnel deployment/cloudflared

# Check recent tunnel events
kubectl logs -n cloudflare-tunnel deployment/cloudflared --tail=50
```

### Test Service Access
```bash
# Test HTTPS access (should redirect from HTTP)
curl -I https://prometheus.254carbon.com
curl -I https://alertmanager.254carbon.com
curl -I https://clickhouse.254carbon.com

# Test with verbose output
curl -v https://prometheus.254carbon.com 2>&1 | head -20
```

### Check SSL Certificates
```bash
# List all TLS secrets
kubectl get secrets -A | grep tls

# View certificate details
kubectl describe secret prometheus-tls -n monitoring
```

## Troubleshooting

### Service Not Accessible
1. Check tunnel pod status:
   ```bash
   kubectl get pods -n cloudflare-tunnel
   ```
2. View tunnel logs:
   ```bash
   kubectl logs -n cloudflare-tunnel deployment/cloudflared
   ```
3. Verify ingress resource:
   ```bash
   kubectl describe ingress <service>-ingress -n <namespace>
   ```
4. Check DNS resolution:
   ```bash
   nslookup <service>.254carbon.com
   ```

### SSL Certificate Issues
1. Check cert-manager status:
   ```bash
   kubectl get certificates -A
   kubectl describe certificate <service>-tls -n <namespace>
   ```
2. View cert-manager logs:
   ```bash
   kubectl logs -n cert-manager deployment/cert-manager
   ```

### 404 Errors
1. Verify service exists:
   ```bash
   kubectl get svc <service-name> -n <namespace>
   ```
2. Check service port matches ingress:
   ```bash
   kubectl get svc <service-name> -n <namespace> -o yaml | grep targetPort
   ```

### Connection Refused
1. Verify service is running:
   ```bash
   kubectl get pods -n <namespace> -l app=<service>
   ```
2. Check service readiness:
   ```bash
   kubectl describe pod <pod-name> -n <namespace>
   ```

## Maintenance

### Adding New Services
1. Create service ingress resource in `/k8s/ingress/`
2. Add hostname to `k8s/cloudflare-tunnel-ingress.yaml`
3. Add entry to `services.json`
4. Apply changes:
   ```bash
   kubectl apply -f k8s/ingress/<new-service>-ingress.yaml
   kubectl apply -f k8s/cloudflare-tunnel-ingress.yaml
   kubectl rollout restart deployment cloudflared -n cloudflare-tunnel
   ```

### Updating Certificates
- Certificates are automatically renewed by cert-manager before expiration
- Manual renewal (if needed):
  ```bash
  kubectl delete secret <service>-tls -n <namespace>
  kubectl rollout restart deployment <service> -n <namespace>
  ```

### Monitoring Tunnel Health
- Check Cloudflare dashboard for tunnel status
- Monitor tunnel pod resources:
  ```bash
  kubectl top pods -n cloudflare-tunnel
  ```

## Security Best Practices

1. **Use HTTPS Only**: All connections redirected to HTTPS
2. **Authentication**: Enable Cloudflare Access for sensitive services
3. **Network Policies**: Apply Kubernetes network policies
4. **RBAC**: Restrict access to ingress resources
5. **Monitoring**: Enable logging and alerting on tunnel events

## Related Files

- **Tunnel Config**: `k8s/cloudflare-tunnel-ingress.yaml`
- **Ingress Resources**: `k8s/ingress/*.yaml`
- **Service Catalog**: `services.json`
- **Deployment Script**: `scripts/deploy-cloudflare-ingress.sh`
- **Cert-Manager Config**: `k8s/cert-manager/`

## References

- [Cloudflare Tunnel Documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
- [Kubernetes Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- [cert-manager Documentation](https://cert-manager.io/docs/)
