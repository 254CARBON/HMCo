# 254Carbon SSO Portal

A modern, responsive landing portal and service catalog for the 254Carbon data platform with integrated Single Sign-On (SSO) via Cloudflare Access.

## Overview

The 254Carbon Portal serves as the central entry point for accessing all cluster services through a unified authentication experience. It provides:

- **Service Discovery**: Browse and access all available platform services
- **Single Sign-On**: Unified authentication via Cloudflare Access
- **Service Catalog**: Organized by category (Monitoring, Data, Compute, Storage, Workflow)
- **Status Dashboard**: Real-time platform health indicators
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile

## Architecture

```
User Browser
    ↓
254carbon.com (Portal Landing Page)
    ↓
Cloudflare Access (Authentication)
    ↓
Kubernetes Ingress (NGINX)
    ↓
Portal Service (Next.js - 2 replicas)
    ↓
Service Links (Redirect to subdomains)
```

## Services

The portal provides access to 9 cluster services organized in 5 categories:

### Monitoring & Visualization
- **Grafana** - Real-time monitoring dashboards
- **Apache Superset** - BI and data visualization

### Data Governance
- **DataHub** - Metadata platform for data discovery

### Compute & Query
- **Trino** - Distributed SQL query engine
- **Apache Doris** - Columnar OLAP database

### Storage & Secrets
- **HashiCorp Vault** - Secrets management
- **MinIO** - S3-compatible object storage
- **LakeFS** - Data lake version control

### Workflow & Orchestration
- **DolphinScheduler** - Workflow orchestration

## Installation

### Prerequisites
- Kubernetes cluster with NGINX Ingress Controller
- Cloudflare Tunnel configured for 254carbon.com
- Docker runtime for building images

### Build Docker Image

```bash
cd portal
docker build -t 254carbon-portal:latest .
```

### Deploy to Kubernetes

```bash
# Create the portal deployment and service
kubectl apply -f k8s/ingress/portal-deployment.yaml

# Create the ingress rules for portal
kubectl apply -f k8s/ingress/portal-ingress.yaml

# Verify deployment
kubectl get pods -n data-platform | grep portal
kubectl get svc -n data-platform | grep portal
```

### Verify Deployment

```bash
# Check pod status
kubectl get pods -n data-platform -l app=portal

# View logs
kubectl logs -n data-platform -l app=portal -f

# Test connectivity
curl -v https://254carbon.com
```

## Configuration

### Environment Variables

The portal supports the following environment variables:

- `NEXT_PUBLIC_PORTAL_URL` - Base URL for the portal (default: https://254carbon.com)
- `NEXT_PUBLIC_API_URL` - API endpoint URL (default: https://254carbon.com/api)
- `NEXT_PUBLIC_CLOUDFLARE_ACCESS_DOMAIN` - Cloudflare Access domain
- `NODE_ENV` - Deployment environment (default: production)

### Kubernetes Configuration

The deployment includes:

- **Replicas**: 2 (high availability)
- **Resource Limits**: 100m CPU / 256Mi memory (request), 500m CPU / 512Mi memory (limit)
- **Health Checks**: Liveness and readiness probes
- **Security**: Non-root user, read-only filesystem, dropped capabilities
- **Pod Anti-Affinity**: Spread across different nodes when possible

## Development

### Local Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Open http://localhost:8080 in browser
```

### Build for Production

```bash
npm run build
npm start
```

## Technology Stack

- **Framework**: Next.js 14
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Runtime**: Node.js 20
- **Container**: Docker
- **Orchestration**: Kubernetes

## Security

### Implementation

- Non-root user execution (UID 1001)
- Read-only root filesystem
- Dropped all Linux capabilities
- No privilege escalation allowed
- Regular pod anti-affinity scheduling

### Cloudflare Access

- All traffic flows through Cloudflare Access
- Centralized authentication and authorization
- Session management handled by Cloudflare
- Audit logging of all access attempts
- DDoS and WAF protection included

## Monitoring

### Health Checks

```bash
# Liveness probe
curl http://localhost:8080/

# Container metrics
kubectl top pods -n data-platform -l app=portal
```

### Logs

```bash
# Real-time logs
kubectl logs -n data-platform -l app=portal -f

# Last 100 lines
kubectl logs -n data-platform -l app=portal -n 100
```

## Troubleshooting

### Portal not accessible

1. Verify ingress rules:
```bash
kubectl get ingress -n data-platform
kubectl describe ingress portal-ingress -n data-platform
```

2. Check pod status:
```bash
kubectl get pods -n data-platform -l app=portal
kubectl logs -n data-platform -l app=portal
```

3. Verify Cloudflare tunnel:
```bash
kubectl logs -n cloudflare-tunnel -f
```

### Services not loading

1. Verify service endpoints:
```bash
kubectl get svc -n data-platform | grep -E "grafana|superset|datahub"
```

2. Check ingress rules:
```bash
kubectl get ingress -A | grep 254carbon
```

3. Test DNS resolution:
```bash
nslookup 254carbon.com
nslookup grafana.254carbon.com
```

## Performance

### Load Testing

The portal is designed to handle:
- 1000+ concurrent users
- Sub-100ms response times
- 99.9% uptime with 2 replicas

### Resource Usage

- Memory: ~256Mi per replica (typical)
- CPU: ~100m per replica (idle), up to 500m under load
- Network: <1Mbps per replica (typical usage)

## Roadmap

### Planned Enhancements

- [ ] User profile management
- [ ] Service bookmarking/favorites
- [ ] Advanced search across services
- [ ] Usage analytics and metrics
- [ ] Role-based service visibility
- [ ] Mobile app

## Contributing

When modifying the portal:

1. Maintain TypeScript strict mode
2. Follow Tailwind CSS conventions
3. Add proper error handling
4. Update documentation
5. Test responsive design

## Maintenance

### Updates

```bash
# Build new image
docker build -t 254carbon-portal:new-tag .

# Update deployment
kubectl set image deployment/portal portal=254carbon-portal:new-tag -n data-platform
```

### Cleanup

```bash
# Delete deployment
kubectl delete deployment portal -n data-platform

# Delete service
kubectl delete svc portal -n data-platform

# Delete ingress
kubectl delete ingress portal-ingress -n data-platform
```

## Support

For issues or questions:
1. Check pod logs: `kubectl logs -n data-platform -l app=portal`
2. Review ingress configuration
3. Verify Cloudflare tunnel status
4. Test authentication flow

## License

Part of the 254Carbon Data Platform
