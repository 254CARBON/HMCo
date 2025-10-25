# JupyterHub Cloudflare Tunnel Configuration

This document describes how to configure the Cloudflare tunnel to expose JupyterHub at `jupyter.254carbon.com`.

## Prerequisites

- Cloudflare account with access to 254carbon.com domain
- Existing Cloudflare tunnel configured
- Access to tunnel configuration

## Tunnel Configuration

Add the following ingress route to your Cloudflare tunnel configuration:

```yaml
ingress:
  # ... existing routes ...
  - hostname: jupyter.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  # ... additional routes ...
```

## Complete Tunnel Configuration

The full tunnel configuration should include:

```yaml
tunnel: <tunnel-id>
credentials-file: /etc/cloudflared/config.json

ingress:
  - hostname: portal.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: www.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: jupyter.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: datahub.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: grafana.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: superset.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: trino.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: vault.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: minio.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: dolphin.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: dolphinscheduler.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: harbor.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: lakefs.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  - hostname: rapids.254carbon.com
    service: http://ingress-nginx-controller.ingress-nginx:80
  # Catch-all for unmapped routes
  - service: http_status:404
```

## Configuration Steps

1. **Access Cloudflare Dashboard**
   - Log in to Cloudflare
   - Navigate to Zero Trust → Tunnels
   - Select your tunnel

2. **Update Public Hostname**
   - Click "Public Hostnames" tab
   - Click "Add a public hostname"
   - Add the following:
     - **Subdomain**: jupyter
     - **Domain**: 254carbon.com
     - **Service**: HTTP
     - **URL**: http://ingress-nginx-controller.ingress-nginx:80

3. **DNS Records**
   - Cloudflare automatically creates CNAME records
   - Verify in DNS management that `jupyter.254carbon.com` points to your tunnel

4. **Test Access**
   ```bash
   curl https://jupyter.254carbon.com
   ```

## Verify Tunnel Status

```bash
# Check tunnel is running
cloudflared tunnel list

# Check tunnel logs
cloudflared tunnel logs <tunnel-name>

# Test specific route
curl -v https://jupyter.254carbon.com/hub/health
```

## Troubleshooting

### Tunnel not connecting
- Verify tunnel token is correctly set in Kubernetes secret
- Check tunnel logs: `kubectl logs -n cloudflare-tunnel deployment/cloudflared`
- Verify ingress routes are correct

### JupyterHub returns 502 Bad Gateway
- Check JupyterHub proxy service is running: `kubectl get svc -n jupyter`
- Verify ingress rule is correctly configured: `kubectl get ingress -n jupyter`
- Check JupyterHub proxy logs: `kubectl logs -n jupyter svc/jupyterhub-proxy-public`

### DNS not resolving
- Wait 5-10 minutes for DNS propagation
- Clear browser cache and cookies
- Verify CNAME record: `nslookup jupyter.254carbon.com`

## Security Configuration

1. **Enable Cloudflare Access** (if not already enabled)
   - Go to Access → Applications
   - Create application for JupyterHub
   - Add policy to allow authenticated users

2. **Set WAF Rules**
   - Enable DDoS protection
   - Configure rate limiting if needed

3. **Monitor Tunnel Health**
   - Set up Cloudflare alerts for tunnel status changes
