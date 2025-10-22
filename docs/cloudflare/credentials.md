# Cloudflare Tunnel Credentials & Rotation

Use this page to manage and rotate Cloudflare Tunnel credentials for Kubernetes.

Prerequisites
- Have a working tunnel in the Cloudflare dashboard
- Obtain `TUNNEL_ID`, `ACCOUNT_TAG`, and `AUTH_TOKEN` from the dashboard

Update credentials (scripted)
```bash
./scripts/update-cloudflare-credentials.sh TUNNEL_ID ACCOUNT_TAG AUTH_TOKEN
```

What the script does
- Creates/updates secret: `cloudflare-tunnel-credentials` in `cloudflare-tunnel` namespace
- Restarts the `cloudflared` deployment
- Waits for rollout and performs a basic health check

Manual verification
```bash
kubectl get secret -n cloudflare-tunnel cloudflare-tunnel-credentials -o jsonpath='{.data.credentials\.json}' | base64 -d | jq .
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --tail=100 | grep -i "registered\|connected\|error"
```

Getting credentials from Cloudflare
- docs/cloudflare/get-certificate.md
- docs/cloudflare/get-proper-credentials.md

Troubleshooting (including Error 1033)
- docs/cloudflare/troubleshooting.md
