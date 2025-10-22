# Getting the Cloudflare Tunnel Certificate (Canonical)

## Issue

cloudflared needs the actual **certificate.pem** file, not just the tunnel ID. This file authenticates the connector to Cloudflare.

## Solution

### Step 1: Get the Tunnel Certificate from Cloudflare

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Navigate to **Networks** → **Tunnels** → **254carbon-cluster**
3. Click **Configure** tab
4. Look for "**Cloudflared**" section
5. You should see installation instructions showing:
   ```bash
   cloudflared service install <TOKEN>
   ```
   - Extract the **TOKEN** (long string after "service install")
   - This is your certificate

### Step 2: OR Download Credentials JSON (Alternative Method)

Some versions of Cloudflare show a "Download credentials" button:

1. Go to tunnel configuration
2. Click **Download credentials**
3. This should download a JSON file with authentication info
4. The file will be similar to:
   ```json
   {
     "AccountTag": "...",
     "TunnelID": "...",
     "TunnelName": "...",
     "TunnelSecret": "..."
   }
   ```

### Step 3: Update the Kubernetes Secret

Once you have the proper credential file:

**Option A: If you have certificate.pem:**
```bash
kubectl create secret generic cloudflare-tunnel-credentials \
  -n cloudflare-tunnel \
  --from-file=cert.pem=/path/to/cert.pem \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl set env deployment/cloudflared \
  -n cloudflare-tunnel \
  TUNNEL_ORIGIN_CERT=/etc/cloudflare-tunnel/creds/cert.pem

kubectl rollout restart deployment/cloudflared -n cloudflare-tunnel
```

**Option B: If you have tunnel token:**
```bash
kubectl create secret generic cloudflare-tunnel-credentials \
  -n cloudflare-tunnel \
  --from-literal=token=YOUR_TOKEN_HERE \
  --dry-run=client -o yaml | kubectl apply -f -

# Then use token-based auth in config
```

### Step 4: Update ConfigMap (If Using Token)

If using token-based authentication, update the config:

```yaml
tunnel: 254carbon-cluster
credentialsFile: /etc/cloudflare-tunnel/creds/token
metrics: 0.0.0.0:2000
# ... rest of config
```

## Troubleshooting

If you still get "Cannot determine default origin certificate path":

1. **Verify the credentials file exists in the pod:**
   ```bash
   kubectl exec -it deployment/cloudflared -n cloudflare-tunnel -- ls -la /etc/cloudflare-tunnel/creds/
   ```

2. **Check what credentials were actually provided:**
   ```bash
   kubectl get secret cloudflare-tunnel-credentials -n cloudflare-tunnel -o jsonpath='{.data}'
   ```

3. **Verify config references correct path:**
   ```bash
   kubectl get cm cloudflared-config -n cloudflare-tunnel -o jsonpath='{.data.config\.yaml}' | head -5
   ```

## Next Steps

1. Go to your Cloudflare dashboard and extract the proper certificate
2. Update the secret with this certificate
3. Restart the pods
4. Check logs for "INF Connection established" message
