# Getting Proper Cloudflare Tunnel Credentials for Kubernetes (Canonical)

## Issue

The token you provided (`eyJhIjoiMGM5M2M3NGQ1MjY5YTIyOGU5MWQ0YmY5MWM1NDdmNTYiLCJ0IjoiMjkxYmMyODktZTNjMy00NDQ2LWE5YWQtOGUzMjc2NjBlY2Q1IiwicyI6Ik9XVXhNbU0zT1RZdE1USTFaaTAwTVRobUxUazRZall0Wm1JeFl6Rm1aREl6TjJGaCJ9`) is a **Cloudflared Service Token**, which is used for `cloudflared service install` on Linux servers, not for Kubernetes tunnel connectors.

For Kubernetes, we need the **Tunnel Configuration** in a different format.

## Solution: Get Tunnel Configuration from Cloudflare Dashboard

### Step 1: Go to Cloudflare Zero Trust Console

1. Visit: https://one.dash.cloudflare.com/
2. Click your profile icon → **Settings**
3. Look for **Product** → **Tunnels**

### Step 2: Find Your Tunnel

1. **Networks** → **Tunnels** → **254carbon-cluster**
2. Click **Configure**
3. Look for the **Public Hostname** tab

###  Step 3: Get the Tunnel Token

On the Configure page, you should see:
- A section for " **Install and run a connector**"
- Options for Docker, Kubernetes, etc.
- **If there's a Kubernetes option, use that!**

Or look for:
- **"Token"** field showing a base64-encoded value
- This is the **correct format** for Kubernetes tunnels

### Step 4: Alternative - Download Credentials File

1. Look for **"Download credentials"** or **"JSON credentials"** button
2. This downloads a JSON file in format:
```json
{
  "AccountTag": "your-account-id",
  "TunnelID": "your-tunnel-uuid",
  "TunnelName": "254carbon-cluster",
  "TunnelSecret": "your-tunnel-secret"
}
```

If you have this JSON format, use the `TunnelSecret` value - this is what goes in Kubernetes!

## Applying the Correct Token

Once you have the **proper Tunnel Token** (not the service install token), update it:

```bash
# Replace TOKEN_HERE with your actual tunnel token from the dashboard
kubectl delete secret cloudflare-tunnel-credentials -n cloudflare-tunnel 2>/dev/null || true

kubectl create secret generic cloudflare-tunnel-credentials \
  -n cloudflare-tunnel \
  --from-literal=token=TOKEN_HERE

# Then apply the deployment
kubectl apply -f k8s/cloudflare/cloudflared-deployment.yaml
```

## Understanding the Difference

| Type | Format | Use Case | Source |
|------|--------|----------|--------|
| Service Install Token | Base64 JSON with `a`, `t`, `s` fields | Linux `sudo cloudflared service install` | `sudo cloudflared service install` output |
| Tunnel Token | Base64-encoded string | Kubernetes/Docker connectors | Cloudflare dashboard Tunnel Config |
| Credentials JSON | `{AccountTag, TunnelID, TunnelName, TunnelSecret}` | Manual setup | Download button in dashboard |

## Where to Find the Right Token

**In Cloudflare Zero Trust Dashboard:**
1. Go to **Networks** → **Tunnels** → Your Tunnel
2. Click **Configure**
3. Look for a tab or section labeled:
   - "Install connector"
   - "Kubernetes"
   - "Docker"
   - "Copy token"
4. The long base64 string you see there is the **correct token for Kubernetes**

**It will look like:** `eyJhI...` (much longer than your current one)

## Next Steps

1. Go to your Cloudflare dashboard
2. Find the **Kubernetes tunnel token** (not the service install token)
3. Provide that token
4. I'll update the deployment and everything will work!

## Questions?

If unsure which token is correct:
- ✅ **Correct**: From "Kubernetes" section of tunnel configuration
- ❌ **Wrong**: From `sudo cloudflared service install` command
- ✅ **Also Correct**: The `TunnelSecret` from downloaded JSON credentials
