# URGENT: Fix Cloudflare "Too Many Redirects" Error

## ⚠️ Problem

Getting "ERR_TOO_MANY_REDIRECTS" when accessing any 254carbon.com service.

## 🎯 Root Cause

Cloudflare SSL/TLS settings are causing redirect loops. The API tokens don't have permission to change these settings, so **you must do this manually in the Cloudflare Dashboard**.

## ✅ IMMEDIATE FIX (5 minutes)

### Step 1: Login to Cloudflare Dashboard

Go to: **https://dash.cloudflare.com**

### Step 2: Select Your Domain

Click on: **254carbon.com**

### Step 3: Configure SSL/TLS Settings

1. **In left sidebar**, click: **SSL/TLS**

2. **Under "Overview"**, look for: **"Your SSL/TLS encryption mode"**

3. **Change the mode to**: **Flexible**
   
   Options shown:
   - Off (not encrypted)
   - Flexible ← **SELECT THIS**
   - Full
   - Full (strict)

4. **Click**: The "Flexible" option

5. **Wait**: You'll see "Encryption mode updated successfully"

### Step 4: Disable Auto HTTPS Redirect

1. **Stay in SSL/TLS section**

2. **Click**: **Edge Certificates** (sub-menu under SSL/TLS)

3. **Scroll down** to find: **"Always Use HTTPS"**

4. **Toggle to**: **OFF** (gray/disabled)

5. **If you see "Automatic HTTPS Rewrites"**, also set to **OFF**

### Step 5: Wait and Test

1. **Wait 30-60 seconds** for Cloudflare to propagate changes globally

2. **Clear your browser cache** (or use Incognito mode)

3. **Test these URLs** (in order):
   - https://portal.254carbon.com
   - https://grafana.254carbon.com
   - https://superset.254carbon.com/superset/login

---

## 🔍 Verification

### Check Settings Are Correct

In Cloudflare Dashboard:

```
SSL/TLS > Overview
└─ Encryption mode: Flexible ✓

SSL/TLS > Edge Certificates
├─ Always Use HTTPS: Off ✓
└─ Automatic HTTPS Rewrites: Off ✓
```

### Test from Command Line

```bash
# Should return HTTP 200 (not 301/302/307)
curl -I https://portal.254carbon.com

# Check for redirect loops
curl -L -I https://portal.254carbon.com 2>&1 | grep -E "HTTP|Location"
```

---

## 📖 Why This Configuration?

**Flexible SSL Mode**:
- Browser → Cloudflare: **HTTPS** (encrypted, secure) ✓
- Cloudflare → Tunnel → Kubernetes: **HTTP** (encrypted inside tunnel) ✓
- **Result**: Visitors see HTTPS, no certificate management needed

**Always Use HTTPS = OFF**:
- Prevents Cloudflare from forcing HTTP→HTTPS redirects
- Your ingress controller (`ssl-redirect: false`) doesn't redirect
- **Result**: No redirect loop!

---

## 🚨 If Still Getting Redirects After This

### 1. Clear Browser Data Completely

**Chrome/Edge**:
```
1. Press F12 (Developer Tools)
2. Right-click the refresh button
3. Select "Empty Cache and Hard Reload"
```

**Or use Incognito**: `Ctrl + Shift + N`

### 2. Check for Cloudflare Page Rules

In Cloudflare Dashboard:
1. Go to: **Rules** > **Page Rules**
2. Look for any rules redirecting HTTP to HTTPS
3. **Disable** or **Delete** any such rules for 254carbon.com

### 3. Check Cloudflare Access Settings

1. Go to: **Zero Trust** > **Access** > **Applications**
2. Find: **Superset.254Carbon** (and others)
3. Click **Edit**
4. Check: **Session Duration** is not causing issues
5. Verify: **Application domain** is exactly: `superset.254carbon.com` (no extra paths)

### 4. Verify Tunnel Is Running

```bash
kubectl logs -n cloudflare-tunnel -l app.kubernetes.io/name=cloudflare-tunnel --tail=10

# Should show:
# INF Registered tunnel connection
```

---

## 🎯 Expected Behavior After Fix

### Portal
- URL: https://portal.254carbon.com
- Should load immediately without redirects
- Shows your Portal UI

### Superset  
- URL: https://superset.254carbon.com/superset/login
- Shows Cloudflare Access login page
- After login, shows Superset login
- Credentials: admin / admin

### DolphinScheduler
- URL: https://dolphinscheduler.254carbon.com
- Shows login directly
- Credentials: admin / admin

### Grafana
- URL: https://grafana.254carbon.com
- Shows Grafana login
- Use your configured credentials

---

## 🔧 Technical Details

### Current Infrastructure Setup

```
Browser (HTTPS)
    ↓
Cloudflare Edge (SSL Mode: Flexible)
    ↓
Cloudflare Tunnel (encrypted connection)
    ↓
Kubernetes Ingress NGINX (HTTP, ssl-redirect: false)
    ↓
Services (HTTP)
```

### Why API Tokens Don't Work

The tokens you provided:
- `DNS_API_TOKEN`: DNS records only ✓ (used successfully)
- `TUNNEL_EDIT_API_TOKEN`: Tunnel config only
- `APPS_API_TOKEN`: Zero Trust/Access only

**None have**: Zone Settings (SSL/TLS) permissions

**Solution**: Manual dashboard configuration (5 minutes, one-time)

---

## ✅ Summary of Manual Steps

1. **Cloudflare Dashboard** → 254carbon.com
2. **SSL/TLS** → Overview → Set mode to **Flexible**
3. **SSL/TLS** → Edge Certificates → **Always Use HTTPS = OFF**
4. **Wait 60 seconds**
5. **Clear browser cache** or use Incognito
6. **Test**: https://portal.254carbon.com

**Total time**: ~5 minutes

---

## 📞 After You Fix This

Once you can access the portal, we'll move on to:
1. ✅ Import DolphinScheduler workflows
2. ✅ Configure Superset dashboards
3. ✅ Set up data pipelines
4. ✅ Run first data quality checks

Let me know when you've made the Cloudflare changes! 🚀





