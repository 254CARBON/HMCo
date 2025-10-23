# API Keys Setup Guide

**Security First:** This guide explains how to securely configure API keys for the DolphinScheduler workflows without committing secrets to version control.

---

## 🔐 Security Approach

### What We Do
✅ **Environment Variables** - Keys loaded from environment  
✅ **Template File** - `api-keys.env.example` shows format  
✅ **Git Ignored** - `api-keys.env` excluded from version control  
✅ **No Hardcoding** - Scripts validate keys at runtime  
✅ **Production Ready** - Compatible with secrets managers

### What We Don't Do
❌ **No Hardcoded Keys** - Never commit actual API keys  
❌ **No Plain Text in Repo** - Only templates committed  
❌ **No Manual Entry** - Automated validation and loading

---

## 📋 Quick Start

### Step 1: Create Your API Keys File

```bash
cd /home/m/tff/254CARBON/HMCo
cp api-keys.env.example api-keys.env
```

### Step 2: Edit with Your Keys

```bash
# Use your preferred editor
nano api-keys.env
# or
vim api-keys.env
# or
code api-keys.env
```

### Step 3: Load and Use

```bash
# Load API keys into environment
source api-keys.env

# Verify keys are loaded
echo $ALPHAVANTAGE_API_KEY  # Should show your key

# Run automation
./scripts/setup-dolphinscheduler-complete.sh
```

---

## 🔑 Required API Keys

### 1. AlphaVantage
**Purpose:** Commodity futures data (CL, NG, HO, RB)  
**Get Key:** https://www.alphavantage.co/support/#api-key  
**Free Tier:** 25 requests/day  
**Variable:** `ALPHAVANTAGE_API_KEY`

### 2. Polygon.io
**Purpose:** Real-time market data  
**Get Key:** https://polygon.io/dashboard/api-keys  
**Free Tier:** 5 requests/minute  
**Variable:** `POLYGON_API_KEY`

### 3. EIA (Energy Information Administration)
**Purpose:** Energy prices and statistics  
**Get Key:** https://www.eia.gov/opendata/  
**Free Tier:** Unlimited  
**Variable:** `EIA_API_KEY`

### 4. GIE (Gas Infrastructure Europe)
**Purpose:** European gas storage data  
**Get Key:** https://www.gie.eu/transparency/  
**Free Tier:** Available  
**Variable:** `GIE_API_KEY`

### 5. US Census Bureau
**Purpose:** Economic indicators  
**Get Key:** https://api.census.gov/data/key_signup.html  
**Free Tier:** Unlimited  
**Variable:** `CENSUS_API_KEY`

### 6. NOAA
**Purpose:** Weather data  
**Get Key:** https://www.ncdc.noaa.gov/cdo-web/token  
**Free Tier:** 1000 requests/day  
**Variable:** `NOAA_API_KEY`

### 7. FRED (Optional)
**Purpose:** Federal Reserve economic data  
**Get Key:** https://fred.stlouisfed.org/docs/api/api_key.html  
**Free Tier:** Unlimited  
**Variable:** `FRED_API_KEY`

---

## 📝 api-keys.env Format

```bash
# AlphaVantage API Key
export ALPHAVANTAGE_API_KEY="your-actual-key-here"

# Polygon.io API Key
export POLYGON_API_KEY="your-actual-key-here"

# EIA API Key
export EIA_API_KEY="your-actual-key-here"

# GIE API Key
export GIE_API_KEY="your-actual-key-here"

# US Census API Key
export CENSUS_API_KEY="your-actual-key-here"

# NOAA API Key
export NOAA_API_KEY="your-actual-key-here"

# FRED API Key (Optional)
export FRED_API_KEY="your-actual-key-here"
```

---

## ✅ Validation

The automation scripts will validate that all required keys are present before proceeding:

```bash
source api-keys.env
./scripts/configure-dolphinscheduler-credentials.sh

# Output:
# ✓ ALPHAVANTAGE_API_KEY loaded
# ✓ POLYGON_API_KEY loaded
# ✓ EIA_API_KEY loaded
# ✓ GIE_API_KEY loaded
# ✓ CENSUS_API_KEY loaded
# ✓ NOAA_API_KEY loaded
```

If any keys are missing:
```bash
# Output:
# ✗ Missing required API keys:
#   - ALPHAVANTAGE_API_KEY
#   - POLYGON_API_KEY
# 
# Please provide API keys via environment variables
```

---

## 🏢 Production Deployment

### Option 1: Kubernetes Secrets (Recommended)

```bash
# Create secret from file
kubectl create secret generic dolphinscheduler-api-keys \
  --from-env-file=api-keys.env \
  --namespace=data-platform

# Or create from individual keys
kubectl create secret generic dolphinscheduler-api-keys \
  --from-literal=ALPHAVANTAGE_API_KEY="your-key" \
  --from-literal=POLYGON_API_KEY="your-key" \
  --namespace=data-platform
```

### Option 2: HashiCorp Vault

```bash
# Store in Vault
vault kv put secret/dolphinscheduler \
  ALPHAVANTAGE_API_KEY="your-key" \
  POLYGON_API_KEY="your-key"

# Load from Vault
export ALPHAVANTAGE_API_KEY=$(vault kv get -field=ALPHAVANTAGE_API_KEY secret/dolphinscheduler)
```

### Option 3: AWS Secrets Manager

```bash
# Store in AWS
aws secretsmanager create-secret \
  --name dolphinscheduler/api-keys \
  --secret-string file://api-keys.json

# Load from AWS
export ALPHAVANTAGE_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id dolphinscheduler/api-keys \
  --query SecretString --output text | jq -r .ALPHAVANTAGE_API_KEY)
```

### Option 4: CI/CD Environment Variables

```yaml
# GitHub Actions
env:
  ALPHAVANTAGE_API_KEY: ${{ secrets.ALPHAVANTAGE_API_KEY }}
  POLYGON_API_KEY: ${{ secrets.POLYGON_API_KEY }}

# GitLab CI
variables:
  ALPHAVANTAGE_API_KEY: $ALPHAVANTAGE_API_KEY
  POLYGON_API_KEY: $POLYGON_API_KEY
```

---

## 🔒 Security Best Practices

### ✅ Do's

1. **Keep api-keys.env local** - Never commit it
2. **Use .gitignore** - Already configured
3. **Rotate keys regularly** - Every 90 days recommended
4. **Use secrets manager in production** - Vault, AWS Secrets Manager, etc.
5. **Limit key permissions** - Use read-only keys where possible
6. **Monitor key usage** - Set up alerts for unusual activity

### ❌ Don'ts

1. **Don't commit api-keys.env** - Use .env.example instead
2. **Don't share keys in chat** - Use secure channels
3. **Don't store in plain text** - Use encryption at rest
4. **Don't use same keys for dev/prod** - Separate environments
5. **Don't embed in source code** - Always use environment variables
6. **Don't log API keys** - Sanitize logs

---

## 🐛 Troubleshooting

### Problem: Keys Not Loading

```bash
# Check if file exists
ls -la api-keys.env

# Check file permissions
chmod 600 api-keys.env  # Only owner can read/write

# Verify content
cat api-keys.env  # Should show export commands

# Try sourcing again
source api-keys.env

# Verify keys are loaded
env | grep API_KEY
```

### Problem: Invalid Key Format

```bash
# Keys should not have quotes in the actual value
# Wrong: export EIA_API_KEY="abc"123"  
# Right: export EIA_API_KEY="abc123"

# Keys should not have spaces
# Wrong: export EIA_API_KEY= "abc123"
# Right: export EIA_API_KEY="abc123"
```

### Problem: Script Still Asks for Keys

```bash
# Make sure you sourced the file in the SAME shell session
source api-keys.env

# Run script in same session
./scripts/setup-dolphinscheduler-complete.sh

# Or export inline
ALPHAVANTAGE_API_KEY="key" ./scripts/setup-dolphinscheduler-complete.sh
```

---

## 📚 Related Documentation

- [Workflow Import Guide](./WORKFLOW_IMPORT_GUIDE.md)
- [Implementation Summary](./IMPLEMENTATION_SUMMARY.md)
- [Scripts README](./scripts/README.md)

---

## 🎯 Summary

**Three Simple Steps:**

1. Copy template: `cp api-keys.env.example api-keys.env`
2. Add your keys: Edit `api-keys.env` with actual values
3. Load and run: `source api-keys.env && ./scripts/setup-dolphinscheduler-complete.sh`

**Security is built-in:**
- ✅ No manual entry required
- ✅ No hardcoded secrets
- ✅ Git-safe by default
- ✅ Production-ready

---

**Last Updated:** October 23, 2025  
**Status:** Production Ready ✅

