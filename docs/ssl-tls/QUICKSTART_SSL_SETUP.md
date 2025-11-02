# Quick SSL/TLS Setup Guide

## Current Status

5 out of 6 certificates are failing due to ACME HTTP-01 challenges not completing. This is likely due to Cloudflare Tunnel configuration or ingress-nginx setup.

## Option 1: Fix ACME Challenges (Automated, Free)

### Diagnosis

```bash
# Check certificate status
kubectl get certificate -A

# Check ACME solver pods
kubectl get pods -n data-platform | grep cm-acme-http-solver

# Check if ACME solver is accessible
kubectl get ingress -n data-platform | grep cm-acme-http-solver
```

### Solution

The issue is that Let's Encrypt cannot reach the ACME HTTP-01 challenge endpoints through Cloudflare Tunnel.

**Fix**: Configure Cloudflare to allow /.well-known/acme-challenge paths or use DNS-01 challenge instead.

#### Automated Cloudflare Access exemption + renewal

```bash
# 1. Reconcile Access apps and add ACME bypass policies
CLOUDFLARE_API_TOKEN=... \
CLOUDFLARE_ACCOUNT_ID=... \
./scripts/create-cloudflare-access-apps.sh \
  --mode zone \
  --zone-domain 254carbon.com \
  --force

# 2. Trigger new ACME orders once bypass is in place
./scripts/reissue-letsencrypt-certs.sh \
  --cert data-platform/spark-history-tls \
  --cert data-platform/graphql-gateway-tls \
  --wait
```

Step 1 creates dedicated Access applications scoped to `/.well-known/acme-challenge/*` for each hostname and attaches a `decision: bypass` policy so Let’s Encrypt can fetch challenge tokens without authenticating. Step 2 forces cert-manager to renew the affected certificates after the bypass is active.

## Option 2: Cloudflare Origin Certificates (Recommended)

Cloudflare Origin Certificates provide free, 15-year certificates specifically designed for use with Cloudflare.

### Step 1: Generate Origin Certificate

1. Log into Cloudflare Dashboard
2. Go to SSL/TLS > Origin Server
3. Click "Create Certificate"
4. Settings:
   - Private key type: RSA (2048)
   - Hostnames: `*.254carbon.com, 254carbon.com`
   - Certificate Validity: 15 years
5. Click "Create"
6. **Save both the certificate and private key**

### Step 2: Create Kubernetes Secrets

```bash
# Save certificate and key to files
cat > origin-cert.pem << 'EOF'
-----BEGIN CERTIFICATE-----
<paste certificate here>
-----END CERTIFICATE-----
EOF

cat > origin-key.pem << 'EOF'
-----BEGIN PRIVATE KEY-----
<paste private key here>
-----END PRIVATE KEY-----
EOF

# Create secrets for each namespace/service
kubectl create secret tls portal-tls \
  --cert=origin-cert.pem \
  --key=origin-key.pem \
  -n data-platform

kubectl create secret tls datahub-tls \
  --cert=origin-cert.pem \
  --key=origin-key.pem \
  -n data-platform

kubectl create secret tls superset-tls \
  --cert=origin-cert.pem \
  --key=origin-key.pem \
  -n data-platform

kubectl create secret tls trino-tls \
  --cert=origin-cert.pem \
  --key=origin-key.pem \
  -n data-platform

kubectl create secret tls grafana-tls \
  --cert=origin-cert.pem \
  --key=origin-key.pem \
  -n monitoring

kubectl create secret tls harbor-tls \
  --cert=origin-cert.pem \
  --key=origin-key.pem \
  -n registry
```

### Step 3: Update Ingress Resources

The ingress resources are already configured to use these secrets. Once you create the secrets, the certificates will be automatically applied.

### Step 4: Verify

```bash
# Check secrets are created
kubectl get secrets -A | grep tls

# Test HTTPS access
curl -I https://portal.254carbon.com
curl -I https://datahub.254carbon.com
curl -I https://grafana.254carbon.com
```

## Option 3: Self-Signed Certificates (Quick Testing)

For quick testing only (browsers will show warnings):

```bash
# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout selfsigned-key.pem \
  -out selfsigned-cert.pem \
  -subj "/CN=*.254carbon.com/O=254Carbon"

# Create secrets (same as Option 2 but with selfsigned files)
```

## Recommended Approach

**Use Option 2 (Cloudflare Origin Certificates)** because:

1. ✓ Free and valid for 15 years
2. ✓ No renewal needed
3. ✓ Designed for Cloudflare Tunnel
4. ✓ Full browser trust
5. ✓ Works immediately

## Cloudflare SSL/TLS Settings

In Cloudflare Dashboard > SSL/TLS:

1. **SSL/TLS encryption mode**: Full (strict)
2. **Always Use HTTPS**: On
3. **Minimum TLS Version**: 1.2
4. **Opportunistic Encryption**: On
5. **TLS 1.3**: Enabled

## Troubleshooting

### Certificates Not Applied

```bash
# Delete old certificate requests
kubectl delete certificate --all -n data-platform

# Ingress will automatically pick up the new secrets
kubectl get ingress -n data-platform
```

### Browser Still Shows Insecure

1. Check Cloudflare SSL mode is "Full (strict)"
2. Clear browser cache
3. Check certificate in browser (should show Cloudflare Origin)
4. Verify secret is mounted: `kubectl describe ingress portal-ingress -n data-platform`

### ACME Solvers Still Running

```bash
# These will automatically clean up after certificate objects are deleted
kubectl delete certificate --all -n data-platform
kubectl delete ingress -n data-platform $(kubectl get ingress -n data-platform -o name | grep cm-acme-http-solver)
```

## Automation Script

Created automation helper: `k8s/certificates/cloudflare-origin-cert-setup.yaml`

This includes:
- ServiceAccount with proper permissions
- ConfigMap with certificate creation script
- Job template for automated deployment

## Next Steps

1. Generate Cloudflare Origin Certificate (5 minutes)
2. Create Kubernetes secrets (2 minutes)
3. Verify HTTPS access (1 minute)

Total time: ~10 minutes

## Security Notes

- Store certificate files securely (they're valid for 15 years)
- Use Kubernetes secrets (not ConfigMaps) for certificates
- Rotate certificates before expiry
- Monitor certificate expiration dates






