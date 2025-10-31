# Cloudflare Origin Certificates Setup Guide

**Purpose**: Configure production-grade SSL/TLS certificates for 254carbon.com using Cloudflare Origin Certificates  
**Last Updated**: October 21, 2025  
**Status**: Implementation Required

---

## Overview

Cloudflare Origin Certificates provide free, long-lived (up to 15 years) SSL/TLS certificates specifically designed for securing the connection between Cloudflare's edge and your origin server (Kubernetes cluster). This is the recommended approach for production environments.

### Benefits

- **Long Validity**: Up to 15 years (vs 90 days for Let's Encrypt)
- **Zero Maintenance**: No renewal automation needed
- **Cloudflare Integration**: Designed specifically for Cloudflare Tunnel
- **Strong Security**: Full SSL/TLS encryption from edge to origin
- **Cost**: Free with any Cloudflare plan

---

## Step 1: Generate Origin Certificate in Cloudflare Dashboard

### 1.1 Navigate to SSL/TLS Settings

1. Log into Cloudflare Dashboard: https://dash.cloudflare.com/
2. Select your domain: **254carbon.com**
3. Go to **SSL/TLS** → **Origin Server**
4. Click **Create Certificate**

### 1.2 Configure Certificate

**Settings**:
- **Private key type**: RSA (2048) - recommended
- **Certificate signing request (CSR)**: Generate private key and CSR with Cloudflare
- **Hostnames**: 
  - `*.254carbon.com` (wildcard for all subdomains)
  - `254carbon.com` (root domain)
- **Certificate validity**: 15 years
- **Signature algorithm**: SHA-256 (default)

Click **Create** to generate the certificate.

### 1.3 Save Certificate and Private Key

⚠️ **IMPORTANT**: The private key is shown only once. You must save it immediately.

You will receive:
1. **Origin Certificate** (Public certificate)
2. **Private Key** (Secret - save securely!)

Download or copy both to secure locations:
- `254carbon-origin.pem` (certificate)
- `254carbon-origin-key.pem` (private key)

---

## Step 2: Create Kubernetes Secret

### 2.1 Prepare Certificate Files

On your local machine or jump host:

```bash
# Create directory for certificates
mkdir -p ~/cloudflare-certs
cd ~/cloudflare-certs

# Save the certificate (copy from Cloudflare dashboard)
cat > 254carbon-origin.pem << 'EOF'
-----BEGIN CERTIFICATE-----
[PASTE YOUR ORIGIN CERTIFICATE HERE]
-----END CERTIFICATE-----
EOF

# Save the private key (copy from Cloudflare dashboard)
cat > 254carbon-origin-key.pem << 'EOF'
-----BEGIN PRIVATE KEY-----
[PASTE YOUR PRIVATE KEY HERE]
-----END PRIVATE KEY-----
EOF

# Secure the files
chmod 600 *.pem
```

### 2.2 Create Kubernetes TLS Secrets

Create secrets for each namespace that needs SSL/TLS:

```bash
# For data-platform namespace
kubectl create secret tls cloudflare-origin-cert \
  --cert=254carbon-origin.pem \
  --key=254carbon-origin-key.pem \
  -n data-platform \
  --dry-run=client -o yaml | kubectl apply -f -

# For monitoring namespace
kubectl create secret tls cloudflare-origin-cert \
  --cert=254carbon-origin.pem \
  --key=254carbon-origin-key.pem \
  -n monitoring \
  --dry-run=client -o yaml | kubectl apply -f -

# For registry namespace
kubectl create secret tls cloudflare-origin-cert \
  --cert=254carbon-origin.pem \
  --key=254carbon-origin-key.pem \
  -n registry \
  --dry-run=client -o yaml | kubectl apply -f -

# For any other namespaces with ingress
# kubectl create secret tls cloudflare-origin-cert ...
```

### 2.3 Verify Secret Creation

```bash
# List secrets
kubectl get secret cloudflare-origin-cert -n data-platform
kubectl get secret cloudflare-origin-cert -n monitoring
kubectl get secret cloudflare-origin-cert -n registry

# Verify certificate details
kubectl get secret cloudflare-origin-cert -n data-platform -o jsonpath='{.data.tls\.crt}' | base64 -d | openssl x509 -text -noout | grep -A2 "Validity"
```

---

## Step 3: Update Ingress Resources

### 3.1 Update Existing Ingresses

For each ingress that uses TLS, update to reference the new Cloudflare origin certificate:

#### Example: Portal Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: portal-ingress
  namespace: data-platform
  annotations:
    kubernetes.io/ingress.class: nginx
    # Remove cert-manager annotations if present
    # cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - portal.254carbon.com
    secretName: cloudflare-origin-cert  # Updated to use Cloudflare cert
  rules:
  - host: portal.254carbon.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: portal
            port:
              number: 8080
```

### 3.2 Automated Update Script

Use the provided script to update all ingresses:

```bash
cd /home/m/tff/254CARBON/HMCo

# This script will update all ingress resources
./scripts/update-ingress-certificates.sh cloudflare-origin-cert
```

Or manually update each ingress:

```bash
# Update portal ingress
kubectl patch ingress portal-ingress -n data-platform --type='json' \
  -p='[{"op": "replace", "path": "/spec/tls/0/secretName", "value": "cloudflare-origin-cert"}]'

# Update grafana ingress
kubectl patch ingress grafana-ingress -n monitoring --type='json' \
  -p='[{"op": "replace", "path": "/spec/tls/0/secretName", "value": "cloudflare-origin-cert"}]'

# Update other ingresses similarly...
```

### 3.3 Remove cert-manager Annotations

If ingresses have cert-manager annotations, remove them:

```bash
# Remove cert-manager annotations from all ingresses
kubectl annotate ingress --all cert-manager.io/cluster-issuer- -n data-platform
kubectl annotate ingress --all cert-manager.io/issuer- -n data-platform
```

---

## Step 4: Configure Cloudflare SSL/TLS Settings

### 4.1 Set SSL/TLS Encryption Mode

1. Go to Cloudflare Dashboard → **254carbon.com** → **SSL/TLS** → **Overview**
2. Set encryption mode to: **Full (strict)**
   - This ensures end-to-end encryption with certificate validation
   - Cloudflare will verify your origin certificate

### 4.2 Configure Additional Settings

**Recommended settings**:

1. **SSL/TLS** → **Edge Certificates**:
   - **Always Use HTTPS**: ON
   - **Minimum TLS Version**: TLS 1.2
   - **Opportunistic Encryption**: ON
   - **TLS 1.3**: ON
   - **Automatic HTTPS Rewrites**: ON

2. **SSL/TLS** → **Origin Server**:
   - **Authenticated Origin Pulls**: ON (REQUIRED for production hardening)
     - This ensures only Cloudflare can reach your origin server
     - Even with Cloudflare Tunnel, this provides an additional security layer
     - See detailed setup in Section 6 below

---

## Step 5: Verification

### 5.1 Test Certificate Installation

```bash
# Test from outside the cluster
curl -vI https://portal.254carbon.com 2>&1 | grep -A5 "SSL connection"

# Check certificate details
echo | openssl s_client -connect portal.254carbon.com:443 -servername portal.254carbon.com 2>/dev/null | openssl x509 -noout -issuer -subject -dates
```

### 5.2 Verify All Services

Test each service endpoint:

```bash
for service in portal grafana superset datahub trino vault minio harbor; do
  echo -n "$service.254carbon.com: "
  curl -s -o /dev/null -w "%{http_code} - SSL: %{ssl_verify_result}\n" https://$service.254carbon.com
done
```

Expected output: `200 - SSL: 0` (or 302 for redirects)

### 5.3 Browser Verification

1. Open https://portal.254carbon.com in a browser
2. Click the padlock icon in the address bar
3. Verify:
   - Connection is secure
   - Certificate is valid
   - Certificate issuer is Cloudflare

---

## Step 6: Enable Authenticated Origin Pulls (Security Hardening)

Authenticated Origin Pulls ensures that **only Cloudflare** can connect to your origin server by validating client certificates. This prevents direct attacks on your origin infrastructure.

### 6.1 Download Cloudflare Origin CA Certificate

The Cloudflare authenticated origin pull certificate is publicly available and must be installed on your origin server:

```bash
# Download the Cloudflare Origin CA certificate
curl -o ~/cloudflare-certs/origin-pull-ca.pem \
  https://developers.cloudflare.com/ssl/static/authenticated_origin_pull_ca.pem

# Verify the certificate fingerprint
openssl x509 -in ~/cloudflare-certs/origin-pull-ca.pem -noout -fingerprint -sha256

# Verify the certificate details
openssl x509 -in ~/cloudflare-certs/origin-pull-ca.pem -text -noout | head -20
```

Expected issuer: `CN=Cloudflare Inc ECC CA-3, O=Cloudflare, Inc., C=US`

**IMPORTANT Security Note**: 
- **ALWAYS** verify the certificate fingerprint against the **current** official [Cloudflare documentation](https://developers.cloudflare.com/ssl/origin-configuration/authenticated-origin-pull/set-up/) before using it in production
- Do NOT rely solely on example fingerprints in this document as they may become outdated
- Certificate fingerprints are published by Cloudflare and should be verified from their official sources
- Check the certificate's validity period and ensure it has not expired

### 6.2 Create Kubernetes Secret for Origin Pull CA

```bash
# Create secret in ingress-nginx namespace (where NGINX Ingress Controller runs)
kubectl create secret generic cloudflare-origin-pull-ca \
  --from-file=ca.crt=~/cloudflare-certs/origin-pull-ca.pem \
  -n ingress-nginx \
  --dry-run=client -o yaml | kubectl apply -f -

# Verify secret creation
kubectl get secret cloudflare-origin-pull-ca -n ingress-nginx
```

### 6.3 Update NGINX Ingress Controller Configuration

Add the Cloudflare origin pull CA to NGINX Ingress Controller to verify client certificates:

```bash
# Update the ingress-nginx ConfigMap
kubectl patch configmap ingress-nginx-controller \
  -n ingress-nginx \
  --type merge \
  -p '{"data":{"ssl-client-cert":"/etc/ingress-controller/ssl/cloudflare-origin-pull-ca/ca.crt","enable-ssl-chain-completion":"false"}}'

# Mount the secret in the ingress controller deployment
# This is typically done via the helm values or by patching the deployment
# Check if already mounted:
kubectl get deployment ingress-nginx-controller -n ingress-nginx -o yaml | grep cloudflare-origin-pull-ca

# If not present, you'll need to update the deployment to mount this secret
# This is usually handled by updating the ingress-nginx helm values
```

Alternative approach using Helm values (recommended):

```yaml
# ingress-nginx-values.yaml
controller:
  extraVolumes:
    - name: cloudflare-origin-pull-ca
      secret:
        secretName: cloudflare-origin-pull-ca
  extraVolumeMounts:
    - name: cloudflare-origin-pull-ca
      mountPath: /etc/ingress-controller/ssl/cloudflare-origin-pull-ca
      readOnly: true
  config:
    ssl-client-cert: /etc/ingress-controller/ssl/cloudflare-origin-pull-ca/ca.crt
    enable-ssl-chain-completion: "false"
```

Then apply:

```bash
helm upgrade ingress-nginx ingress-nginx/ingress-nginx \
  -n ingress-nginx \
  -f ingress-nginx-values.yaml
```

### 6.4 Enable Authenticated Origin Pulls in Cloudflare Dashboard

1. Log into Cloudflare Dashboard: https://dash.cloudflare.com/
2. Select your domain: **254carbon.com**
3. Go to **SSL/TLS** → **Origin Server**
4. Scroll to **Authenticated Origin Pulls**
5. Toggle **ON** (this enables it for all origins)

For per-hostname control (advanced):
- Use Cloudflare API to enable per-zone or per-hostname authenticated origin pulls
- See: https://developers.cloudflare.com/ssl/origin-configuration/authenticated-origin-pull/

### 6.5 Verify Authenticated Origin Pulls

After enabling, test that direct connections (bypassing Cloudflare) are rejected:

```bash
# This should FAIL (direct connection without Cloudflare client cert)
curl -v https://portal.254carbon.com --resolve portal.254carbon.com:443:YOUR_ORIGIN_IP 2>&1 | grep -i "ssl"

# This should SUCCEED (via Cloudflare)
curl -v https://portal.254carbon.com 2>&1 | grep -i "200 OK"
```

Check NGINX logs for client certificate validation:

```bash
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx | grep -i "certificate"
```

### 6.6 Troubleshooting Authenticated Origin Pulls

**Issue**: Services return 400/495/496 errors after enabling

**Solution**: Ensure the Cloudflare Origin Pull CA certificate is correctly mounted in NGINX

```bash
# Check if certificate is accessible in NGINX pod
kubectl exec -n ingress-nginx deployment/ingress-nginx-controller -- \
  ls -l /etc/ingress-controller/ssl/cloudflare-origin-pull-ca/

# Verify certificate content
kubectl exec -n ingress-nginx deployment/ingress-nginx-controller -- \
  cat /etc/ingress-controller/ssl/cloudflare-origin-pull-ca/ca.crt | openssl x509 -text -noout | grep Issuer
```

**Issue**: Some requests fail intermittently

**Solution**: This might indicate some requests are bypassing Cloudflare. Verify:
1. All DNS records are proxied (orange cloud) in Cloudflare
2. No direct IP access to origin
3. Firewall rules block non-Cloudflare IPs

---

## Step 7: Clean Up Old Certificates

### 7.1 Remove cert-manager Generated Certificates (Optional)

If you're fully migrating to Cloudflare Origin Certificates:

```bash
# List all cert-manager certificates
kubectl get certificate -A

# Delete old certificates (they're no longer needed)
kubectl delete certificate --all -n data-platform
kubectl delete certificate --all -n monitoring

# Optionally, keep cert-manager for future use or remove it
# kubectl delete namespace cert-manager
```

### 7.2 Delete Old TLS Secrets

```bash
# List all TLS secrets
kubectl get secrets -A | grep tls

# Delete old Let's Encrypt or self-signed secrets
# kubectl delete secret <old-secret-name> -n <namespace>
```

---

## Certificate Management

### Renewal Process

**Good news**: Cloudflare Origin Certificates are valid for up to 15 years!

**Set a reminder** to regenerate the certificate 14 years from now:
- Reminder date: **October 21, 2039**
- Process: Repeat Steps 1-3 above

### Rotation Best Practices

Even though certificates are long-lived, consider rotating them:
- Every 2-5 years for security best practices
- After any security incident
- When changing infrastructure significantly

To rotate:
1. Generate new certificate in Cloudflare dashboard
2. Create new Kubernetes secret (different name)
3. Update ingresses to use new secret
4. Test thoroughly
5. Delete old secret after confirming all services work

### Backup

⚠️ **Critical**: Back up your certificate and private key securely!

```bash
# Encrypt and backup
cd ~/cloudflare-certs
tar czf 254carbon-certs-$(date +%Y%m%d).tar.gz *.pem
gpg --symmetric 254carbon-certs-$(date +%Y%m%d).tar.gz
rm 254carbon-certs-$(date +%Y%m%d).tar.gz

# Store the encrypted .gpg file in multiple secure locations:
# - Password manager
# - Secure cloud storage
# - Offline encrypted backup
```

---

## Troubleshooting

### Issue: SSL Handshake Failures

**Symptoms**: `SSL handshake failed` or certificate verification errors

**Solutions**:
1. Verify Cloudflare SSL/TLS mode is "Full (strict)"
2. Confirm secret is created in the correct namespace
3. Check ingress references the correct secret name
4. Verify certificate includes both wildcard and root domain

```bash
# Debug certificate
kubectl get secret cloudflare-origin-cert -n data-platform -o jsonpath='{.data.tls\.crt}' | base64 -d | openssl x509 -text -noout
```

### Issue: Certificate Not Found

**Symptoms**: Ingress controller can't find the TLS secret

**Solutions**:
1. Verify secret exists in the same namespace as ingress
2. Check secret name matches ingress specification
3. Confirm secret type is `kubernetes.io/tls`

```bash
# Check secret in namespace
kubectl get secret cloudflare-origin-cert -n data-platform -o yaml
```

### Issue: Mixed Content Warnings

**Symptoms**: Browser shows mixed content warnings

**Solutions**:
1. Enable "Automatic HTTPS Rewrites" in Cloudflare
2. Update application to use relative URLs or `https://`
3. Add HSTS headers via NGINX ingress

---

## Security Best Practices

1. **Never commit private keys** to version control
2. **Restrict access** to certificate files (chmod 600)
3. **Use RBAC** to limit who can read secrets in Kubernetes
4. **Enable audit logging** for secret access
5. **Rotate certificates** every few years even if not expired
6. **Monitor certificate expiration** (set calendar reminders)
7. **Test certificate renewal process** annually

---

## Automated Script

A helper script is provided for the entire process:

```bash
# Run the automated setup script
cd /home/m/tff/254CARBON/HMCo
./scripts/setup-cloudflare-origin-certs.sh

# This script will:
# 1. Prompt for certificate and key files
# 2. Create secrets in all namespaces
# 3. Update all ingresses
# 4. Verify the setup
# 5. Generate verification report
```

---

## References

- [Cloudflare Origin CA](https://developers.cloudflare.com/ssl/origin-configuration/origin-ca/)
- [Kubernetes TLS Secrets](https://kubernetes.io/docs/concepts/configuration/secret/#tls-secrets)
- [NGINX Ingress TLS](https://kubernetes.github.io/ingress-nginx/user-guide/tls/)

---

## Next Steps

After completing this setup:

1. ✅ Verify all services are accessible via HTTPS
2. ✅ Confirm certificate validity in browsers
3. ✅ Test SSL/TLS grade at https://www.ssllabs.com/ssltest/
4. ✅ Update monitoring to alert on certificate expiration
5. ✅ Document certificate location and backup procedures
6. ✅ Set calendar reminder for renewal in 14 years

---

**Status**: Ready for implementation  
**Estimated Time**: 30-45 minutes  
**Risk Level**: Low (can roll back easily)
