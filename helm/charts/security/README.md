# Security Policies Helm Chart

This Helm chart provides admission control policies for container image signing and SBOM verification using Kyverno.

## Overview

The security chart implements supply chain security measures by:

1. **Verifying Image Signatures**: Ensures container images are signed using Cosign keyless signing with Sigstore/Fulcio
2. **Verifying SBOM Attestations**: Ensures container images have Software Bill of Materials (SBOM) attestations
3. **Namespace-based Enforcement**: Applies policies selectively based on namespace patterns

## Prerequisites

- Kubernetes cluster with Kyverno installed
- Images signed with Cosign using keyless signing (OIDC)
- SBOM attestations created with `cosign attest`

## Installation

```bash
# Install Kyverno first if not already installed
helm repo add kyverno https://kyverno.github.io/kyverno/
helm repo update
helm install kyverno kyverno/kyverno --namespace kyverno --create-namespace

# Install security policies
helm install security ./helm/charts/security --namespace kyverno
```

## Configuration

### Image Signing Policy

The `imageSigning.enabled` value controls whether the image signing verification policy is active.

**Enforcement Mode:**
- `enforce`: Blocks deployments of unsigned images (production mode)
- `audit`: Logs policy violations without blocking (testing mode)

```yaml
imageSigning:
  enabled: true
  validationFailureAction: enforce
```

### Namespace Configuration

**Include Namespaces:**
Namespaces where the policy applies (supports glob patterns):

```yaml
imageSigning:
  includeNamespaces:
    - "*"  # Apply to all namespaces
```

**Exclude Namespaces:**
Namespaces exempt from image signing requirements:

```yaml
imageSigning:
  excludeNamespaces:
    - kube-system          # System namespace
    - kube-public          # Public namespace
    - kube-node-lease      # Node lease namespace
    - "*-dev"              # Development namespaces (e.g., myapp-dev)
    - "*-development"      # Development namespaces (e.g., myapp-development)
    - local-path-storage   # Local storage provisioner
    - ingress-nginx        # Ingress controller
    - cert-manager         # Certificate management
    - monitoring           # Monitoring stack
    - logging              # Logging stack
```

### Trusted Registries

Define registries that require image signing:

```yaml
imageSigning:
  trustedRegistries:
    - registry: "ghcr.io/254carbon/*"
      keyless: true  # Use Sigstore keyless verification
```

For keyless verification, images must be signed with:
- **Subject**: `https://github.com/254CARBON/*`
- **Issuer**: `https://token.actions.githubusercontent.com`
- **Transparency Log**: Rekor (https://rekor.sigstore.dev)

### Allowed Unsigned Registries

Registries from which unsigned images are permitted (typically for well-known base images and system components):

```yaml
imageSigning:
  allowedUnsignedRegistries:
    - "docker.io/library/*"      # Official Docker Hub images
    - "registry.k8s.io/*"        # Kubernetes official images
    - "quay.io/jetstack/*"       # Jetstack images (cert-manager)
    - "quay.io/prometheus/*"     # Prometheus images
    - "grafana/*"                # Grafana images
    - "prom/*"                   # Prometheus community images
```

## Exception List Documentation

### System Components

The following namespaces are excluded from image signing requirements to allow system components to function:

1. **kube-system**: Core Kubernetes system components (kube-proxy, coredns, etc.)
2. **kube-public**: Publicly accessible cluster information
3. **kube-node-lease**: Node heartbeat and lease management
4. **local-path-storage**: Local persistent volume provisioner
5. **ingress-nginx**: NGINX Ingress Controller
6. **cert-manager**: TLS certificate automation
7. **monitoring**: Prometheus, Grafana, and other monitoring tools
8. **logging**: Logging infrastructure (Fluentd, Loki, etc.)

### Development Environments

Development and testing environments are excluded using glob patterns:

- `*-dev`: Any namespace ending with `-dev` (e.g., `myapp-dev`, `backend-dev`)
- `*-development`: Any namespace ending with `-development` (e.g., `api-development`)

### Unsigned Image Registries

Images from the following registries are allowed without signature verification:

1. **docker.io/library/***: Official Docker Hub base images (e.g., `nginx`, `postgres`, `redis`)
2. **registry.k8s.io/***: Kubernetes official components and addons
3. **quay.io/jetstack/***: Cert-manager and related tools
4. **quay.io/prometheus/***: Prometheus operator and exporters
5. **grafana/***: Grafana dashboards and plugins
6. **prom/***: Prometheus community exporters and tools

**Rationale**: These are well-established, trusted registries providing core infrastructure components that may not support Cosign signing.

## Policy Behavior

### Image Signature Verification

When a Pod is created or updated:

1. **Signed Images**: The policy verifies the signature using Cosign and Sigstore
   - Checks the signature against the Rekor transparency log
   - Validates the signing identity (GitHub Actions OIDC)
   - Allows deployment if verification succeeds

2. **Unsigned Images from Trusted Registries**: Deployment is blocked

3. **Images from Allowed Unsigned Registries**: Deployment is allowed without verification

4. **Images from Unknown Registries**: Deployment is blocked

### SBOM Attestation Verification

When a Pod is created or updated in production namespaces:

1. Verifies that an SBOM attestation is attached to the image
2. Validates the attestation signature using Cosign
3. Uses `failurePolicy: Ignore` to avoid blocking deployments if SBOM is missing (warning mode)

## Verification Commands

After deploying images, verify signatures and attestations:

```bash
# Verify image signature
cosign verify ghcr.io/254carbon/hmco-backend@sha256:abc123...

# Verify SBOM attestation
cosign verify-attestation --type spdx ghcr.io/254carbon/hmco-backend@sha256:abc123...

# Extract SBOM from attestation
cosign verify-attestation --type spdx ghcr.io/254carbon/hmco-backend@sha256:abc123... | jq -r '.payload' | base64 -d | jq .
```

## Testing the Policy

### Audit Mode

Start with audit mode to test the policy without blocking deployments:

```yaml
imageSigning:
  validationFailureAction: audit
```

Review policy violations in Kyverno:

```bash
kubectl get policyreport -A
kubectl describe policyreport <report-name> -n <namespace>
```

### Enforce Mode

Once testing is complete, switch to enforce mode:

```yaml
imageSigning:
  validationFailureAction: enforce
```

## Troubleshooting

### Deployment Blocked

If a deployment is blocked by the policy:

1. Check if the namespace is in the exclude list
2. Verify the image is signed: `cosign verify <image>`
3. Check if the image registry is in `trustedRegistries` or `allowedUnsignedRegistries`
4. Review Kyverno logs: `kubectl logs -n kyverno -l app.kubernetes.io/name=kyverno`

### Adding Exceptions

To add a namespace exception:

```yaml
imageSigning:
  excludeNamespaces:
    - my-special-namespace
```

To allow unsigned images from a new registry:

```yaml
imageSigning:
  allowedUnsignedRegistries:
    - "myregistry.io/*"
```

## Security Considerations

1. **Keyless Signing**: Uses GitHub Actions OIDC tokens for identity verification
2. **Transparency Log**: All signatures recorded in Rekor for auditability
3. **Namespace Isolation**: Different enforcement levels for dev vs. prod
4. **Grace Period**: SBOM attestation uses `Ignore` failure policy during rollout

## Maintenance

### Updating Excluded Namespaces

Review and update the exclude list quarterly to ensure:
- Deprecated namespaces are removed
- New system namespaces are added
- Dev/test namespace patterns are current

### Updating Trusted Registries

When adding new trusted registries:
1. Document the business justification
2. Verify the registry supports Cosign signing
3. Test in audit mode first
4. Update the `trustedRegistries` list

## References

- [Kyverno Documentation](https://kyverno.io/)
- [Cosign Documentation](https://docs.sigstore.dev/cosign/)
- [Sigstore](https://www.sigstore.dev/)
- [SPDX SBOM Specification](https://spdx.dev/)
