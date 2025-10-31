# Supply Chain Security Implementation

This document describes the supply chain hardening improvements implemented in this repository.

## Overview

Three key security measures have been implemented to enhance supply chain security:

1. **Pinned Base Images and SBOMs** - All container base images are pinned by digest, and SBOMs are generated for all images
2. **Keyless Image Signing and SBOM Attestation** - Images are signed using Cosign with Sigstore/Fulcio OIDC
3. **Admission Policy for Signed Images** - Kyverno policies enforce signed image requirements in production namespaces

## T1.1: Build Images and SBOMs for All Containers

### Base Image Pinning

All Dockerfiles have been updated to pin base images by digest to prevent supply chain attacks via image tag hijacking:

- **backend/Dockerfile**: 
  - `node:20@sha256:c11ae157cdd9f8b522d5a65e7f3f5f5c34cf45a8bd883c15e8f2028a2673dec7`
  - `node:20-slim@sha256:cba1d7bb8433bb920725193cd7d95d09688fb110b170406f7d4de948562f9850`

- **services/portal-services/Dockerfile**: 
  - `node:lts-alpine@sha256:f36fed0b2129a8492535e2853c64fbdbd2d29dc1219ee3217023ca48aebd3787`

- **docker/jupyter-notebook/Dockerfile**: 
  - `jupyter/datascience-notebook:latest@sha256:476c6e673e7d5d8b5059f8680b1c6a988942a79263da651bf302dc696ab311f2`

### SBOM Generation

The CI workflow generates SBOMs in SPDX JSON format for all container images:

- `sbom.backend.spdx.json` - Backend service SBOM
- `sbom.portal-services.spdx.json` - Portal services SBOM
- `sbom.jupyter-notebook.spdx.json` - Jupyter notebook SBOM

SBOMs are:
- Generated using Syft
- Uploaded as CI artifacts with 90-day retention
- Signed with Cosign
- Attested to their respective container images

### Verification

Check that SBOMs are generated in CI:

```bash
# Download SBOM artifacts from GitHub Actions
gh run download <run-id> -n sbom-files

# Verify SBOM content
cat sbom.backend.spdx.json | jq .
```

## T1.2: Keyless Image Signing and SBOM Attestation

### Implementation

Images are signed using Cosign with keyless signing via GitHub Actions OIDC:

- **Identity Provider**: GitHub Actions OIDC (`https://token.actions.githubusercontent.com`)
- **Subject**: `https://github.com/254CARBON/*`
- **Transparency Log**: Rekor (https://rekor.sigstore.dev)

The workflow performs:
1. Build and push container images to GHCR
2. Generate SBOM for each image using Syft
3. Sign images with `cosign sign --yes $IMAGE_DIGEST`
4. Attest SBOMs with `cosign attest --yes --predicate <sbom> --type spdx $IMAGE_DIGEST`

### Verification

Verify image signatures and attestations:

```bash
# Set COSIGN_EXPERIMENTAL for keyless verification
export COSIGN_EXPERIMENTAL=1

# Verify image signature
cosign verify ghcr.io/254carbon/hmco-backend@sha256:...

# Verify SBOM attestation
cosign verify-attestation --type spdx ghcr.io/254carbon/hmco-backend@sha256:...

# Extract and view SBOM from attestation
cosign verify-attestation --type spdx ghcr.io/254carbon/hmco-backend@sha256:... | \
  jq -r '.payload' | base64 -d | jq .predicate
```

### CI/CD Integration

The `.github/workflows/supplychain-and-e2e.yml` workflow:
- Runs on push to main and develop branches
- Has OIDC permissions (`id-token: write`)
- Uses Cosign installer action
- Signs images only on main branch (not on PRs)
- Provides verification commands in the workflow summary

## T1.3: Admission Policy to Require Signed Images

### Kyverno Policy

A Helm chart has been created at `helm/charts/security` that deploys Kyverno policies to enforce image signing requirements.

**Policy: `verify-image-signatures`**
- Verifies container images are signed with Cosign
- Uses keyless verification with GitHub Actions OIDC
- Validates signatures against Rekor transparency log
- Enforcement mode: `enforce` (blocks unsigned images) or `audit` (logs violations)

**Policy: `verify-sbom-attestations`**
- Verifies container images have SBOM attestations
- Uses same keyless verification mechanism
- Failure policy: `Ignore` (warning mode during rollout)

### Namespace Configuration

**Enforcement applies to:**
- All namespaces (`*`) by default

**Excluded namespaces (exception list):**
- `kube-system` - Kubernetes system components
- `kube-public` - Public cluster information
- `kube-node-lease` - Node heartbeat management
- `*-dev` - Development namespaces (e.g., `myapp-dev`)
- `*-development` - Development namespaces (e.g., `api-development`)
- `local-path-storage` - Local volume provisioner
- `ingress-nginx` - Ingress controller
- `cert-manager` - Certificate management
- `monitoring` - Monitoring stack (Prometheus, Grafana)
- `logging` - Logging infrastructure

**Allowed unsigned registries:**
- `docker.io/library/*` - Official Docker Hub images
- `registry.k8s.io/*` - Kubernetes official images
- `quay.io/jetstack/*` - Cert-manager images
- `quay.io/prometheus/*` - Prometheus images
- `grafana/*` - Grafana images
- `prom/*` - Prometheus community images

### Installation

```bash
# Install Kyverno
helm repo add kyverno https://kyverno.github.io/kyverno/
helm install kyverno kyverno/kyverno --namespace kyverno --create-namespace

# Install security policies
helm install security ./helm/charts/security --namespace kyverno
```

### Testing

Test in audit mode first:

```bash
# Install with audit mode
helm install security ./helm/charts/security \
  --namespace kyverno \
  --set imageSigning.validationFailureAction=audit

# Check policy reports
kubectl get policyreport -A
kubectl describe policyreport <report-name> -n <namespace>

# Upgrade to enforce mode when ready
helm upgrade security ./helm/charts/security \
  --namespace kyverno \
  --set imageSigning.validationFailureAction=enforce
```

### Exception List Maintenance

The exception list is documented in:
- `helm/charts/security/values.yaml` - Configuration
- `helm/charts/security/README.md` - Detailed documentation

To add exceptions:

```yaml
# In values.yaml or override file
imageSigning:
  excludeNamespaces:
    - my-special-namespace
  
  allowedUnsignedRegistries:
    - "myregistry.io/*"
```

## Verification Checklist

- [x] All Dockerfiles use pinned base images with digest
- [x] CI workflow generates SBOMs for all container images
- [x] SBOM artifacts are uploaded on every CI run
- [x] Images are signed with Cosign on main branch
- [x] SBOMs are attested to images using `cosign attest`
- [x] Verification commands documented in workflow summary
- [x] Kyverno policy chart created at `helm/charts/security`
- [x] Policy enforces signed images in production namespaces
- [x] Exception list for dev namespaces documented
- [x] Exception list for trusted unsigned registries documented

## Security Considerations

1. **Keyless Signing**: Uses GitHub Actions OIDC tokens, eliminating the need to manage signing keys
2. **Transparency**: All signatures are recorded in Rekor public transparency log
3. **Auditability**: SBOMs provide transparency into software components
4. **Defense in Depth**: Multiple layers (pinned images, signatures, admission control)
5. **Gradual Rollout**: Audit mode allows testing before enforcement

## Maintenance

### Updating Base Images

When updating base images:

```bash
# Pull the new image
docker pull node:20

# Get the digest
docker inspect node:20 | jq -r '.[0].RepoDigests[0]'

# Update Dockerfile with new digest
# FROM node:20@sha256:<new-digest>
```

### Rotating Exception List

Review exception lists quarterly:
- Remove deprecated namespaces
- Add new system components
- Validate dev namespace patterns
- Document changes in git commit

### Monitoring

Monitor policy violations:

```bash
# View policy reports
kubectl get policyreport -A -o wide

# Check for violations
kubectl get clusterpolicyreport -o json | \
  jq '.items[] | select(.results[].result=="fail")'
```

## References

- [Sigstore](https://www.sigstore.dev/)
- [Cosign](https://docs.sigstore.dev/cosign/)
- [Kyverno](https://kyverno.io/)
- [SPDX](https://spdx.dev/)
- [Syft](https://github.com/anchore/syft)
