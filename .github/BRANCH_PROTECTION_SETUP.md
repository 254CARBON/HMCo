# Branch Protection Setup Guide

This document provides instructions for setting up branch protection rules on the `main` branch to enforce the required status checks.

## Overview

Branch protection rules ensure that:
1. All required CI checks pass before merging
2. Code quality and security standards are maintained
3. The main branch remains stable and deployable

## Required Status Checks

The following status checks must pass before a PR can be merged to `main`:

1. **k8s_validate** - Kubernetes manifest validation
2. **sbom_and_sign** - SBOM generation and artifact signing
3. **e2e** - End-to-end tests

## Setup Instructions

### Prerequisites

- Repository administrator access
- The `supplychain-and-e2e.yml` workflow must be committed and visible in the Actions tab

### Step 1: Navigate to Branch Protection Settings

1. Go to the repository on GitHub: `https://github.com/254CARBON/HMCo`
2. Click on **Settings** tab
3. In the left sidebar, click on **Branches**
4. Under "Branch protection rules", click **Add rule** (or edit existing rule for `main`)

### Step 2: Configure Branch Protection Rule

#### Basic Settings

- **Branch name pattern**: `main`

#### Protection Rules to Enable

Check the following boxes:

- [x] **Require a pull request before merging**
  - [x] Require approvals (recommended: 1)
  - [x] Dismiss stale pull request approvals when new commits are pushed
  - [ ] Require review from Code Owners (optional, if CODEOWNERS file exists)

- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging
  
  **Required status checks** (add these):
  - `k8s_validate`
  - `sbom_and_sign`
  - `e2e`
  
  > **Note**: These checks will only appear in the list after the workflow has run at least once. You may need to:
  > 1. Create a test PR to trigger the workflow
  > 2. Return to this page to add the checks once they appear

- [x] **Require conversation resolution before merging** (recommended)

- [x] **Do not allow bypassing the above settings** (recommended for production)

#### Additional Recommended Settings

- [x] **Require linear history** (optional - enforces rebase/squash merges)
- [x] **Include administrators** (recommended - applies rules to admins too)

### Step 3: Save Changes

1. Scroll to the bottom of the page
2. Click **Create** (or **Save changes** if editing existing rule)

### Step 4: Verify Setup

1. Create a test PR
2. Verify that the three required checks run automatically
3. Verify that the merge button is blocked until all checks pass
4. Attempt to merge - should be blocked if any check fails

## Verification Checklist

- [ ] Branch protection rule created for `main` branch
- [ ] Required status checks configured: `k8s_validate`, `sbom_and_sign`, `e2e`
- [ ] Workflow runs successfully on PRs
- [ ] Merge blocked until all three checks pass
- [ ] PR template appears when creating new PRs

## Troubleshooting

### Status checks not appearing in the list

**Problem**: The required checks don't show up when configuring branch protection.

**Solution**: 
1. Ensure the workflow file is merged to the main branch
2. Create a test PR to trigger the workflow
3. Once the workflow runs, the checks will appear in the branch protection settings
4. Return to branch protection settings and add them

### Workflow failing

**Problem**: One or more required checks are failing.

**Solution**:
1. Check the Actions tab to see which job failed
2. Review the logs for the failing job
3. Common issues:
   - Kubernetes manifests with validation errors → Fix YAML syntax
   - SBOM generation failures → Check dependencies
   - E2E test failures → Review test logs and fix issues

### Merge button still enabled despite failing checks

**Problem**: PR can be merged even with failing checks.

**Solution**:
1. Verify branch protection rule is applied to correct branch (`main`)
2. Ensure "Require status checks to pass before merging" is checked
3. Verify the exact check names match the workflow job names
4. Ensure "Include administrators" is checked if testing with admin account

## GitHub Actions Permissions

Ensure the following permissions are configured for GitHub Actions:

### Repository Settings

1. Go to **Settings** → **Actions** → **General**
2. Under "Workflow permissions", ensure:
   - [x] Read and write permissions (for GHCR push)
   - [x] Allow GitHub Actions to create and approve pull requests

### GHCR (GitHub Container Registry) Access

The workflow requires `packages: write` permission to push container images to GHCR. This is already configured in the workflow file:

```yaml
permissions:
  contents: read
  packages: write
  id-token: write
  security-events: write
```

No additional GHCR setup is required - the workflow uses `GITHUB_TOKEN` for authentication.

## Pinned Tool Versions

The following tools are pinned to specific versions in the workflow:

| Tool | Version | Job |
|------|---------|-----|
| Helm | v3.13.3 | k8s_validate |
| kubeconform | v0.6.4 | k8s_validate |
| Syft | v1.0.1 | sbom_and_sign |
| Cosign | v2.2.3 | sbom_and_sign |
| Grype | v0.74.7 | sbom_and_sign |
| Kind | v0.20.0 | e2e |
| kubectl | v1.28.0 | e2e |
| Python | 3.11 | e2e |

These versions are pinned to ensure reproducible builds and security compliance.

## Maintenance

### Updating Tool Versions

To update tool versions:

1. Edit `.github/workflows/supplychain-and-e2e.yml`
2. Update version numbers for specific tools
3. Test the workflow in a PR
4. Document version changes in commit message

### Modifying Required Checks

To add or remove required status checks:

1. Update the workflow file to add/remove jobs
2. Update this document to reflect changes
3. Update branch protection settings to match

## Support

For issues or questions:
- Check workflow logs in the Actions tab
- Review this documentation
- Contact the platform team
- Open an issue in the repository

## References

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub Actions Permissions](https://docs.github.com/en/actions/security-guides/automatic-token-authentication#permissions-for-the-github_token)
- [Kubernetes Manifest Validation Tools](https://github.com/yannh/kubeconform)
- [SBOM with Syft](https://github.com/anchore/syft)
- [Sigstore Cosign](https://github.com/sigstore/cosign)
