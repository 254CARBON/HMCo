# Release Management Guide

This guide documents the release management improvements implemented for the HMCo platform.

## Overview

The HMCo platform now has a comprehensive release management system that ensures:
- **Reproducible releases** per environment
- **Version pinning** for all Helm charts
- **Structured promotion** pipeline from dev → staging → prod
- **Change tracking** via changelogs
- **Approval-based** production deployments

## Key Components

### 1. Chart Versioning

All Helm charts now follow **Semantic Versioning** (semver):

```
MAJOR.MINOR.PATCH
```

- **Current Version**: 1.0.0 (initial stable release)
- **Location**: `helm/charts/*/Chart.yaml`

#### Version Bump Guidelines

| Change Type | Version Change | Example |
|------------|----------------|---------|
| Bug fixes, patches | PATCH | 1.0.0 → 1.0.1 |
| New features (backward-compatible) | MINOR | 1.0.0 → 1.1.0 |
| Breaking changes | MAJOR | 1.0.0 → 2.0.0 |

### 2. Changelogs

Each top-level chart has a `CHANGELOG.md` file documenting all changes:

```
helm/charts/
├── api-gateway/CHANGELOG.md
├── cloudflare-tunnel/CHANGELOG.md
├── data-platform/CHANGELOG.md
├── jupyterhub/CHANGELOG.md
├── ml-platform/CHANGELOG.md
├── monitoring/CHANGELOG.md
├── networking/CHANGELOG.md
├── platform-policies/CHANGELOG.md
├── portal-services/CHANGELOG.md
├── security/CHANGELOG.md
├── service-mesh/CHANGELOG.md
├── storage/CHANGELOG.md
└── vault/CHANGELOG.md
```

Format follows [Keep a Changelog](https://keepachangelog.com/) standard.

### 3. Environment-Specific Deployments

GitOps manifests are now split by environment:

```
environments/
├── dev/
│   └── argocd-applications.yaml      # Development environment
├── staging/
│   └── argocd-applications.yaml      # Staging environment
└── prod/
    └── argocd-applications.yaml      # Production environment (manual sync)
```

#### Environment Characteristics

| Environment | Auto-Sync | Namespace Pattern | Purpose |
|------------|-----------|-------------------|---------|
| Development | ✅ Yes | `*-dev` | Active development & testing |
| Staging | ✅ Yes | `*-staging` | Pre-production validation |
| Production | ❌ No (Manual) | Standard | Live production workloads |

### 4. Promotion Pipeline

Automated chart promotion via GitHub Actions workflow.

#### Workflow: `.github/workflows/chart-promotion.yml`

**Features:**
- Validates promotion paths (dev→staging→prod)
- Updates Chart.yaml version
- Updates CHANGELOG.md
- Creates pull request with changes
- Runs helm lint validation
- Supports draft PRs for production promotions

#### Usage

1. Navigate to **Actions** → **Chart Promotion Pipeline**
2. Click **Run workflow**
3. Fill in parameters:
   - Chart name (e.g., `data-platform`)
   - New version (e.g., `1.0.1`)
   - Source environment (`dev` or `staging`)
   - Target environment (`staging` or `prod`)

#### Valid Promotion Paths

```
✅ dev → staging
✅ staging → prod
❌ dev → prod (blocked - must go through staging)
```

## Implementation Details

### T9.1: Chart Versioning and Changelogs ✅

**Completed:**
- ✅ Bumped all Chart.yaml versions from 0.1.0 to 1.0.0 (22 charts total)
- ✅ Created CHANGELOG.md for all 13 top-level charts
- ✅ Added release notes template (`.github/RELEASE_TEMPLATE.md`)
- ✅ Added comprehensive documentation

**Charts Updated:**
- Top-level (13): api-gateway, cloudflare-tunnel, data-platform, jupyterhub, ml-platform, monitoring, networking, platform-policies, portal-services, security, service-mesh, storage, vault
- Sub-charts (9): clickhouse, dolphinscheduler, trino, spark-operator, data-lake, datahub, superset, kubeflow, mlflow

### T9.2: Promotion Pipeline ✅

**Completed:**
- ✅ Created environment-specific GitOps manifests (environments/dev, staging, prod)
- ✅ Split ArgoCD applications by environment with distinct namespaces
- ✅ Created promotion workflow in GitHub Actions
- ✅ Added PR-based promotion with approval requirements
- ✅ Configured manual sync for production (no auto-deploy)
- ✅ Added environment-specific AppProjects for RBAC

**Production Safeguards:**
- Manual sync required (no auto-sync)
- Slack notifications on deployments
- Retry policies with backoff
- Draft PR for production promotions
- Approval required before merge

## Usage Examples

### Example 1: Promote a Chart from Dev to Staging

```bash
# Via GitHub Actions UI
1. Go to Actions → Chart Promotion Pipeline
2. Run workflow:
   - chart_name: data-platform
   - new_version: 1.0.1
   - source_env: dev
   - target_env: staging
3. Review and approve the auto-generated PR
4. Merge PR
5. ArgoCD auto-syncs to staging
```

### Example 2: Promote to Production

```bash
# Via GitHub Actions UI
1. Go to Actions → Chart Promotion Pipeline
2. Run workflow:
   - chart_name: data-platform
   - new_version: 1.0.1
   - source_env: staging
   - target_env: prod
3. Review the DRAFT PR created
4. Get approvals from:
   - Team lead
   - SRE team member
5. Mark PR as ready for review
6. Merge PR
7. MANUALLY sync in ArgoCD UI (production requires manual approval)
```

### Example 3: Manual Version Update

```bash
# 1. Create branch
git checkout -b bump-ml-platform-version

# 2. Update Chart.yaml
cd helm/charts/ml-platform
# Edit Chart.yaml: version: 1.0.1

# 3. Update CHANGELOG.md
cat >> CHANGELOG.md << 'EOF'

## [1.0.1] - $(date +%Y-%m-%d)

### Fixed
- Fixed MLflow authentication issues
- Corrected Kubeflow pipeline defaults

[1.0.1]: https://github.com/254CARBON/HMCo/releases/tag/ml-platform-1.0.1
EOF

# 4. Commit and push
git add Chart.yaml CHANGELOG.md
git commit -m "Bump ml-platform to 1.0.1"
git push origin bump-ml-platform-version

# 5. Create PR and get approval
```

## ArgoCD Integration

### Chart Version Tracking

ArgoCD now tracks exact chart versions. All applications reference:
- **Repository**: `https://github.com/254CARBON/HMCo.git`
- **Target Revision**: `main` branch
- **Chart Path**: `helm/charts/<chart-name>`
- **Helm Version**: `v3`

The chart version is automatically picked up from `Chart.yaml` in the repository.

### Sync Policies

#### Development & Staging
```yaml
syncPolicy:
  automated:
    prune: false
    selfHeal: true
    allowEmpty: false
  syncOptions:
    - CreateNamespace=true
```

#### Production
```yaml
syncPolicy:
  automated: null  # ← Manual sync required
  syncOptions:
    - CreateNamespace=true
  retry:
    limit: 3
    backoff:
      duration: 5s
      factor: 2
```

## Rollback Procedures

### Quick Rollback via Git

```bash
# 1. Find the previous working commit
git log --oneline helm/charts/<chart-name>/

# 2. Create rollback PR
git checkout -b rollback-<chart-name>-<version>
git revert <commit-hash>
git push origin rollback-<chart-name>-<version>

# 3. Fast-track PR approval with 'rollback' label
# 4. Merge and sync in ArgoCD
```

### Emergency Rollback via ArgoCD

```bash
# Use ArgoCD CLI
argocd app rollback <app-name> <revision>

# Or via UI
1. Open ArgoCD UI
2. Select application
3. Click "History and Rollback"
4. Select previous revision
5. Click "Rollback"
```

## Monitoring

### Metrics to Track

- **Chart Version Drift**: Ensure all environments use intended versions
- **Deployment Frequency**: Track promotion velocity
- **Rollback Rate**: Monitor deployment success rate
- **Time to Production**: Measure dev→prod cycle time

### Dashboards

Create Grafana dashboards to track:
- Chart versions per environment
- Deployment history timeline
- Failed vs successful deployments
- ArgoCD sync status

## Compliance & Audit

### Audit Trail

All changes are tracked through:
1. **Git commits**: Full history of chart changes
2. **Pull requests**: Review and approval history
3. **ArgoCD sync history**: Deployment timeline
4. **GitHub Actions logs**: Promotion workflow execution

### Compliance Checks

- ✅ No floating versions (no `latest` tags)
- ✅ Pinned chart versions in ArgoCD
- ✅ Mandatory PR approvals for production
- ✅ Automated validation in CI
- ✅ Change documentation in CHANGELOGs

## Best Practices

### 1. Version Management
- Always bump version when making changes
- Follow semantic versioning strictly
- Update CHANGELOG with every version bump
- Test version upgrades in dev first

### 2. Promotion Strategy
- Always promote through all environments (dev→staging→prod)
- Never skip staging validation
- Allow sufficient soak time in each environment
- Monitor metrics before promoting to next environment

### 3. Production Deployments
- Schedule production deployments during maintenance windows
- Have on-call engineer available
- Review full change history before deployment
- Prepare rollback plan before deployment
- Notify stakeholders of deployment schedule

### 4. Documentation
- Keep CHANGELOGs up to date
- Document breaking changes clearly
- Provide upgrade/migration instructions
- Update runbooks for operational changes

## Troubleshooting

### Issue: Promotion workflow fails

**Solution:**
```bash
# Check workflow logs
gh workflow view "Chart Promotion Pipeline"

# Common issues:
# - Invalid version format (must be semver)
# - Invalid promotion path (must be dev→staging or staging→prod)
# - Chart not found (check chart name spelling)
```

### Issue: ArgoCD not syncing

**Solution:**
```bash
# Check sync status
argocd app get <app-name>

# Force sync if needed (dev/staging only)
argocd app sync <app-name>

# Check for sync errors
argocd app get <app-name> --show-operation
```

### Issue: Version conflict

**Solution:**
```bash
# Check current versions
grep "^version:" helm/charts/*/Chart.yaml

# Ensure version is unique and higher than previous
# Update Chart.yaml and CHANGELOG.md
# Commit and push changes
```

## References

### Documentation
- [Environments README](environments/README.md) - Environment-specific deployment guide
- [Helm Charts README](helm/charts/README.md) - Chart development guide
- [Release Template](.github/RELEASE_TEMPLATE.md) - Release notes template

### External Resources
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [Helm Best Practices](https://helm.sh/docs/chart_best_practices/)

## Support

- **Chart Issues**: Label with `helm-chart`
- **Promotion Issues**: Label with `promotion`
- **Production Incidents**: Follow incident response procedures
- **Questions**: Use GitHub Discussions

---

**Last Updated**: 2025-10-31
**Version**: 1.0.0
**Status**: Active
