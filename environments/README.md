# Environment-Specific GitOps Manifests

This directory contains environment-specific ArgoCD application manifests for the HMCo platform.

## Directory Structure

```
environments/
├── dev/                    # Development environment
│   └── argocd-applications.yaml
├── staging/                # Staging environment
│   └── argocd-applications.yaml
├── prod/                   # Production environment
│   └── argocd-applications.yaml
└── README.md
```

## Environments

### Development (dev)
- **Purpose**: Active development and testing
- **Sync Policy**: Automated (auto-heal enabled)
- **Namespace Pattern**: `*-dev`
- **Promotion Target**: Staging

### Staging (staging)
- **Purpose**: Pre-production validation
- **Sync Policy**: Automated (auto-heal enabled)
- **Namespace Pattern**: `*-staging`
- **Promotion Source**: Development
- **Promotion Target**: Production

### Production (prod)
- **Purpose**: Live production workloads
- **Sync Policy**: **Manual sync only** (requires approval)
- **Namespace Pattern**: Standard namespaces
- **Promotion Source**: Staging
- **Notifications**: Slack notifications on successful sync

## Chart Versioning Strategy

All Helm charts follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version: Incompatible API changes
- **MINOR** version: Backward-compatible functionality additions
- **PATCH** version: Backward-compatible bug fixes

Current stable version: **1.0.0**

## Promotion Process

### Automated Promotion via GitHub Actions

Use the Chart Promotion workflow to promote charts between environments:

1. Navigate to **Actions** → **Chart Promotion Pipeline**
2. Click **Run workflow**
3. Fill in the promotion details:
   - **Chart name**: The chart to promote (e.g., `data-platform`)
   - **New version**: Semantic version (e.g., `1.0.1`)
   - **Source environment**: `dev` or `staging`
   - **Target environment**: `staging` or `prod`
4. The workflow will:
   - Validate the promotion path
   - Update Chart.yaml version
   - Update CHANGELOG.md
   - Create a pull request with changes
   - Run helm lint validation

### Valid Promotion Paths

```
dev → staging → prod
```

- ✅ dev → staging
- ✅ staging → prod
- ❌ dev → prod (must go through staging)

### Manual Promotion Steps

If you need to promote manually:

1. **Update Chart Version**
   ```bash
   # Edit helm/charts/<chart-name>/Chart.yaml
   version: 1.0.1  # Bump version
   ```

2. **Update CHANGELOG**
   ```bash
   # Edit helm/charts/<chart-name>/CHANGELOG.md
   ## [1.0.1] - YYYY-MM-DD
   ### Changed
   - Promoted to <target-env> environment
   ```

3. **Create Pull Request**
   ```bash
   git checkout -b promote/<chart-name>-<version>-to-<env>
   git add helm/charts/<chart-name>/
   git commit -m "Promote <chart-name> to <version> in <env>"
   git push origin promote/<chart-name>-<version>-to-<env>
   ```

4. **Get Approval**
   - Development → Staging: 1 approval required
   - Staging → Production: 2 approvals required (including SRE team)

5. **Merge and Deploy**
   - For dev/staging: ArgoCD auto-syncs after merge
   - For production: Manual sync required in ArgoCD UI after merge

## Production Deployment Checklist

Before promoting to production:

- [ ] Chart version follows semantic versioning
- [ ] CHANGELOG.md is updated with changes
- [ ] Changes tested thoroughly in staging
- [ ] Performance impact assessed
- [ ] Rollback plan documented
- [ ] SRE team approval obtained
- [ ] Deployment window scheduled (if required)
- [ ] Monitoring alerts configured
- [ ] On-call engineer notified

## ArgoCD Sync Policies

### Development & Staging
```yaml
syncPolicy:
  automated:
    prune: false
    selfHeal: true
    allowEmpty: false
```

### Production
```yaml
syncPolicy:
  automated: null  # Manual sync only
  retry:
    limit: 3
    backoff:
      duration: 5s
      factor: 2
```

## Rollback Procedure

### Quick Rollback (Production)

1. Identify the previous working version from git history
2. Create rollback branch:
   ```bash
   git checkout -b rollback/<chart-name>-to-<previous-version>
   ```
3. Revert Chart.yaml to previous version
4. Update CHANGELOG with rollback note
5. Create PR with label `rollback` and `urgent`
6. Fast-track approval process
7. Merge and manually sync in ArgoCD

### Emergency Rollback

For critical production issues:
1. Use ArgoCD UI to sync to previous git commit
2. Create post-incident rollback PR to match ArgoCD state
3. Follow incident response procedures

## Monitoring and Alerts

- **Dev/Staging**: Standard monitoring, non-critical alerts
- **Production**: Full monitoring suite with PagerDuty integration
  - Slack channel: `#prod-deployments`
  - On-call rotation notified of all production changes

## Support

For questions or issues:
- **Chart issues**: Create issue with label `helm-chart`
- **Promotion issues**: Create issue with label `promotion`
- **Production incidents**: Follow incident response runbook

## References

- [Helm Chart Guidelines](../../helm/charts/README.md)
- [ArgoCD Best Practices](https://argo-cd.readthedocs.io/en/stable/user-guide/best_practices/)
- [Semantic Versioning](https://semver.org/)
