# Release Notes Template

Use this template when creating release notes for chart versions.

## Release Title Format
```
<chart-name> v<version> - <environment>
```

Example: `data-platform v1.0.1 - Production`

---

## Release Notes Template

```markdown
# [Chart Name] v[Version]

**Release Date**: YYYY-MM-DD
**Environment**: [Dev/Staging/Production]
**Previous Version**: [Previous Version]

## 📋 Summary

[Brief description of what this release contains]

## ✨ What's New

### Added
- [New feature 1]
- [New feature 2]

### Changed
- [Changed behavior 1]
- [Changed behavior 2]

### Fixed
- [Bug fix 1]
- [Bug fix 2]

### Removed
- [Deprecated feature 1]
- [Deprecated feature 2]

## 🔧 Configuration Changes

[List any configuration changes required for this version]

- [ ] Update values.yaml parameter `x` to `y`
- [ ] Add new ConfigMap for feature Z
- [ ] Migrate secret format from A to B

## 📊 Performance & Resource Impact

- **CPU**: [Impact on CPU usage]
- **Memory**: [Impact on memory usage]
- **Storage**: [Impact on storage requirements]
- **Network**: [Impact on network traffic]

## 🔐 Security Updates

[List any security-related changes]

- [ ] Updated base image to address CVE-XXXX-XXXX
- [ ] Enhanced RBAC policies
- [ ] Added network policies for pod isolation

## 🚀 Deployment Notes

### Prerequisites
- [ ] Kubernetes version: [minimum version]
- [ ] Helm version: [minimum version]
- [ ] Dependencies: [list required dependencies]

### Deployment Steps

1. **Pre-deployment**
   ```bash
   # Backup current state
   kubectl get all -n <namespace> -o yaml > backup-$(date +%Y%m%d).yaml
   ```

2. **Deploy**
   ```bash
   # Using ArgoCD (recommended)
   argocd app sync <app-name>
   
   # Or using Helm directly
   helm upgrade --install <release> helm/charts/<chart-name> \
     --namespace <namespace> \
     --values helm/charts/<chart-name>/values/<env>.yaml \
     --version <version>
   ```

3. **Post-deployment**
   ```bash
   # Verify deployment
   kubectl get pods -n <namespace>
   kubectl logs -n <namespace> <pod-name>
   ```

## ✅ Verification Steps

- [ ] All pods are running
- [ ] Services are accessible
- [ ] Health checks passing
- [ ] Monitoring dashboards show expected metrics
- [ ] No error logs detected

## 🔄 Rollback Procedure

If issues occur:

```bash
# Using Helm
helm rollback <release> <previous-revision> -n <namespace>

# Using ArgoCD
argocd app rollback <app-name> <revision>
```

Or use the automated rollback procedure in [environments/README.md](../environments/README.md).

## 📈 Monitoring & Alerts

### Key Metrics to Monitor
- Pod restart count
- CPU and memory utilization
- Request latency (if applicable)
- Error rate

### Dashboards
- [Link to Grafana dashboard]
- [Link to application metrics]

### Alerts
- [List of configured alerts for this release]

## 🐛 Known Issues

[List any known issues or limitations]

- **Issue 1**: [Description and workaround]
- **Issue 2**: [Description and workaround]

## 📚 Documentation

- [CHANGELOG](../../helm/charts/<chart-name>/CHANGELOG.md)
- [Chart README](../../helm/charts/<chart-name>/README.md)
- [Values Documentation](../../helm/charts/<chart-name>/values.yaml)

## 👥 Contributors

Thanks to all contributors who made this release possible!

- @username1
- @username2

## 🔗 References

- [Pull Request #XXX](link-to-pr)
- [Issue #XXX](link-to-issue)
- [Design Document](link-to-doc)

---

**Release Manager**: @username
**Reviewed By**: @reviewer1, @reviewer2
**Deployed By**: @deployer
```

---

## Quick Reference

### Version Numbering

| Change Type | Example | When to Use |
|------------|---------|-------------|
| PATCH | 1.0.0 → 1.0.1 | Bug fixes, minor updates |
| MINOR | 1.0.0 → 1.1.0 | New features, backward-compatible |
| MAJOR | 1.0.0 → 2.0.0 | Breaking changes, incompatible updates |

### Release Checklist

**Before Release:**
- [ ] Version bumped in Chart.yaml
- [ ] CHANGELOG.md updated
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Security scan completed
- [ ] Performance tested
- [ ] Peer review completed

**During Release:**
- [ ] Create release branch
- [ ] Tag release in git
- [ ] Deploy to target environment
- [ ] Monitor deployment
- [ ] Verify functionality

**After Release:**
- [ ] Update release notes
- [ ] Notify stakeholders
- [ ] Update runbooks
- [ ] Schedule retrospective (if needed)

### Communication Templates

**Slack Announcement:**
```
🚀 [Chart Name] v[Version] deployed to [Environment]

Release Notes: [Link]
Deployment Time: [Duration]
Status: ✅ Success / ⚠️ Issues Detected

@channel Please verify functionality in your areas.
```

**Email Announcement:**
```
Subject: [Chart Name] v[Version] Released to [Environment]

Team,

We've successfully deployed [Chart Name] version [Version] to [Environment].

Key Changes:
- [Change 1]
- [Change 2]

Please review the full release notes: [Link]

Next Steps:
- Verify your integrations
- Report any issues to #incidents channel
- Update dependent services if needed

Thanks,
Release Team
```
