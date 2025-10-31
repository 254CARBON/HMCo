## Description
<!-- Provide a brief description of the changes in this PR -->

## Type of Change
<!-- Mark the relevant option with an 'x' -->
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Infrastructure/DevOps change
- [ ] Refactoring (no functional changes)

## Changes Made
<!-- List the specific changes made in this PR -->
- 
- 
- 

## Related Issues
<!-- Link to related issues using #issue_number -->
Fixes #
Related to #

## Testing Performed
<!-- Describe the testing you have performed -->
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] E2E tests added/updated
- [ ] Manual testing performed
- [ ] Tested in development environment
- [ ] Tested in staging environment

## Checklist
<!-- Ensure all items are completed before requesting review -->

### Code Quality
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code has been performed
- [ ] Code is well-commented, particularly in hard-to-understand areas
- [ ] No unnecessary console logs or debug statements

### Documentation
- [ ] Documentation has been updated (README, inline docs, etc.)
- [ ] API documentation updated (if applicable)
- [ ] Helm chart values documented (if applicable)

### Infrastructure & Kubernetes
- [ ] Kubernetes manifests validated with kubeval/kubeconform
- [ ] Helm charts linted and templates rendered successfully
- [ ] Resource limits and requests defined appropriately
- [ ] Security contexts configured correctly
- [ ] ConfigMaps/Secrets properly referenced

### Security & Supply Chain
- [ ] No secrets or sensitive data committed
- [ ] Dependencies scanned for vulnerabilities
- [ ] SBOM can be generated for changes
- [ ] Container images use specific tags (not 'latest')
- [ ] Security best practices followed

### CI/CD
- [ ] All CI checks pass (unit tests, integration tests, linting)
- [ ] Kubernetes validation passes (k8s_validate)
- [ ] SBOM generation and signing passes (sbom_and_sign)
- [ ] End-to-end tests pass (e2e)
- [ ] No breaking changes to existing workflows

### Data Platform Specific (if applicable)
- [ ] Data schema changes documented
- [ ] Migration scripts provided (if needed)
- [ ] Data lineage updated in DataHub
- [ ] Spark/Flink job configurations validated
- [ ] Iceberg table changes reviewed

### Monitoring & Observability
- [ ] Appropriate logging added
- [ ] Metrics/monitoring configured
- [ ] Alerts configured (if needed)
- [ ] Grafana dashboards updated (if applicable)

## Screenshots/Recordings
<!-- If applicable, add screenshots or recordings to demonstrate the changes -->

## Deployment Notes
<!-- Any special deployment instructions or considerations -->

## Rollback Plan
<!-- Describe how to rollback these changes if needed -->

## Additional Context
<!-- Add any other context about the PR here -->

---
**Reviewer Checklist**
- [ ] Code changes reviewed and approved
- [ ] Tests are comprehensive and passing
- [ ] Documentation is clear and complete
- [ ] No security concerns identified
- [ ] Ready to merge
