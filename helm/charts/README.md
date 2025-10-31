# HMCo Helm Charts

This directory contains all Helm charts for the HMCo platform infrastructure and services.

## Available Charts

### Core Infrastructure

- **[storage](./storage/)** - Persistent storage configuration and provisioners
- **[networking](./networking/)** - Network policies and ingress controllers
- **[security](./security/)** - Security scanning and certificate management
- **[vault](./vault/)** - HashiCorp Vault for secret management
- **[platform-policies](./platform-policies/)** - Kyverno policies for compliance

### Platform Services

- **[data-platform](./data-platform/)** - Integrated data platform with:
  - ClickHouse (analytics database)
  - DolphinScheduler (workflow orchestration)
  - Trino (distributed SQL)
  - Spark Operator (big data processing)
  - Data Lake (storage management)
  - DataHub (data catalog)
  - Superset (visualization)

- **[ml-platform](./ml-platform/)** - Machine learning platform with:
  - MLflow (experiment tracking)
  - Kubeflow (ML pipelines)

- **[monitoring](./monitoring/)** - Observability stack with Kubecost
- **[service-mesh](./service-mesh/)** - Istio service mesh
- **[api-gateway](./api-gateway/)** - API Gateway for service routing
- **[cloudflare-tunnel](./cloudflare-tunnel/)** - Secure tunnel management
- **[portal-services](./portal-services/)** - Portal backend services
- **[jupyterhub](./jupyterhub/)** - Multi-user Jupyter notebook environment

## Chart Structure

Each chart follows this structure:

```
<chart-name>/
├── Chart.yaml              # Chart metadata and version
├── CHANGELOG.md            # Version history and changes
├── values.yaml             # Default configuration values
├── values/                 # Environment-specific values
│   ├── dev.yaml
│   ├── staging.yaml
│   └── prod.yaml
├── templates/              # Kubernetes manifest templates
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ...
└── charts/                 # Sub-chart dependencies (if any)
```

## Versioning

All charts follow [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH
```

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward-compatible)
- **PATCH**: Bug fixes (backward-compatible)

Current stable version: **1.0.0**

## Chart Development

### Creating a New Chart

```bash
helm create helm/charts/<chart-name>
```

### Linting Charts

```bash
helm lint helm/charts/<chart-name>
```

### Testing Chart Rendering

```bash
helm template <release-name> helm/charts/<chart-name> \
  --values helm/charts/<chart-name>/values/dev.yaml \
  --debug
```

### Packaging Charts

```bash
helm package helm/charts/<chart-name>
```

## Making Changes

### 1. Version Bump Guidelines

Increment the version in `Chart.yaml` according to the change type:

- **Bug fix**: 1.0.0 → 1.0.1
- **New feature**: 1.0.0 → 1.1.0
- **Breaking change**: 1.0.0 → 2.0.0

### 2. Update CHANGELOG

Add an entry to `CHANGELOG.md`:

```markdown
## [1.0.1] - 2025-10-31

### Fixed
- Fixed memory leak in service container
- Corrected ingress path configuration

### Added
- Added resource limits for pods

### Changed
- Updated dependency versions
```

### 3. Test Changes

```bash
# Lint the chart
helm lint helm/charts/<chart-name>

# Render templates
helm template test helm/charts/<chart-name> \
  --values helm/charts/<chart-name>/values/dev.yaml

# Install in test environment
helm upgrade --install <release> helm/charts/<chart-name> \
  --namespace <namespace> \
  --values helm/charts/<chart-name>/values/dev.yaml
```

### 4. Submit Pull Request

Include in your PR description:
- Chart name and new version
- Summary of changes
- Testing performed
- Impact assessment

## Environment-Specific Values

Each chart supports environment-specific value overrides:

### Development (`values/dev.yaml`)
- Lower resource limits
- Debug logging enabled
- Mock external services
- Relaxed security policies

### Staging (`values/staging.yaml`)
- Production-like resources
- Standard logging
- Real external services
- Standard security policies

### Production (`values/prod.yaml`)
- Full resource allocation
- Structured logging
- High availability
- Strict security policies
- Backup and monitoring enabled

## Dependencies

Charts may depend on:
- **Sub-charts**: Located in `charts/` directory
- **External charts**: Defined in `Chart.yaml` dependencies

Update dependencies:
```bash
helm dependency update helm/charts/<chart-name>
```

## Best Practices

### 1. Resource Management
- Always set resource requests and limits
- Use horizontal pod autoscaling where appropriate
- Configure pod disruption budgets for high availability

### 2. Security
- Use least-privilege RBAC
- Implement network policies
- Scan images for vulnerabilities
- Never commit secrets (use Vault or Kubernetes secrets)

### 3. Configuration
- Use ConfigMaps for configuration
- Support environment-specific overrides
- Document all values in `values.yaml` comments

### 4. Documentation
- Keep CHANGELOG.md up to date
- Document breaking changes clearly
- Provide upgrade instructions for major versions

### 5. Testing
- Test with all environment value files
- Validate against Kubernetes schemas
- Run helm lint before committing

## Promotion Process

Charts are promoted through environments:

```
dev → staging → prod
```

Use the GitHub Actions workflow or see [environments/README.md](../../environments/README.md) for details.

## Troubleshooting

### Chart Won't Install

```bash
# Check for template errors
helm template <release> helm/charts/<chart-name> --debug

# Validate against Kubernetes API
helm install <release> helm/charts/<chart-name> --dry-run --debug
```

### Version Conflicts

```bash
# List installed releases
helm list -A

# Check chart history
helm history <release> -n <namespace>

# Rollback if needed
helm rollback <release> <revision> -n <namespace>
```

### Dependency Issues

```bash
# Update dependencies
helm dependency update helm/charts/<chart-name>

# List dependencies
helm dependency list helm/charts/<chart-name>
```

## CI/CD Integration

Charts are automatically validated in CI:
- **Lint**: All charts are linted on PR
- **Template**: Charts are rendered to catch syntax errors
- **Security**: Scanned for vulnerabilities
- **Version**: Verified to follow semver

See [.github/workflows/ci.yml](../../.github/workflows/ci.yml) for details.

## Support

- **Issues**: Create a GitHub issue with label `helm-chart`
- **Questions**: Use GitHub Discussions
- **Urgent**: Contact SRE team

## References

- [Helm Documentation](https://helm.sh/docs/)
- [Helm Best Practices](https://helm.sh/docs/chart_best_practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [ArgoCD Helm Integration](https://argo-cd.readthedocs.io/en/stable/user-guide/helm/)
