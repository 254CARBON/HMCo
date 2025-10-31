# Cloudflare Access: Applications & Authentication Policies (as Code)

This guide provisions and manages Cloudflare Access applications and policies as code, ensuring reproducible SSO configuration for all externally exposed services on `254carbon.com`. All configuration is version controlled and can be recreated from scripts.

## Overview

**Philosophy**: Access policies should be **version controlled**, **reproducible**, and **auditable**. Manual changes in the Cloudflare dashboard are discouraged; all changes should be made through the scripts in this repository.

## What this sets up

- 14+ self-hosted Access apps covering all major services:
  - Portal (+ apex + www)
  - Data Platform: Grafana, Superset, DataHub, Trino, ClickHouse
  - Storage & Orchestration: Vault, MinIO, DolphinScheduler, LakeFS
  - ML & Analytics: MLflow, Spark History
- Per-app session durations (shorter for sensitive services like Vault/Trino)
- Authentication policies allowing only authorized email domains
- Optional restrictions by IdP, country codes, or specific users
- ACME HTTP-01 challenge bypass policies (for certificate renewal)
- NGINX ingress annotations for seamless integration

## Prerequisites

- Cloudflare Zero Trust account and Team name (e.g., `qagi`)
- API token with Access: Apps (write) scope
- Account ID (from Zero Trust → Settings → Account)
- `jq` installed locally

## Provision apps via script (Infrastructure as Code)

### Initial Setup

Use `scripts/create-cloudflare-access-apps.sh` to create all applications and policies:

```bash
export CLOUDFLARE_API_TOKEN=<your-api-token>
export CLOUDFLARE_ACCOUNT_ID=<your-account-id>

# Zone mode (recommended – apps live at <sub>.254carbon.com)
./scripts/create-cloudflare-access-apps.sh \
  --mode zone \
  --zone-domain 254carbon.com \
  --allowed-email-domains 254carbon.com

# With additional security constraints
./scripts/create-cloudflare-access-apps.sh \
  --mode zone \
  --zone-domain 254carbon.com \
  --allowed-email-domains 254carbon.com \
  --countries US,GB,KE \
  --force
```

### Idempotent Updates (Reconciliation)

The script is **fully idempotent** and can be run multiple times safely:

```bash
# Update existing apps to match desired state (reconciliation)
./scripts/create-cloudflare-access-apps.sh \
  --mode zone \
  --zone-domain 254carbon.com \
  --allowed-email-domains 254carbon.com \
  --force

# Create new apps only, skip existing (no updates)
./scripts/create-cloudflare-access-apps.sh \
  --mode zone \
  --zone-domain 254carbon.com \
  --allowed-email-domains 254carbon.com \
  --skip-existing

# Dry-run to preview changes without applying
./scripts/create-cloudflare-access-apps.sh \
  --mode zone \
  --zone-domain 254carbon.com \
  --dry-run
```

### Export Current Configuration

Export your current Access configuration for backup and version control:

```bash
# Export all apps and policies to JSON
./scripts/export-cloudflare-access-apps.sh \
  --token <your-api-token> \
  --account-id <your-account-id> \
  --output docs/cloudflare/access-config-backup.json

# Version control the export
git add docs/cloudflare/access-config-backup.json
git commit -m "Backup Cloudflare Access configuration"
```

### Team-domain mode (alternative)

For using Cloudflare Zero Trust team domain (e.g., `qagi.cloudflareaccess.com`):

```bash
./scripts/create-cloudflare-access-apps.sh \
  --mode team \
  --team-domain qagi.cloudflareaccess.com \
  --allowed-email-domains 254carbon.com
```

### Defaults

- Mode: `team` (override with `--mode zone` for `254carbon.com` hostnames)
- Team domain: derived from `--team-domain` or `<team-name>.cloudflareaccess.com`
- Allowed email domains: `254carbon.com` (override with `--allowed-email-domains`)
- Applications include portal/root/www, Grafana, Superset, DataHub, Trino, Doris, Vault, MinIO, DolphinScheduler, LakeFS, MLflow, Spark History

### Policy model

- Include: list of `email_domain` and/or specific `email`
- Require: optional `login_method` (IdP) and/or allowed `geo` country codes
- Exclude: optional list of blocked `email`

## Ingress annotations (NGINX)

Use your Team domain (not the account ID) for auth endpoints. Example for Grafana:

```yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/auth-url: "https://qagi.cloudflareaccess.com/cdn-cgi/access/authorize"
    nginx.ingress.kubernetes.io/auth-signin: "https://qagi.cloudflareaccess.com/cdn-cgi/access/login?redirect_url=$escaped_request_uri"
    nginx.ingress.kubernetes.io/auth-response-headers: "cf-access-jwt-assertion"
```

See `k8s/ingress/ingress-cloudflare-sso.yaml` or `k8s/ingress/ingress-sso-rules.yaml` for complete examples. Replace `qagi` with your Team name.

## Reproducible Workflow (GitOps-style)

### 1. Initial Provisioning

```bash
# Create all applications and policies
./scripts/create-cloudflare-access-apps.sh \
  --mode zone \
  --zone-domain 254carbon.com \
  --allowed-email-domains 254carbon.com \
  --force

# Export the configuration for version control
./scripts/export-cloudflare-access-apps.sh \
  --output docs/cloudflare/access-baseline.json

# Commit to git
git add docs/cloudflare/access-baseline.json
git commit -m "Initial Cloudflare Access baseline"
```

### 2. Making Changes

**DO NOT** make manual changes in the Cloudflare dashboard. Instead:

```bash
# 1. Update the script parameters or APPLICATIONS array in create-cloudflare-access-apps.sh
# 2. Run with --dry-run to preview changes
./scripts/create-cloudflare-access-apps.sh --mode zone --zone-domain 254carbon.com --dry-run

# 3. Apply changes with --force
./scripts/create-cloudflare-access-apps.sh --mode zone --zone-domain 254carbon.com --force

# 4. Export and commit the new state
./scripts/export-cloudflare-access-apps.sh --output docs/cloudflare/access-baseline.json
git add docs/cloudflare/access-baseline.json
git commit -m "Update Access policies: <describe change>"
```

### 3. Drift Detection

Detect when manual changes have been made in the dashboard:

```bash
# Export current state
./scripts/export-cloudflare-access-apps.sh --output /tmp/access-current.json

# Compare with baseline
diff docs/cloudflare/access-baseline.json /tmp/access-current.json

# If drift detected, reconcile back to desired state
./scripts/create-cloudflare-access-apps.sh --mode zone --zone-domain 254carbon.com --force
```

### 4. Disaster Recovery

Restore Access configuration from backup:

```bash
# The baseline JSON serves as documentation
# To restore, simply re-run the create script with --force
./scripts/create-cloudflare-access-apps.sh \
  --mode zone \
  --zone-domain 254carbon.com \
  --allowed-email-domains 254carbon.com \
  --force

# This will recreate all applications and policies to match the script configuration
```

## Configuration Management Best Practices

1. **Version Control Everything**: Always export and commit Access configuration after changes
2. **No Manual Changes**: Avoid making changes in the Cloudflare dashboard; use scripts
3. **Dry-Run First**: Always test with `--dry-run` before applying changes
4. **Document Changes**: Use descriptive git commit messages for Access policy changes
5. **Regular Exports**: Schedule weekly exports to detect drift: `cron: 0 0 * * 0 ./scripts/export-cloudflare-access-apps.sh`
6. **Review Changes**: Use `git diff` to review Access configuration changes before deploying
7. **Backup Secrets**: Securely backup API tokens used for Access management

## Notes and tips

- Team domain is found at Zero Trust → Settings → Team (e.g., `qagi.cloudflareaccess.com`) and can be passed explicitly with `--team-domain`.
- If you choose zone mode (`--zone-domain 254carbon.com`), ensure your DNS zone is on Cloudflare and your tunnel/public hostnames route there.
- Zone mode automatically provisions apex (`254carbon.com`) and `www` alongside `portal.254carbon.com`; disable or remove them if a public landing page must remain open.
- You can re-run with `--force` to reconcile apps and policies idempotently - this is the recommended way to ensure desired state.
- Vault and other sensitive apps use shorter sessions by default; adjust in the `APPLICATIONS` array in `create-cloudflare-access-apps.sh` if needed.
- The `--skip-existing` flag is useful when adding new applications without touching existing ones.

## Troubleshooting

- 1033 errors: see `ERROR_1033_DIAGNOSIS_AND_FIX.md`
- Tunnel troubleshooting: `scripts/verify-tunnel.sh`
- Credentials update: `scripts/update-cloudflare-credentials.sh`
