# Cloudflare Access: Applications & Authentication Policies

This guide provisions the Cloudflare Access applications and policies that guard every externally exposed hostname on `254carbon.com`. Automation covers the portal aliases plus every service ingress (Grafana, Superset, DataHub, Trino, Doris, Vault, MinIO, DolphinScheduler, lakeFS, MLflow, Spark History).

## What this sets up

- 14 self-hosted Access apps (portal + apex + `www` + Grafana, Superset, DataHub, Trino, Doris, Vault, MinIO, DolphinScheduler, LakeFS, MLflow, Spark History)
- Per-app session durations (shorter for Vault/Trino/Doris)
- Policies that allow only company email domains by default (overrideable)
- Optional restrictions by IdP and country codes
- NGINX ingress annotations pointing to your Zero Trust Team domain

## Prerequisites

- Cloudflare Zero Trust account and Team name (e.g., `qagi`)
- API token with Access: Apps (write) scope
- Account ID (from Zero Trust → Settings → Account)
- `jq` installed locally

## Provision apps via script

Use `scripts/create-cloudflare-access-apps.sh`:

```bash
export CLOUDFLARE_API_TOKEN=REDACTED
export CLOUDFLARE_ACCOUNT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Team-domain mode (use Zero Trust domain like qagi.cloudflareaccess.com)
./scripts/create-cloudflare-access-apps.sh \
  --mode team \
  --team-domain qagi.cloudflareaccess.com \
  --allowed-email-domains 254carbon.com

# Zone mode (recommended – apps live at <sub>.254carbon.com)
./scripts/create-cloudflare-access-apps.sh \
  --mode zone \
  --zone-domain 254carbon.com \
  --allowed-email-domains 254carbon.com

# Add stronger constraints (optional)
./scripts/create-cloudflare-access-apps.sh \
  --mode team \
  --team-domain qagi.cloudflareaccess.com \
  --allowed-email-domains 254carbon.com \
  --idp-id 11111111-2222-3333-4444-555555555555 \
  --countries US,GB,DE \
  --force

# Dry-run to inspect payloads only
./scripts/create-cloudflare-access-apps.sh --mode zone --zone-domain 254carbon.com --dry-run
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

## Notes and tips

- Team domain is found at Zero Trust → Settings → Team (e.g., `qagi.cloudflareaccess.com`) and can be passed explicitly with `--team-domain`.
- If you choose zone mode (`--zone-domain 254carbon.com`), ensure your DNS zone is on Cloudflare and your tunnel/public hostnames route there.
- Zone mode automatically provisions apex (`254carbon.com`) and `www` alongside `portal.254carbon.com`; disable or remove them if a public landing page must remain open.
- You can re-run with `--force` to reconcile apps and policies idempotently.
- Vault and other sensitive apps use shorter sessions by default; adjust in the `APPLICATIONS` table in the script if needed.

## Troubleshooting

- 1033 errors: see `ERROR_1033_DIAGNOSIS_AND_FIX.md`
- Tunnel troubleshooting: `scripts/verify-tunnel.sh`
- Credentials update: `scripts/update-cloudflare-credentials.sh`
