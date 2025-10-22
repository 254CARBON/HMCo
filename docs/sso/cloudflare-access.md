# SSO Phase 2: Cloudflare Access Configuration – Implementation Guide

## Overview

Cloudflare Access acts as the policy gate in front of every externally exposed service on `254carbon.com`. This guide captures the exact application, policy, and session settings required for Phase 2 of the SSO rollout and shows how to provision them either via automation or the Cloudflare Zero Trust dashboard.

## Current Status

- ✅ Cloudflare Tunnel online (deployment: `cloudflared`, 2 replicas)  
- ✅ Portal and service ingresses deployed for `254carbon.com`  
- ⏳ Cloudflare Access applications and policies still need to be created

## Access Applications Required

Create one Access application per hostname (12 service hosts + optional apex/`www`). All applications use **Decision: Allow → Include: Company email domains** unless stated otherwise.

| # | Hostname | Application name | Session duration | Policy name |
|---|----------|------------------|------------------|-------------|
| 1 | `portal.254carbon.com` | `254Carbon Portal` | 24h | `Allow Portal Access` |
| 2 | `254carbon.com` | `254Carbon Root` | 24h | `Allow Portal Access` |
| 3 | `www.254carbon.com` | `254Carbon WWW` | 24h | `Allow Portal Access` |
| 4 | `grafana.254carbon.com` | `Grafana.254Carbon` | 24h | `Allow Grafana Access` |
| 5 | `superset.254carbon.com` | `Superset.254Carbon` | 24h | `Allow Superset Access` |
| 6 | `datahub.254carbon.com` | `DataHub.254Carbon` | 12h | `Allow DataHub Access` |
| 7 | `trino.254carbon.com` | `Trino.254Carbon` | 8h | `Allow Trino Access` |
| 8 | `doris.254carbon.com` | `Doris.254Carbon` | 8h | `Allow Doris Access` |
| 9 | `vault.254carbon.com` | `Vault.254Carbon` | 2h | `Allow Vault Access` |
|10 | `minio.254carbon.com` | `MinIO.254Carbon` | 8h | `Allow MinIO Access` |
|11 | `dolphin.254carbon.com` | `DolphinScheduler.254Carbon` | 12h | `Allow DolphinScheduler Access` |
|12 | `lakefs.254carbon.com` | `LakeFS.254Carbon` | 12h | `Allow LakeFS Access` |
|13 | `mlflow.254carbon.com` | `MLflow.254Carbon` | 12h | `Allow MLflow Access` |
|14 | `spark-history.254carbon.com` | `Spark History.254Carbon` | 12h | `Allow Spark History Access` |

> **Policy template**  
> - **Decision**: `Allow`  
> - **Include**: `Email domain -> 254carbon.com` (add additional domains or specific emails as needed)  
> - **Require** *(optional)*: `Email` (OTP) and/or IdP login method  
> - **Session duration**: Set per table above  

## Recommended: Automated Provisioning

1. Ensure you have an API token with **Access: Apps (write)** scope and the 32‑character Cloudflare **Account ID**.
2. Export the variables and run the helper script in zone mode:

```bash
export CLOUDFLARE_API_TOKEN=YOUR_API_TOKEN
export CLOUDFLARE_ACCOUNT_ID=YOUR_ACCOUNT_ID

# Optional policy inputs (comma separated lists)
export CLOUDFLARE_ACCESS_ALLOWED_EMAIL_DOMAINS=254carbon.com
# export CLOUDFLARE_ACCESS_ALLOWED_EMAILS=person@254carbon.com
# export CLOUDFLARE_ACCESS_IDP_ID=11111111-2222-3333-4444-555555555555

./scripts/create-cloudflare-access-apps.sh \
  --mode zone \
  --zone-domain 254carbon.com \
  --force
```

- Use `--dry-run` first to inspect payloads without creating apps.  
- Set `CLOUDFLARE_ACCESS_MODE=team` with `--team-domain <team>.cloudflareaccess.com` if you prefer the Zero Trust team domain instead of the public zone.

The script is idempotent; rerun with `--force` to reconcile settings if changes were made manually.

## Manual Dashboard Configuration

1. Log in to <https://dash.cloudflare.com> and open **Zero Trust → Access → Applications**.
2. For each hostname listed in the table:
   - Click **Add an application → Self-hosted**.
   - In **Domain**, enter the full hostname (e.g., `grafana.254carbon.com`).  
     The UI will automatically recognize the `254carbon.com` zone.
   - Set **Session Duration** according to the table.
   - Leave **App launcher visibility** enabled and **HTTP Only Cookie** enabled.
3. Under **Policies**, create (or reuse) the policy shown in the template above.
4. Save the application and repeat for the remaining hostnames.

### Enable Audit Logging

1. Navigate to **Zero Trust → Settings → Logpush**.
2. Enable Access audit logs (Logpush or email notifications).
3. Confirm logs are being delivered to the configured destination.

## Verification Checklist

- `curl -I https://portal.254carbon.com` → `302` to `https://<account>.cloudflareaccess.com/cdn-cgi/access/login`
- Cloudflare Zero Trust → **Access → Applications** shows all 14 entries in **Healthy** state.
- Attempting to reach any service prompts Cloudflare Access email/OTP (or the configured IdP).
- After login, services load successfully and Kubernetes ingress annotations continue to reference `https://<team>.cloudflareaccess.com/cdn-cgi/access/...`.

## Generated URLs (Post Authentication)

- Portal: `https://portal.254carbon.com` and `https://254carbon.com`
- Grafana: `https://grafana.254carbon.com`
- Superset: `https://superset.254carbon.com`
- DataHub: `https://datahub.254carbon.com`
- Trino: `https://trino.254carbon.com`
- Doris: `https://doris.254carbon.com`
- Vault: `https://vault.254carbon.com`
- MinIO: `https://minio.254carbon.com`
- DolphinScheduler API: `https://dolphin.254carbon.com`
- lakeFS: `https://lakefs.254carbon.com`
- MLflow: `https://mlflow.254carbon.com`
- Spark History: `https://spark-history.254carbon.com`

## Next Steps

Once all Access applications are live:

1. Apply the SSO-enabled ingress manifests (`k8s/ingress/ingress-sso-rules.yaml`).
2. Execute **Phase 3** automation: `./scripts/sso-setup-phase3.sh`.
3. Run **Phase 4** validation: `./scripts/sso-validate-phase4.sh`.
