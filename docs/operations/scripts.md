# Scripts Index

Purpose and quick usage for operational scripts in `scripts/`.

Cloudflare & SSO
- `scripts/update-cloudflare-credentials.sh` — Rotates tunnel credentials and restarts cloudflared
  - See [Cloudflare credentials guide](../cloudflare/credentials.md)
- `scripts/setup-cloudflare-dns.sh` — Creates/updates Cloudflare DNS records for exposed services
- `scripts/update-tunnel-routes.sh` — Refreshes Cloudflare tunnel ingress routes via API
- `scripts/create-cloudflare-access-apps.sh` — Helper to create Access apps (if used)
- `scripts/sso-setup-phase2.sh` — Applies ingress updates and service-side auth changes after Phase 2

Platform Operations
- `scripts/mirror-images.sh` — Mirrors container images to a private registry
- `scripts/setup-private-registry.sh` — Bootstraps Harbor/ECR/GCR/ACR access
- `scripts/initialize-vault-production.sh` — Initializes and configures Vault for production
- `scripts/phase4-testing.sh` — Runs Phase 4 end-to-end tests
- `scripts/verify-tunnel.sh` — Checks tunnel health and common failure modes

Backup & DR
- `scripts/deploy-velero-backup.sh` — Installs Velero, creates credentials secret, and applies backup schedules
- `scripts/velero-restore-validate.sh` — Automates namespace/full restores with optional cleanup and waiting

See also
- [Operations runbook](deployment/operations-runbook.md) — Operational procedures
- [Deployment troubleshooting](deployment/troubleshooting.md) — Common issues and fixes
