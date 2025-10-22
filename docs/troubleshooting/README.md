# Troubleshooting Playbooks

Operational incident guides and remediation notes collected from recent work. Each file captures the timeline, root cause, and fix steps for a specific issue. Use this directory when you need to diagnose an issue quickly or trace past resolution work.

## Quick Lookup

- **Portal / Authentication**
  - `502-error-investigation.md` – Pod crashes and image pull issues
  - `502-after-auth-resolution.md` – Portal API missing backend deployment
  - `redirect-loop.md` – Loop caused by conflicting TLS redirects
  - `hanging-after-auth.md` – Network policy blocking ingress traffic
  - `portal-all-issues-resolved.md` – Consolidated checklist covering all fixes
  - `sso-portal-deployment-fixes.md` – Build and deployment fixes for the portal app

- **Gateway / Timeouts**
  - `504-timeout.md` – NGINX timeout configuration
  - `connectivity-immediate-remediation.md` – Quick actions for network flakiness
  - `network-issue-index.md` & `network-issue-summary.md` – Deep-dive on Kind networking failures

## Related Resources

- Cloudflare tunnel operations and troubleshooting: `docs/cloudflare/troubleshooting.md`
- SSO implementation steps and validation: `docs/sso/`
- Historical incident timelines: `docs/history/incidents/`
