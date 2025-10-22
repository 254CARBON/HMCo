# SSO Documentation Index (Canonical)

Welcome to the home page for every active SSO document. Follow this map to move from Phase 1 completion to full Cloudflare Access enforcement across all services.

---

## Start Here

1. **Overview & Timeline** → [`overview.md`](overview.md)  
   Recap of achievements to date, remaining scope, and delivery timeline.
2. **Implementation Guide** → [`guide.md`](guide.md)  
   Canonical step-by-step instructions for Phases 2‑4.
3. **Quick Reference** → [`quick-reference.md`](quick-reference.md)  
   Fast lookup for service ports, session durations, and verification commands.
4. **Checklist** → [`checklist.md`](checklist.md)  
   Task-by-task progress tracking with sign-off sections.
5. **Status & Automation** → [`status.md`](status.md)  
   Ready-state confirmation plus Phase 3/4 automation scripts.
6. **Cloudflare Access Configuration** → [`cloudflare-access.md`](cloudflare-access.md)  
   Exact application, policy, and session settings for all hostnames.
7. **Hands-on Quickstart** → [`quickstart.md`](quickstart.md)  
   Command-driven walkthrough for operators already familiar with the stack.

Keep the [Quick Reference](quick-reference.md) and [Checklist](checklist.md) open while executing the implementation guide.

---

## Document Map

| Area | Document | Purpose |
|------|----------|---------|
| Planning | [`overview.md`](overview.md) | Executive summary, effort estimates, and timeline |
| Execution | [`guide.md`](guide.md) | Detailed instructions for Phases 2‑4 |
| Execution | [`cloudflare-access.md`](cloudflare-access.md) | Cloudflare Access app matrix and policy details |
| Execution | [`quickstart.md`](quickstart.md) | Concise command flow per phase |
| Tracking | [`checklist.md`](checklist.md) | Mark completion of every task and verification |
| Reference | [`quick-reference.md`](quick-reference.md) | Service list, session durations, verification commands |
| Validation | [`validation.md`](validation.md) | End-to-end testing scripts and manual checks |
| Status | [`status.md`](status.md) | Pre-flight checklist, automation entry points, expected outcomes |

---

## Supporting Artifacts

- **Ingress & Kubernetes manifests**: `k8s/ingress/` and `k8s/cloudflare/` contain the resources referenced throughout the guide.
- **Automation scripts**: `scripts/sso-setup-phase3.sh` and `scripts/sso-validate-phase4.sh` power automated integration and test runs.
- **Portal application**: `portal/` hosts the Next.js portal used in all user-facing flows.

---

## Historical Records

Earlier delivery notes, phase summaries, and retrospectives remain in `docs/history/`. Reference them for background context; the live runbooks stay in this directory.

---

## Quick Routing

- Need detailed steps? → [`guide.md`](guide.md)  
- Configuring Cloudflare Access? → [`cloudflare-access.md`](cloudflare-access.md)  
- Looking for automation entry points? → [`status.md`](status.md)  
- Verifying progress? → [`checklist.md`](checklist.md) + [`quick-reference.md`](quick-reference.md)  
- Troubleshooting portal or ingress issues? → `docs/troubleshooting/`

---

Follow the documents in the order above to complete Phases 2‑4 and keep them on hand for ongoing operations.
