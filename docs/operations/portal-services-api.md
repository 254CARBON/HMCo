# Portal Services API (ConfigMap + SSO-aware health)

This API powers the portal’s Service Directory. It reads a ConfigMap registry and exposes:

- GET `/api/services` — List of services (id, name, url, category, description, icon)
- GET `/api/services/status?mode=auto|external|internal` — Health snapshot with per-service status code, latency, and checkedAt
- POST `/api/services/reload` — Reloads the registry from the mounted ConfigMap
- POST `/api/services/discover?namespaces=data-platform,monitoring,vault-prod&mode=merge|replace` — Discover from Ingress annotations and merge or replace the registry
- GET `/healthz` — Liveness/readiness

## Deploy

1) Build and push the image (or use your registry)
- `docker build -t 254carbon/portal-services:latest services/portal-services`
- `docker push 254carbon/portal-services:latest`

2) Apply manifests
- `kubectl apply -f k8s/portal/portal-services.yaml`
- `kubectl apply -f k8s/ingress/portal-ingress.yaml`

This adds a route on portal hosts for `/api/services` (Prefix), so `/api/services/status` works as well.

## ConfigMap format

ConfigMap `portal-services-registry` mounts `services.json` to `/config/services.json` inside the API container.

Each service entry supports:
- `id` (string) unique key
- `name`, `url`, `category`, `description`, `icon`
- `healthPath` path appended to `url` or `internalUrl`
- `useCloudflareAccess` true to try external SSO-protected check first
- `internalUrl` optional in-cluster URL used for `mode=internal` or fallback
- `timeoutMs` optional per-service timeout (default 6000)

Example entries are pre-seeded in `k8s/portal/portal-services.yaml`:1.

## SSO-aware health checks

- The API can probe external URLs protected by Cloudflare Access by including the service token headers:
  - `Cf-Access-Client-Id: $CF_ACCESS_CLIENT_ID`
  - `Cf-Access-Client-Secret: $CF_ACCESS_CLIENT_SECRET`
- Provide global headers via Secret `cloudflare-access-service-token` (keys `client_id`, `client_secret`).
- Per-service tokens: set `cfAccessSecretRef` on the service entry, e.g. `{ name, namespace, clientIdKey, clientSecretKey }`.
  - The API reads those Secrets via the Kubernetes API with a dedicated ServiceAccount and RBAC (see k8s/portal/portal-services.yaml:1).
  - Only `get` access to exact Secret names is granted.

Probe strategy
- `mode=auto` (default): if `useCloudflareAccess=true`, tries external URL with CF Access headers; else uses `internalUrl`.
- `mode=external`: force external via `url` and Access headers if present.
- `mode=internal`: force in-cluster `internalUrl`.

Response shape
- `[ { id, status: ok|warn|err|unknown, code, latencyMs, checkedAt, url } ]`

## Portal integration

- Frontend fetches `/api/services` to list; `/api/services/status` to show status chips.
- Keyboard palette can call `Open: <service>` by using the listed URLs.
- No CORS issues: requests are same-origin via portal ingress.

## Operational notes

- Update the registry: `kubectl edit cm portal-services-registry -n data-platform` and call `POST /api/services/reload` or restart the pod.
- For per-service tokens, create secrets like `cf-access-<service>` in the appropriate namespace with keys `client_id` and `client_secret`, then ensure the Role `portal-services-secrets-read` has those `resourceNames`.
- Tighten RBAC if exposing in restricted namespaces.

## Ingress annotations for auto-discovery

Add annotations (prefix `portal.254carbon.com/`) on your Ingress to auto-discover services:
- `service-id`: stable id (default: leftmost host label)
- `service-name`: display name (default: capitalized id)
- `service-category`: e.g., Catalog, BI, Monitoring, OLAP, SQL, Security, Data Lake, Orchestration, Storage
- `service-description`: short description
- `service-icon`: emoji/icon text (e.g., 📊)
- `service-health-path`: path for health checks (default `/`)
- `use-cloudflare-access`: `true`/`false` (default `true`)
- `cf-access-secret-name`: Secret name with client_id/client_secret
- `cf-access-secret-namespace`: Secret namespace (default: Ingress namespace)
- `cf-access-client-id-key`: key in the Secret (default `client_id`)
- `cf-access-client-secret-key`: key in the Secret (default `client_secret`)

Example:
```
metadata:
  annotations:
    portal.254carbon.com/service-id: "grafana"
    portal.254carbon.com/service-name: "Grafana"
    portal.254carbon.com/service-category: "Monitoring"
    portal.254carbon.com/service-description: "Observability dashboards and alerts."
    portal.254carbon.com/service-icon: "📈"
    portal.254carbon.com/service-health-path: "/api/health"
    portal.254carbon.com/use-cloudflare-access: "false"
    portal.254carbon.com/cf-access-secret-name: "cf-access-grafana"
    portal.254carbon.com/cf-access-secret-namespace: "monitoring"
```

RBAC: The service account `portal-services` is granted list/get for Ingress in `data-platform`, `monitoring`, and `vault-prod`. Adjust `k8s/portal/portal-services.yaml`:1 if you use different namespaces.

Auto-discovery options
- Endpoint: `POST /api/services/discover?namespaces=<csv>&mode=merge|replace`
- Env: `DISCOVER_ON_START=true` to discover at boot; `DISCOVERY_NAMESPACES` to set namespaces (default `data-platform,monitoring,vault-prod`).
- If you run Cloudflare Zero Trust policies, ensure the service token is scoped least-privilege and rotated on a schedule.
