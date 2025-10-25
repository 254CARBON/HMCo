# OIDC SSO Configuration (Grafana, Superset, DataHub, ArgoCD)

The platform now authenticates the core user-facing services with the Keycloak realm at `https://auth.254carbon.com/auth/realms/master`.  
This document captures the settings, required secrets, and RBAC alignment for each application.

## Identity Provider Details

- **Issuer**: `https://auth.254carbon.com/auth/realms/master`
- **Authorization URL**: `.../protocol/openid-connect/auth`
- **Token URL**: `.../protocol/openid-connect/token`
- **UserInfo URL**: `.../protocol/openid-connect/userinfo`
- **Logout URL**: `.../protocol/openid-connect/logout`
- **Scopes**: `openid profile email groups`
- **Group claim**: `groups`
- **Email claim**: `email`
- **Name claim**: `name`
- **Username claim**: `email`

OIDC groups are mapped consistently across services:

| OIDC Group        | Role/Permissions                           |
|-------------------|---------------------------------------------|
| `platform-admins` | Full admin rights                           |
| `data-engineers`  | Editor / deploy permissions                 |
| `data-analysts`   | Read-only access                            |

## Application Configuration Summary

| Service   | Namespace       | Client ID           | Redirect URI                                             | Secret (namespace/key)                             |
|-----------|-----------------|---------------------|-----------------------------------------------------------|----------------------------------------------------|
| Grafana   | `monitoring`    | `grafana`           | `https://grafana.254carbon.com/login/generic_oauth`       | `monitoring/grafana-oidc-secret:client-secret`      |
| Superset  | `data-platform` | `superset`          | `https://superset.254carbon.com/oauth-authorized/oidc`    | `data-platform/superset-oidc-secret:client-secret`  |
| DataHub   | `data-platform` | `datahub`           | `https://datahub.254carbon.com/callback/oidc`             | `data-platform/datahub-oidc-secret:client-secret`   |
| ArgoCD    | `argocd`        | `argocd`            | `https://argocd.254carbon.com/auth/callback` (implicit)   | `argocd-secret:oidc.clientSecret` (merge)           |

Staging clients follow the same structure with `*-staging` suffixes and environment-specific hostnames (e.g., `grafana-staging.254carbon.com`). Ensure the corresponding Kubernetes secrets (`grafana-staging-oidc-secret`, `superset-staging-oidc-secret`, `datahub-staging-oidc-secret`) are populated before syncing staging releases.

> **Important:** Secrets are not checked into the repo. Create them manually before syncing ArgoCD.

### Secret Creation Snippets

```bash
# Grafana
kubectl -n monitoring create secret generic grafana-oidc-secret \
  --from-literal=client-secret='<grafana-client-secret>'

# Superset
kubectl -n data-platform create secret generic superset-oidc-secret \
  --from-literal=client-secret='<superset-client-secret>'

# DataHub
kubectl -n data-platform create secret generic datahub-oidc-secret \
  --from-literal=client-secret='<datahub-client-secret>'

# ArgoCD (merge into existing secret)
kubectl -n argocd patch secret argocd-secret \
  --type merge \
  --patch '{"stringData":{"oidc.clientSecret":"<argocd-client-secret>"}}'

# Staging variants (example)
kubectl -n monitoring create secret generic grafana-staging-oidc-secret \
  --from-literal=client-secret='<grafana-staging-client-secret>'
kubectl -n data-platform-staging create secret generic superset-staging-oidc-secret \
  --from-literal=client-secret='<superset-staging-client-secret>'
kubectl -n data-platform-staging create secret generic datahub-staging-oidc-secret \
  --from-literal=client-secret='<datahub-staging-client-secret>'
```

Replace each `<...>` with the client secret provisioned in Keycloak.

## ArgoCD RBAC Mapping

Apply the ArgoCD config map and RBAC:

```bash
kubectl apply -f k8s/gitops/argocd-oidc.yaml
```

`k8s/gitops/argocd-oidc.yaml` configures:

- OIDC authentication against the Keycloak realm.
- RBAC policies aligned with the shared groups:
  - `platform-admins` → `role:admin`
  - `data-engineers` → `role:editor` (create/update/sync applications)
  - `data-analysts` → `role:readonly`
- Default policy remains read-only for all other users.

After creating the client secret, sync the new config map and restart `argocd-server` to apply changes:

```bash
kubectl -n argocd rollout restart deploy/argocd-server
```

## Verification Checklist

1. **Grafana**: Navigate to `https://grafana.254carbon.com` and confirm redirect to Keycloak; verify role mapping via `Settings → Users`.
2. **Superset**: Confirm login redirects through OIDC; check `Your Profile → Roles` reflects mapped role.
3. **DataHub**: Login via SSO and ensure the top-right user menu shows the Keycloak identity; optional—verify groups under **Settings → Policies & Access**.
4. **ArgoCD**: Access `https://argocd.254carbon.com`, authenticate via SSO, and confirm permissions align with your group membership.

If any application fails to redirect, double-check the client ID/secret, redirect URI registration in Keycloak, and the secret values in the cluster.
