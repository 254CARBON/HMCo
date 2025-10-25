# Vault External Secrets Integration

This document explains how Vault now feeds application credentials into the Kubernetes cluster by way of the External Secrets Operator. The Helm chart `helm/charts/vault` deploys Vault (HA Raft), the Vault Agent injector, and the External Secrets Operator, while each application chart declares the `ExternalSecret` resources it needs. Secrets are no longer committed to Git – they live in Vault and are synced on demand into Kubernetes Secrets.

## Components

1. **Vault** (`vault` namespace)
   - HA Raft storage with optional TLS
   - Vault Agent injector enabled for future sidecar use
   - Helper script `scripts/initialize-vault-production.sh` configures Kubernetes auth, enables KV v2/database engines, and now creates the `external-secrets-read` policy + Kubernetes role.
2. **External Secrets Operator** (`vault` namespace)
   - Watches the `ClusterSecretStore/platform-vault` definition emitted by the Vault Helm chart
   - Uses Kubernetes auth to obtain short-lived Vault tokens tied to the `external-secrets-read` policy
3. **Application ExternalSecrets**
   - `helm/charts/data-platform/templates/externalsecrets.yaml` renders ExternalSecret resources for every data-platform/MLflow secret
   - Sub-charts (Superset, MLflow, etc.) skip shipping hard-coded `Secret` manifests when Vault-backed sync is enabled

## Vault Paths & Secret Mapping

All application credentials are stored under the KV v2 mount path `secret/`. The following table lists the expected Vault document paths and properties alongside the Kubernetes secret that will be created.

| Vault Path | Properties (keys) | Kubernetes Secret |
|------------|-------------------|-------------------|
| `secret/data-platform/minio` | `access-key`, `secret-key` | `minio-secret`
| `secret/data-platform/postgres/shared` | `password` | `postgres-shared-secret`
| `secret/data-platform/postgres/workflow` | `password` | `postgres-workflow-secret`
| `secret/data-platform/datahub` | `DATAHUB_SECRET` | `datahub-secret`
| `secret/data-platform/superset` | `secret-key`, `database-uri`, `admin-password`, plus the uppercase Superset keys (`SUPERSET_SECRET_KEY`, `DATABASE_URI`, `DATABASE_USERNAME`, `DATABASE_PASSWORD`, `SUPERSET_DORIS_URI`, `SUPERSET_TRINO_URI`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `ADMIN_EMAIL`, `ADMIN_FIRSTNAME`, `ADMIN_LASTNAME`) | `superset-secret`, `superset-secrets`
| `secret/data-platform/mlflow/backend` | `db_user`, `db_password`, `db_host`, `db_port`, `db_name`, `backend_store_uri` | `mlflow-backend-secret`
| `secret/data-platform/mlflow/artifact` | `aws_access_key_id`, `aws_secret_access_key`, `s3_endpoint_url`, `artifact_root` | `mlflow-artifact-secret`

> **Tip:** the external-secrets policy grants access to the entire `secret/data-platform/*` tree. If you add a new secret, drop it anywhere under this path, add the mapping to `helm/charts/data-platform/values.yaml`, and re-sync ArgoCD.

## Loading Secrets into Vault

After deploying the updated Helm charts:

1. Run the initializer script (once Vault pods are running and unsealed):

   ```bash
   ./scripts/initialize-vault-production.sh init
   # or, for existing clusters
   ./scripts/initialize-vault-production.sh config
   ```

   This now enables Kubernetes auth, configures secret engines, and writes the policy/role used by the External Secrets Operator.

2. Login to Vault using the root token (or an operator token with the right capabilities):

   ```bash
   kubectl exec -n vault-prod vault-0 -- vault login
   ```

3. Populate the required secrets. Example (replace placeholder values):

   ```bash
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/data-platform/minio \
       access-key="REDACTED" \
       secret-key="REDACTED"

   kubectl exec -n vault-prod vault-0 -- vault kv put secret/data-platform/postgres/shared \
       password="REDACTED"

   kubectl exec -n vault-prod vault-0 -- vault kv put secret/data-platform/superset \
       SUPERSET_SECRET_KEY="$(openssl rand -hex 32)" \
       DATABASE_URI="postgresql://superset:..." \
       DATABASE_USERNAME="superset" \
       DATABASE_PASSWORD="REDACTED" \
       ADMIN_USERNAME="admin" \
       ADMIN_PASSWORD="REDACTED" \
       ADMIN_EMAIL="admin@example.com" \
       ADMIN_FIRSTNAME="Superset" \
       ADMIN_LASTNAME="Admin"
   ```

4. Verify sync:

   ```bash
   kubectl get externalsecret -n data-platform
   kubectl describe externalsecret minio-secret -n data-platform
   kubectl get secret minio-secret -n data-platform -o yaml
   ```

   Status should report `Ready: True`, and the Kubernetes `Secret` objects will be recreated automatically if rotated in Vault.

## Toggling Vault-backed Secrets

- Set `global.vault.enabled=false` (or `vault.externalSecrets.enabled=false`) in the Helm values to fall back to static Kubernetes `Secret` manifests.
- The Superset and MLflow subcharts automatically skip their embedded `Secret` templates when Vault federation is enabled. This prevents double-definition conflicts.
- Additional applications can be migrated by adding entries to `vault.externalSecrets.secrets` and, if necessary, gating their legacy `Secret` manifests behind the same `global.vault.enabled` flag.

## Troubleshooting

- `ExternalSecret` stuck in `ErrorObserving` – confirm the Vault policy/role exists (`vault policy read external-secrets-read`) and the Kubernetes `ServiceAccount` `external-secrets-operator` in namespace `vault` has tokens.
- `permission denied` errors – ensure the secret path matches exactly (`secret/data/data-platform/...`). KV v2 requires both the `/data/` and `/metadata/` rules to be present.
- Updating secret values – write new values to Vault; the operator refreshes every 1 hour by default. Force an immediate refresh by deleting the generated Kubernetes `Secret` object – the operator will recreate it using the latest Vault value.
