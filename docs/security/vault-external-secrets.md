# Vault External Secrets Integration

This document explains how Vault now feeds application credentials into the Kubernetes cluster by way of the External Secrets Operator. The Helm chart `helm/charts/vault` deploys Vault (HA Raft), the Vault Agent injector, and the External Secrets Operator, while each application chart declares the `ExternalSecret` resources it needs. Secrets are no longer committed to Git – they live in Vault and are synced on demand into Kubernetes Secrets.

## Components

1. **Vault** (`vault-prod` namespace)
   - HA Raft storage with optional TLS
   - Vault Agent injector enabled for future sidecar use
   - Helper script `scripts/initialize-vault-production.sh` configures:
     - Kubernetes authentication method
     - KV v2, database, and SSH secret engines
     - Least-privilege policies for each application and namespace
     - Service account role bindings using Kubernetes auth
     - Access control testing to verify cross-namespace isolation
2. **External Secrets Operator** (`vault` namespace)
   - Watches the `ClusterSecretStore/platform-vault` definition emitted by the Vault Helm chart
   - Uses Kubernetes auth to obtain short-lived Vault tokens tied to appropriate read policies
   - Syncs secrets from Vault into Kubernetes Secret objects on-demand
3. **Application ExternalSecrets**
   - `helm/charts/data-platform/templates/externalsecrets.yaml` renders ExternalSecret resources for all data-platform secrets
   - Additional charts (ClickHouse, Superset, MLflow, API Gateway, Monitoring, Cloudflare Tunnel) include ExternalSecret templates
   - Legacy `Secret` manifests are gated behind `vault.enabled=false` flag to prevent double-definition conflicts

## Vault Paths & Secret Mapping

All application credentials are stored under the KV v2 mount path `secret/`. The following table lists the expected Vault document paths and properties alongside the Kubernetes secret that will be created.

| Vault Path | Properties (keys) | Kubernetes Secret | Namespace |
|------------|-------------------|-------------------|-----------|
| `secret/data-platform/minio` | `access-key`, `secret-key` | `minio-secret` | data-platform |
| `secret/data-platform/postgres/shared` | `password` | `postgres-shared-secret` | data-platform |
| `secret/data-platform/postgres/workflow` | `password` | `postgres-workflow-secret` | data-platform |
| `secret/data-platform/datahub` | `DATAHUB_SECRET` | `datahub-secret` | data-platform |
| `secret/data-platform/clickhouse` | `password` | `clickhouse-secret` | data-platform |
| `secret/data-platform/superset` | `secret-key`, `database-uri`, `admin-password`, `SUPERSET_SECRET_KEY`, `DATABASE_URI`, `DATABASE_USERNAME`, `DATABASE_PASSWORD`, `SUPERSET_CLICKHOUSE_URI`, `SUPERSET_TRINO_URI`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `ADMIN_EMAIL`, `ADMIN_FIRSTNAME`, `ADMIN_LASTNAME` | `superset-secret`, `superset-secrets` | data-platform |
| `secret/data-platform/mlflow/backend` | `db_user`, `db_password`, `db_host`, `db_port`, `db_name`, `backend_store_uri` | `mlflow-backend-secret` | data-platform |
| `secret/data-platform/mlflow/artifact` | `aws_access_key_id`, `aws_secret_access_key`, `s3_endpoint_url`, `artifact_root` | `mlflow-artifact-secret` | data-platform |
| `secret/api-gateway/postgres` | `postgres-password`, `kong-password` | `kong-postgres` | kong |
| `secret/api-gateway/jwt` | `kongCredType`, `key`, `algorithm`, `rsa_public_key` | `portal-jwt-credential` | kong |
| `secret/monitoring/alertmanager` | `config` (YAML content) | `alertmanager-config` | monitoring |
| `secret/cloudflare/tunnel` | `token` | `cloudflare-tunnel-token` | default/cloudflare |

> **Note:** Least-privilege policies ensure each application can only read its own secrets. Cross-namespace and cross-application access is denied by Vault policies.

## Vault Initialization and Policy Configuration

The `scripts/initialize-vault-production.sh` script automates the complete Vault setup with least-privilege access control.

### First-Time Initialization

For new Vault deployments, run the full initialization:

```bash
./scripts/initialize-vault-production.sh init
```

This performs the following:
1. Initializes Vault with 3 key shares (2 threshold)
2. Unseals all Vault replicas
3. Enables Kubernetes authentication method
4. Configures secret engines (KV v2, database, SSH)
5. Creates least-privilege policies for each application and namespace
6. Binds service accounts to appropriate Vault roles
7. Tests access control to verify cross-namespace isolation

### Existing Vault Configuration

For already-initialized Vault instances, update policies and roles:

```bash
./scripts/initialize-vault-production.sh config
```

### Access Control Testing

Verify least-privilege policies are working correctly:

```bash
./scripts/initialize-vault-production.sh test-access
```

This tests:
- Each application can read its own secrets
- Cross-application reads are denied
- Cross-namespace reads are denied
- External Secrets Operator has appropriate broad access

## Vault Policies

The initialization script creates the following least-privilege policies:

| Policy Name | Scope | Purpose |
|-------------|-------|---------|
| `external-secrets-read` | `secret/data/data-platform/*` | External Secrets Operator - broad access to data-platform |
| `data-platform-read` | `secret/data/data-platform/*` | Generic data-platform service accounts |
| `clickhouse-read` | `secret/data/data-platform/clickhouse` | ClickHouse service account only |
| `mlflow-read` | `secret/data/data-platform/mlflow/*` | MLflow service account only |
| `superset-read` | `secret/data/data-platform/superset` | Superset service account only |
| `api-gateway-read` | `secret/api-gateway/*` | Kong API Gateway service account |
| `monitoring-read` | `secret/monitoring/*` | Alertmanager/Prometheus service accounts |
| `cloudflare-read` | `secret/cloudflare/*` | Cloudflare Tunnel service account |

## Service Account Role Bindings

Each service account is bound to its corresponding policy via Kubernetes authentication:

| Service Account | Namespace(s) | Policy | TTL |
|-----------------|-------------|--------|-----|
| `external-secrets-operator` | `vault` | `external-secrets-read` | 24h |
| `clickhouse` | `data-platform` | `clickhouse-read` | 1h |
| `mlflow` | `data-platform` | `mlflow-read` | 1h |
| `superset` | `data-platform` | `superset-read` | 1h |
| `kong` | `kong` | `api-gateway-read` | 1h |
| `alertmanager`, `prometheus` | `monitoring` | `monitoring-read` | 1h |
| `cloudflared` | `default`, `cloudflare` | `cloudflare-read` | 1h |

## Loading Secrets into Vault

After running the initialization script:

1. Login to Vault using the root token (or an operator token with write capabilities):

   ```bash
   kubectl exec -n vault-prod vault-0 -- vault login
   ```

2. Populate the required secrets for each application:

   ### Data Platform Secrets

   ```bash
   # MinIO credentials
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/data-platform/minio \
       access-key="minioadmin" \
       secret-key="REPLACE_WITH_STRONG_PASSWORD"

   # PostgreSQL shared database
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/data-platform/postgres/shared \
       password="REPLACE_WITH_STRONG_PASSWORD"

   # PostgreSQL workflow database
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/data-platform/postgres/workflow \
       password="REPLACE_WITH_STRONG_PASSWORD"

   # ClickHouse database
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/data-platform/clickhouse \
       password="REPLACE_WITH_STRONG_PASSWORD"

   # DataHub
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/data-platform/datahub \
       DATAHUB_SECRET="$(openssl rand -hex 32)"

   # Superset
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/data-platform/superset \
       secret-key="$(openssl rand -hex 32)" \
       database-uri="postgresql://superset:PASSWORD@postgres-shared-service.data-platform.svc.cluster.local:5432/superset" \
       admin-password="REPLACE_WITH_STRONG_PASSWORD" \
       SUPERSET_SECRET_KEY="$(openssl rand -hex 32)" \
       DATABASE_URI="postgresql://superset:PASSWORD@postgres-shared-service.data-platform.svc.cluster.local:5432/superset" \
       DATABASE_USERNAME="superset" \
       DATABASE_PASSWORD="REPLACE_WITH_STRONG_PASSWORD" \
       SUPERSET_CLICKHOUSE_URI="clickhouse://default:PASSWORD@clickhouse-service.data-platform:9000/default" \
       SUPERSET_TRINO_URI="trino://admin@trino-coordinator.data-platform:8080/iceberg/default?http_scheme=http" \
       ADMIN_USERNAME="admin" \
       ADMIN_PASSWORD="REPLACE_WITH_STRONG_PASSWORD" \
       ADMIN_EMAIL="admin@254carbon.com" \
       ADMIN_FIRSTNAME="Superset" \
       ADMIN_LASTNAME="Admin"

   # MLflow backend
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/data-platform/mlflow/backend \
       db_user="mlflow" \
       db_password="REPLACE_WITH_STRONG_PASSWORD" \
       db_host="postgres-shared-service.data-platform.svc.cluster.local" \
       db_port="5432" \
       db_name="mlflow" \
       backend_store_uri="postgresql://mlflow:PASSWORD@postgres-shared-service.data-platform.svc.cluster.local:5432/mlflow"

   # MLflow artifact storage
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/data-platform/mlflow/artifact \
       aws_access_key_id="minioadmin" \
       aws_secret_access_key="REPLACE_WITH_STRONG_PASSWORD" \
       s3_endpoint_url="http://minio-service.data-platform.svc.cluster.local:9000" \
       artifact_root="s3://mlflow-artifacts"
   ```

   ### API Gateway Secrets

   ```bash
   # Kong PostgreSQL credentials
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/api-gateway/postgres \
       postgres-password="REPLACE_WITH_STRONG_PASSWORD" \
       kong-password="REPLACE_WITH_STRONG_PASSWORD"

   # JWT credentials (generate RSA key pair first)
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/api-gateway/jwt \
       kongCredType="jwt" \
       key="portal-service-issuer" \
       algorithm="RS256" \
       rsa_public_key="-----BEGIN PUBLIC KEY-----
   MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...
   -----END PUBLIC KEY-----"
   ```

   ### Monitoring Secrets

   ```bash
   # Alertmanager configuration with SMTP credentials
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/monitoring/alertmanager \
       config="$(cat <<'EOF'
   global:
     resolve_timeout: 5m
     smtp_smarthost: 'smtp.example.com:587'
     smtp_from: 'alertmanager@254carbon.com'
     smtp_auth_username: 'alertmanager@254carbon.com'
     smtp_auth_password: 'REPLACE_WITH_SMTP_PASSWORD'
     smtp_require_tls: true
   route:
     group_by: ['alertname', 'namespace', 'service']
     group_wait: 30s
     group_interval: 5m
     repeat_interval: 12h
     receiver: 'team-email'
   receivers:
   - name: 'team-email'
     email_configs:
     - to: 'ops@254carbon.com'
   EOF
   )"
   ```

   ### Cloudflare Tunnel Secret

   ```bash
   # Cloudflare Tunnel token
   kubectl exec -n vault-prod vault-0 -- vault kv put secret/cloudflare/tunnel \
       token="REPLACE_WITH_CLOUDFLARE_TUNNEL_TOKEN"
   ```

3. Verify ExternalSecret sync:

   ```bash
   # Check ExternalSecret status
   kubectl get externalsecret -n data-platform
   kubectl get externalsecret -n kong
   kubectl get externalsecret -n monitoring
   
   # Verify a specific ExternalSecret
   kubectl describe externalsecret minio-secret -n data-platform
   
   # Check that Kubernetes Secret was created
   kubectl get secret minio-secret -n data-platform
   
   # Verify only managed secrets exist (no literal credentials)
   kubectl get externalsecrets,secret -A | grep -v kubernetes.io
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
