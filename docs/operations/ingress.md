# Ingress Rules Overview

Canonical reference for external routes, TLS, and annotations for 254Carbon services.

Source manifest
- k8s/ingress/ingress-rules.yaml

Hosts and backends (summary)
- datahub.254carbon.com → data-platform/datahub-frontend:9002 (+ /api → datahub-gms:8080)
- superset.254carbon.com → data-platform/superset:8088
- grafana.254carbon.com → monitoring/grafana:3000
- doris.254carbon.com → data-platform/doris-fe-service:8030
- trino.254carbon.com → data-platform/trino-coordinator:8080
- vault.254carbon.com → data-platform/vault:8200
- lakefs.254carbon.com → data-platform/lakefs:8000
- dolphin.254carbon.com → data-platform/dolphinscheduler-api:12345
- minio.254carbon.com → data-platform/minio-console:9001

TLS and annotations
- All hosts use `cert-manager.io/cluster-issuer: "letsencrypt-prod"`
- SSL enforced with `nginx.ingress.kubernetes.io/ssl-redirect: "true"`
- Vault sets backend-protocol HTTP via `nginx.ingress.kubernetes.io/backend-protocol: "HTTP"`

Related
- Cloudflare deployment: docs/cloudflare/deployment.md
- SSO/authentication annotations: docs/sso/guide.md

