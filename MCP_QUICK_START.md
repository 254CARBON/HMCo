# 254Carbon MCP Servers - Quick Start Setup Guide
**Last Updated**: October 24, 2025  
**Setup Time**: 30 minutes  
**Difficulty**: Intermediate

---

## Step 1: Install Required Tools

Before setting up MCP servers, ensure you have these installed:

```bash
# Install Node.js (v18+) - required for MCP
node --version  # Should be v18+

# Install global npm packages for MCP
npm install -g @anthropic-sdks/mcp

# Verify kubectl (for K8s access)
kubectl version --client

# Verify Helm (for chart operations)
helm version
```

---

## Step 2: Configure Cursor Settings

### Access Cursor Settings

1. Open Cursor
2. Press `Cmd+,` (macOS) or `Ctrl+,` (Linux/Windows)
3. Search for "MCP" or scroll to Extensions
4. Click "Edit in settings.json" or find the MCP configuration section

### Add MCP Servers Configuration

Paste this into your Cursor settings (typically `~/.cursor/settings.json` or via GUI):

```json
{
  "mcp": {
    "servers": {
      "kubernetes": {
        "command": "npx",
        "args": ["-y", "@anthropic-sdks/mcp-kubernetes"],
        "enabled": true,
        "priority": 1
      },
      "docker": {
        "command": "npx",
        "args": ["-y", "@anthropic-sdks/mcp-docker"],
        "enabled": true,
        "priority": 2
      },
      "github": {
        "command": "npx",
        "args": ["-y", "@anthropic-sdks/mcp-github"],
        "env": {
          "GITHUB_TOKEN": "YOUR_GITHUB_TOKEN_HERE"
        },
        "enabled": true,
        "priority": 3
      },
      "postgresql": {
        "command": "npx",
        "args": ["-y", "@anthropic-sdks/mcp-postgresql"],
        "env": {
          "PG_HOST": "localhost",
          "PG_PORT": "5432",
          "PG_DATABASE": "postgres",
          "PG_USER": "postgres",
          "PG_PASSWORD": "your_postgres_password"
        },
        "enabled": true,
        "priority": 4
      }
    }
  }
}
```

---

## Step 3: Get Required Credentials

### GitHub Token
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token with scopes: `repo`, `workflow`, `read:org`
3. Copy token and add to Cursor settings

### PostgreSQL Credentials
1. Port-forward to PostgreSQL:
   ```bash
   kubectl port-forward -n data-platform svc/kong-postgresql 5432:5432 &
   ```
2. Get credentials from Vault or Kubernetes secret:
   ```bash
   kubectl get secret -n data-platform kong-postgresql-secret -o jsonpath='{.data.password}' | base64 -d
   ```

### Vault Token (optional, for secrets)
1. Login to Vault:
   ```bash
   vault login -method=kubernetes role=254carbon
   ```
2. Copy token from `~/.vault-token`

---

## Step 4: Test MCP Connections

After restarting Cursor, test each MCP server:

### Test Kubernetes MCP
```bash
# In Cursor terminal or chat:
@kubernetes list-pods -n data-platform
```

**Expected output**: List of pods in data-platform namespace

### Test GitHub MCP
```bash
@github list-repositories --owner 254CARBON --limit 5
```

**Expected output**: List of your GitHub repositories

### Test PostgreSQL MCP
```bash
@postgresql query --database postgres "SELECT version();"
```

**Expected output**: PostgreSQL version info

### Test Docker MCP
```bash
@docker list-containers
```

**Expected output**: List of running/stopped containers

---

## Step 5: Phase 1 - Install CRITICAL Servers (Week 1)

Add these to your Cursor settings progressively:

### Configuration Template (Copy & Adapt)

```json
{
  "mcp": {
    "servers": {
      "kubernetes": {
        "command": "npx",
        "args": ["-y", "@anthropic-sdks/mcp-kubernetes"],
        "enabled": true
      },
      "docker": {
        "command": "npx",
        "args": ["-y", "@anthropic-sdks/mcp-docker"],
        "enabled": true
      },
      "helm": {
        "command": "npx",
        "args": ["-y", "@anthropic-sdks/mcp-helm"],
        "enabled": true
      },
      "argocd": {
        "command": "npx",
        "args": ["-y", "@anthropic-sdks/mcp-argocd"],
        "env": {
          "ARGOCD_SERVER": "argocd.254carbon.com",
          "ARGOCD_AUTH_TOKEN": "YOUR_ARGOCD_TOKEN"
        },
        "enabled": true
      },
      "vault": {
        "command": "npx",
        "args": ["-y", "@anthropic-sdks/mcp-vault"],
        "env": {
          "VAULT_ADDR": "https://vault.254carbon.com",
          "VAULT_TOKEN": "YOUR_VAULT_TOKEN"
        },
        "enabled": true
      }
    }
  }
}
```

---

## Step 6: Phase 2 - Add RECOMMENDED Servers (Week 2-3)

Once Phase 1 is stable, add:

```json
{
  "grafana": {
    "command": "npx",
    "args": ["-y", "@anthropic-sdks/mcp-grafana"],
    "env": {
      "GRAFANA_URL": "https://grafana.254carbon.com",
      "GRAFANA_API_KEY": "YOUR_GRAFANA_API_KEY"
    },
    "enabled": true
  },
  "prometheus": {
    "command": "npx",
    "args": ["-y", "@anthropic-sdks/mcp-prometheus"],
    "env": {
      "PROMETHEUS_URL": "http://prometheus.monitoring:9090"
    },
    "enabled": true
  },
  "slack": {
    "command": "npx",
    "args": ["-y", "@anthropic-sdks/mcp-slack"],
    "env": {
      "SLACK_BOT_TOKEN": "xoxb-YOUR_SLACK_TOKEN"
    },
    "enabled": true
  },
  "kafka": {
    "command": "npx",
    "args": ["-y", "@anthropic-sdks/mcp-kafka"],
    "env": {
      "KAFKA_BROKERS": "kafka-0.kafka-headless.data-platform:9092,kafka-1.kafka-headless.data-platform:9092,kafka-2.kafka-headless.data-platform:9092"
    },
    "enabled": true
  }
}
```

---

## Step 7: Common Usage Patterns

### Pattern 1: Deploy a New Service Version

```
User: Deploy a new version of portal-services using Helm
@helm upgrade portal-services ./helm/charts/portal-services \
  --set image.tag=v1.2.0 \
  --namespace data-platform

@kubernetes wait-deployment portal-services -n data-platform
@grafana query-dashboard "Portal Services Health"
```

### Pattern 2: Debug a Pod Issue

```
User: Why is the trino-coordinator pod not starting?
@kubernetes describe-pod trino-coordinator -n data-platform
@kubernetes logs trino-coordinator -n data-platform --tail 100
@postgres query "SELECT * FROM pg_stat_statements LIMIT 10"
```

### Pattern 3: Query Data Lake

```
User: Show me recent files in MinIO data lake
@minio list-objects --bucket data-lake --recursive
@trino execute "SELECT * FROM iceberg.data_lake.commodities LIMIT 10"
```

### Pattern 4: Monitor Performance

```
User: What's the current CPU/memory usage of all data-platform pods?
@prometheus query 'container_cpu_usage_seconds_total{namespace="data-platform"}'
@grafana dashboard-info "Data Platform Overview"
@loki query '{namespace="data-platform"} | rate(bytes_written[5m])'
```

### Pattern 5: Manage Credentials

```
User: Rotate the MLflow database password
@vault read secret/data/mlflow/postgres/password
@vault write secret/data/mlflow/postgres/password password=new_secure_password
@kubernetes restart-deployment mlflow -n data-platform
```

---

## Step 8: Environment-Specific Setup

### Local Development

```bash
# Forward services locally
kubectl port-forward -n data-platform svc/trino 8080:8080 &
kubectl port-forward -n data-platform svc/kafka 9092:9092 &
kubectl port-forward -n data-platform svc/minio 9000:9000 &

# In Cursor, update MCP servers to use localhost:
# PostgreSQL: PG_HOST=localhost
# Trino: TRINO_HOST=localhost:8080
# Kafka: KAFKA_BROKERS=localhost:9092
```

### Staging Environment

```bash
# Set KUBECONFIG to staging cluster
export KUBECONFIG=~/.kube/staging-config.yaml

# Update MCP URLs
ARGOCD_SERVER=argocd-staging.254carbon.com
GRAFANA_URL=https://grafana-staging.254carbon.com
VAULT_ADDR=https://vault-staging.254carbon.com
```

### Production Environment

```bash
# Set KUBECONFIG to production cluster
export KUBECONFIG=~/.kube/prod-config.yaml

# Update MCP URLs with prod endpoints
ARGOCD_SERVER=argocd.254carbon.com
GRAFANA_URL=https://grafana.254carbon.com
VAULT_ADDR=https://vault.254carbon.com
```

---

## Step 9: Troubleshooting

### MCP Server Not Connecting

**Symptom**: "MCP server unavailable" error

**Solutions**:
```bash
# 1. Check if service is running
kubectl get pod -n argocd  # For ArgoCD
kubectl get pod -n data-platform -l app=postgres  # For PostgreSQL

# 2. Check network connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  sh -c "curl -v http://postgres.data-platform:5432"

# 3. Verify credentials in Cursor settings
vault token lookup  # Check Vault token expiry
echo $GITHUB_TOKEN | wc -c  # Verify token length
```

### Kubernetes Context Switching

**Problem**: Cursor connects to wrong cluster

**Solution**:
```bash
# Set default context
kubectl config use-context production

# Verify in Cursor MCP - it should detect context automatically
@kubernetes current-context
```

### PostgreSQL Connection Issues

**Problem**: "Connection refused" from PostgreSQL MCP

**Solution**:
```bash
# 1. Port-forward PostgreSQL
kubectl port-forward -n data-platform svc/kong-postgresql 5432:5432 &

# 2. Test connection
psql -h localhost -U postgres -d postgres -c "SELECT 1"

# 3. Update Cursor settings to use localhost:5432
```

---

## Step 10: Best Practices

### ✅ DO:
- **Always test in staging first** - Use staging environment URLs before production
- **Enable audit logging** - Keep MCP operations logged for compliance
- **Rotate credentials regularly** - Update tokens/passwords monthly
- **Document custom queries** - Save useful MCP queries to repository
- **Use namespaces** - Always specify `-n namespace` for K8s operations
- **Validate YAML** - Always validate Helm charts before deployment

### ❌ DON'T:
- **Don't hardcode credentials** - Always use environment variables or Vault
- **Don't deploy to production directly** - Always test in staging first
- **Don't disable MFA** - Keep GitHub/Vault MFA enabled
- **Don't leave port-forwards running** - Clean up after testing
- **Don't query sensitive data** - Minimize access to production databases
- **Don't commit secrets** - Use .gitignore for credential files

---

## Useful MCP Queries for 254Carbon

### Health Check (Full Platform)
```
@kubernetes get-pods -n data-platform
@kubernetes get-pods -n monitoring
@prometheus query 'up{job="kubernetes-pods"}'
@grafana query-panel "Platform Health Score"
```

### Deployment Status
```
@argocd app-status data-platform
@argocd app-sync data-platform
@helm status data-platform
@kubernetes get-deployments -n data-platform
```

### Data Pipeline Status
```
@kafka list-topics
@kafka describe-topic commodities
@trino execute "SHOW CATALOGS"
@postgres query "SELECT * FROM dolphinscheduler.t_ds_process_definition"
```

### Performance Analysis
```
@prometheus query 'rate(container_cpu_usage_seconds_total[5m])'
@prometheus query 'container_memory_working_set_bytes'
@loki query '{namespace="data-platform"}'
@grafana dashboard-info "Data Platform Metrics"
```

### Incident Response
```
@kubernetes logs pod-name -n namespace --tail 1000
@postgres query "SELECT * FROM logs WHERE severity='ERROR' ORDER BY timestamp DESC LIMIT 50"
@prometheus query 'rate(errors_total[5m])'
@slack send-message --channel #incidents "Platform alert: [description]"
```

---

## Environment Variables Checklist

Create `.env.mcp` file in project root:

```bash
# GitHub
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"

# ArgoCD
export ARGOCD_SERVER="argocd.254carbon.com"
export ARGOCD_AUTH_TOKEN="argocd_token_xxxxx"

# Vault
export VAULT_ADDR="https://vault.254carbon.com"
export VAULT_TOKEN="hvs.xxxxx"

# PostgreSQL
export PG_HOST="localhost"
export PG_PORT="5432"
export PG_USER="postgres"
export PG_PASSWORD="xxxxx"
export PG_DATABASE="postgres"

# Grafana
export GRAFANA_URL="https://grafana.254carbon.com"
export GRAFANA_API_KEY="eyJxxxx"

# Slack
export SLACK_BOT_TOKEN="xoxb-xxxx"

# Kafka
export KAFKA_BROKERS="kafka-0:9092,kafka-1:9092,kafka-2:9092"

# MinIO
export MINIO_URL="http://minio.data-platform:9000"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin"

# Prometheus
export PROMETHEUS_URL="http://prometheus.monitoring:9090"

# Loki
export LOKI_URL="http://loki.monitoring:3100"

# Trino
export TRINO_HOST="trino-coordinator.data-platform"
export TRINO_PORT="8080"

# MLflow
export MLFLOW_TRACKING_URI="http://mlflow.ml-platform:5000"

# Ray
export RAY_HEAD_NODE="ray-head.ml-platform"
export RAY_PORT="10001"
```

Load before using Cursor:
```bash
source .env.mcp
```

---

## Next Steps

1. **Today**: Complete Steps 1-4 (Installation & Credentials)
2. **Tomorrow**: Complete Steps 5-7 (Configuration & Testing)
3. **This Week**: Enable Phase 1 CRITICAL servers
4. **Next Week**: Gradually add Phase 2 RECOMMENDED servers
5. **Documentation**: Add team guidelines for MCP usage in wiki

---

## Support & Resources

- **Cursor MCP Docs**: https://cursor.com/docs/mcp
- **254Carbon Repo**: https://github.com/254CARBON/HMCo
- **Kubernetes Docs**: https://kubernetes.io/docs
- **Platform Health**: https://grafana.254carbon.com
- **Troubleshooting**: See `docs/troubleshooting/README.md`

---

**Setup Complete!** 🚀

Your AI-assisted development experience is now supercharged with direct access to all platform infrastructure and tools.
