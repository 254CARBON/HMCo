# How to Get Data in Grafana Dashboards

## ✅ Your Datasources Are Working!

Victoria Metrics has **20+ metrics** including:
- `up` - Service health status
- GPU metrics (DCGM_*)
- Kubernetes API server metrics
- Pod/Service metrics

Loki has **logs from 99+ pods** with labels like:
- `namespace`
- `app`
- `pod`
- `container`

## 🎯 Create a Working Dashboard

### Option 1: Use Explore Tab (Fastest)

1. **Go to Grafana**: https://grafana.254carbon.com
   - Login: admin / datahub_admin_password

2. **Click "Explore"** (compass icon on left sidebar)

3. **For Metrics** (Select VictoriaMetrics datasource):
   ```promql
   up{kubernetes_namespace="data-platform"}
   ```
   Click "Run query" - You'll see all data-platform pods!

4. **For Logs** (Switch to Loki datasource):
   ```logql
   {namespace="data-platform"}
   ```
   Click "Run query" - You'll see live logs!

### Option 2: Create Dashboard Manually

1. **Click "+" → Create → Dashboard**

2. **Add Panel** → Select "Time series"

3. **In Query tab**:
   - Datasource: VictoriaMetrics
   - Query: `up{kubernetes_namespace="data-platform"}`
   - Legend: `{{app}} - {{kubernetes_pod_name}}`

4. **Click "Apply"** - You now have data!

### Option 3: Import Pre-Made Dashboard

Copy this JSON and import it:

1. Click "+" → "Import"
2. Paste this JSON:

```json
{
  "title": "Data Platform Status",
  "panels": [
    {
      "title": "Pod Health",
      "targets": [{
        "expr": "up{kubernetes_namespace=\"data-platform\"}",
        "legendFormat": "{{app}}"
      }],
      "type": "stat",
      "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0}
    }
  ],
  "refresh": "30s",
  "time": {"from": "now-1h", "to": "now"}
}
```

3. Click "Load"
4. Select "VictoriaMetrics" as datasource
5. Click "Import"

## 📊 Useful Queries

### Pod Status Queries
```promql
# All pods in data-platform
up{kubernetes_namespace="data-platform"}

# Specific service
up{app="dolphinscheduler-api"}

# Count running pods
count(up{kubernetes_namespace="data-platform"} == 1)

# Kong gateway status
up{app="kong"}

# All cluster services
up
```

### Log Queries
```logql
# All data-platform logs
{namespace="data-platform"}

# DolphinScheduler logs only
{namespace="data-platform", app="dolphinscheduler-api"}

# Error logs
{namespace="data-platform"} |= "ERROR"

# Last 5 minutes
{namespace="data-platform"} [5m]
```

## 🔍 Why Previous Dashboards Didn't Work

The pre-configured dashboards likely had:
1. Wrong datasource UIDs (they need to match Grafana's auto-generated UIDs)
2. Wrong metric queries (e.g., looking for `kube_pod_status_phase` which requires kube-state-metrics)
3. Missing datasource template variables

## ✅ Current Working Setup

- **VictoriaMetrics**: http://victoria-metrics.victoria-metrics.svc.cluster.local:8428
- **Loki**: http://loki.victoria-metrics.svc.cluster.local:3100
- **VMAgent**: Scraping 19+ targets successfully
- **Fluent Bit**: Collecting logs from both nodes
- **Data**: LIVE and flowing ✅

## 🎯 Quick Test

In Grafana Explore tab, run these to confirm data:

1. Switch to VictoriaMetrics → Query: `up`
   - Should show 20+ results with app names and namespaces

2. Switch to Loki → Query: `{namespace="data-platform"}`
   - Should show streaming logs

**If you see results, YOUR DATA IS WORKING!** 🎉

Just create panels using the working queries above.
