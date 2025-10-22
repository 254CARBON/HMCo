# Commodity Data Platform - Quick Start Guide

**Get your commodity data platform running in 30 minutes**

---

## Prerequisites

✅ Kubernetes cluster deployed (DONE)  
✅ All services running (DONE)  
✅ DolphinScheduler accessible (DONE)  
✅ Superset accessible (DONE)

**What you need:**
- API keys for data sources (FRED, EIA, etc.)
- 30 minutes of your time

---

## Step 1: Configure API Keys (5 minutes)

Edit the secret with your API credentials:

```bash
kubectl edit secret seatunnel-api-keys -n data-platform
```

Update these fields (base64 encoded):

```yaml
stringData:
  FRED_API_KEY: "your-fred-api-key"      # Get from: https://fred.stlouisfed.org/docs/api/api_key.html
  EIA_API_KEY: "your-eia-api-key"        # Get from: https://www.eia.gov/opendata/
  NOAA_API_KEY: "optional"                # NOAA API is mostly public
```

Quick encode helper:
```bash
echo -n "your-actual-api-key" | base64
```

---

## Step 2: Import DolphinScheduler Workflows (10 minutes)

### 2.1 Access DolphinScheduler

⚠️ **Important**: Use the full path including `/dolphinscheduler/ui/`

Open: **https://dolphinscheduler.254carbon.com/dolphinscheduler/ui/**  
Login: `admin` / `dolphinscheduler123`

**Note**: The root URL (without /dolphinscheduler/ui/) will show a blank page or 404.

### 2.2 Create Project

1. Click "Project Management" in left menu
2. Click "Create Project"
3. Name: `Commodity Data Platform`
4. Click "Submit"

### 2.3 Import Workflows

Extract workflow JSON files:

```bash
# Get all workflows
kubectl get configmap dolphinscheduler-commodity-workflows -n data-platform -o yaml > workflows.yaml

# The workflows are in the data section:
# - market-data-daily.json
# - economic-indicators-daily.json
# - weather-data-hourly.json
# - alternative-data-weekly.json
# - data-quality-checks.json
```

For each workflow:
1. Copy the JSON content
2. In DolphinScheduler: Project > Workflow Definition > Import
3. Paste JSON
4. Click "Import"

### 2.4 Test a Workflow

1. Select "Daily Market Data Ingestion"
2. Click "Run"
3. Monitor execution in "Workflow Instances"

---

## Step 3: Set Up Superset Dashboards (10 minutes)

### 3.1 Access Superset

Open: https://superset.254carbon.com/superset/login  
Login: `admin` / `admin`

### 3.2 Add Database Connections

**Add Trino connection:**
1. Settings > Database Connections > + Database
2. Select "Trino"
3. SQLAlchemy URI: `trino://trino-coordinator:8080/iceberg_catalog/commodity_data`
4. Test Connection
5. Save

**Add PostgreSQL connection:**
1. Settings > Database Connections > + Database
2. Select "PostgreSQL"
3. SQLAlchemy URI: `postgresql://postgres:postgres@postgres-shared-service:5432/datahub`
4. Save

### 3.3 Import Dashboards

Extract dashboard JSON:

```bash
kubectl get configmap superset-commodity-dashboards -n data-platform -o yaml > dashboards.yaml
```

For each dashboard:
1. Dashboards > Import Dashboard
2. Upload dashboard JSON
3. Select database connection
4. Import

---

## Step 4: Configure Grafana (5 minutes)

### 4.1 Access Grafana

Open: https://grafana.254carbon.com

### 4.2 Add Trino Data Source

1. Configuration > Data Sources > Add data source
2. Search for "PostgreSQL" (we'll use it to query Trino via federated queries)
3. Or use Grafana's Trino plugin if available

### 4.3 View Commodity Dashboards

The dashboards auto-load from ConfigMaps:
1. Dashboards > Browse
2. Look for dashboards tagged with "commodity"
3. Open "Commodity Market Overview"

---

## Step 5: Run First Data Ingestion (5 minutes)

### Option A: Automated (Recommended)

Enable the Daily Market Data workflow:
1. In DolphinScheduler, select "Daily Market Data Ingestion"
2. Click "Online" to enable scheduling
3. Wait for 2 AM UTC or run manually

### Option B: Manual Test

```bash
# Test market data connector
kubectl exec -n data-platform deploy/seatunnel-engine -- sh -c '
  echo "Testing market data ingestion..."
  # This will test the connector configuration
  ls -la /opt/seatunnel/config/connectors/
'

# Check if SeaTunnel is ready
kubectl get pods -n data-platform -l app=seatunnel
```

### Option C: Direct SQL Test

```bash
# Access Trino
kubectl port-forward -n data-platform svc/trino-coordinator 8080:8080

# In another terminal, run:
trino --server localhost:8080 --catalog iceberg_catalog --schema commodity_data

# Or via kubectl exec:
kubectl exec -n data-platform deploy/trino-coordinator -- \
  trino --execute "SHOW TABLES IN commodity_data"
```

---

## Verify Everything Works

### Check Workflow Execution

```bash
# Check DolphinScheduler master logs
kubectl logs -n data-platform -l app=dolphinscheduler-master --tail=50

# Check worker logs
kubectl logs -n data-platform -l app=dolphinscheduler-worker --tail=50
```

### Check Data Quality

```bash
# View quality metrics
kubectl logs -n data-platform -l app=data-quality-exporter --tail=20

# Check for alerts
kubectl port-forward -n monitoring svc/kube-prometheus-stack-alertmanager 9093:9093
# Open http://localhost:9093
```

### Query Data

```bash
# Port-forward Trino
kubectl port-forward -n data-platform svc/trino-coordinator 8080:8080

# Connect and query
trino --server localhost:8080 --catalog iceberg_catalog --schema commodity_data

# Sample query
trino> SHOW TABLES;
trino> SELECT COUNT(*) FROM energy_prices;
trino> SELECT * FROM economic_indicators LIMIT 10;
```

---

## Common Issues & Solutions

### Issue: "No data in tables"
**Solution:** 
- Verify API keys are configured correctly
- Run workflows manually in DolphinScheduler
- Check SeaTunnel connector logs: `kubectl logs -n data-platform -l app=seatunnel`

### Issue: "Workflow fails"
**Solution:**
- Check DolphinScheduler worker logs
- Verify database connectivity
- Check if required services are running (Trino, Iceberg, PostgreSQL)

### Issue: "Dashboards show no data"
**Solution:**
- Verify database connections in Superset/Grafana
- Ensure data ingestion has run at least once
- Check Iceberg table existence: `SHOW TABLES IN commodity_data`

### Issue: "RAPIDS Jupyter not accessible"
**Solution:**
- Add DNS record for rapids.254carbon.com in Cloudflare
- Or use port-forwarding: `kubectl port-forward -n data-platform svc/rapids-service 8888:8888`
- Access at: http://localhost:8888

---

## Next Steps After Quick Start

1. **Customize Workflows**
   - Add your specific commodities
   - Adjust ingestion schedules
   - Add custom validation rules

2. **Set Up Alerts**
   - Configure email notifications in AlertManager
   - Set up Slack/PagerDuty integrations
   - Customize alert thresholds

3. **Build Custom Dashboards**
   - Create commodity-specific views
   - Add your KPIs and metrics
   - Share with stakeholders

4. **Enable GPU Analytics**
   - Access RAPIDS Jupyter Lab
   - Run time series analysis
   - Implement price forecasting models

5. **Scale Up**
   - Add more data sources
   - Increase connector replicas
   - Allocate more GPU resources

---

## Success Criteria

After completing this quick start, you should have:

- ✅ At least 1 workflow running successfully
- ✅ Data visible in Trino/Iceberg tables
- ✅ Dashboards showing metrics
- ✅ Data quality validation passing
- ✅ Alerts configured and working

---

## Getting Help

### Check Logs

```bash
# DolphinScheduler
kubectl logs -n data-platform -l app=dolphinscheduler-master --tail=100

# SeaTunnel
kubectl logs -n data-platform -l app=seatunnel --tail=100

# Data Quality
kubectl logs -n data-platform -l app=data-quality-exporter --tail=100

# Trino
kubectl logs -n data-platform -l app=trino-coordinator --tail=100
```

### View Events

```bash
# Data platform events
kubectl get events -n data-platform --sort-by='.lastTimestamp' | head -20

# Specific component
kubectl describe pod -n data-platform <pod-name>
```

### Documentation

- Full deployment guide: `COMMODITY_PLATFORM_DEPLOYMENT.md`
- Main README: `README.md`
- Troubleshooting: `docs/troubleshooting/`

---

**Estimated Time to Production Data**: 30-60 minutes after API key configuration

**Questions?** Check the comprehensive guide in `COMMODITY_PLATFORM_DEPLOYMENT.md`

