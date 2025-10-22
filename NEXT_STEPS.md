# 254Carbon Commodity Platform - Next Steps

**Current Status**: ✅ Platform deployed and stable  
**Ready for**: Production data ingestion  
**Estimated Time to Production**: 10 minutes (automated) or 60-90 minutes (manual)

---

## 🚀 NEW: Automated Setup (Recommended)

**Complete platform configuration in ~10 minutes with a single command!**

```bash
./scripts/setup-commodity-platform.sh
```

This automated script will:
1. ✅ Configure API keys (interactive prompts)
2. ✅ Import DolphinScheduler workflows automatically
3. ✅ Set up Superset dashboards and database connections
4. ✅ Verify platform health

**See**: `docs/automation/AUTOMATION_GUIDE.md` for complete documentation.

### Non-Interactive Mode (CI/CD)

```bash
export FRED_API_KEY="your-fred-key"
export EIA_API_KEY="your-eia-key"

./scripts/setup-commodity-platform.sh --non-interactive
```

---

## Manual Setup (Alternative)

If you prefer step-by-step manual configuration, follow the sections below:

---

## Immediate Actions Required (User)

### 1. Configure API Keys (15 minutes) - **REQUIRED**

Update the secret with your actual API credentials:

```bash
kubectl edit secret seatunnel-api-keys -n data-platform
```

Replace these placeholders:

```yaml
stringData:
  FRED_API_KEY: "your-fred-api-key-here"          # Get from: https://fred.stlouisfed.org/docs/api/api_key.html
  EIA_API_KEY: "your-eia-api-key-here"            # Get from: https://www.eia.gov/opendata/
  WORLD_BANK_API_KEY: "your-world-bank-key-here"  # Optional
  NOAA_API_KEY: "not-required"                     # NOAA API is public
```

**Quick encode**: `echo -n "your-key" | base64`

### 2. Import DolphinScheduler Workflows (20 minutes) - **REQUIRED**

**Access DolphinScheduler**: https://dolphinscheduler.254carbon.com  
**Login**: admin / admin  

**Steps:**
1. Create project: "Commodity Data Platform"
2. Extract workflow JSON:
   ```bash
   kubectl get configmap dolphinscheduler-commodity-workflows -n data-platform -o yaml > workflows.yaml
   ```
3. Import each workflow via UI (5 workflows total)
4. Test "Daily Market Data Ingestion" manually

**Detailed Instructions**: See `COMMODITY_QUICKSTART.md`

### 3. Set Up Superset Dashboards (15 minutes) - **REQUIRED**

**Access Superset**: https://superset.254carbon.com/superset/login  
**Login**: admin / admin

**Steps:**
1. Add Trino connection: `trino://trino-coordinator:8080/iceberg_catalog/commodity_data`
2. Extract dashboard JSON:
   ```bash
   kubectl get configmap superset-commodity-dashboards -n data-platform -o yaml > dashboards.yaml
   ```
3. Import 5 dashboards via UI
4. Customize for your commodities

### 4. Configure Cloudflare DNS for RAPIDS (5 minutes) - **OPTIONAL**

Add DNS record for GPU analytics environment:

1. Login to Cloudflare Dashboard
2. Add A/CNAME record: `rapids.254carbon.com` → your tunnel
3. Or use port-forward: `kubectl port-forward -n data-platform svc/rapids-service 8888:8888`

---

## Optional Enhancements

### Install NVIDIA GPU Operator (30 minutes)

**If you want to use your 196GB GPU:**

```bash
# Install NVIDIA GPU Operator
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator

# Verify GPU detection
kubectl get nodes -o json | jq '.items[].status.capacity'

# Update RAPIDS to use GPU
kubectl edit deployment rapids-commodity-processor -n data-platform
# Uncomment the nvidia.com/gpu: 2 lines
```

**Without GPU Operator**: RAPIDS will use CPU (still fast, just not GPU-accelerated)

### Scale DolphinScheduler Workers (if needed)

```bash
# If you have many concurrent workflows
kubectl scale deployment dolphinscheduler-worker -n data-platform --replicas=8
```

### Enable Kafka Connect (for real-time streams)

```bash
# Deploy Kafka Connect
kubectl apply -f k8s/streaming/kafka-connect.yaml

# Configure connectors for WebSocket price feeds
```

---

## Verification Checklist

After completing the required steps, verify:

- [ ] DolphinScheduler workflows imported and visible
- [ ] At least 1 workflow executed successfully
- [ ] Data visible in Trino: `SELECT * FROM commodity_data.energy_prices LIMIT 10;`
- [ ] Superset dashboards showing data
- [ ] Grafana "Commodity Market Overview" dashboard accessible
- [ ] No critical alerts firing in AlertManager

---

## Expected Results

### After First Workflow Run

**You should see:**
- Records in Iceberg tables (query via Trino)
- Data freshness metrics updating
- Superset charts populating
- Grafana dashboard showing metrics
- Quality score > 95%

**Timeline:**
- T+0: Configure API keys
- T+20: Import workflows
- T+30: Run first workflow
- T+45: Data appears in dashboards
- T+60: Platform fully operational

---

## Troubleshooting

### "Workflow fails immediately"
**Likely cause**: API keys not configured or incorrect  
**Solution**: Double-check secret values, test API manually

### "No data in tables"
**Likely cause**: Workflow hasn't run yet or API returned empty  
**Solution**: Check DolphinScheduler logs, verify API endpoints

### "Dashboards empty"
**Likely cause**: No data ingested yet  
**Solution**: Wait for first workflow run, then refresh dashboards

### "SeaTunnel pods not starting"
**Current status**: Installing dependencies on startup  
**Solution**: Wait 2-3 minutes for pip install to complete  
**Workaround**: Use Spark/Python directly for custom ingestion scripts

### "RAPIDS not using GPU"
**Cause**: GPU operator not installed  
**Solution**: Install NVIDIA GPU operator (see optional enhancements)  
**Workaround**: RAPIDS works on CPU, just slower

---

## Support

**Quick Start Guide**: `COMMODITY_QUICKSTART.md`  
**Full Documentation**: `COMMODITY_PLATFORM_DEPLOYMENT.md`  
**Technical Reference**: `docs/commodity-data/README.md`

**Logs:**
```bash
kubectl logs -n data-platform -l app=dolphinscheduler-master --tail=100
kubectl logs -n data-platform -l app=seatunnel --tail=100
kubectl logs -n data-platform -l app=data-quality-exporter --tail=100
```

---

## Success Metrics

**Target (After API Configuration):**
- ✅ All 5 workflows imported
- ✅ Daily ingestion running automatically
- ✅ Data quality > 99%
- ✅ Dashboard refresh < 30 seconds
- ✅ Query performance < 5 seconds
- ✅ Zero data loss
- ✅ Automated alerts working

**Current Platform State:**
- ✅ Infrastructure: 100% deployed
- ✅ Stability: DolphinScheduler, DataHub, Trino all stable
- ✅ Monitoring: Grafana + Superset ready
- ⏳ Data: Waiting for API keys + first run
- ⏳ GPU: Waiting for NVIDIA operator (optional)

---

**Next Step**: Follow `COMMODITY_QUICKSTART.md` to get your first data flowing!

**Estimated Time to First Data**: 60 minutes from now

