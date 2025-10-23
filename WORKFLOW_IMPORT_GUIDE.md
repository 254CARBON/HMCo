# DolphinScheduler Workflow Import Guide

**Status:** ✅ Ready for Import  
**Date:** October 23, 2025  
**Workflows Available:** 11

## Quick Start

### 1. Access DolphinScheduler

DolphinScheduler is accessible at: **https://dolphin.254carbon.com**

Default credentials (if not changed):
- Username: `admin`
- Password: `dolphinscheduler123`

### 2. Recommended Import Order

**Start with the comprehensive workflow to test all APIs at once:**

1. **Workflow #11** - All Sources Daily (recommended first)
   - File: `/home/m/tff/254CARBON/HMCo/workflows/11-all-sources-daily.json`
   - Tests: All 6 data sources in one workflow
   - Duration: ~30-45 minutes
   - Best for: Validating all API integrations

**Then import individual workflows as needed:**

2-10. Individual source workflows (optional)
11. Data quality checks workflow

## Import Instructions

### Method 1: Web UI Import (Recommended)

1. **Navigate to Project**
   - Click "Project Management" in the top menu
   - Select or create a project (e.g., "Commodity Data Platform")

2. **Import Workflow**
   - Click "Workflow Definition" in the left sidebar
   - Click "Import Workflow" button
   - Choose "Upload File" option
   - Select the JSON file from `/home/m/tff/254CARBON/HMCo/workflows/`
   - Click "Import"

3. **Verify Import**
   - Check that the workflow appears in the list
   - Click on the workflow name to view the DAG
   - Verify all tasks are properly connected

### Method 2: API Import (Advanced)

```bash
# Set variables
DOLPHIN_API="http://dolphinscheduler-api.data-platform:12345"
PROJECT_CODE="your-project-code"
WORKFLOW_FILE="/home/m/tff/254CARBON/HMCo/workflows/11-all-sources-daily.json"

# Get authentication token (adjust credentials)
TOKEN=$(curl -X POST "$DOLPHIN_API/dolphinscheduler/login" \
  -d "userName=admin&userPassword=dolphinscheduler123" | \
  jq -r '.data.token')

# Import workflow
curl -X POST "$DOLPHIN_API/dolphinscheduler/projects/$PROJECT_CODE/process/import" \
  -H "token: $TOKEN" \
  -F "file=@$WORKFLOW_FILE"
```

## Workflow Details

### 11. All Sources Daily (Recommended First) ⭐

**File:** `11-all-sources-daily.json`

**Purpose:** Comprehensive data collection from all sources

**Data Sources:**
1. EIA - Energy Information Administration
2. FRED - Federal Reserve Economic Data
3. AlphaVantage - Market data & commodity futures
4. Polygon.io - Real-time market data
5. GIE - European gas storage data
6. US Census - Economic indicators

**Schedule:** Daily at 1:00 AM UTC

**Duration:** 30-45 minutes

**Tasks:**
- Fetch data from 6 APIs
- Validate data quality
- Load to Iceberg tables via Kafka
- Send completion notifications

### Individual Workflows (Optional)

| # | Name | File | Schedule | Sources |
|---|------|------|----------|---------|
| 1 | Market Data Daily | 01-market-data-daily.json | Daily 2 AM | EIA, NOAA |
| 2 | Economic Indicators | 02-economic-indicators-daily.json | Daily 3 AM | FRED, World Bank |
| 3 | Weather Data | 03-weather-data-hourly.json | Every 4 hours | NOAA |
| 4 | Alternative Data | 04-alternative-data-weekly.json | Sun 4 AM | MinIO/S3 |
| 5 | Quality Checks | 05-data-quality-checks.json | Daily 6 AM | Trino SQL |
| 6 | AlphaVantage | 06-alphavantage-daily.json | Daily 4 AM | AlphaVantage |
| 7 | Polygon Data | 07-polygon-market-data.json | Daily 5 AM | Polygon.io |
| 8 | GIE Storage | 08-gie-storage-daily.json | Daily 6 AM | GIE AGSI/ALSI |
| 9 | Census Data | 09-census-economic-daily.json | Daily 7 AM | US Census |
| 10 | OpenFIGI Mapping | 10-openfigi-mapping-weekly.json | Mon 8 AM | OpenFIGI |

## Configuration Requirements

### API Credentials

Before running workflows, configure these API keys in DolphinScheduler:

1. **Global Variables** (Project Settings → Variables):
   - `EIA_API_KEY` - EIA data access
   - `FRED_API_KEY` - Federal Reserve data
   - `ALPHAVANTAGE_API_KEY` - Market data
   - `POLYGON_API_KEY` - Real-time data
   - `GIE_API_KEY` - European gas storage
   - `CENSUS_API_KEY` - US Census data

2. **Connection Settings**:
   - Kafka broker: `kafka.data-platform:9092`
   - Iceberg catalog: `iceberg-rest-catalog.data-platform:8181`
   - Trino coordinator: `trino-coordinator.data-platform:8080`

### How to Set Variables

1. Go to Project Settings → Variables
2. Click "Create Variable"
3. Set:
   - Variable Name: `EIA_API_KEY`
   - Variable Value: `your-api-key-here`
   - Variable Type: `VARCHAR`
   - Click "Submit"
4. Repeat for all API keys

## Testing Workflow Execution

### Manual Test Run

1. **Start a Workflow**
   - Navigate to: Workflow Definition
   - Find your imported workflow
   - Click "Run" button
   - Select: Run Type = "Run Once"
   - Click "Run Workflow"

2. **Monitor Execution**
   - Go to: Workflow Instance
   - Find your running instance
   - Click to view real-time DAG status
   - Each task shows: Running/Success/Failed

3. **Check Logs**
   - Click on any task in the DAG
   - Click "View Log" to see execution details
   - Check for errors or warnings

### Verify Data Ingestion

After workflow completes, verify data in Trino:

```sql
-- Connect to Trino
-- Host: trino-coordinator.data-platform:8080

-- Check tables exist
SHOW TABLES FROM commodity_data;

-- Verify data loaded
SELECT 
  COUNT(*) as record_count,
  MIN(date) as earliest_date,
  MAX(date) as latest_date
FROM commodity_data.energy_prices;

-- Sample data
SELECT * FROM commodity_data.energy_prices
ORDER BY date DESC
LIMIT 10;
```

## Troubleshooting

### Issue: Import Fails

**Solutions:**
1. Check JSON file is valid: `jq . workflow.json`
2. Ensure project exists before importing
3. Check file size isn't too large (should be <10MB)
4. Verify DolphinScheduler API is accessible

### Issue: Workflow Runs But No Data

**Solutions:**
1. Check API credentials are configured correctly
2. Verify Kafka is accessible: `kubectl get svc kafka -n data-platform`
3. Check worker logs: `kubectl logs -n data-platform -l app=dolphinscheduler-worker`
4. Validate network policies allow egress to external APIs

### Issue: Tasks Fail with Timeout

**Solutions:**
1. Increase task timeout in workflow settings
2. Check worker resources: `kubectl top pods -n data-platform -l app=dolphinscheduler-worker`
3. Scale workers if needed: `kubectl scale deployment dolphinscheduler-worker -n data-platform --replicas=3`

### Issue: API Rate Limits

**Solutions:**
1. AlphaVantage: Free tier = 25 requests/day (use paid tier for more)
2. Polygon: Free tier = 5 requests/minute
3. Adjust workflow schedules to spread requests
4. Consider caching responses in Redis

## Monitoring & Alerts

### Key Metrics to Monitor

1. **Workflow Success Rate**
   - Go to: Dashboard → Statistics
   - Monitor: Success/Failed workflows
   - Target: >95% success rate

2. **Data Freshness**
   - Check latest data timestamps in Trino
   - Alert if data is >24 hours old
   - Query: `SELECT MAX(date) FROM commodity_data.energy_prices`

3. **API Health**
   - Monitor API response times
   - Check for rate limit errors in logs
   - Set up alerts in Grafana

### Grafana Dashboards

Access monitoring at: **https://grafana.254carbon.com**

Relevant dashboards:
- "Commodity Data Pipeline" - Workflow metrics
- "Data Platform Health" - Overall platform status
- "Kafka Metrics" - Message throughput

## Best Practices

### 1. Start Small
- Import workflow #11 first
- Run manually to test
- Verify data lands correctly
- Then enable scheduling

### 2. Configure Alerts
- Set up email/Slack notifications
- Alert on workflow failures
- Monitor data quality metrics
- Track API quota usage

### 3. Schedule Management
- Avoid overlapping workflows
- Consider API rate limits
- Schedule during low-usage periods
- Stagger workflows by 30-60 minutes

### 4. Data Quality
- Run workflow #05 (quality checks) daily
- Monitor null rates and anomalies
- Set up automated data validation
- Create alerts for data quality issues

## Next Steps After Import

1. ✅ Import workflow #11 (comprehensive)
2. ✅ Configure all API credentials
3. ✅ Run manual test execution
4. ✅ Verify data in Trino
5. ✅ Enable scheduling if successful
6. ✅ Set up monitoring alerts
7. ✅ Import remaining workflows as needed
8. ✅ Document any custom modifications

## Support

### Logs Location
- DolphinScheduler: `kubectl logs -n data-platform -l app=dolphinscheduler-api`
- Workers: `kubectl logs -n data-platform -l app=dolphinscheduler-worker`
- Master: `kubectl logs -n data-platform -l app=dolphinscheduler-master`

### Useful Commands
```bash
# Check workflow execution
kubectl get pods -n data-platform | grep dolphin

# View worker resources
kubectl top pods -n data-platform -l app=dolphinscheduler-worker

# Restart API if needed
kubectl rollout restart deployment dolphinscheduler-api -n data-platform

# Scale workers
kubectl scale deployment dolphinscheduler-worker -n data-platform --replicas=3
```

### Documentation
- DolphinScheduler Docs: https://dolphinscheduler.apache.org/docs/latest
- Workflow README: `/home/m/tff/254CARBON/HMCo/workflows/README.md`
- Platform Docs: `/home/m/tff/254CARBON/HMCo/docs/`

## Conclusion

You're now ready to import and run commodity data workflows! Start with the comprehensive workflow (#11) to validate all integrations, then import individual workflows as needed for your use case.

---

**Last Updated:** October 23, 2025  
**Status:** Ready for Production Use  
**Next Review:** After first successful workflow run

