# Alternative Methods to Import DolphinScheduler Workflows

**Problem**: File upload in DolphinScheduler UI doesn't work  
**Solution**: Use alternative import methods

---

## Method 1: Create Workflows Manually in UI (Recommended)

Since file upload isn't working, create workflows directly in the UI:

### Quick Start: Test Workflow

1. **Log in**: https://dolphinscheduler.254carbon.com/dolphinscheduler/ui/
   - Username: admin
   - Password: dolphinscheduler123

2. **Create Project**:
   - Click "Project Management"
   - Click "Create Project"
   - Name: "Commodity Data Platform"
   - Submit

3. **Create a Simple Test Workflow**:
   - Click on your project
   - Click "Workflow Definition"
   - Click "Create Workflow"
   - Name: "Test EIA Data"
   
4. **Add a SHELL Task**:
   - Drag "SHELL" from left panel
   - Double-click to configure
   - Task Name: "test_eia_api"
   - Script:
     ```bash
     #!/bin/bash
     echo "Testing EIA API..."
     curl -s "https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key=$EIA_API_KEY&frequency=daily&length=5" | jq .
     echo "EIA API test complete"
     ```
   - Save

5. **Save and Run**:
   - Click "Save"
   - Click "Online" (to enable)
   - Click "Run" to test

---

## Method 2: Use DolphinScheduler API Directly

Import workflows via REST API:

### Get Session Token

```bash
# Login and get session
SESSION=$(kubectl exec -n data-platform deploy/dolphinscheduler-api -- curl -s -X POST http://localhost:12345/dolphinscheduler/login \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'userName=admin&userPassword=dolphinscheduler123' | jq -r '.data.sessionId')

echo "Session: $SESSION"
```

### Create Project via API

```bash
kubectl exec -n data-platform deploy/dolphinscheduler-api -- curl -s -X POST \
  "http://localhost:12345/dolphinscheduler/projects" \
  -H "sessionId: $SESSION" \
  -H 'Content-Type: application/json' \
  -d '{
    "projectName": "Commodity Data Platform",
    "description": "Automated commodity data ingestion"
  }' | jq .
```

---

## Method 3: Simplified Shell Script Workflows

Create simple shell-based workflows that don't require JSON import:

### Create EIA Data Collection Workflow

```bash
# In DolphinScheduler UI:
# 1. Create Workflow: "EIA Energy Prices"
# 2. Add SHELL task
# 3. Paste this script:

#!/bin/bash
set -e

echo "Fetching EIA petroleum spot prices..."

# Fetch data
curl -s "https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key=$EIA_API_KEY&frequency=daily&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&length=30" \
  -o /tmp/eia_data.json

# Check response
if [ -s /tmp/eia_data.json ]; then
  echo "✅ EIA data fetched successfully"
  jq '.response.data | length' /tmp/eia_data.json
  echo "records retrieved"
else
  echo "❌ No data received from EIA"
  exit 1
fi

echo "EIA petroleum data collection complete"
```

### Create FRED Indicators Workflow

```bash
#!/bin/bash
set -e

echo "Fetching FRED economic indicators..."

# FRED series to collect
indicators="DCOILWTICO DHHNGSP DPROPANEMBTX GASREGW ELECPRICE"

for series in $indicators; do
  echo "Fetching $series..."
  curl -s "https://api.stlouisfed.org/fred/series/observations?series_id=$series&api_key=$FRED_API_KEY&file_type=json&limit=30" \
    -o /tmp/fred_${series}.json
  
  if [ -s /tmp/fred_${series}.json ]; then
    count=$(jq '.observations | length' /tmp/fred_${series}.json)
    echo "  ✅ $series: $count observations"
  fi
done

echo "FRED indicators collection complete"
```

### Create AlphaVantage Workflow

```bash
#!/bin/bash
set -e

echo "Fetching AlphaVantage commodity futures..."

# Commodity symbols
symbols="CL=F NG=F HO=F RB=F"

for symbol in $symbols; do
  echo "Fetching $symbol..."
  curl -s "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=$symbol&apikey=$ALPHAVANTAGE_API_KEY&outputsize=compact" \
    -o /tmp/av_${symbol}.json
  
  if [ -s /tmp/av_${symbol}.json ]; then
    # Check if we got data
    if grep -q "Time Series" /tmp/av_${symbol}.json; then
      echo "  ✅ $symbol: Data received"
    else
      echo "  ⚠️  $symbol: Check API response"
    fi
  fi
  
  # Rate limit: wait 12 seconds between calls (5 calls/min)
  sleep 12
done

echo "AlphaVantage futures collection complete"
```

---

## Method 4: Python Task Type

Create workflows using PYTHON task type (no file upload needed):

### Python Task for Polygon.io

```python
import requests
import os
from datetime import datetime, timedelta

API_KEY = os.getenv('POLYGON_API_KEY')
TICKERS = ['C:CL', 'C:NG', 'C:HO']

end_date = datetime.now().date()
start_date = end_date - timedelta(days=7)

for ticker in TICKERS:
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}'
    headers = {'Authorization': f'Bearer {API_KEY}'}
    
    response = requests.get(url, headers=headers)
    print(f'{ticker}: Status {response.status_code}')
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        print(f'  Records: {len(results)} days')
    else:
        print(f'  Error: {response.text}')

print('Polygon.io data collection complete')
```

---

## Method 5: Use Existing DolphinScheduler Features

### Create via Workflow Designer

1. **Use DAG Designer**:
   - Click "Create Workflow" in UI
   - Drag and drop tasks from left panel
   - Connect tasks to create dependencies
   - Configure each task directly in UI
   - No JSON file needed!

2. **Task Types Available**:
   - **SHELL**: For curl commands, bash scripts
   - **PYTHON**: For Python API calls
   - **SQL**: For Trino queries
   - **HTTP**: For simple REST API calls

3. **Workflow Structure**:
   ```
   Start → Fetch EIA Data (SHELL) 
         → Fetch FRED Data (SHELL)
         → Fetch AlphaVantage (PYTHON)
         → Validate Data (SQL)
         → Send Notification (HTTP)
         → End
   ```

---

## Quick Win: Create One Working Workflow

Let's create a simple comprehensive workflow manually:

### Workflow: "Daily Commodity Data Collection"

**1. Create Workflow** in UI

**2. Add Task 1 - EIA Data**:
- Type: SHELL
- Name: collect_eia_energy
- Script:
```bash
curl -s "https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key=$EIA_API_KEY&frequency=daily&length=5" | jq '.' > /tmp/eia.json && echo "EIA: OK"
```

**3. Add Task 2 - FRED Data**:
- Type: SHELL
- Name: collect_fred_indicators
- Depends on: collect_eia_energy
- Script:
```bash
curl -s "https://api.stlouisfed.org/fred/series/observations?series_id=DCOILWTICO&api_key=$FRED_API_KEY&limit=5&file_type=json" | jq '.' > /tmp/fred.json && echo "FRED: OK"
```

**4. Add Task 3 - AlphaVantage**:
- Type: PYTHON
- Name: collect_alphavantage
- Depends on: collect_fred_indicators
- Script:
```python
import requests, os
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=CL=F&apikey={os.getenv('ALPHAVANTAGE_API_KEY')}&outputsize=compact"
r = requests.get(url)
print(f"AlphaVantage: {r.status_code}")
```

**5. Add Task 4 - Polygon**:
- Type: PYTHON  
- Name: collect_polygon
- Depends on: collect_alphavantage
- Script:
```python
import requests, os
from datetime import datetime, timedelta
end = datetime.now().date()
start = end - timedelta(days=7)
url = f"https://api.polygon.io/v2/aggs/ticker/C:CL/range/1/day/{start}/{end}"
r = requests.get(url, headers={'Authorization': f'Bearer {os.getenv("POLYGON_API_KEY")}'})
print(f"Polygon: {r.status_code}, Records: {len(r.json().get('results', []))}")
```

**6. Save and Run**:
- Click "Save"
- Click "Online"
- Click "Run"

This will test all your major APIs in one workflow!

---

## Method 6: Copy Workflow Content to UI

For each workflow JSON file:

1. **Open the JSON file** on your local machine:
   ```bash
   cat /home/m/tff/254CARBON/HMCo/workflows/11-all-sources-daily.json
   ```

2. **Copy the task definitions** from the JSON

3. **Recreate manually** in DolphinScheduler UI:
   - Create workflow with same name
   - Add tasks one by one
   - Configure dependencies
   - Set schedule

---

## Recommended Approach

**For now, create 1-2 simple workflows manually to test**:

1. Create "Test All APIs" workflow with 6 SHELL/PYTHON tasks
2. Each task tests one API (EIA, FRED, AlphaVantage, Polygon, GIE, Census)
3. Run manually to verify all APIs work
4. Once working, create more complex workflows

**Benefits**:
- No file upload issues
- Can test immediately
- Learn DolphinScheduler workflow designer
- Faster than troubleshooting JSON import

---

## Need Help?

I can create a simplified version of the workflows that you can manually recreate in the UI.

**Want me to**:
1. Create a single comprehensive workflow with all the shell/python scripts?
2. Make a step-by-step guide for manually creating workflows in UI?
3. Try another method to fix the file upload issue?

Let me know which approach you'd prefer!

