# DolphinScheduler Workflow JSON Files

**Location**: `/home/m/tff/254CARBON/HMCo/workflows/`  
**Total Files**: 11 workflow definitions  
**Status**: ✅ Ready for import into DolphinScheduler

---

## Workflow Files (Import in this order)

### 🌟 Start Here (Recommended First)
**11-all-sources-daily.json** - Comprehensive Commodity Data Collection
- **Schedule**: Daily at 1 AM UTC
- **Description**: Collects data from ALL 6 sources in one workflow
- **Sources**: EIA, FRED, AlphaVantage, Polygon.io, GIE, US Census
- **Duration**: ~30 minutes
- **Best for**: Testing all APIs at once

### Original Workflows (5)

**01-market-data-daily.json** - Daily Market Data Ingestion
- **Schedule**: Daily at 2 AM UTC
- **Sources**: EIA energy prices, NOAA weather
- **Commodities**: Crude oil, natural gas, electricity, LNG
- **Duration**: ~15-30 minutes

**02-economic-indicators-daily.json** - Daily Economic Indicators
- **Schedule**: Daily at 3 AM UTC
- **Sources**: FRED, World Bank
- **Indicators**: WTI oil, Henry Hub gas, propane, gasoline, electricity prices
- **Duration**: ~10-20 minutes

**03-weather-data-hourly.json** - Hourly Weather Data
- **Schedule**: Every 4 hours
- **Source**: NOAA
- **Locations**: Houston, Chicago, New York
- **Duration**: ~5 minutes

**04-alternative-data-weekly.json** - Weekly Alternative Data
- **Schedule**: Sundays at 4 AM UTC
- **Source**: MinIO/S3 custom files
- **Format**: Parquet, CSV
- **Duration**: Variable

**05-data-quality-checks.json** - Daily Quality Validation
- **Schedule**: Daily at 6 AM UTC
- **Type**: Data quality checks via Trino SQL
- **Checks**: Freshness, completeness, validity
- **Duration**: ~5-10 minutes

### Advanced Workflows (6)

**06-alphavantage-daily.json** - AlphaVantage Commodity Futures
- **Schedule**: Daily at 4 AM UTC
- **Symbols**: CL=F (crude oil), NG=F (nat gas), HO=F (heating oil), RB=F (gasoline)
- **Source**: AlphaVantage API
- **Limit**: 25 requests/day (free tier)

**07-polygon-market-data.json** - Polygon.io Market Data
- **Schedule**: Daily at 5 AM UTC
- **Tickers**: C:CL, C:NG, C:HO (commodity futures)
- **Features**: OHLCV, VWAP, transaction count
- **Source**: Polygon.io API

**08-gie-storage-daily.json** - European Gas Storage
- **Schedule**: Daily at 6 AM UTC
- **Source**: GIE AGSI/ALSI API
- **Data**: EU gas storage levels, injection/withdrawal rates
- **Coverage**: All EU countries

**09-census-economic-daily.json** - US Census Economic Data
- **Schedule**: Daily at 7 AM UTC
- **Source**: US Census Bureau
- **Dataset**: Economic indicators, construction data
- **Update Frequency**: Daily (data changes slowly)

**10-openfigi-mapping-weekly.json** - OpenFIGI Instrument Mapping
- **Schedule**: Weekly on Mondays at 8 AM UTC
- **Source**: OpenFIGI API
- **Purpose**: Map commodity tickers to FIGI identifiers
- **Commodities**: CL, NG, HO, RB

**11-all-sources-daily.json** - Comprehensive Collection ⭐
- **Schedule**: Daily at 1 AM UTC
- **Sources**: ALL 6 APIs (EIA, FRED, AlphaVantage, Polygon, GIE, Census)
- **Purpose**: One workflow to collect everything
- **Duration**: ~30-45 minutes
- **Best for**: Daily comprehensive data refresh

---

## Import Instructions

### Method 1: Automated Import (Recommended) ⭐

**One-command setup:**

```bash
cd /home/m/tff/254CARBON/HMCo
./scripts/setup-dolphinscheduler-complete.sh
```

This automation:
- ✅ Configures all API credentials
- ✅ Imports all 11 workflows automatically
- ✅ Tests execution (optional)
- ✅ Verifies data ingestion

**Or import workflows only:**

```bash
python3 ./scripts/import-workflows-from-files.py --port-forward
```

---

### Method 2: File Upload (Manual)

1. In DolphinScheduler UI, go to: Project > Workflow Definition
2. Click "Import Workflow"
3. Click "Upload" or "Choose File"
4. Select a JSON file from this directory
5. Click "Import"
6. Repeat for all 11 files

### Method 3: Copy/Paste (Manual)

1. Open a JSON file in your editor (they're in this directory)
2. Copy the entire contents
3. In DolphinScheduler: Project > Workflow Definition > Import Workflow
4. Paste the JSON content
5. Click "Import"

---

## Workflow Details

### What Each Workflow Does

**Market Data (01)**: 
- Ingests crude oil, natural gas, electricity, LNG prices from EIA
- Validates data completeness
- Sends success notification

**Economic Indicators (02)**:
- Collects 5 FRED economic series
- Fetches World Bank commodity prices
- Aggregates summary statistics

**Weather Data (03)**:
- Collects forecasts for 3 US regions
- Analyzes weather impact on commodities
- Updates every 4 hours

**Alternative Data (04)**:
- Scans MinIO for new parquet/CSV files
- Ingests custom proprietary data
- Runs weekly to process batch uploads

**Data Quality (05)**:
- Checks data freshness (<24h)
- Validates completeness (no nulls)
- Identifies invalid records

**AlphaVantage (06)**:
- Fetches commodity futures prices
- 4 symbols: crude, gas, heating oil, gasoline
- Daily time series data

**Polygon (07)**:
- Real-time/historical commodity prices
- High-frequency market data
- VWAP and volume data

**GIE Storage (08)**:
- European gas storage levels
- Injection/withdrawal rates
- Storage fullness percentage

**Census (09)**:
- US economic indicators
- Construction data
- Economic time series

**OpenFIGI (10)**:
- Maps commodity tickers to FIGIs
- Standardized instrument identification
- Updates weekly (instruments don't change often)

**Comprehensive (11)** ⭐:
- Runs ALL the above in one workflow
- Tests all API connections
- Most efficient for daily updates

---

## Testing Workflows

### Test Individual API

To test a single API, import and run the corresponding workflow:
- Test EIA: Run workflow #01
- Test FRED: Run workflow #02
- Test AlphaVantage: Run workflow #06
- Test Polygon: Run workflow #07
- Test GIE: Run workflow #08
- Test Census: Run workflow #09

### Test All APIs

Import and run workflow #11 (Comprehensive Collection) - tests everything!

---

## Scheduling

All workflows are pre-configured with schedules:

| Time (UTC) | Workflow | What It Does |
|------------|----------|--------------|
| 1:00 AM | Comprehensive Collection | ALL sources |
| 2:00 AM | Market Data | EIA prices |
| 3:00 AM | Economic Indicators | FRED |
| 4:00 AM | AlphaVantage | Futures |
| 5:00 AM | Polygon.io | Markets |
| 6:00 AM | GIE Storage + Quality | EU gas + validation |
| 7:00 AM | Census | Economic |
| 8:00 AM (Mon) | OpenFIGI | Mapping |
| Every 4h | Weather | NOAA forecasts |
| Sun 4 AM | Alternative Data | Custom sources |

**Note**: If you only want one comprehensive workflow, use #11 and disable the others to avoid duplicate data collection.

---

## Troubleshooting

### Import Fails
- Check: Project exists
- Check: JSON is valid
- Check: No duplicate workflow names

### Workflow Runs But Fails
- Check: API keys configured correctly
- Check: Worker pods are running
- View logs: Workflow Instances > Click instance > View Log

### No Data After Run
- Check: Workflow completed successfully (not failed)
- Query Trino: `SELECT COUNT(*) FROM commodity_data.energy_prices;`
- Check: Data quality metrics in Grafana

---

## Automation Scripts

Complete automation tools are available in `../scripts/`:

### Quick Start (One Command)
```bash
# Complete setup: credentials + import + test + verify
./scripts/setup-dolphinscheduler-complete.sh

# Or step-by-step:
./scripts/configure-dolphinscheduler-credentials.sh  # 1. API keys
python3 ./scripts/import-workflows-from-files.py --port-forward  # 2. Import
./scripts/test-dolphinscheduler-workflows.sh  # 3. Test
./scripts/verify-workflow-data-ingestion.sh  # 4. Verify
```

### Available Scripts

| Script | Purpose | Duration |
|--------|---------|----------|
| `setup-dolphinscheduler-complete.sh` | Master automation (all steps) | 5-50 min |
| `configure-dolphinscheduler-credentials.sh` | Configure 6 API keys | 1 min |
| `import-workflows-from-files.py` | Import all 11 workflows | 2 min |
| `test-dolphinscheduler-workflows.sh` | Run test workflow | 30-45 min |
| `verify-workflow-data-ingestion.sh` | Verify data in Trino | 1 min |

See [Automation Guide](../docs/automation/AUTOMATION_GUIDE.md) for detailed documentation.

---

## Next Steps

### Automated (Recommended)
```bash
./scripts/setup-dolphinscheduler-complete.sh
```

### Manual
1. **Import workflow #11** (Comprehensive Collection)
2. **Run it manually** to test all APIs
3. **Check execution logs** for any errors
4. **Query Trino** to verify data landed in Iceberg
5. **Enable scheduling** if successful
6. **Import other workflows** as needed for specific use cases

---

**Files Ready**: `/home/m/tff/254CARBON/HMCo/workflows/`  
**Automation**: `/home/m/tff/254CARBON/HMCo/scripts/`  
**Documentation**: [Automation Guide](../docs/automation/AUTOMATION_GUIDE.md)  
**Status**: ✅ Ready for import!

🎊 Run `./scripts/setup-dolphinscheduler-complete.sh` or log in to DolphinScheduler! 🎊
