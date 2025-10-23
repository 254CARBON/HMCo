# Automation Scripts

Collection of automation scripts for the 254Carbon Commodity Platform.

## Quick Start

**One-command setup** (recommended):
```bash
./setup-commodity-platform.sh
```

This single script automates the entire platform configuration in ~10 minutes.

---

## Available Scripts

### Platform Setup

#### `setup-commodity-platform.sh`
**All-in-one orchestration script**

Complete platform configuration with a single command.

```bash
# Interactive mode
./setup-commodity-platform.sh

# Non-interactive (CI/CD)
export FRED_API_KEY="your-key"
export EIA_API_KEY="your-key"
./setup-commodity-platform.sh --non-interactive

# Skip specific steps
./setup-commodity-platform.sh --skip-workflows --skip-dashboards
```

**What it does**:
1. Configure API keys
2. Wait for services
3. Import DolphinScheduler workflows
4. Import Superset dashboards
5. Verify platform health

**Time**: ~10 minutes

---

### DolphinScheduler Workflow Automation

#### `setup-dolphinscheduler-complete.sh` ⭐
**Complete DolphinScheduler setup automation**

One command to set up all workflows, API keys, and verification.

```bash
# Full setup (recommended)
./setup-dolphinscheduler-complete.sh

# Skip specific steps
./setup-dolphinscheduler-complete.sh --skip-test
./setup-dolphinscheduler-complete.sh --verify-only
```

**What it does**:
1. Configure API credentials (6 data sources)
2. Import 11 workflow definitions
3. Run comprehensive test workflow (optional)
4. Verify data ingestion in Trino

**Time**: 5-50 minutes (depending on options)

---

#### `configure-dolphinscheduler-credentials.sh`
**Configure API credentials for DolphinScheduler workflows**

Sets up Kubernetes secrets and mounts to worker pods.

```bash
./configure-dolphinscheduler-credentials.sh
./configure-dolphinscheduler-credentials.sh --namespace data-platform
```

**API Keys configured**:
- AlphaVantage
- Polygon.io
- EIA (Energy Information Administration)
- GIE (European Gas Infrastructure)
- US Census Bureau
- NOAA

**Time**: 1 minute

---

#### `import-workflows-from-files.py`
**Import workflows from local JSON files**

Python script to bulk import workflow definitions.

```bash
# Auto port-forward
python3 import-workflows-from-files.py --port-forward

# Custom directory
python3 import-workflows-from-files.py --workflow-dir /path/to/workflows

# Skip existing workflows
python3 import-workflows-from-files.py --skip-existing
```

**Features**:
- Reads from `/home/m/tff/254CARBON/HMCo/workflows/`
- Validates JSON structure
- Creates project if needed
- Handles multipart file upload
- Auto port-forward support

**Time**: 2 minutes

---

#### `test-dolphinscheduler-workflows.sh`
**Test workflow execution**

Runs comprehensive test workflow and monitors progress.

```bash
./test-dolphinscheduler-workflows.sh
./test-dolphinscheduler-workflows.sh --workflow-name "Comprehensive Commodity Data Collection"
```

**Monitoring**:
- Real-time task progress
- Success/failure status
- Execution duration
- Failed task details
- Log extraction

**Time**: 30-45 minutes (workflow execution)

---

#### `verify-workflow-data-ingestion.sh`
**Verify data landed in Trino/Iceberg**

Connects to Trino and validates data ingestion.

```bash
./verify-workflow-data-ingestion.sh
./verify-workflow-data-ingestion.sh --catalog iceberg --schema commodity_data
```

**Checks**:
- Schema and table existence
- Record counts per table
- Data freshness (<48 hours)
- Date ranges
- Generates summary report

**Time**: 1 minute

---

### API Configuration

#### `configure-api-keys.sh`
**Interactive API key configuration**

Configure API keys for external data sources (FRED, EIA, NOAA, etc.).

```bash
# Interactive prompts
./configure-api-keys.sh

# Non-interactive
FRED_API_KEY="xxx" EIA_API_KEY="xxx" ./configure-api-keys.sh --non-interactive

# From file
./configure-api-keys.sh --from-file api-keys.env
```

**Features**:
- Interactive prompts with helpful URLs
- Format validation
- Automatic secret update
- Pod restart

**Time**: ~5 minutes (interactive)

---

### Workflow Management

#### `import-dolphinscheduler-workflows.py`
**Import DolphinScheduler workflows**

Automatically imports workflow definitions from Kubernetes ConfigMap.

```bash
# Basic usage
python3 import-dolphinscheduler-workflows.py

# With port-forward (from outside cluster)
kubectl port-forward -n data-platform svc/dolphinscheduler-api-service 12345:12345 &
python3 import-dolphinscheduler-workflows.py --dolphinscheduler-url http://localhost:12345
```

**What it does**:
- Reads workflows from ConfigMap
- Creates project if needed
- Imports all workflow definitions
- Skips existing workflows (idempotent)

**Time**: ~2 minutes

---

### Dashboard Management

#### `import-superset-dashboards.py`
**Import Superset dashboards and setup databases**

Automatically configures database connections and imports dashboards.

```bash
# Basic usage (includes database setup)
python3 import-superset-dashboards.py --setup-databases

# With port-forward (from outside cluster)
kubectl port-forward -n data-platform svc/superset 8088:8088 &
python3 import-superset-dashboards.py --superset-url http://localhost:8088 --setup-databases
```

**What it does**:
- Creates Trino connection
- Creates PostgreSQL connection
- Imports dashboard definitions
- Updates existing dashboards (with --overwrite)

**Time**: ~2 minutes

---

### Platform Verification

#### `verify-platform-complete.sh`
**Comprehensive health check**

Verifies all platform components are functioning correctly.

```bash
./verify-platform-complete.sh
```

**Checks**:
- Kubernetes connectivity
- Pod health (30+ pods)
- API keys configuration
- Service endpoints
- Database connectivity
- DolphinScheduler workflows
- Superset dashboards
- Data quality framework
- Monitoring stack
- Ingress and DNS

**Output**:
- Color-coded status
- Detailed summary
- Quick links to services

**Time**: ~1 minute

**Exit Codes**:
- `0`: All checks passed
- `1`: One or more failures

---

### Utility Scripts

#### `deploy-automation-configmap.sh`
**Deploy automation scripts to Kubernetes**

Updates the ConfigMap used by Kubernetes Jobs.

```bash
./deploy-automation-configmap.sh
```

**What it does**:
- Creates/updates `automation-scripts` ConfigMap
- Includes Python import scripts
- Used by workflow-import-job and dashboard-import-job

**Time**: <1 minute

---

## Usage Examples

### First-Time Setup

```bash
# 1. Configure API keys
./configure-api-keys.sh

# 2. Import workflows
python3 import-dolphinscheduler-workflows.py

# 3. Import dashboards
python3 import-superset-dashboards.py --setup-databases

# 4. Verify everything
./verify-platform-complete.sh
```

**OR just use the orchestrator**:
```bash
./setup-commodity-platform.sh
```

---

### CI/CD Pipeline

```bash
#!/bin/bash
# deploy-platform.sh

# Deploy Kubernetes resources
kubectl apply -f k8s/

# Configure with environment variables
export FRED_API_KEY="${FRED_API_KEY_SECRET}"
export EIA_API_KEY="${EIA_API_KEY_SECRET}"

# Run automated setup
./scripts/setup-commodity-platform.sh --non-interactive

# Verify deployment
./scripts/verify-platform-complete.sh || exit 1

echo "Deployment successful!"
```

---

### Partial Updates

```bash
# Only update API keys
./configure-api-keys.sh

# Only import new workflows
python3 import-dolphinscheduler-workflows.py --skip-existing

# Only verify health
./verify-platform-complete.sh
```

---

### Development/Testing

```bash
# Quick setup without verification (faster iterations)
./setup-commodity-platform.sh --skip-verification

# Re-import workflows after changes
python3 import-dolphinscheduler-workflows.py

# Check specific components
./verify-platform-complete.sh | grep -A5 "DolphinScheduler"
```

---

## Environment Variables

### API Keys (for non-interactive mode)
```bash
export FRED_API_KEY="your-fred-api-key"
export EIA_API_KEY="your-eia-api-key"
export NOAA_API_KEY="your-noaa-api-key"
export WORLD_BANK_API_KEY="your-wb-key"
export WEATHER_API_KEY="your-weather-key"
export ICE_API_KEY="your-ice-key"
export API_KEY="your-market-data-key"
```

### Service URLs (optional overrides)
```bash
export DOLPHINSCHEDULER_URL="http://custom-url:12345"
export SUPERSET_URL="http://custom-url:8088"
```

---

## Troubleshooting

### Script fails with "kubectl not found"
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
```

### API key validation fails
```bash
# Check secret contents
kubectl get secret seatunnel-api-keys -n data-platform -o yaml

# Manually update
kubectl edit secret seatunnel-api-keys -n data-platform
```

### Workflow import fails
```bash
# Check DolphinScheduler API
kubectl get pods -n data-platform -l app=dolphinscheduler-api
kubectl logs -n data-platform -l app=dolphinscheduler-api --tail=50

# Try with port-forward
kubectl port-forward -n data-platform svc/dolphinscheduler-api-service 12345:12345 &
python3 import-dolphinscheduler-workflows.py --dolphinscheduler-url http://localhost:12345
```

### Dashboard import fails
```bash
# Check Superset health
kubectl get pods -n data-platform -l app=superset
kubectl exec -n data-platform $(kubectl get pod -n data-platform -l app=superset -o jsonpath='{.items[0].metadata.name}') -- curl -s http://localhost:8088/health
```

---

## Dependencies

**Required**:
- `kubectl` - Kubernetes CLI
- `python3` - Python 3.8+
- `bash` - Shell (v4.0+)

**Optional** (for verification):
- `jq` - JSON processor
- `curl` - HTTP client

**Python Modules**:
- Standard library only (no pip install needed)

---

## Documentation

- **Automation Guide**: `../docs/automation/AUTOMATION_GUIDE.md`
- **Quick Start**: `../NEXT_STEPS.md`
- **Implementation Summary**: `../AUTOMATION_IMPLEMENTATION_SUMMARY.md`
- **Main README**: `../README.md`

---

## Support

### Getting Help

1. Check script `--help` flag:
   ```bash
   ./setup-commodity-platform.sh --help
   python3 import-dolphinscheduler-workflows.py --help
   ```

2. Review documentation in `docs/automation/`

3. Check logs for errors:
   ```bash
   kubectl logs -n data-platform <pod-name>
   ```

4. Run verification:
   ```bash
   ./verify-platform-complete.sh
   ```

---

## Contributing

When adding new scripts:
1. Include `--help` flag
2. Add usage examples to this README
3. Update `AUTOMATION_GUIDE.md`
4. Make script executable: `chmod +x script.sh`
5. Test both interactive and non-interactive modes
6. Add error handling and clear error messages
7. Include progress indicators

---

## License

Internal use - 254Carbon Data Platform

---

**Last Updated**: October 21, 2025  
**Scripts Version**: 1.0.0


