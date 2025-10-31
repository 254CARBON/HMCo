# Quick Start Guide: Power/LNG Trading Analytics

This guide will get you up and running with the 10 advanced trading features.

## Prerequisites

- Python 3.9+
- Node.js 18+
- Docker & Kubernetes (for deployment)
- ClickHouse database
- MLflow tracking server (optional)

## Installation

### 1. Install Python Dependencies

```bash
# Install ML/DL frameworks and analytics libraries
pip install -r requirements-models.txt

# Or install specific components
pip install torch pytorch-lightning torch-geometric
pip install pandas numpy scipy statsmodels
pip install fastapi uvicorn clickhouse-driver mlflow
```

### 2. Install Portal Dependencies

```bash
cd portal
npm install
```

## Running Services

### LMP Nowcast API

```bash
cd services/lmp-nowcast-api

# Set environment variables
export MODEL_PATH=/models/lmp-nowcast-latest.pt
export MLFLOW_TRACKING_URI=http://mlflow:5000

# Run server
python server.py

# Server will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

**Test the API:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "iso": "CAISO",
    "nodes": ["NODE_0001", "NODE_0002"],
    "include_diagnostics": true
  }'
```

### Congestion Simulator

```bash
cd services/congestion-sim

# Run server
python simulator.py

# Server will be available at http://localhost:8001
```

**Test the simulator:**

```bash
curl -X POST http://localhost:8001/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "iso": "MISO",
    "outages": [
      {
        "from_node": "NODE_0010",
        "to_node": "NODE_0020",
        "start_time": "2025-01-31T16:00:00Z",
        "duration_minutes": 120
      }
    ]
  }'
```

### Portal UI

```bash
cd portal

# Set environment variables
export LMP_NOWCAST_URL=http://localhost:8000
export CONGESTION_SIM_URL=http://localhost:8001

# Run development server
npm run dev

# Portal will be available at http://localhost:3000
```

## Database Setup

### Initialize ClickHouse Schemas

```bash
# Connect to ClickHouse
clickhouse-client

# Create database
CREATE DATABASE IF NOT EXISTS trading;

# Load schemas
USE trading;
SOURCE clickhouse/ddl/rt_forecasts.sql;
SOURCE clickhouse/ddl/fact_cross_asset.sql;
SOURCE clickhouse/ddl/curve_scenarios.sql;
SOURCE clickhouse/ddl/embeddings.sql;
```

## Training Models

### 1. LMP Nowcast Model

```python
from models.lmp_nowcast.trainer import LMPNowcastTrainer
from models.lmp_nowcast.dataprep import LMPDataPreparation
from datetime import datetime

# Prepare data
dataprep = LMPDataPreparation()
dataset = dataprep.prepare_training_dataset(
    iso='CAISO',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

# Configure model
model_config = {
    'num_nodes': 100,
    'node_features': 32,
    'hidden_dim': 128,
    'num_heads': 8,
    'num_layers': 4,
    'forecast_horizon': 12,
    'quantiles': [0.1, 0.5, 0.9]
}

# Train
trainer = LMPNowcastTrainer(
    model_config=model_config,
    data_config={},
    training_config={'max_epochs': 100}
)

trainer.train(train_loader, val_loader, mlflow_tracking=True)
trainer.save_model('lmp-nowcast-caiso.pt')
```

### 2. Node Embeddings

```python
from models.node2grid.graphsage_model import NodeEmbedding
from features.graph.topology import GraphTopology

# Load topology
topo = GraphTopology()
graph = topo.load_topology('CAISO')

# Train embeddings
model = NodeEmbedding(in_channels=32, out_channels=256)
# ... training loop ...

# Save embeddings to ClickHouse
# INSERT INTO embeddings.node_hub_v1 ...
```

### 3. Regime Detector

```python
from models.regime.bayes_hmm import BayesianHMM
import numpy as np

# Prepare features
features = np.random.randn(1000, 10)  # Load real features

# Fit HMM
hmm = BayesianHMM(num_states=3, feature_dim=10)
hmm.fit(features, max_iter=100)

# Predict regimes
states, probs = hmm.predict_regime(features)
```

## Using the Features

### Cross-Commodity Analytics

```python
from analytics.signals.cross_commodity.spark_spread import SparkSpreadCalculator

calc = SparkSpreadCalculator()

# Calculate spark spread
spread = calc.calculate_spark_spread(
    power_lmp=45.0,      # $/MWh
    gas_price=5.0,       # $/MMBtu
    heat_rate=7.0,       # MMBtu/MWh
    variable_om=2.0      # $/MWh
)

print(f"Spark Spread: ${spread:.2f}/MWh")

# Implied heat rate
ihr = calc.calculate_implied_heat_rate(
    power_lmp=45.0,
    gas_price=5.0
)

print(f"Implied Heat Rate: {ihr:.2f} MMBtu/MWh")
```

### Strategy Optimizer

```python
from strategies.rl_hedger.cql_agent import CQLHedger
import torch

# Initialize agent
agent = CQLHedger(state_dim=64, action_dim=10)

# Current market state
state = torch.randn(1, 64)

# Optimize with constraints
action, metrics = agent.optimize_with_constraints(
    state,
    risk_budget=100000,
    position_limits={'max_long': 100, 'max_short': 50}
)

print(f"Recommended Action: {action}")
print(f"Expected Value: ${metrics['expected_value']:.2f}")
print(f"Risk Used: {metrics['risk_used']:.1%}")
```

## Portal UI Features

### Scenarios Dashboard

Navigate to http://localhost:3000/scenarios

1. Select ISO (CAISO, MISO, SPP, etc.)
2. Add outage configurations
3. Run simulation (<2s target)
4. View LMP deltas with confidence bands

### Trader Copilot

Navigate to http://localhost:3000/copilot

1. Describe scenario (e.g., "Outage on LINE_X @ 16:00")
2. Select hubs and time horizon
3. Run copilot analysis (<60s target)
4. Review:
   - Recommended actions (bids/hedges)
   - Risk metrics (VaR, Sharpe, drawdown)
   - Key drivers (SHAP attribution)
5. Export report or submit for approval

## Validation

Run the validation script to verify everything is working:

```bash
python validate_implementation.py
```

Expected output:
```
✅ ALL FEATURES SUCCESSFULLY IMPLEMENTED!
Results: 30/30 checks passed (100.0%)
```

## Performance Monitoring

### Check API Performance

```bash
# LMP Nowcast latency
curl http://localhost:8000/model-info

# Should show:
# - target_inference_ms: 500
# - target_mape_reduction: 20
# - target_crps_reduction: 15
```

### Monitor ClickHouse

```sql
-- Check forecast metrics
SELECT 
    iso,
    hour,
    avg_inference_time_ms,
    p95_inference_time_ms
FROM rt_forecasts_metrics
WHERE hour >= now() - INTERVAL 24 HOUR
ORDER BY hour DESC;

-- Verify p95 < 500ms target
SELECT 
    quantile(0.95)(inference_time_ms) as p95_ms
FROM rt_forecasts
WHERE timestamp >= now() - INTERVAL 1 HOUR;
```

## Troubleshooting

### Model Loading Issues

If models fail to load:
```bash
# Check model file exists
ls -lh /models/lmp-nowcast-latest.pt

# Verify PyTorch can load it
python -c "import torch; torch.load('/models/lmp-nowcast-latest.pt')"
```

### ClickHouse Connection

```python
from clickhouse_driver import Client

client = Client('localhost')
result = client.execute('SELECT version()')
print(f"ClickHouse version: {result[0][0]}")
```

### Portal Build Issues

```bash
cd portal

# Clear cache and rebuild
rm -rf .next node_modules
npm install
npm run build
```

## Docker Deployment

### Build Services

```bash
# LMP Nowcast API
docker build -t lmp-nowcast-api:latest services/lmp-nowcast-api/

# Congestion Simulator
docker build -t congestion-sim:latest services/congestion-sim/

# Portal
docker build -t trading-portal:latest portal/
```

### Deploy to Kubernetes

```bash
# Apply Kubernetes manifests (to be created)
kubectl apply -f k8s/lmp-nowcast-api.yaml
kubectl apply -f k8s/congestion-sim.yaml
kubectl apply -f k8s/trading-portal.yaml
```

## Next Steps

1. **Load Historical Data**: Import ISO data into ClickHouse
2. **Train Production Models**: Train on full historical datasets
3. **Shadow Mode**: Run alongside existing systems
4. **Validate Performance**: Measure against DoD targets
5. **Gradual Rollout**: Enable features incrementally
6. **Monitor & Iterate**: Track metrics and improve

## Support

For issues or questions:
- See detailed documentation in `TRADING_FEATURES.md`
- Check API docs at service `/docs` endpoints
- Review test files in `tests/unit/`

---

**Ready to transform power/LNG trading with AI! 🚀**
