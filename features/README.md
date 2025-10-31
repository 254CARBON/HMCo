# Features

Feature engineering modules for ML models.

## Modules

### 1. Graph (`graph/`)
**Topology Signals** - Network-based features for grid sensitivity models.

- `topology_signals.py`: Co-movement, flow proxies, centrality metrics, PCA

### 2. LNG (`lng/`)
**LNG Features** - AIS tracking and regasification capacity.

- `ais_fusion.py`: AIS data fusion with port operations and weather
- `regas_capacity.py`: Regasification terminal capacity and utilization

### 3. Carbon (`carbon/`)
**Carbon Features** - Marginal intensity and carbon price coupling.

- `marginal_intensity.py`: Nodal marginal carbon intensity estimates

## Usage

Features are typically computed as part of model training workflows:

```bash
# Topology signals
python features/graph/topology_signals.py \
  --flow-data flows.csv \
  --lmp-data lmps.csv \
  --nodes nodes.csv \
  --lines lines.csv \
  --output topology_signals.csv
```

Features are cached in ClickHouse for reuse across models.
