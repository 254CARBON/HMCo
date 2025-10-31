# Analytics

Analytics modules for pattern discovery, signal generation, and P&L attribution.

## Modules

### 1. Patterns (`patterns/`)
**Topology Motif Miner** - Recurrent congestion pattern discovery.

- `congestion_motifs.py`: SAX/Matrix Profile + HDBSCAN on LMP spreads & flows
- **DoD**: Motif alerts capture ≥60% of high-congestion intervals with >2:1 precision:recall

**Usage:**
```bash
python analytics/patterns/congestion_motifs.py \
  --lmp-spreads data/spreads.csv \
  --flow-data data/flows.csv \
  --output motifs.csv
```

### 2. Signals (`signals/`)
**Imbalance Cost Signals** - Hub/node risk premia generation.

- `imbalance_cost.py`: Calculate P10/P50/P90 imbalance cost curves

**Usage:**
```bash
python analytics/signals/imbalance_cost.py \
  --history data/imbalance_history.csv \
  --output risk_premia.csv
```

### 3. Attribution (`attribution/`)
**Decision Shapley** - Driver→P&L decomposition using Shapley values.

- `decision_shapley.py`: Shapley over decisions with counterfactual scenarios
- **DoD**: Top-3 drivers explain ≥70% of P&L variance for target books

**Usage:**
```bash
python analytics/attribution/decision_shapley.py \
  --decisions data/decisions.csv \
  --pnl data/pnl.csv \
  --output attribution.csv
```

## Integration

Analytics outputs are:
1. Stored in ClickHouse for historical tracking
2. Surfaced in Portal Copilot dashboards
3. Used by trading strategies for real-time decisions

## Workflow Orchestration

All analytics run on DolphinScheduler schedules:
- Motif mining: Daily
- Signal generation: Hourly
- Attribution: Daily after market close
