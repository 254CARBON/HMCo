# Advanced Analytics Implementation

This document describes the 10 proprietary data logic & analytics features implemented for P&L optimization.

## Overview

All features are designed to move P&L, not just dashboards. Each has concrete Definition of Done (DoD) metrics tied to business outcomes.

## Implemented Features

### 1. Data-driven PTDF/LODF Estimator
**Location:** `models/grid_sensitivities/`, `features/graph/`, `clickhouse/ddl/grid_sensitivities.sql`

Near-real-time network sensitivities using compressed sensing + sparse VAR/Granger + ridge regression.

**DoD:** On historical outages, predicted ΔLMP correlation ≥ 0.8; confidence tracks error.

**Key Components:**
- `train.py`: PTDF/LODF model training with cross-validation
- `serve.py`: Real-time sensitivity prediction service
- `topology_signals.py`: Co-movement patterns and flow proxies
- ClickHouse tables for sensitivity matrices and predictions

### 2. DA↔RT Imbalance Cost Model
**Location:** `models/imbalance/`, `analytics/signals/`, `clickhouse/ddl/imbalance_risk.sql`

Schedule risk pricing with quantile regression + conformal prediction.

**DoD:** Backtest shows hedge cuts realized imbalance costs by ≥20% at same load.

**Key Components:**
- `features.py`: DA-RT spread, forecast error, ramping features
- `train.py`: Quantile regression (P10/P50/P90)
- `infer.py`: Real-time imbalance cost prediction
- `imbalance_cost.py`: Risk premia signal generation

### 3. Ancillary Services Co-forecast & Co-opt
**Location:** `models/ancillary/`, `clickhouse/ddl/ancillary_forecasts.sql`

Joint energy + AS optimization (reg/spin/non-spin) with multi-task learning.

**DoD:** Historical replay shows portfolio P&L +10% with no SLO breach.

**Key Components:**
- Multi-task model with shared features
- Co-optimization for energy/AS split
- Portfolio P&L tracking with AS positions

### 4. Topology Motif Miner
**Location:** `analytics/patterns/`, `clickhouse/ddl/congestion_motifs.sql`

Recurrent congestion pattern discovery using SAX/Matrix Profile + HDBSCAN.

**DoD:** Motif alerts capture ≥60% of high-congestion intervals with >2:1 precision:recall.

**Key Components:**
- `congestion_motifs.py`: Pattern discovery and matching
- Motif library with entry/exit signals
- Real-time motif detection and alerts

### 5. Extreme-Tail Spike Engine
**Location:** `models/tailrisk/`, `clickhouse/ddl/tail_events.sql`

EVT (Extreme Value Theory) + generative oversampling for price spike prediction.

**DoD:** Spike coverage ≥90% at targeted α with <10% false-alarm growth.

**Key Components:**
- `evt_fit.py`: POT/GEV parameter estimation
- `oversample.py`: GAN-based synthetic tail scenarios
- Calibrated exceedance probabilities

### 6. Unit Commitment Probability Surfaces
**Location:** `models/unit_commit/`, `datasets/unit_status/`, `clickhouse/ddl/unit_commit_surfaces.sql`

Start/stop/ramp predictions using survival/hazard models.

**DoD:** Brier score improves ≥15% vs baseline; better ramp errors on peak hours.

**Key Components:**
- `hazard.py`: Hazard models with weather, outages, fuel spreads
- `calibration.py`: Probability calibration
- Unit status reconstruction from awards/telemetry

### 7. LNG→Power Coupling
**Location:** `features/lng/`, `models/lng_power/`, `clickhouse/ddl/lng_impact.sql`

AIS/regas/pipeline impact on hub prices using causal inference.

**DoD:** Event study shows predicted impact explains ≥30% of spread variance on arrival windows.

**Key Components:**
- `ais_fusion.py`: AIS vessel tracking with port operations
- `regas_capacity.py`: Regasification terminal capacity
- `impact_model.py`: Causal forest/IV for price impact

### 8. Execution & Market-Impact Optimizer
**Location:** `execution/impact/`, `execution/scheduler/`, `clickhouse/ddl/execution_metrics.sql`

Liquidity-aware routing with Almgren-Chriss + nonlinear impact models.

**DoD:** Live paper-trade shows ≥30% slippage reduction vs simple POV at matched risk.

**Key Components:**
- `fit.py`: Market impact curve estimation
- `simulate.py`: Execution simulation
- Smart schedulers: POV, TWAP, IS, hybrid
- Cross-venue routing (ICE/CME/EEX)

### 9. Carbon-Adjusted Hedging
**Location:** `features/carbon/`, `models/carbon_coupling/`

Marginal intensity & EUA/CCA coupling for carbon risk immunization.

**DoD:** Backtest shows variance −10% with neutralized carbon factor, same mean return.

**Key Components:**
- `marginal_intensity.py`: Nodal carbon intensity estimates
- VARX/State-space models for carbon-energy coupling
- Hedge overlays with carbon exposure constraints

### 10. Alpha Attribution & Decision Shapley
**Location:** `analytics/attribution/`, `clickhouse/ddl/alpha_attribution.sql`, `portal/app/(dashboard)/copilot/attribution/`

Driver→P&L decomposition using Shapley values over counterfactual scenarios.

**DoD:** Top-3 drivers explain ≥70% of P&L variance for target books; attribution stable across regimes.

**Key Components:**
- `decision_shapley.py`: Shapley value computation (exact & sampling)
- Per-trade/strategy attribution
- Portal dashboard for visualization

## Architecture

### Data Flow
1. **Ingestion**: Market data, weather, fundamentals → ClickHouse
2. **Feature Engineering**: `features/` modules compute derived features
3. **Model Training**: `models/` modules train on historical data
4. **Inference**: Real-time predictions via serving modules
5. **Analytics**: `analytics/` modules generate signals and attribution
6. **Execution**: `execution/` modules optimize trading
7. **Visualization**: Portal dashboards display results

### Storage Layer (ClickHouse)
All features write to dedicated ClickHouse tables:
- `grid_sensitivities.sql`: PTDF/LODF matrices and predictions
- `imbalance_risk.sql`: Imbalance cost forecasts and risk premia
- `ancillary_forecasts.sql`: AS price/demand forecasts and allocations
- `congestion_motifs.sql`: Motif library and occurrences
- `tail_events.sql`: EVT parameters and spike probabilities
- `unit_commit_surfaces.sql`: Unit commitment probabilities
- `lng_impact.sql`: LNG vessel tracking and price impact
- `execution_metrics.sql`: Order execution and slippage
- `alpha_attribution.sql`: Decision Shapley values and factor attribution

### Orchestration Layer (DolphinScheduler)
Workflows under `workflows/`:
- `12-grid-sensitivities.json`: PTDF/LODF training (every 6 hours)
- `13-imbalance-risk.json`: Imbalance model training (daily)
- `14-alpha-attribution.json`: Attribution calculation (daily)
- Additional workflows for other features

### Presentation Layer (Portal)
- `portal/app/(dashboard)/copilot/attribution/`: Attribution dashboard
- Integration with existing Copilot UI
- Real-time alerts and scenario analysis

## Development Workflow

### Adding a New Feature
1. Create module under `models/`, `analytics/`, `features/`, or `execution/`
2. Define ClickHouse schema in `clickhouse/ddl/`
3. Create DolphinScheduler workflow in `workflows/`
4. Add portal visualization if needed
5. Write tests and documentation

### Model Training
```bash
# Train PTDF/LODF model
python models/grid_sensitivities/train.py \
  --flow-data data/flows.csv \
  --lmp-data data/lmps.csv \
  --topology-signals data/topology.csv \
  --output-model models/ptdf_lodf.pkl

# Train imbalance model
python models/imbalance/train.py \
  --features data/imbalance_features.csv \
  --output models/imbalance_model.pkl
```

### Real-time Inference
```bash
# PTDF/LODF prediction
python models/grid_sensitivities/serve.py \
  --model-path models/ptdf_lodf.pkl \
  --flow-changes '{"line1": 100}' \
  --topology-signals '{}'

# Imbalance cost prediction
python models/imbalance/infer.py \
  --model models/imbalance_model.pkl \
  --features data/current_features.csv \
  --output predictions.csv
```

## Monitoring & Validation

All models write performance metrics to ClickHouse for continuous monitoring:
- Prediction accuracy (MAE, RMSE, correlation)
- Calibration (Brier score, quantile coverage)
- Business impact (P&L, hedge effectiveness, slippage reduction)

Dashboards in Portal show:
- Model performance over time
- Feature importance and Shapley values
- Alert history and hit rates
- P&L attribution

## Next Steps

1. **Backtest Validation**: Run historical backtests for each DoD metric
2. **Live Testing**: Deploy to staging for paper trading
3. **Integration**: Connect to production trading systems
4. **Monitoring**: Set up alerting for model degradation
5. **Iteration**: Refine models based on live performance

## References

- PTDF/LODF: Power Transfer Distribution Factors for congestion management
- EVT: Extreme Value Theory for tail risk
- Shapley Values: Game-theoretic attribution method
- Almgren-Chriss: Optimal execution framework
- Causal Forest: ML method for heterogeneous treatment effects
