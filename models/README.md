# Models

This directory contains machine learning models for proprietary analytics and P&L optimization.

## Modules

### 1. Grid Sensitivities (`grid_sensitivities/`)
**PTDF/LODF Estimator** - Near-real-time network sensitivities for nodal price prediction.

- `train.py`: Train PTDF/LODF models using compressed sensing + sparse VAR/Granger + ridge
- `serve.py`: Real-time inference service for sensitivity predictions
- **DoD**: Predicted ΔLMP correlation ≥ 0.8 on historical outages

### 2. Imbalance Risk (`imbalance/`)
**DA↔RT Imbalance Cost Model** - Schedule risk pricing and hedge optimization.

- `features.py`: Feature engineering for DA-RT spreads, forecast errors, ramping
- `train.py`: Quantile regression + conformal prediction
- `infer.py`: Real-time inference for expected imbalance costs
- **DoD**: Hedge cuts realized imbalance costs by ≥20% at same load

### 3. Ancillary Services (`ancillary/`)
**Co-forecast & Co-opt** - Joint energy + ancillary services optimization.

- `prep.py`: Data preparation for multi-task learning
- `train.py`: Multi-task model with shared features
- `infer.py`: Joint DA/RT forecasts for energy/AS split
- **DoD**: Portfolio P&L +10% with no SLO breach

### 4. Tail Risk (`tailrisk/`)
**Extreme-Tail Spike Engine** - EVT + generative oversampling for price spikes.

- `evt_fit.py`: Extreme Value Theory (POT/GEV) fitting
- `oversample.py`: GAN-based synthetic tail scenario generation
- **DoD**: Spike coverage ≥90% at targeted α with <10% false alarms

### 5. Unit Commitment (`unit_commit/`)
**Probability Surfaces** - Start/stop/ramp predictions with hazard models.

- `hazard.py`: Survival/hazard models with weather, outages, fuel spreads
- `calibration.py`: Calibration and Brier score optimization
- **DoD**: Brier score improves ≥15% vs baseline; better ramp errors on peak hours

### 6. LNG Power (`lng_power/`)
**LNG→Power Coupling** - AIS/regas/pipeline impact on hub prices.

- `impact_model.py`: Causal forest/IV models for price impact
- **DoD**: Predicted impact explains ≥30% of spread variance on arrival windows

### 7. Carbon Coupling (`carbon_coupling/`)
**Carbon-Adjusted Hedging** - Marginal intensity & EUA/CCA coupling.

- `train.py`: VARX/State-space models for carbon-energy coupling
- `infer.py`: Real-time carbon risk factor inference
- **DoD**: Variance −10% with neutralized carbon factor, same mean return

## Usage

Each module is designed to be used both standalone and as part of DolphinScheduler workflows.

```bash
# Example: Train PTDF/LODF model
python models/grid_sensitivities/train.py \
  --flow-data data/flows.csv \
  --lmp-data data/lmps.csv \
  --topology-signals data/topology.csv \
  --output-model models/ptdf_lodf.pkl

# Example: Inference
python models/grid_sensitivities/serve.py \
  --model-path models/ptdf_lodf.pkl \
  --flow-changes '{"line1": 100, "line2": -50}' \
  --topology-signals '{"topology_signal1": 0.5}'
```

## Data Pipeline

Models integrate with:
- **ClickHouse**: Store predictions and metrics
- **MLflow**: Track experiments and models
- **DolphinScheduler**: Orchestrate training and inference
- **Portal**: Visualize results in Copilot dashboard

## Performance Monitoring

All models write performance metrics to ClickHouse for monitoring:
- Prediction accuracy (MAE, RMSE, correlation)
- Calibration metrics (Brier score, quantile coverage)
- Business metrics (P&L impact, hedge effectiveness)
