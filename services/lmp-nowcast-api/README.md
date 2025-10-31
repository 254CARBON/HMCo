# LMP Nowcast API

Real-time LMP forecasting service with physics-aware graph transformers.

## Features

- **Sub-500ms inference**: P95 latency <500ms for 5k nodes
- **Quantile predictions**: P10, P50, P90 with calibration
- **MLflow integration**: Model registry and version management
- **Diagnostics**: Performance metrics and model monitoring

## DoD Targets

- MAPE reduction: **-20%** vs baseline
- CRPS reduction: **-15%** vs baseline
- P95 inference: **<500ms** for 5k nodes

## API Endpoints

### `POST /predict`
Generate LMP predictions for specified ISO/nodes.

**Request:**
```json
{
  "iso": "CAISO",
  "nodes": ["NODE_0001", "NODE_0002"],
  "include_diagnostics": true
}
```

**Response:**
```json
{
  "run_id": "run_20250131_120000",
  "timestamp": "2025-01-31T12:00:00Z",
  "predictions": {
    "q10": [...],
    "q50": [...],
    "q90": [...]
  },
  "diagnostics": {
    "inference_time_ms": 385.2
  }
}
```

### `GET /`
Health check endpoint.

### `GET /model-info`
Model metadata and configuration.

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MODEL_PATH=/models/lmp-nowcast-latest.pt

# Run server
python server.py
```

## Docker

```bash
docker build -t lmp-nowcast-api .
docker run -p 8000:8000 lmp-nowcast-api
```
