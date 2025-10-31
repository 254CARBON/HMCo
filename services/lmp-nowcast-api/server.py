"""
LMP Nowcast API Service
gRPC/HTTP inference service with MLflow registry integration
Target: <500ms p95 inference for 5k nodes
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import logging
import time
from datetime import datetime
import sys
import os

# Add models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="LMP Nowcast API",
    description="Real-time LMP forecasting with physics-aware graph transformers",
    version="0.1.0"
)


class PredictionRequest(BaseModel):
    """Request model for LMP prediction"""
    iso: str
    nodes: Optional[List[str]] = None
    include_diagnostics: bool = True
    include_quantiles: bool = True


class PredictionResponse(BaseModel):
    """Response model for LMP prediction"""
    run_id: str
    timestamp: str
    iso: str
    num_nodes: int
    predictions: Dict
    diagnostics: Optional[Dict] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float


# Global state
inference_engine = None
start_time = time.time()
model_version = "0.1.0"


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup"""
    global inference_engine
    
    logger.info("Initializing LMP Nowcast API...")
    
    try:
        # Load model from MLflow registry
        model_path = os.getenv("MODEL_PATH", "/models/lmp-nowcast-latest.pt")
        
        # Import inference engine
        from models.lmp_nowcast.infer import LMPNowcastInference
        
        inference_engine = LMPNowcastInference(
            model_path=model_path,
            device='cpu'  # Use GPU if available
        )
        
        logger.info(f"Model loaded from {model_path}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will run in mock mode")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if inference_engine else "degraded",
        model_loaded=inference_engine is not None,
        version=model_version,
        uptime_seconds=time.time() - start_time
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_lmp(request: PredictionRequest):
    """
    Generate LMP predictions
    
    Returns quantile predictions (P10, P50, P90) with diagnostics
    """
    start_time_ms = time.time()
    
    try:
        # Generate run ID
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        if inference_engine is None:
            # Mock response for testing
            predictions = _generate_mock_predictions(request.iso, request.nodes or [])
        else:
            # Prepare features
            features = _prepare_features(request.iso, request.nodes)
            
            # Run inference
            result = inference_engine.predict(
                features,
                return_diagnostics=request.include_diagnostics
            )
            
            predictions = result
        
        # Format response
        response = PredictionResponse(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            iso=request.iso,
            num_nodes=len(request.nodes) if request.nodes else 100,
            predictions=predictions.get('predictions', predictions),
            diagnostics=predictions.get('diagnostics') if request.include_diagnostics else None
        )
        
        # Log performance
        elapsed_ms = (time.time() - start_time_ms) * 1000
        logger.info(f"Prediction completed in {elapsed_ms:.2f}ms for {response.num_nodes} nodes")
        
        # Check p95 target
        if elapsed_ms > 500:
            logger.warning(f"Inference time {elapsed_ms:.2f}ms exceeds 500ms target")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
async def batch_predict(requests: List[PredictionRequest], background_tasks: BackgroundTasks):
    """
    Batch prediction endpoint
    
    Processes multiple ISOs/scenarios in parallel
    """
    results = []
    
    for req in requests:
        result = await predict_lmp(req)
        results.append(result)
    
    return {
        "batch_id": f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "num_requests": len(requests),
        "results": results
    }


@app.get("/model-info")
async def model_info():
    """Get model information and metadata"""
    return {
        "model_version": model_version,
        "model_type": "PhysicsGuidedSTTransformer",
        "framework": "PyTorch Lightning",
        "quantiles": [0.1, 0.5, 0.9],
        "target_inference_ms": 500,
        "target_mape_reduction": 20,
        "target_crps_reduction": 15,
        "loaded": inference_engine is not None
    }


def _prepare_features(iso: str, nodes: Optional[List[str]]) -> Dict:
    """Prepare features for inference"""
    # In production, load from ClickHouse and feature stores
    features = {
        'iso': iso,
        'num_nodes': len(nodes) if nodes else 100,
        'node_ids': nodes or [f'NODE_{i:04d}' for i in range(100)],
        'seq_len': 36,  # 3 hours of 5-min data
        'feature_dim': 32,
        'base_time': datetime.utcnow()
    }
    
    return features


def _generate_mock_predictions(iso: str, nodes: List[str]) -> Dict:
    """Generate mock predictions for testing"""
    import numpy as np
    
    num_nodes = len(nodes) if nodes else 100
    forecast_horizon = 12  # 60min / 5min
    
    # Generate synthetic predictions
    base_lmp = np.random.uniform(25, 45, (num_nodes, forecast_horizon))
    
    predictions = {
        'q10': base_lmp * 0.9,
        'q50': base_lmp,
        'q90': base_lmp * 1.1,
        'node_ids': nodes or [f'{iso}_NODE_{i:04d}' for i in range(num_nodes)],
        'timestamps': [
            (datetime.utcnow().timestamp() + i * 300) for i in range(forecast_horizon)
        ]
    }
    
    return predictions


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
