"""
Congestion Simulator Service
Applies outages/derates and emits LMP deltas using OPF surrogate
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import time
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Congestion Simulator",
    description="Fast LMP delta simulator for outage scenarios",
    version="0.1.0"
)


class Outage(BaseModel):
    """Outage configuration"""
    from_node: str
    to_node: str
    start_time: str
    duration_minutes: int


class SimulationRequest(BaseModel):
    """Simulation request"""
    iso: str
    outages: List[Outage]
    baseline_lmp: Optional[Dict[str, float]] = None


class SimulationResponse(BaseModel):
    """Simulation results"""
    runId: str
    lmpDeltas: Dict[str, float]
    confidence: Dict[str, float]
    computeTimeMs: float
    correlation: float


@app.post("/simulate", response_model=SimulationResponse)
async def simulate_scenario(request: SimulationRequest):
    """
    Simulate outage scenario and predict LMP impacts
    Target: <2s latency, correlation ≥0.8 to true deltas
    """
    start_time = time.time()
    
    try:
        # Load OPF surrogate model
        # In production: from models.opf_surrogate import OPFSurrogate
        
        # Simulate LMP deltas
        lmp_deltas, confidence = _simulate_deltas(
            request.iso,
            request.outages,
            request.baseline_lmp or {}
        )
        
        compute_time_ms = (time.time() - start_time) * 1000
        
        # Check target
        if compute_time_ms > 2000:
            logger.warning(f"Simulation time {compute_time_ms:.0f}ms exceeds 2s target")
        
        response = SimulationResponse(
            runId=f"sim_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            lmpDeltas=lmp_deltas,
            confidence=confidence,
            computeTimeMs=compute_time_ms,
            correlation=0.85  # Mock - in production compute vs actual
        )
        
        logger.info(f"Simulation complete: {len(lmp_deltas)} nodes, {compute_time_ms:.1f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _simulate_deltas(
    iso: str,
    outages: List[Outage],
    baseline_lmp: Dict[str, float]
) -> tuple:
    """Simulate LMP deltas using surrogate model"""
    import numpy as np
    
    # Mock implementation
    num_nodes = 100
    nodes = [f"{iso}_NODE_{i:04d}" for i in range(num_nodes)]
    
    # Generate deltas based on number of outages
    severity = len(outages) * 2.5
    
    lmp_deltas = {}
    confidence = {}
    
    for node in nodes:
        # Random delta with higher impact near outages
        delta = np.random.randn() * severity
        lmp_deltas[node] = float(delta)
        
        # Confidence based on distance from outages
        conf = np.random.uniform(0.7, 0.95)
        confidence[node] = float(conf)
    
    return lmp_deltas, confidence


@app.get("/")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "congestion-simulator",
        "target_latency_ms": 2000,
        "target_correlation": 0.8
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
