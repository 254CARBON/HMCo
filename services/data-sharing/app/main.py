"""
Data Sharing Service - Partner entitlements and token-based access
Provides secure, audited access to curated datasets for external partners
"""
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from pydantic import BaseModel
from enum import Enum

app = FastAPI(
    title="HMCo Data Sharing Service",
    description="Secure data sharing with time-scoped tokens and audit logging",
    version="1.0.0"
)

security = HTTPBearer()
logger = logging.getLogger(__name__)


class AccessScope(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"


class DatasetType(str, Enum):
    LMP = "lmp"
    WEATHER = "weather"
    OUTAGES = "outages"
    TRADING = "trading"


class Partner(BaseModel):
    partner_id: str
    partner_name: str
    email: str
    organization: str
    is_active: bool = True
    created_at: datetime = datetime.utcnow()


class DatasetEntitlement(BaseModel):
    partner_id: str
    dataset_type: DatasetType
    dataset_name: str
    scope: AccessScope = AccessScope.READ_ONLY
    row_filters: Optional[dict] = None  # SQL WHERE conditions
    column_masks: Optional[List[str]] = None  # Columns to mask
    expires_at: Optional[datetime] = None
    created_at: datetime = datetime.utcnow()


class AccessToken(BaseModel):
    token: str
    partner_id: str
    scope: AccessScope
    datasets: List[str]
    expires_at: datetime
    issued_at: datetime = datetime.utcnow()


class AccessLog(BaseModel):
    partner_id: str
    dataset_name: str
    query: str
    rows_returned: int
    bytes_scanned: int
    execution_time_ms: float
    timestamp: datetime = datetime.utcnow()


# In-memory storage (replace with database in production)
partners_db = {}
entitlements_db = []
access_logs = []


@app.post("/partners/", response_model=Partner)
async def create_partner(partner: Partner):
    """Register a new data sharing partner"""
    if partner.partner_id in partners_db:
        raise HTTPException(status_code=400, detail="Partner already exists")
    
    partners_db[partner.partner_id] = partner
    logger.info(f"Created partner: {partner.partner_id}")
    return partner


@app.get("/partners/", response_model=List[Partner])
async def list_partners():
    """List all registered partners"""
    return list(partners_db.values())


@app.post("/entitlements/", response_model=DatasetEntitlement)
async def grant_entitlement(entitlement: DatasetEntitlement):
    """Grant dataset access to a partner"""
    if entitlement.partner_id not in partners_db:
        raise HTTPException(status_code=404, detail="Partner not found")
    
    entitlements_db.append(entitlement)
    logger.info(f"Granted entitlement: {entitlement.partner_id} -> {entitlement.dataset_name}")
    return entitlement


@app.get("/entitlements/{partner_id}", response_model=List[DatasetEntitlement])
async def get_partner_entitlements(partner_id: str):
    """Get all entitlements for a partner"""
    return [e for e in entitlements_db if e.partner_id == partner_id]


@app.post("/tokens/", response_model=AccessToken)
async def issue_token(
    partner_id: str,
    datasets: List[str],
    scope: AccessScope = AccessScope.READ_ONLY,
    duration_hours: int = 24
):
    """Issue a time-scoped access token for a partner"""
    if partner_id not in partners_db:
        raise HTTPException(status_code=404, detail="Partner not found")
    
    partner = partners_db[partner_id]
    if not partner.is_active:
        raise HTTPException(status_code=403, detail="Partner is inactive")
    
    # Verify partner has entitlements for requested datasets
    partner_datasets = {
        e.dataset_name for e in entitlements_db 
        if e.partner_id == partner_id and (
            e.expires_at is None or e.expires_at > datetime.utcnow()
        )
    }
    
    for dataset in datasets:
        if dataset not in partner_datasets:
            raise HTTPException(
                status_code=403,
                detail=f"Partner not entitled to dataset: {dataset}"
            )
    
    # Generate token (simplified - use proper JWT in production)
    token = f"hmco_{partner_id}_{datetime.utcnow().timestamp()}"
    expires_at = datetime.utcnow() + timedelta(hours=duration_hours)
    
    access_token = AccessToken(
        token=token,
        partner_id=partner_id,
        scope=scope,
        datasets=datasets,
        expires_at=expires_at
    )
    
    logger.info(f"Issued token to {partner_id} for datasets: {datasets}")
    return access_token


@app.post("/access-log/")
async def log_access(log: AccessLog):
    """Log dataset access for audit purposes"""
    access_logs.append(log)
    logger.info(
        f"Access logged: {log.partner_id} accessed {log.dataset_name}, "
        f"{log.rows_returned} rows, {log.bytes_scanned} bytes"
    )
    return {"status": "logged"}


@app.get("/access-log/{partner_id}", response_model=List[AccessLog])
async def get_access_logs(
    partner_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Retrieve access logs for a partner"""
    logs = [l for l in access_logs if l.partner_id == partner_id]
    
    if start_date:
        logs = [l for l in logs if l.timestamp >= start_date]
    if end_date:
        logs = [l for l in logs if l.timestamp <= end_date]
    
    return logs


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "partners_count": len(partners_db),
        "entitlements_count": len(entitlements_db),
        "access_logs_count": len(access_logs)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
