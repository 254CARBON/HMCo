"""
Federation Gateway - Token-scoped Iceberg REST exposure for partner access.
Allows external Trino/Starburst to query your Iceberg catalog with strict scopes.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import jwt
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import redis
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
auth_attempts = Counter('federation_auth_attempts_total', 'Total authentication attempts')
auth_failures = Counter('federation_auth_failures_total', 'Authentication failures')
queries_total = Counter('federation_queries_total', 'Total queries by partner', ['partner', 'catalog'])
query_duration = Histogram('federation_query_duration_seconds', 'Query duration')

app = FastAPI(title="HMCo Federation Gateway", version="1.0.0")
security = HTTPBearer()


class PartnerScope(BaseModel):
    """Partner access scope definition"""
    partner_id: str
    allowed_catalogs: List[str]
    allowed_schemas: List[str]
    allowed_tables: List[str]  # Can use wildcards like "public.*"
    row_filters: Dict[str, str]  # Column-level filters, e.g., {"iso": "CAISO"}
    denied_columns: List[str]  # Columns to exclude
    max_queries_per_hour: int
    max_scan_bytes: int
    expires_at: datetime


class QueryRequest(BaseModel):
    """Federated query request"""
    catalog: str
    schema: str
    table: str
    columns: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None


class FederationGateway:
    """Gateway for federated data sharing"""
    
    def __init__(self, 
                 iceberg_rest_url: str,
                 jwt_secret: str,
                 redis_host: str = 'localhost'):
        self.iceberg_rest_url = iceberg_rest_url
        self.jwt_secret = jwt_secret
        self.redis = redis.Redis(host=redis_host, decode_responses=True)
    
    def verify_token(self, token: str) -> PartnerScope:
        """Verify JWT token and extract partner scope"""
        auth_attempts.inc()
        
        try:
            # Decode JWT
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Extract scope
            scope = PartnerScope(
                partner_id=payload['partner_id'],
                allowed_catalogs=payload.get('allowed_catalogs', []),
                allowed_schemas=payload.get('allowed_schemas', []),
                allowed_tables=payload.get('allowed_tables', []),
                row_filters=payload.get('row_filters', {}),
                denied_columns=payload.get('denied_columns', []),
                max_queries_per_hour=payload.get('max_queries_per_hour', 100),
                max_scan_bytes=payload.get('max_scan_bytes', 1024**3),  # 1GB default
                expires_at=datetime.fromtimestamp(payload['exp'])
            )
            
            # Check expiration
            if scope.expires_at < datetime.now():
                auth_failures.inc()
                raise HTTPException(status_code=401, detail="Token expired")
            
            # Check rate limit
            if not self._check_rate_limit(scope):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            return scope
            
        except jwt.ExpiredSignatureError:
            auth_failures.inc()
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError as e:
            auth_failures.inc()
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    
    def _check_rate_limit(self, scope: PartnerScope) -> bool:
        """Check if partner is within rate limit"""
        key = f"rate_limit:{scope.partner_id}:hour"
        current = self.redis.get(key)
        
        if current is None:
            # First query this hour
            self.redis.setex(key, 3600, 1)  # Expire in 1 hour
            return True
        
        count = int(current)
        if count >= scope.max_queries_per_hour:
            return False
        
        self.redis.incr(key)
        return True
    
    def validate_query_access(self, 
                              scope: PartnerScope,
                              query: QueryRequest) -> bool:
        """Validate that query is within partner's scope"""
        
        # Check catalog access
        if query.catalog not in scope.allowed_catalogs:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied to catalog: {query.catalog}"
            )
        
        # Check schema access
        if query.schema not in scope.allowed_schemas and '*' not in scope.allowed_schemas:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied to schema: {query.schema}"
            )
        
        # Check table access
        table_fqn = f"{query.schema}.{query.table}"
        allowed = any(
            self._matches_pattern(table_fqn, pattern)
            for pattern in scope.allowed_tables
        )
        
        if not allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied to table: {table_fqn}"
            )
        
        # Check column access
        if query.columns:
            denied = set(query.columns) & set(scope.denied_columns)
            if denied:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied to columns: {denied}"
                )
        
        return True
    
    def _matches_pattern(self, table: str, pattern: str) -> bool:
        """Check if table matches pattern (supports wildcards)"""
        import re
        regex_pattern = pattern.replace('*', '.*')
        return bool(re.match(f"^{regex_pattern}$", table))
    
    def apply_row_filters(self,
                          scope: PartnerScope,
                          query: QueryRequest) -> QueryRequest:
        """Apply row-level filters from scope"""
        
        if not scope.row_filters:
            return query
        
        # Merge scope filters with query filters
        merged_filters = {**scope.row_filters}
        if query.filters:
            # Scope filters take precedence (cannot be overridden)
            merged_filters.update({
                k: v for k, v in query.filters.items()
                if k not in scope.row_filters
            })
        
        query.filters = merged_filters
        return query
    
    def log_query_access(self,
                         scope: PartnerScope,
                         query: QueryRequest,
                         scan_bytes: int):
        """Log query for usage metering and billing"""
        
        # Update metrics
        queries_total.labels(
            partner=scope.partner_id,
            catalog=query.catalog
        ).inc()
        
        # Store in Redis for billing
        usage_key = f"usage:{scope.partner_id}:{datetime.now().strftime('%Y%m')}"
        
        usage = {
            'query_count': 1,
            'scan_bytes': scan_bytes,
            'timestamp': datetime.now().isoformat(),
            'catalog': query.catalog,
            'table': f"{query.schema}.{query.table}"
        }
        
        # Append to list (for detailed logs)
        self.redis.rpush(usage_key, str(usage))
        self.redis.expire(usage_key, 90 * 24 * 3600)  # Keep 90 days
        
        # Update aggregates
        agg_key = f"usage_agg:{scope.partner_id}:{datetime.now().strftime('%Y%m')}"
        self.redis.hincrby(agg_key, 'query_count', 1)
        self.redis.hincrby(agg_key, 'scan_bytes', scan_bytes)
        self.redis.expire(agg_key, 90 * 24 * 3600)
        
        logger.info(f"Query logged: partner={scope.partner_id}, table={query.schema}.{query.table}, scan_bytes={scan_bytes}")


# Global gateway instance
gateway = None


def get_gateway() -> FederationGateway:
    """Get gateway instance"""
    global gateway
    if gateway is None:
        import os
        gateway = FederationGateway(
            iceberg_rest_url=os.getenv('ICEBERG_REST_URL', 'http://iceberg-rest:8181'),
            jwt_secret=os.getenv('JWT_SECRET'),
            redis_host=os.getenv('REDIS_HOST', 'localhost')
        )
    return gateway


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.post("/v1/query")
async def execute_query(
    query: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    gateway: FederationGateway = Depends(get_gateway)
):
    """Execute federated query with scope validation"""
    
    # Verify token and get scope
    scope = gateway.verify_token(credentials.credentials)
    
    # Validate access
    gateway.validate_query_access(scope, query)
    
    # Apply row filters
    query = gateway.apply_row_filters(scope, query)
    
    # Execute query (delegate to Iceberg REST)
    # In production: forward to Iceberg REST catalog
    result = {
        "partner_id": scope.partner_id,
        "catalog": query.catalog,
        "table": f"{query.schema}.{query.table}",
        "filters_applied": query.filters,
        "status": "query_executed"
    }
    
    # Log for billing
    scan_bytes = 1024 * 1024  # Mock: 1MB scanned
    gateway.log_query_access(scope, query, scan_bytes)
    
    return result


@app.get("/v1/usage/{partner_id}")
async def get_usage(
    partner_id: str,
    month: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    gateway: FederationGateway = Depends(get_gateway)
):
    """Get usage metrics for partner"""
    
    # Verify token
    scope = gateway.verify_token(credentials.credentials)
    
    # Only allow partner to see their own usage (or admin)
    if scope.partner_id != partner_id and scope.partner_id != 'admin':
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get usage
    month_key = month or datetime.now().strftime('%Y%m')
    agg_key = f"usage_agg:{partner_id}:{month_key}"
    
    usage = gateway.redis.hgetall(agg_key)
    
    if not usage:
        return {"partner_id": partner_id, "month": month_key, "usage": {}}
    
    return {
        "partner_id": partner_id,
        "month": month_key,
        "usage": {
            "query_count": int(usage.get('query_count', 0)),
            "scan_bytes": int(usage.get('scan_bytes', 0)),
            "scan_gb": round(int(usage.get('scan_bytes', 0)) / (1024**3), 2)
        }
    }


@app.post("/v1/admin/create_token")
async def create_partner_token(
    scope: PartnerScope,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    gateway: FederationGateway = Depends(get_gateway)
):
    """Create JWT token for partner (admin only)"""
    
    # Verify admin token
    admin_scope = gateway.verify_token(credentials.credentials)
    if admin_scope.partner_id != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Create token
    payload = {
        'partner_id': scope.partner_id,
        'allowed_catalogs': scope.allowed_catalogs,
        'allowed_schemas': scope.allowed_schemas,
        'allowed_tables': scope.allowed_tables,
        'row_filters': scope.row_filters,
        'denied_columns': scope.denied_columns,
        'max_queries_per_hour': scope.max_queries_per_hour,
        'max_scan_bytes': scope.max_scan_bytes,
        'exp': int(scope.expires_at.timestamp()),
        'iat': int(datetime.now().timestamp())
    }
    
    token = jwt.encode(payload, gateway.jwt_secret, algorithm='HS256')
    
    return {
        "partner_id": scope.partner_id,
        "token": token,
        "expires_at": scope.expires_at.isoformat()
    }
