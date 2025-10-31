"""
Catalog metadata store for tracking materialized views and cost hints.
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json
import redis

logger = logging.getLogger(__name__)


@dataclass
class MaterializedViewMetadata:
    """Metadata for a materialized view"""
    name: str
    system: str  # 'clickhouse' or 'trino'
    source_table: str
    aggregation_type: str
    time_granularity: Optional[str]
    cost_hint: float  # Relative cost compared to raw table (0.1 = 10x cheaper)
    latency_hint: float  # Relative latency (0.2 = 5x faster)
    data_freshness_minutes: int
    query_pattern: str
    created_at: str
    updated_at: str


class CatalogMetadataStore:
    """Manages catalog metadata for query rewriting"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.mv_prefix = 'mv:'
        self.cost_hint_prefix = 'cost:'
    
    def register_materialized_view(self, mv: MaterializedViewMetadata) -> None:
        """Register a materialized view in the catalog"""
        key = f"{self.mv_prefix}{mv.name}"
        value = json.dumps(asdict(mv))
        
        self.redis_client.set(key, value)
        
        # Also index by source table for quick lookup
        source_key = f"source:{mv.source_table}"
        self.redis_client.sadd(source_key, mv.name)
        
        logger.info(f"Registered MV: {mv.name} for source {mv.source_table}")
    
    def get_materialized_view(self, mv_name: str) -> Optional[MaterializedViewMetadata]:
        """Retrieve MV metadata"""
        key = f"{self.mv_prefix}{mv_name}"
        value = self.redis_client.get(key)
        
        if not value:
            return None
        
        data = json.loads(value)
        return MaterializedViewMetadata(**data)
    
    def find_mvs_by_source(self, source_table: str) -> List[MaterializedViewMetadata]:
        """Find all MVs for a source table"""
        source_key = f"source:{source_table}"
        mv_names = self.redis_client.smembers(source_key)
        
        mvs = []
        for mv_name in mv_names:
            mv = self.get_materialized_view(mv_name)
            if mv:
                mvs.append(mv)
        
        return mvs
    
    def find_mvs_by_pattern(self, pattern_type: str) -> List[MaterializedViewMetadata]:
        """Find MVs matching a query pattern type"""
        # Scan all MVs and filter by pattern
        mvs = []
        cursor = 0
        
        while True:
            cursor, keys = self.redis_client.scan(cursor, match=f"{self.mv_prefix}*")
            
            for key in keys:
                value = self.redis_client.get(key)
                if value:
                    mv_data = json.loads(value)
                    mv = MaterializedViewMetadata(**mv_data)
                    if pattern_type in mv.query_pattern:
                        mvs.append(mv)
            
            if cursor == 0:
                break
        
        return mvs
    
    def update_cost_hint(self, mv_name: str, cost_hint: float) -> None:
        """Update cost hint for an MV"""
        mv = self.get_materialized_view(mv_name)
        if mv:
            mv.cost_hint = cost_hint
            self.register_materialized_view(mv)
    
    def tag_view_with_cost(self, view_name: str, system: str, cost_hint: float) -> None:
        """Tag a view or MV with cost hint"""
        key = f"{self.cost_hint_prefix}{system}:{view_name}"
        self.redis_client.set(key, str(cost_hint))
    
    def get_view_cost(self, view_name: str, system: str) -> Optional[float]:
        """Get cost hint for a view"""
        key = f"{self.cost_hint_prefix}{system}:{view_name}"
        value = self.redis_client.get(key)
        
        if value:
            return float(value)
        return None
    
    def list_all_mvs(self) -> List[MaterializedViewMetadata]:
        """List all registered MVs"""
        mvs = []
        cursor = 0
        
        while True:
            cursor, keys = self.redis_client.scan(cursor, match=f"{self.mv_prefix}*")
            
            for key in keys:
                value = self.redis_client.get(key)
                if value:
                    mv_data = json.loads(value)
                    mvs.append(MaterializedViewMetadata(**mv_data))
            
            if cursor == 0:
                break
        
        return mvs


def seed_example_mvs(store: CatalogMetadataStore):
    """Seed example MVs for power trading use case"""
    from datetime import datetime
    
    # LMP time aggregation MV
    store.register_materialized_view(MaterializedViewMetadata(
        name='mv_lmp_5min_agg',
        system='clickhouse',
        source_table='curated.rt_lmp_raw',
        aggregation_type='time_window',
        time_granularity='5min',
        cost_hint=0.05,  # 20x cheaper
        latency_hint=0.1,  # 10x faster
        data_freshness_minutes=5,
        query_pattern='time_window_agg',
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    ))
    
    # Hub/node rollup MV
    store.register_materialized_view(MaterializedViewMetadata(
        name='mv_lmp_hub_node_rollup',
        system='clickhouse',
        source_table='curated.rt_lmp_5m',
        aggregation_type='hub_node_rollup',
        time_granularity='hourly',
        cost_hint=0.02,  # 50x cheaper
        latency_hint=0.05,  # 20x faster
        data_freshness_minutes=60,
        query_pattern='hub_node_rollup',
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    ))
    
    # Load factor MV
    store.register_materialized_view(MaterializedViewMetadata(
        name='mv_load_hourly_agg',
        system='clickhouse',
        source_table='curated.load_raw',
        aggregation_type='time_window',
        time_granularity='hourly',
        cost_hint=0.1,  # 10x cheaper
        latency_hint=0.15,  # ~7x faster
        data_freshness_minutes=60,
        query_pattern='time_window_agg',
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    ))
    
    logger.info("Seeded 3 example MVs")
