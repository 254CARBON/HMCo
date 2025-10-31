"""
ClickHouse Materialized View Optimizer
Analyzes query patterns and automatically creates/manages MVs for performance
"""
import logging
import re
import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from clickhouse_driver import Client
import sqlparse
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Metrics
mv_created = Counter('ch_mv_optimizer_created_total', 'Total MVs created')
mv_dropped = Counter('ch_mv_optimizer_dropped_total', 'Total MVs dropped')
query_analysis_duration = Histogram('ch_mv_optimizer_analysis_seconds', 'Query analysis duration')
mv_count = Gauge('ch_mv_optimizer_active_mvs', 'Number of active MVs')


@dataclass
class QueryPattern:
    query_hash: str
    query_template: str
    count: int
    avg_duration_ms: float
    p95_duration_ms: float
    tables: List[str]
    last_seen: datetime


@dataclass
class MaterializedView:
    name: str
    database: str
    table: str
    definition: str
    created_at: datetime
    last_accessed: Optional[datetime]
    rows: int
    bytes: int


class ClickHouseMVOptimizer:
    """Automatic materialized view optimizer for ClickHouse"""
    
    def __init__(self, config_path: str = "config/policy.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.client = Client(host='localhost')  # Configure as needed
        logger.info("ClickHouse MV Optimizer initialized")
    
    def analyze_queries(self) -> List[QueryPattern]:
        """Analyze query_log to find optimization candidates"""
        analysis_window = self.config['optimizer']['analysis_window_hours']
        min_query_count = self.config['optimizer']['min_query_count']
        min_avg_time = self.config['optimizer']['min_avg_query_time_ms']
        
        query = f"""
        SELECT
            cityHash64(normalizeQuery(query)) AS query_hash,
            normalizeQuery(query) AS query_template,
            count() AS query_count,
            avg(query_duration_ms) AS avg_duration_ms,
            quantile(0.95)(query_duration_ms) AS p95_duration_ms,
            groupArray(distinct tables) AS tables,
            max(event_time) AS last_seen
        FROM system.query_log
        WHERE 
            event_time >= now() - interval {analysis_window} hour
            AND type = 'QueryFinish'
            AND query NOT LIKE '%system.%'
            AND query NOT LIKE '%SHOW%'
            AND query NOT LIKE '%DESCRIBE%'
        GROUP BY query_hash, query_template
        HAVING 
            query_count >= {min_query_count}
            AND avg_duration_ms >= {min_avg_time}
        ORDER BY query_count DESC
        LIMIT 50
        """
        
        with query_analysis_duration.time():
            results = self.client.execute(query)
        
        patterns = []
        for row in results:
            pattern = QueryPattern(
                query_hash=str(row[0]),
                query_template=row[1],
                count=row[2],
                avg_duration_ms=row[3],
                p95_duration_ms=row[4],
                tables=row[5],
                last_seen=row[6]
            )
            patterns.append(pattern)
        
        logger.info(f"Analyzed {len(patterns)} query patterns")
        return patterns
    
    def should_create_mv(self, pattern: QueryPattern) -> bool:
        """Determine if an MV should be created for this pattern"""
        # Check if already has an MV
        existing_mvs = self.get_existing_mvs()
        for mv in existing_mvs:
            if mv.table in pattern.tables:
                table_mv_count = sum(1 for m in existing_mvs if m.table == mv.table)
                max_mvs = self.config['guardrails']['max_mvs_per_table']
                if table_mv_count >= max_mvs:
                    logger.debug(f"Max MVs reached for table {mv.table}")
                    return False
        
        # Check storage limits
        storage_limit = self.config['guardrails']['max_storage_percent']
        if self._get_mv_storage_percent() >= storage_limit:
            logger.warning("MV storage limit reached")
            return False
        
        return True
    
    def generate_mv_definition(self, pattern: QueryPattern) -> Optional[str]:
        """Generate MV definition for a query pattern"""
        for template in self.config['patterns']:
            if re.search(template['match_pattern'], pattern.query_template, re.IGNORECASE):
                # Extract components from query
                parsed = sqlparse.parse(pattern.query_template)[0]
                
                # This is simplified - real implementation would parse SQL properly
                mv_name = f"mv_auto_{pattern.query_hash[:8]}"
                
                definition = template['template'].format(
                    mv_name=mv_name,
                    base_table=pattern.tables[0] if pattern.tables else "unknown",
                    # Add other extracted components
                )
                
                logger.info(f"Generated MV definition: {mv_name}")
                return definition
        
        return None
    
    def create_mv(self, definition: str) -> bool:
        """Create a materialized view"""
        try:
            self.client.execute(definition)
            mv_created.inc()
            logger.info(f"Created MV successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create MV: {e}")
            return False
    
    def get_existing_mvs(self) -> List[MaterializedView]:
        """Get list of existing materialized views"""
        query = """
        SELECT
            database,
            name,
            engine,
            create_table_query,
            metadata_modification_time,
            total_rows,
            total_bytes
        FROM system.tables
        WHERE engine LIKE '%MaterializedView%'
        OR name LIKE 'mv_%'
        """
        
        results = self.client.execute(query)
        
        mvs = []
        for row in results:
            mv = MaterializedView(
                name=row[1],
                database=row[0],
                table="",  # Extract from definition
                definition=row[3],
                created_at=row[4],
                last_accessed=None,  # Get from query_log
                rows=row[5],
                bytes=row[6]
            )
            mvs.append(mv)
        
        mv_count.set(len(mvs))
        return mvs
    
    def cleanup_unused_mvs(self):
        """Drop MVs that haven't been used recently"""
        retention_days = self.config['guardrails']['unused_retention_days']
        cutoff = datetime.now() - timedelta(days=retention_days)
        
        for mv in self.get_existing_mvs():
            # Check if MV was accessed in query_log
            query = f"""
            SELECT max(event_time) as last_access
            FROM system.query_log
            WHERE 
                query LIKE '%{mv.name}%'
                AND event_time >= now() - interval {retention_days} day
            """
            
            result = self.client.execute(query)
            last_access = result[0][0] if result and result[0][0] else None
            
            if last_access is None or last_access < cutoff:
                logger.info(f"Dropping unused MV: {mv.name}")
                self.drop_mv(mv)
    
    def drop_mv(self, mv: MaterializedView):
        """Drop a materialized view"""
        try:
            self.client.execute(f"DROP VIEW IF EXISTS {mv.database}.{mv.name}")
            mv_dropped.inc()
            logger.info(f"Dropped MV: {mv.name}")
        except Exception as e:
            logger.error(f"Failed to drop MV {mv.name}: {e}")
    
    def _get_mv_storage_percent(self) -> float:
        """Calculate storage overhead of all MVs"""
        query = """
        SELECT
            sum(total_bytes) / 
            (SELECT sum(total_bytes) FROM system.tables) * 100 as mv_percent
        FROM system.tables
        WHERE name LIKE 'mv_%'
        """
        
        result = self.client.execute(query)
        return result[0][0] if result else 0.0
    
    def run(self):
        """Run optimization cycle"""
        logger.info("Starting MV optimization cycle")
        
        # Analyze query patterns
        patterns = self.analyze_queries()
        
        # Create MVs for hot queries
        for pattern in patterns:
            if self.should_create_mv(pattern):
                definition = self.generate_mv_definition(pattern)
                if definition:
                    self.create_mv(definition)
        
        # Cleanup unused MVs
        self.cleanup_unused_mvs()
        
        logger.info("Optimization cycle complete")


if __name__ == "__main__":
    import schedule
    import time
    
    logging.basicConfig(level=logging.INFO)
    optimizer = ClickHouseMVOptimizer()
    
    # Schedule optimization runs
    interval = optimizer.config['optimizer']['interval_minutes']
    schedule.every(interval).minutes.do(optimizer.run)
    
    logger.info(f"MV Optimizer started, running every {interval} minutes")
    
    while True:
        schedule.run_pending()
        time.sleep(60)
