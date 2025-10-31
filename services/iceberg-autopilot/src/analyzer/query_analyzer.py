"""
Query analyzer that reads Trino/ClickHouse query logs and Iceberg metadata
to identify optimization opportunities for partition and sort order changes.
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from trino.dbapi import connect as trino_connect
from clickhouse_driver import Client as ClickHouseClient

logger = logging.getLogger(__name__)


@dataclass
class QueryPattern:
    """Represents a query pattern extracted from logs"""
    table_name: str
    filter_columns: List[str]
    order_by_columns: List[str]
    query_count: int
    avg_scan_bytes: float
    avg_duration_ms: float
    date_range: tuple


@dataclass
class OptimizationRecommendation:
    """Represents a partition or sort order optimization recommendation"""
    table_name: str
    recommendation_type: str  # 'partition' or 'sort_order'
    current_spec: Dict[str, Any]
    proposed_spec: Dict[str, Any]
    expected_scan_reduction_pct: float
    affected_query_count: int
    confidence_score: float
    reasoning: str


class QueryLogAnalyzer:
    """Analyzes query logs from Trino and ClickHouse to identify optimization opportunities"""
    
    def __init__(self, 
                 trino_host: str,
                 trino_port: int,
                 trino_catalog: str,
                 clickhouse_host: str,
                 clickhouse_port: int,
                 lookback_days: int = 7):
        self.trino_host = trino_host
        self.trino_port = trino_port
        self.trino_catalog = trino_catalog
        self.clickhouse_host = clickhouse_host
        self.clickhouse_port = clickhouse_port
        self.lookback_days = lookback_days
        
    def _connect_trino(self):
        """Create Trino connection"""
        return trino_connect(
            host=self.trino_host,
            port=self.trino_port,
            catalog=self.trino_catalog,
            schema='information_schema'
        )
    
    def _connect_clickhouse(self):
        """Create ClickHouse connection"""
        return ClickHouseClient(
            host=self.clickhouse_host,
            port=self.clickhouse_port
        )
    
    def extract_trino_query_patterns(self) -> List[QueryPattern]:
        """Extract query patterns from Trino query history"""
        logger.info("Extracting query patterns from Trino...")
        
        query = f"""
        SELECT 
            query,
            query_id,
            created,
            elapsed_time_ms,
            queued_time_ms,
            analysis_time_ms,
            planning_time_ms,
            execution_time_ms,
            peak_memory_bytes,
            physical_input_bytes,
            physical_input_rows,
            state
        FROM system.runtime.queries
        WHERE created >= CURRENT_TIMESTAMP - INTERVAL '{self.lookback_days}' DAY
            AND state = 'FINISHED'
            AND query LIKE '%FROM iceberg.%'
        ORDER BY created DESC
        """
        
        patterns = []
        try:
            conn = self._connect_trino()
            cursor = conn.cursor()
            cursor.execute(query)
            
            for row in cursor.fetchall():
                pattern = self._parse_query_pattern(row)
                if pattern:
                    patterns.append(pattern)
                    
        except Exception as e:
            logger.error(f"Error extracting Trino patterns: {e}")
            
        logger.info(f"Extracted {len(patterns)} query patterns from Trino")
        return patterns
    
    def _parse_query_pattern(self, query_row: tuple) -> Optional[QueryPattern]:
        """Parse a single query to extract pattern information"""
        query_text = query_row[0]
        
        # Simple pattern extraction (in production, use SQL parser)
        # Extract table name
        table_match = self._extract_table_name(query_text)
        if not table_match:
            return None
            
        # Extract filter columns (WHERE clause)
        filter_cols = self._extract_filter_columns(query_text)
        
        # Extract order by columns
        order_cols = self._extract_order_columns(query_text)
        
        return QueryPattern(
            table_name=table_match,
            filter_columns=filter_cols,
            order_by_columns=order_cols,
            query_count=1,
            avg_scan_bytes=query_row[9] or 0,  # physical_input_bytes
            avg_duration_ms=query_row[7] or 0,  # execution_time_ms
            date_range=(query_row[2], query_row[2])  # created timestamp
        )
    
    def _extract_table_name(self, query: str) -> Optional[str]:
        """Extract table name from query (simplified)"""
        import re
        match = re.search(r'FROM\s+iceberg\.(\w+)\.(\w+)', query, re.IGNORECASE)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        return None
    
    def _extract_filter_columns(self, query: str) -> List[str]:
        """Extract columns used in WHERE clause (simplified)"""
        import re
        # Simple extraction - in production use proper SQL parser
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP BY|ORDER BY|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return []
        
        where_clause = where_match.group(1)
        # Extract column names (simplified - matches identifiers)
        columns = re.findall(r'\b([a-z_][a-z0-9_]*)\s*[=<>]', where_clause, re.IGNORECASE)
        return list(set(columns))
    
    def _extract_order_columns(self, query: str) -> List[str]:
        """Extract columns used in ORDER BY clause (simplified)"""
        import re
        order_match = re.search(r'ORDER BY\s+(.+?)(?:LIMIT|$)', query, re.IGNORECASE)
        if not order_match:
            return []
        
        order_clause = order_match.group(1)
        # Extract column names
        columns = re.findall(r'\b([a-z_][a-z0-9_]*)\b', order_clause, re.IGNORECASE)
        return columns
    
    def aggregate_patterns(self, patterns: List[QueryPattern]) -> Dict[str, List[QueryPattern]]:
        """Aggregate query patterns by table"""
        table_patterns = {}
        
        for pattern in patterns:
            if pattern.table_name not in table_patterns:
                table_patterns[pattern.table_name] = []
            table_patterns[pattern.table_name].append(pattern)
            
        return table_patterns
    
    def generate_recommendations(self, 
                                 table_patterns: Dict[str, List[QueryPattern]],
                                 current_metadata: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on query patterns"""
        recommendations = []
        
        for table_name, patterns in table_patterns.items():
            # Analyze partition optimization opportunities
            partition_rec = self._recommend_partition_changes(table_name, patterns, current_metadata)
            if partition_rec:
                recommendations.append(partition_rec)
            
            # Analyze sort order optimization opportunities
            sort_rec = self._recommend_sort_changes(table_name, patterns, current_metadata)
            if sort_rec:
                recommendations.append(sort_rec)
                
        return recommendations
    
    def _recommend_partition_changes(self,
                                     table_name: str,
                                     patterns: List[QueryPattern],
                                     metadata: Dict[str, Any]) -> Optional[OptimizationRecommendation]:
        """Recommend partition spec changes based on query patterns"""
        # Analyze most frequently filtered columns
        filter_col_counts = {}
        total_queries = len(patterns)
        total_scan_bytes = sum(p.avg_scan_bytes for p in patterns)
        
        for pattern in patterns:
            for col in pattern.filter_columns:
                filter_col_counts[col] = filter_col_counts.get(col, 0) + 1
        
        # Find top filter columns (used in >30% of queries)
        threshold = total_queries * 0.3
        candidate_cols = [col for col, count in filter_col_counts.items() if count > threshold]
        
        if not candidate_cols:
            return None
        
        current_spec = metadata.get(table_name, {}).get('partition_spec', {})
        
        # Simple heuristic: recommend partitioning by most common filter column
        top_col = max(candidate_cols, key=lambda c: filter_col_counts[c])
        
        if top_col in str(current_spec):
            return None  # Already partitioned by this column
        
        proposed_spec = {
            'fields': [
                {'name': top_col, 'transform': 'identity', 'source_id': 1}
            ]
        }
        
        # Estimate scan reduction (simplified heuristic)
        expected_reduction = min(70.0, (filter_col_counts[top_col] / total_queries) * 80)
        
        return OptimizationRecommendation(
            table_name=table_name,
            recommendation_type='partition',
            current_spec=current_spec,
            proposed_spec=proposed_spec,
            expected_scan_reduction_pct=expected_reduction,
            affected_query_count=filter_col_counts[top_col],
            confidence_score=filter_col_counts[top_col] / total_queries,
            reasoning=f"Column '{top_col}' is filtered in {filter_col_counts[top_col]} of {total_queries} queries ({filter_col_counts[top_col]/total_queries*100:.1f}%). Partitioning by this column could reduce scan volume."
        )
    
    def _recommend_sort_changes(self,
                                table_name: str,
                                patterns: List[QueryPattern],
                                metadata: Dict[str, Any]) -> Optional[OptimizationRecommendation]:
        """Recommend sort order changes based on query patterns"""
        # Analyze most frequently ordered columns
        order_col_counts = {}
        total_queries = len(patterns)
        
        for pattern in patterns:
            for col in pattern.order_by_columns:
                order_col_counts[col] = order_col_counts.get(col, 0) + 1
        
        if not order_col_counts:
            return None
        
        # Find top order column (used in >20% of queries)
        threshold = total_queries * 0.2
        candidate_cols = [col for col, count in order_col_counts.items() if count > threshold]
        
        if not candidate_cols:
            return None
        
        current_spec = metadata.get(table_name, {}).get('sort_order', {})
        top_col = max(candidate_cols, key=lambda c: order_col_counts[c])
        
        if top_col in str(current_spec):
            return None  # Already sorted by this column
        
        proposed_spec = {
            'order_id': 1,
            'fields': [
                {'source_id': 1, 'transform': 'identity', 'direction': 'asc', 'null_order': 'nulls-last'}
            ]
        }
        
        # Estimate scan reduction
        expected_reduction = min(50.0, (order_col_counts[top_col] / total_queries) * 60)
        
        return OptimizationRecommendation(
            table_name=table_name,
            recommendation_type='sort_order',
            current_spec=current_spec,
            proposed_spec=proposed_spec,
            expected_scan_reduction_pct=expected_reduction,
            affected_query_count=order_col_counts[top_col],
            confidence_score=order_col_counts[top_col] / total_queries,
            reasoning=f"Column '{top_col}' is used in ORDER BY in {order_col_counts[top_col]} of {total_queries} queries ({order_col_counts[top_col]/total_queries*100:.1f}%). Sorting by this column could improve query performance."
        )
