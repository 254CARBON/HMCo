"""
Query cost analyzer - Static and dynamic analysis of query costs.
Suggests rewrites, warns, or auto-refuses queries above budgets.
"""
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sqlparse

logger = logging.getLogger(__name__)


class CostLevel(Enum):
    """Cost classification levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class CostEstimate:
    """Query cost estimate"""
    estimated_scan_bytes: int
    estimated_scan_tb: float
    estimated_cost_usd: float
    cost_level: CostLevel
    reasoning: str


@dataclass
class QueryOptimization:
    """Query optimization suggestion"""
    original_query: str
    optimized_query: str
    estimated_savings_pct: float
    estimated_savings_usd: float
    optimization_type: str
    explanation: str


class QueryCostAnalyzer:
    """Analyzes query costs and suggests optimizations"""
    
    def __init__(self,
                 cost_per_tb_scanned: float = 5.0,  # $5 per TB
                 mv_metadata: Optional[Dict[str, Any]] = None):
        self.cost_per_tb = cost_per_tb_scanned
        self.mv_metadata = mv_metadata or {}
    
    def estimate_cost(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> CostEstimate:
        """Estimate query cost based on tables scanned"""
        
        # Extract tables from query
        tables = self._extract_tables(query)
        
        # Estimate scan bytes
        total_scan_bytes = 0
        for table in tables:
            table_size = self._get_table_size(table, metadata)
            
            # Check for partition filters
            has_partition_filter = self._has_partition_filter(query, table)
            if has_partition_filter:
                # Assume 10% scan with good partitioning
                table_size = int(table_size * 0.1)
            
            total_scan_bytes += table_size
        
        scan_tb = total_scan_bytes / (1024 ** 4)
        estimated_cost = scan_tb * self.cost_per_tb
        
        # Classify cost level
        if scan_tb < 0.1:
            cost_level = CostLevel.LOW
        elif scan_tb < 1.0:
            cost_level = CostLevel.MEDIUM
        elif scan_tb < 5.0:
            cost_level = CostLevel.HIGH
        else:
            cost_level = CostLevel.VERY_HIGH
        
        reasoning = f"Scans {len(tables)} table(s), estimated {scan_tb:.2f}TB"
        if not any(self._has_partition_filter(query, t) for t in tables):
            reasoning += ". No partition filters detected - full table scan likely."
        
        return CostEstimate(
            estimated_scan_bytes=total_scan_bytes,
            estimated_scan_tb=scan_tb,
            estimated_cost_usd=estimated_cost,
            cost_level=cost_level,
            reasoning=reasoning
        )
    
    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query"""
        tables = []
        
        # Simple regex extraction (in production use SQL parser)
        from_pattern = r'FROM\s+([^\s,;)]+)'
        join_pattern = r'JOIN\s+([^\s,;)]+)'
        
        tables.extend(re.findall(from_pattern, query, re.IGNORECASE))
        tables.extend(re.findall(join_pattern, query, re.IGNORECASE))
        
        return list(set(tables))
    
    def _get_table_size(self, table: str, metadata: Optional[Dict[str, Any]]) -> int:
        """Get table size in bytes"""
        if metadata and table in metadata:
            return metadata[table].get('size_bytes', 1024 ** 4)  # Default 1TB
        
        # Mock table sizes (in production, query from catalog)
        default_sizes = {
            'rt_lmp_raw': 10 * (1024 ** 4),  # 10TB
            'rt_lmp_5m': 1 * (1024 ** 4),    # 1TB (aggregated)
            'load_raw': 5 * (1024 ** 4),      # 5TB
            'weather_hourly': 0.5 * (1024 ** 4),  # 500GB
        }
        
        for key in default_sizes:
            if key in table.lower():
                return default_sizes[key]
        
        return 1024 ** 4  # Default 1TB
    
    def _has_partition_filter(self, query: str, table: str) -> bool:
        """Check if query has partition filter"""
        query_lower = query.lower()
        
        # Look for common partition columns
        partition_cols = ['timestamp', 'date', 'day', 'month', 'year', 'dt']
        
        for col in partition_cols:
            if col in query_lower and ('where' in query_lower or 'and' in query_lower):
                return True
        
        return False
    
    def suggest_optimization(self, query: str, cost: CostEstimate) -> Optional[QueryOptimization]:
        """Suggest query optimization"""
        
        if cost.cost_level == CostLevel.LOW:
            return None  # No optimization needed
        
        # Check for MV opportunities
        mv_optimization = self._suggest_mv_usage(query, cost)
        if mv_optimization:
            return mv_optimization
        
        # Check for partition filter opportunities
        partition_optimization = self._suggest_partition_filter(query, cost)
        if partition_optimization:
            return partition_optimization
        
        # Check for column pruning
        column_optimization = self._suggest_column_pruning(query, cost)
        if column_optimization:
            return column_optimization
        
        return None
    
    def _suggest_mv_usage(self, query: str, cost: CostEstimate) -> Optional[QueryOptimization]:
        """Suggest using materialized view"""
        
        # Check if query matches an MV pattern
        query_lower = query.lower()
        
        # Time-window aggregation
        if 'date_trunc' in query_lower and 'group by' in query_lower:
            if 'rt_lmp' in query_lower:
                return QueryOptimization(
                    original_query=query,
                    optimized_query=query.replace('rt_lmp_raw', 'mv_lmp_5min_agg'),
                    estimated_savings_pct=90.0,
                    estimated_savings_usd=cost.estimated_cost_usd * 0.9,
                    optimization_type='mv_substitution',
                    explanation="Use precomputed MV 'mv_lmp_5min_agg' instead of raw table. 10x faster, 90% cost reduction."
                )
        
        return None
    
    def _suggest_partition_filter(self, query: str, cost: CostEstimate) -> Optional[QueryOptimization]:
        """Suggest adding partition filter"""
        
        tables = self._extract_tables(query)
        
        for table in tables:
            if not self._has_partition_filter(query, table):
                # Suggest adding timestamp filter
                optimized = query
                if 'WHERE' in query.upper():
                    # Add to existing WHERE
                    optimized = query.replace('WHERE', 'WHERE timestamp >= CURRENT_DATE - INTERVAL \'7\' DAY AND')
                else:
                    # Add WHERE clause
                    from_idx = query.upper().find(f'FROM {table.upper()}')
                    if from_idx != -1:
                        insert_pos = from_idx + len(f'FROM {table}')
                        optimized = query[:insert_pos] + ' WHERE timestamp >= CURRENT_DATE - INTERVAL \'7\' DAY' + query[insert_pos:]
                
                return QueryOptimization(
                    original_query=query,
                    optimized_query=optimized,
                    estimated_savings_pct=70.0,
                    estimated_savings_usd=cost.estimated_cost_usd * 0.7,
                    optimization_type='partition_filter',
                    explanation=f"Add partition filter on timestamp to scan only recent data. Reduces full table scan on {table}."
                )
        
        return None
    
    def _suggest_column_pruning(self, query: str, cost: CostEstimate) -> Optional[QueryOptimization]:
        """Suggest selecting specific columns instead of SELECT *"""
        
        if 'SELECT *' in query.upper():
            return QueryOptimization(
                original_query=query,
                optimized_query=query.replace('SELECT *', 'SELECT /* specify columns */'),
                estimated_savings_pct=30.0,
                estimated_savings_usd=cost.estimated_cost_usd * 0.3,
                optimization_type='column_pruning',
                explanation="SELECT * reads all columns. Specify only needed columns to reduce scan size."
            )
        
        return None
    
    def check_budget(self, 
                     cost: CostEstimate,
                     team_budget: float,
                     team: str) -> Tuple[bool, str]:
        """Check if query is within team budget"""
        
        if cost.estimated_cost_usd <= team_budget:
            return True, "Within budget"
        
        over_budget_pct = ((cost.estimated_cost_usd / team_budget) - 1) * 100
        
        message = f"Query exceeds team budget. Estimated: ${cost.estimated_cost_usd:.2f}, Budget: ${team_budget:.2f} ({over_budget_pct:.0f}% over)"
        
        return False, message


class CostCoach:
    """Interactive cost coaching for users"""
    
    def __init__(self, analyzer: QueryCostAnalyzer):
        self.analyzer = analyzer
    
    def coach_query(self, 
                    query: str,
                    team: str,
                    team_budget: float,
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Provide coaching on query cost and optimizations"""
        
        # Estimate cost
        cost = self.analyzer.estimate_cost(query, metadata)
        
        # Check budget
        within_budget, budget_message = self.analyzer.check_budget(cost, team_budget, team)
        
        # Suggest optimization
        optimization = self.analyzer.suggest_optimization(query, cost)
        
        coaching = {
            'query': query,
            'team': team,
            'cost_estimate': {
                'scan_tb': cost.estimated_scan_tb,
                'cost_usd': cost.estimated_cost_usd,
                'cost_level': cost.cost_level.value,
                'reasoning': cost.reasoning
            },
            'budget': {
                'team_budget_usd': team_budget,
                'within_budget': within_budget,
                'message': budget_message
            },
            'optimization': None
        }
        
        if optimization:
            coaching['optimization'] = {
                'type': optimization.optimization_type,
                'optimized_query': optimization.optimized_query,
                'savings_pct': optimization.estimated_savings_pct,
                'savings_usd': optimization.estimated_savings_usd,
                'explanation': optimization.explanation
            }
        
        # Recommendation
        if not within_budget and optimization:
            coaching['recommendation'] = f"⚠️ Query refused (over budget). Use suggested optimization to reduce cost by {optimization.estimated_savings_pct:.0f}%."
        elif not within_budget:
            coaching['recommendation'] = f"⚠️ Query refused (over budget). Contact data team for assistance."
        elif optimization:
            coaching['recommendation'] = f"💡 Hint: You can reduce cost by {optimization.estimated_savings_pct:.0f}% with suggested optimization."
        else:
            coaching['recommendation'] = "✅ Query looks good!"
        
        return coaching
