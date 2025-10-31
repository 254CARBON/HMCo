"""
Pattern matcher that identifies query patterns that can be optimized
by routing to ClickHouse MVs or Trino views.
"""
import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Parenthesis
from sqlparse.tokens import Keyword, DML

logger = logging.getLogger(__name__)


@dataclass
class QueryPattern:
    """Represents a detected query pattern"""
    pattern_type: str  # 'time_window_agg', 'percentile', 'rollup'
    confidence: float
    target_system: str  # 'clickhouse' or 'trino'
    rewrite_hint: Dict[str, Any]


@dataclass
class RewriteCandidate:
    """Represents a query rewrite candidate"""
    original_query: str
    rewritten_query: str
    target_system: str
    estimated_speedup: float
    reasoning: str
    cost_reduction_pct: float


class PatternMatcher:
    """Identifies query patterns suitable for optimization"""
    
    def __init__(self):
        self.patterns = {
            'time_window_agg': self._match_time_window_agg,
            'percentile': self._match_percentile,
            'hub_node_rollup': self._match_hub_node_rollup
        }
    
    def analyze_query(self, query: str) -> Optional[QueryPattern]:
        """Analyze query to identify optimization patterns"""
        parsed = sqlparse.parse(query)[0]
        
        # Try each pattern matcher
        for pattern_name, matcher_fn in self.patterns.items():
            result = matcher_fn(parsed, query)
            if result:
                return result
        
        return None
    
    def _match_time_window_agg(self, parsed, query: str) -> Optional[QueryPattern]:
        """Match time-window aggregation queries (ideal for ClickHouse)"""
        query_lower = query.lower()
        
        # Look for time-based aggregations
        has_time_filter = any([
            'date_trunc' in query_lower,
            'time_bucket' in query_lower,
            'interval' in query_lower and ('hour' in query_lower or 'day' in query_lower)
        ])
        
        has_aggregation = any([
            'avg(' in query_lower,
            'sum(' in query_lower,
            'max(' in query_lower,
            'min(' in query_lower,
            'count(' in query_lower
        ])
        
        has_group_by = 'group by' in query_lower
        
        if has_time_filter and has_aggregation and has_group_by:
            # This pattern is perfect for ClickHouse materialized views
            confidence = 0.9
            if 'window' in query_lower or 'over(' in query_lower:
                confidence = 0.95
            
            return QueryPattern(
                pattern_type='time_window_agg',
                confidence=confidence,
                target_system='clickhouse',
                rewrite_hint={
                    'mv_name': self._suggest_mv_name(query),
                    'aggregation_type': self._extract_agg_functions(query)
                }
            )
        
        return None
    
    def _match_percentile(self, parsed, query: str) -> Optional[QueryPattern]:
        """Match percentile/quantile queries (ClickHouse is much faster)"""
        query_lower = query.lower()
        
        # Look for percentile functions
        percentile_funcs = [
            'percentile_cont',
            'percentile_disc',
            'approx_percentile',
            'quantile',
            'median'
        ]
        
        has_percentile = any(func in query_lower for func in percentile_funcs)
        
        if has_percentile:
            return QueryPattern(
                pattern_type='percentile',
                confidence=0.95,
                target_system='clickhouse',
                rewrite_hint={
                    'use_approximate': True,
                    'ch_function': 'quantile'
                }
            )
        
        return None
    
    def _match_hub_node_rollup(self, parsed, query: str) -> Optional[QueryPattern]:
        """Match hub/node rollup queries (common in power trading)"""
        query_lower = query.lower()
        
        # Look for hub/node aggregations
        has_hub_node = any([
            'hub' in query_lower and 'node' in query_lower,
            'pnode' in query_lower,
            'zone' in query_lower and 'iso' in query_lower
        ])
        
        has_lmp = 'lmp' in query_lower
        has_aggregation = any([
            'avg(' in query_lower,
            'sum(' in query_lower,
            'weighted' in query_lower
        ])
        
        if has_hub_node and (has_lmp or has_aggregation):
            # Check if there's an existing MV for this pattern
            return QueryPattern(
                pattern_type='hub_node_rollup',
                confidence=0.85,
                target_system='clickhouse',
                rewrite_hint={
                    'mv_name': 'lmp_hub_node_rollup',
                    'precomputed': True
                }
            )
        
        return None
    
    def _suggest_mv_name(self, query: str) -> str:
        """Suggest materialized view name based on query pattern"""
        # Simple heuristic - in production, query catalog
        if 'lmp' in query.lower():
            return 'mv_lmp_time_agg'
        elif 'load' in query.lower():
            return 'mv_load_time_agg'
        elif 'weather' in query.lower():
            return 'mv_weather_time_agg'
        else:
            return 'mv_generic_time_agg'
    
    def _extract_agg_functions(self, query: str) -> List[str]:
        """Extract aggregation functions from query"""
        agg_funcs = []
        patterns = [
            r'\b(avg|sum|max|min|count|stddev|variance)\s*\(',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query.lower())
            agg_funcs.extend(matches)
        
        return list(set(agg_funcs))


class QueryRewriter:
    """Rewrites queries to use optimized execution paths"""
    
    def __init__(self, catalog_metadata: Dict[str, Any]):
        self.catalog_metadata = catalog_metadata
        self.pattern_matcher = PatternMatcher()
    
    def rewrite_query(self, query: str) -> Optional[RewriteCandidate]:
        """Attempt to rewrite query for better performance"""
        # Analyze query pattern
        pattern = self.pattern_matcher.analyze_query(query)
        
        if not pattern:
            return None
        
        # Look up materialized view in catalog
        mv_info = self._find_matching_mv(pattern)
        
        if not mv_info:
            logger.info(f"No matching MV found for pattern: {pattern.pattern_type}")
            return None
        
        # Generate rewritten query
        rewritten = self._generate_rewrite(query, pattern, mv_info)
        
        if not rewritten:
            return None
        
        # Estimate performance improvement
        estimated_speedup = self._estimate_speedup(pattern, mv_info)
        cost_reduction = self._estimate_cost_reduction(pattern, mv_info)
        
        return RewriteCandidate(
            original_query=query,
            rewritten_query=rewritten,
            target_system=pattern.target_system,
            estimated_speedup=estimated_speedup,
            reasoning=self._generate_reasoning(pattern, mv_info),
            cost_reduction_pct=cost_reduction
        )
    
    def _find_matching_mv(self, pattern: QueryPattern) -> Optional[Dict[str, Any]]:
        """Find matching materialized view in catalog"""
        # Look up in catalog metadata (simplified)
        mv_name = pattern.rewrite_hint.get('mv_name')
        
        if not mv_name:
            return None
        
        # In production, query actual catalog
        # Return mock MV info
        return {
            'name': mv_name,
            'system': pattern.target_system,
            'cost_hint': 0.1,  # 10x cheaper
            'latency_hint': 0.2  # 5x faster
        }
    
    def _generate_rewrite(self, query: str, pattern: QueryPattern, mv_info: Dict[str, Any]) -> Optional[str]:
        """Generate rewritten query using MV"""
        if pattern.pattern_type == 'time_window_agg':
            # Simple table substitution
            # In production, use proper SQL AST rewriting
            original_table = self._extract_table_name(query)
            if original_table:
                mv_name = mv_info['name']
                rewritten = query.replace(original_table, mv_name)
                return rewritten
        
        elif pattern.pattern_type == 'percentile':
            # Rewrite percentile functions for ClickHouse
            rewritten = query.lower()
            rewritten = rewritten.replace('percentile_cont', 'quantile')
            rewritten = rewritten.replace('percentile_disc', 'quantileDeterministic')
            rewritten = rewritten.replace('approx_percentile', 'quantileTDigest')
            return rewritten
        
        elif pattern.pattern_type == 'hub_node_rollup':
            # Use precomputed rollup MV
            mv_name = mv_info['name']
            # Simplified - in production, use full AST rewrite
            return f"SELECT * FROM {mv_name} WHERE ..."
        
        return None
    
    def _extract_table_name(self, query: str) -> Optional[str]:
        """Extract table name from query (simplified)"""
        match = re.search(r'FROM\s+([^\s,;]+)', query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _estimate_speedup(self, pattern: QueryPattern, mv_info: Dict[str, Any]) -> float:
        """Estimate query speedup from rewrite"""
        base_speedup = {
            'time_window_agg': 5.0,  # 5x faster
            'percentile': 10.0,      # 10x faster with CH quantile functions
            'hub_node_rollup': 8.0   # 8x faster with precomputed rollups
        }
        
        return base_speedup.get(pattern.pattern_type, 2.0) * (1 / mv_info.get('latency_hint', 0.5))
    
    def _estimate_cost_reduction(self, pattern: QueryPattern, mv_info: Dict[str, Any]) -> float:
        """Estimate cost reduction from rewrite"""
        # MVs reduce data scanned, thus reducing cost
        base_reduction = {
            'time_window_agg': 70.0,  # 70% less data scanned
            'percentile': 60.0,
            'hub_node_rollup': 80.0
        }
        
        return base_reduction.get(pattern.pattern_type, 50.0)
    
    def _generate_reasoning(self, pattern: QueryPattern, mv_info: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for rewrite"""
        reasoning_map = {
            'time_window_agg': f"Time-window aggregation query can use precomputed MV '{mv_info['name']}' in ClickHouse for faster execution",
            'percentile': f"Percentile calculation is much faster using ClickHouse's native quantile functions",
            'hub_node_rollup': f"Hub/node rollup query can use precomputed MV '{mv_info['name']}' to avoid scanning raw data"
        }
        
        return reasoning_map.get(pattern.pattern_type, "Query can be optimized")
