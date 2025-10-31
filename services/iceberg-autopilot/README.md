# Iceberg Autopilot

Automated partition and sort order optimization for Apache Iceberg tables based on actual query patterns.

## Purpose

**Data shape changes—hard-coded specs rot.** This service continuously analyzes query logs and proposes optimizations to reduce scan volume and improve query performance.

## Features

- **Query Pattern Analysis**: Analyzes Trino and ClickHouse query logs to identify optimization opportunities
- **Intelligent Recommendations**: Proposes partition and sort order changes based on actual usage
- **Safe Execution**: Creates optimization plans with canary testing on lakeFS branches
- **Automated PR Creation**: Generates bot PRs with spec deltas and expected scan reduction
- **Guarded Rollout**: Validates improvements before merging to production

## Architecture

```
services/iceberg-autopilot/
├── src/
│   ├── analyzer/
│   │   └── query_analyzer.py      # Analyzes query logs and Iceberg metadata
│   └── executor/
│       └── rewrite_planner.py     # Plans and executes RewriteDataFiles
└── requirements.txt
```

## How It Works

1. **Analyze**: Reads Trino/ClickHouse query logs (last 7 days)
2. **Extract Patterns**: Identifies frequently filtered and ordered columns
3. **Recommend**: Proposes partition or sort order changes
4. **Plan**: Creates RewriteDataFiles plan on lakeFS branch
5. **Test**: Runs canary queries to validate scan reduction
6. **PR**: Creates bot PR with expected improvements
7. **Merge**: After approval, merges optimization to main

## Usage

### Run Analysis

```python
from iceberg_autopilot.analyzer import QueryLogAnalyzer

analyzer = QueryLogAnalyzer(
    trino_host='trino.data-platform.svc.cluster.local',
    trino_port=8080,
    trino_catalog='iceberg',
    clickhouse_host='clickhouse.data-platform.svc.cluster.local',
    clickhouse_port=9000,
    lookback_days=7
)

# Extract query patterns
patterns = analyzer.extract_trino_query_patterns()

# Aggregate by table
table_patterns = analyzer.aggregate_patterns(patterns)

# Generate recommendations
recommendations = analyzer.generate_recommendations(table_patterns, current_metadata={})

for rec in recommendations:
    print(f"Table: {rec.table_name}")
    print(f"Type: {rec.recommendation_type}")
    print(f"Expected scan reduction: {rec.expected_scan_reduction_pct:.1f}%")
    print(f"Reasoning: {rec.reasoning}")
```

### Execute Optimization

```python
from iceberg_autopilot.executor import RewritePlanner

planner = RewritePlanner(
    lakefs_endpoint='http://lakefs.data-platform.svc.cluster.local:8000',
    lakefs_access_key='<access_key>',
    lakefs_secret_key='<secret_key>'
)

# Create optimization branch
branch_name = planner.create_optimization_branch(
    table_name='curated.rt_lmp_5m',
    base_branch='main'
)

# Create rewrite plan
plan = planner.plan_rewrite(
    table_name='curated.rt_lmp_5m',
    recommendation=recommendations[0],
    branch_name=branch_name
)

# Execute rewrite
result = planner.execute_rewrite(plan)

# Run canary tests
test_queries = [
    "SELECT * FROM curated.rt_lmp_5m WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '1' DAY",
    "SELECT iso, AVG(lmp) FROM curated.rt_lmp_5m WHERE timestamp >= CURRENT_DATE GROUP BY iso"
]
canary_results = planner.run_canary_test(plan, test_queries)

# Merge if successful
if canary_results['tests_passed'] > 0 and canary_results['scan_reduction'] >= 30.0:
    planner.merge_to_main(
        branch_name=branch_name,
        commit_message=f"Optimize {plan.table_name}: {plan.action_type} - {canary_results['scan_reduction']:.1f}% scan reduction"
    )
else:
    planner.rollback_branch(branch_name)
```

## Optimization Criteria

### Partition Changes
- Column must be filtered in **>30% of queries**
- Not already in current partition spec
- Expected scan reduction **≥30%**

### Sort Order Changes
- Column must be in ORDER BY in **>20% of queries**
- Not already in current sort order
- Expected scan reduction **≥30%**

## DoD (Definition of Done)

✅ After merge, scan bytes for hot queries drop **≥30%** with identical results  
✅ Canary tests validate correctness and performance  
✅ Bot PR includes spec deltas and expected improvements  
✅ Rollback on failure (no production impact)

## Configuration

Environment variables:
- `TRINO_HOST`: Trino coordinator host
- `TRINO_PORT`: Trino coordinator port (default: 8080)
- `TRINO_CATALOG`: Iceberg catalog name (default: iceberg)
- `CLICKHOUSE_HOST`: ClickHouse host
- `CLICKHOUSE_PORT`: ClickHouse port (default: 9000)
- `LAKEFS_ENDPOINT`: lakeFS API endpoint
- `LAKEFS_ACCESS_KEY`: lakeFS access key
- `LAKEFS_SECRET_KEY`: lakeFS secret key
- `LOOKBACK_DAYS`: Days of query logs to analyze (default: 7)

## Deployment

Runs as a CronJob in Kubernetes:
- **Schedule**: Daily at 2 AM UTC
- **Concurrency**: Forbid (only one job at a time)
- **Timeout**: 2 hours

## Monitoring

Metrics exposed on `:9090/metrics`:
- `autopilot_recommendations_total`: Total recommendations generated
- `autopilot_optimizations_executed`: Total optimizations executed
- `autopilot_scan_reduction_pct`: Average scan reduction achieved
- `autopilot_canary_failures`: Canary test failures

## Safety Guarantees

1. **Branch Isolation**: All changes tested on lakeFS branches
2. **Canary Testing**: Validates performance and correctness before merge
3. **Rollback**: Automatic rollback on canary failure
4. **No Split-Brain**: Single writer enforced via Iceberg catalog
5. **Partial Progress**: Rewrite operations support partial commits

## Contact

**Owner:** data-platform@254carbon.com  
**Slack:** #data-platform-autopilot
