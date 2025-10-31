# Query Cost Coach + Fix-It Bot

Static and dynamic query cost analysis with automatic suggestions, warnings, and budget enforcement.

## Purpose

**Most waste is user behavior—fix it automatically.**

This service:
- Analyzes queries before execution
- Estimates cost ($/TB scanned)
- Suggests cheaper alternatives
- Enforces team/desk budgets
- Auto-refuses expensive queries

## Features

- **Cost Estimation**: Predicts scan volume and cost before execution
- **Inline Coaching**: Shows "cheaper equivalent" and "hit MV X instead"
- **Budget Enforcement**: Auto-refuse queries over budget
- **Weekly Reports**: Top offenders by team/desk
- **Portal Integration**: Real-time hints in query UI

## Architecture

```
services/cost-coach/
├── src/
│   ├── analyzer/
│   │   └── query_cost_analyzer.py   # Cost estimation and optimization
│   └── optimizer/
│       └── budget_enforcer.py       # Budget enforcement
└── requirements.txt
```

## Cost Levels

- **LOW**: <0.1 TB scanned (<$0.50)
- **MEDIUM**: 0.1-1 TB scanned ($0.50-$5)
- **HIGH**: 1-5 TB scanned ($5-$25)
- **VERY HIGH**: >5 TB scanned (>$25)

## Usage

### Estimate Query Cost

```python
from cost_coach.analyzer import QueryCostAnalyzer

analyzer = QueryCostAnalyzer(cost_per_tb_scanned=5.0)

query = """
SELECT *
FROM rt_lmp_raw
WHERE timestamp >= CURRENT_DATE - INTERVAL '30' DAY
"""

cost = analyzer.estimate_cost(query)

print(f"Estimated scan: {cost.estimated_scan_tb:.2f}TB")
print(f"Estimated cost: ${cost.estimated_cost_usd:.2f}")
print(f"Cost level: {cost.cost_level.value}")
print(f"Reasoning: {cost.reasoning}")
```

### Get Optimization Suggestions

```python
optimization = analyzer.suggest_optimization(query, cost)

if optimization:
    print(f"Type: {optimization.optimization_type}")
    print(f"Savings: {optimization.estimated_savings_pct:.0f}%")
    print(f"Explanation: {optimization.explanation}")
    print(f"\nOptimized query:\n{optimization.optimized_query}")
```

### Enforce Budget

```python
team_budget = 10.0  # $10 per query limit

within_budget, message = analyzer.check_budget(cost, team_budget, 'trading-analytics')

if not within_budget:
    print(f"⚠️ {message}")
    # Refuse query
else:
    print("✅ Within budget. Execute query.")
```

### Interactive Coaching

```python
from cost_coach.analyzer import CostCoach

coach = CostCoach(analyzer)

coaching = coach.coach_query(
    query=query,
    team='trading-analytics',
    team_budget=10.0
)

print(coaching['recommendation'])

if coaching['optimization']:
    opt = coaching['optimization']
    print(f"\n💡 Optimization available:")
    print(f"Type: {opt['type']}")
    print(f"Savings: ${opt['savings_usd']:.2f} ({opt['savings_pct']:.0f}%)")
    print(f"Explanation: {opt['explanation']}")
```

## Optimization Types

### 1. MV Substitution
Replace raw table scans with materialized views.

**Before:**
```sql
SELECT date_trunc('hour', timestamp) AS hour,
       AVG(lmp) AS avg_lmp
FROM rt_lmp_raw
WHERE timestamp >= CURRENT_DATE
GROUP BY hour
```

**After:**
```sql
SELECT hour, avg_lmp
FROM mv_lmp_5min_agg
WHERE hour >= CURRENT_DATE
```

**Savings:** ~90% cost reduction, 10x faster

### 2. Partition Filter
Add timestamp filters to enable partition pruning.

**Before:**
```sql
SELECT * FROM rt_lmp_raw WHERE iso = 'CAISO'
```

**After:**
```sql
SELECT * FROM rt_lmp_raw 
WHERE timestamp >= CURRENT_DATE - INTERVAL '7' DAY 
  AND iso = 'CAISO'
```

**Savings:** ~70% cost reduction (scan only 7 days vs full table)

### 3. Column Pruning
Replace SELECT * with specific columns.

**Before:**
```sql
SELECT * FROM rt_lmp_5m LIMIT 100
```

**After:**
```sql
SELECT timestamp, iso, node, lmp FROM rt_lmp_5m LIMIT 100
```

**Savings:** ~30% cost reduction (read fewer columns)

## Budget Enforcement

### Team Budgets

```python
# Set budget per query for team
team_budgets = {
    'trading-analytics': 10.0,    # $10/query
    'data-science': 50.0,         # $50/query
    'reporting': 5.0              # $5/query
}

# Check query
team = 'trading-analytics'
if cost.estimated_cost_usd > team_budgets[team]:
    # Refuse query
    print(f"⚠️ Query exceeds budget: ${cost.estimated_cost_usd:.2f} > ${team_budgets[team]:.2f}")
    print(f"Consider optimization or request budget increase")
    raise Exception("Budget exceeded")
```

### Desk Budgets (Aggregate)

```python
# Track cumulative spend by desk
desk_spend = {
    'power-trading': 150.0,  # $150 spent this month
}

desk_budgets = {
    'power-trading': 500.0,  # $500/month limit
}

remaining = desk_budgets['power-trading'] - desk_spend['power-trading']
print(f"Remaining budget: ${remaining:.2f}")

if cost.estimated_cost_usd > remaining:
    print("⚠️ Insufficient desk budget")
```

## Portal Integration

### Inline Hints

Show hints in query UI:

```javascript
// Query editor hint
if (optimization) {
  showHint({
    type: 'info',
    message: `💡 Cheaper equivalent available: ${optimization.type}`,
    action: 'View suggestion',
    savings: `${optimization.savings_pct}% cost reduction`
  });
}
```

### Pre-Execution Warning

```javascript
// Before executing query
if (cost.cost_level === 'VERY_HIGH') {
  showWarning({
    message: `⚠️ This query will scan ${cost.scan_tb}TB and cost ~$${cost.cost_usd}`,
    actions: ['Cancel', 'Optimize', 'Continue']
  });
}
```

## Weekly Reports

### Top Offenders

```python
# Generate weekly report
def generate_weekly_report():
    top_offenders = [
        {'user': 'analyst1', 'queries': 150, 'cost': 450.0, 'avg_cost': 3.0},
        {'user': 'analyst2', 'queries': 80, 'cost': 380.0, 'avg_cost': 4.75},
        {'user': 'analyst3', 'queries': 200, 'cost': 300.0, 'avg_cost': 1.5},
    ]
    
    # Sort by total cost
    top_offenders.sort(key=lambda x: x['cost'], reverse=True)
    
    print("Top 10 Cost Offenders (Weekly)")
    print("-" * 60)
    for i, user in enumerate(top_offenders[:10], 1):
        print(f"{i}. {user['user']}: {user['queries']} queries, ${user['cost']:.2f} total, ${user['avg_cost']:.2f} avg")
```

## DoD (Definition of Done)

✅ **30-day trend: $/TB-scanned down**  
✅ **P95 latency stable or better** (optimizations don't hurt performance)  
✅ Inline coaching in portal  
✅ Budget enforcement working  
✅ Weekly offender reports generated

## Configuration

Environment variables:
- `COST_PER_TB_SCANNED`: Cost per TB (default: $5)
- `DEFAULT_TEAM_BUDGET`: Default per-query budget (default: $10)
- `ENABLE_AUTO_REFUSE`: Auto-refuse over-budget queries (default: true)
- `ENABLE_COACHING`: Show inline hints (default: true)

## Deployment

Runs as:
1. **Query Proxy**: Intercepts queries before execution
2. **API Service**: Provides cost estimates on demand
3. **Scheduled Job**: Generates weekly reports

## Metrics

Exposed on `:9090/metrics`:
- `cost_coach_queries_analyzed`: Total queries analyzed
- `cost_coach_queries_refused`: Queries refused (over budget)
- `cost_coach_optimizations_suggested`: Optimizations suggested
- `cost_coach_cost_saved_usd`: Total cost saved by optimizations
- `cost_coach_avg_cost_per_query`: Average cost per query

## Example Output

```
========================================
Query Cost Analysis
========================================

Estimated Scan: 2.5TB
Estimated Cost: $12.50
Cost Level: HIGH

⚠️ Query exceeds team budget ($10.00)

💡 Optimization Available:
Type: mv_substitution
Savings: $11.25 (90%)

Suggested Query:
SELECT hour, avg_lmp
FROM mv_lmp_5min_agg
WHERE hour >= CURRENT_DATE

Explanation:
Use precomputed MV 'mv_lmp_5min_agg' instead of raw table. 
10x faster, 90% cost reduction.

========================================
```

## Contact

**Owner:** data-platform@254carbon.com  
**Slack:** #cost-optimization
