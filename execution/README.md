# Execution

Execution optimization and market impact minimization.

## Modules

### Impact Models (`impact/`)
**Market Impact Optimizer** - Almgren-Chriss + nonlinear impact models.

- `fit.py`: Fit market impact curves from execution history
- `simulate.py`: Simulate execution strategies with impact

### Schedulers (`scheduler/`)
**Smart Order Routing** - Liquidity-aware scheduling across venues.

- POV (Percentage of Volume)
- TWAP (Time-Weighted Average Price)
- IS (Implementation Shortfall)
- Hybrid schedulers with risk constraints

## DoD

**≥30% slippage reduction** vs simple POV at matched risk in live paper-trade.

## Integration

Execution metrics flow to ClickHouse `execution_metrics` tables:
- Orders, fills, slippage
- Venue liquidity
- Algorithm performance

## Usage

Execution optimizers are called by trading systems with venue-specific parameters:

```python
from execution.scheduler import SmartRouter

router = SmartRouter(venues=['ICE', 'CME', 'EEX'])
schedule = router.optimize_execution(
    instrument='NGH24',
    quantity=1000,
    urgency='MEDIUM',
    risk_limit=0.05
)
```
