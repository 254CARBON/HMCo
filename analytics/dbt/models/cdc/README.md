# CDC Consolidation → Star Schemas

Transforms CDC firehose into queryable star schemas with SCD2 dimensions and fact history.

## Purpose

**CDC firehose ≠ queryable history.**

This transforms raw CDC streams into:
- **SCD Type 2 dimensions** for reference data (nodes, hubs, constraints, assets)
- **Fact tables with history** for transactional data (trades, positions)
- **AS-OF queries** for point-in-time analysis
- **Drift tests** to ensure consistency across day boundaries

## Architecture

```
analytics/dbt/models/cdc/
├── dimensions/
│   ├── dim_nodes_scd2.sql          # Power grid nodes with full history
│   ├── dim_hubs_scd2.sql           # Trading hubs with full history
│   ├── dim_constraints_scd2.sql    # Transmission constraints
│   └── dim_assets_scd2.sql         # Generation assets
└── facts/
    ├── fact_trade_history.sql      # Trade lifecycle events
    └── fact_position_history.sql   # Position changes over time
```

## Dimensions (SCD Type 2)

### Node Dimension
Tracks changes to power grid nodes over time.

**Schema:**
```sql
node_sk              -- Surrogate key (node_id + valid_from)
node_id              -- Natural key
node_name            -- Node name (may change)
iso                  -- ISO/RTO
zone                 -- Price zone
hub                  -- Associated hub
pnode_type           -- Node type
latitude, longitude  -- Location (may change)
capacity_mw          -- Capacity (may change)
is_active            -- Active status
valid_from           -- SCD2 start date
valid_to             -- SCD2 end date (9999-12-31 if current)
is_current           -- Current record flag
version_number       -- Version counter
```

**Usage:**
```sql
-- Current nodes
SELECT * FROM dim_nodes_scd2 WHERE is_current = TRUE

-- AS-OF query (nodes as they were on 2025-01-01)
SELECT * 
FROM dim_nodes_scd2
WHERE '2025-01-01' BETWEEN valid_from AND valid_to

-- Node history
SELECT 
    node_id,
    node_name,
    valid_from,
    valid_to,
    version_number
FROM dim_nodes_scd2
WHERE node_id = 'DLAP_LOAD'
ORDER BY version_number
```

### Hub Dimension
Tracks changes to trading hubs over time.

**Schema:**
```sql
hub_sk                -- Surrogate key
hub_id                -- Natural key
hub_name              -- Hub name
iso                   -- ISO/RTO
hub_type              -- Load zone, Interface, Hub
settlement_location   -- Settlement location
is_active             -- Active status
valid_from, valid_to  -- SCD2 validity period
is_current            -- Current record flag
```

## Facts (History)

### Trade History Fact
Captures complete trade lifecycle.

**Schema:**
```sql
trade_history_sk      -- Surrogate key
trade_id              -- Trade identifier
trade_timestamp       -- Original trade timestamp
trade_status          -- SUBMITTED, EXECUTED, SETTLED, CANCELLED
previous_status       -- Previous status (for state transitions)
trader_id             -- Trader
counterparty_id       -- Counterparty
product_type          -- Product type
delivery_start        -- Delivery period start
delivery_end          -- Delivery period end
quantity_mwh          -- Quantity in MWh
price_per_mwh         -- Price
previous_price        -- Previous price (for amendments)
price_change          -- Calculated price change
total_value           -- Total value
settlement_date       -- Settlement date
cdc_timestamp         -- CDC event timestamp
event_sequence        -- Event order for this trade
```

**Usage:**
```sql
-- Trade lifecycle
SELECT 
    trade_id,
    trade_status,
    previous_status,
    price_per_mwh,
    cdc_timestamp,
    event_sequence
FROM fact_trade_history
WHERE trade_id = 'TRD123'
ORDER BY event_sequence

-- Price amendments
SELECT 
    trade_id,
    price_per_mwh,
    previous_price,
    price_change,
    cdc_timestamp
FROM fact_trade_history
WHERE price_change != 0
```

### Position History Fact
Tracks position changes for risk management.

**Schema:**
```sql
position_history_sk     -- Surrogate key
position_id             -- Position identifier
position_date           -- Position date
book_id                 -- Trading book
trader_id               -- Trader
net_position_mwh        -- Net position
previous_position_mwh   -- Previous position
position_delta_mwh      -- Calculated delta
marked_to_market_value  -- MTM value
previous_mtm_value      -- Previous MTM
pnl_change              -- Calculated P&L change
var_95                  -- Value at Risk (95%)
position_status         -- OPEN, CLOSED, HEDGED
cdc_timestamp           -- CDC event timestamp
```

## ClickHouse Integration

### Merge-on-Read MVs

ClickHouse materialized views for efficient CDC processing:

```sql
-- MV for node dimension updates
CREATE MATERIALIZED VIEW mv_dim_nodes_scd2
ENGINE = ReplacingMergeTree(cdc_timestamp)
PARTITION BY toYYYYMM(valid_from)
ORDER BY (node_id, valid_from)
AS
SELECT
    node_id,
    node_name,
    iso,
    zone,
    hub,
    valid_from,
    valid_to,
    is_current,
    cdc_timestamp
FROM cdc.node_master_cdc;

-- MV for trade history
CREATE MATERIALIZED VIEW mv_fact_trade_history
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(trade_timestamp)
ORDER BY (trade_id, cdc_timestamp)
AS
SELECT
    trade_id,
    trade_timestamp,
    trade_status,
    trader_id,
    quantity_mwh,
    price_per_mwh,
    cdc_timestamp
FROM cdc.trades_cdc;
```

### Queryable History Tables

```sql
-- Final queryable table (merge-on-read)
CREATE TABLE dim_nodes_current
ENGINE = ReplacingMergeTree(cdc_timestamp)
ORDER BY node_id
AS 
SELECT * FROM mv_dim_nodes_scd2 WHERE is_current = TRUE;

-- Historical queries
SELECT * 
FROM dim_nodes_current
FINAL  -- ClickHouse merges on read
WHERE node_id = 'DLAP_LOAD';
```

## AS-OF Queries

Point-in-time analysis for regulatory reporting and audits:

```sql
-- Portfolio value as of 2025-01-01 EOD
WITH positions_asof AS (
    SELECT 
        position_id,
        net_position_mwh,
        marked_to_market_value
    FROM fact_position_history
    WHERE cdc_timestamp <= '2025-01-01 23:59:59'
    AND position_id IN (
        SELECT DISTINCT position_id 
        FROM fact_position_history 
        WHERE cdc_timestamp <= '2025-01-01 23:59:59'
    )
    QUALIFY ROW_NUMBER() OVER (PARTITION BY position_id ORDER BY cdc_timestamp DESC) = 1
)
SELECT 
    SUM(net_position_mwh) AS total_position,
    SUM(marked_to_market_value) AS total_mtm
FROM positions_asof;
```

## Drift Tests

Ensure consistency across day boundaries:

```sql
-- Test: Position at EOD should equal BOD next day
WITH eod_positions AS (
    SELECT 
        position_id,
        net_position_mwh AS eod_position
    FROM fact_position_history
    WHERE DATE(cdc_timestamp) = '2025-01-01'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY position_id ORDER BY cdc_timestamp DESC) = 1
),
bod_positions AS (
    SELECT 
        position_id,
        net_position_mwh AS bod_position
    FROM fact_position_history
    WHERE DATE(cdc_timestamp) = '2025-01-02'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY position_id ORDER BY cdc_timestamp ASC) = 1
)
SELECT 
    e.position_id,
    e.eod_position,
    b.bod_position,
    e.eod_position - b.bod_position AS drift
FROM eod_positions e
INNER JOIN bod_positions b ON e.position_id = b.position_id
WHERE ABS(e.eod_position - b.bod_position) > 0.01;  -- Should be empty
```

## DoD (Definition of Done)

✅ **AS-OF queries return consistent state** across time  
✅ **Drift tests pass** across day boundaries  
✅ **SCD2 dimensions** maintain full history  
✅ **Fact tables** capture all state transitions  
✅ **ClickHouse MVs** provide merge-on-read access

## Deployment

```bash
# Run dbt models
cd analytics/dbt
dbt run --models cdc

# Run drift tests
dbt test --models cdc

# Check lineage in OpenMetadata
# CDC sources → dbt models → BI dashboards
```

## Contact

**Owner:** data-platform@254carbon.com  
**Slack:** #cdc-consolidation
