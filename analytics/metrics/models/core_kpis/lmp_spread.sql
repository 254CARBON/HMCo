{{ config(
    materialized='table',
    tags=['kpi', 'trading', 'governed']
) }}

-- Canonical LMP Spread Metric
-- Governed KPI: Difference between locational marginal price and hub reference price
-- Owner: trading-analytics
-- SLA: 60 minutes

WITH lmp_data AS (
    SELECT
        timestamp AS metric_timestamp,
        iso,
        node,
        hub,
        lmp,
        congestion,
        loss,
        energy
    FROM {{ source('curated', 'rt_lmp_5m') }}
    WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '7' DAY
),

hub_reference AS (
    SELECT
        timestamp AS metric_timestamp,
        iso,
        hub,
        AVG(lmp) AS hub_lmp
    FROM {{ source('curated', 'rt_lmp_5m') }}
    WHERE is_hub = true
        AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '7' DAY
    GROUP BY timestamp, iso, hub
)

SELECT
    l.metric_timestamp,
    l.iso,
    l.node,
    l.hub,
    l.lmp AS node_lmp,
    h.hub_lmp,
    l.lmp - h.hub_lmp AS lmp_spread,
    l.congestion,
    l.loss,
    CURRENT_TIMESTAMP AS calculated_at
FROM lmp_data l
INNER JOIN hub_reference h
    ON l.metric_timestamp = h.metric_timestamp
    AND l.iso = h.iso
    AND l.hub = h.hub
