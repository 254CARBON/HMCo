{{ config(
    materialized='table',
    tags=['kpi', 'trading', 'governed']
) }}

-- Canonical Nodal Congestion Factor Metric
-- Governed KPI: Congestion component as factor of total LMP
-- Owner: trading-analytics
-- SLA: 60 minutes

SELECT
    timestamp AS metric_timestamp,
    iso,
    node,
    lmp,
    congestion,
    CASE 
        WHEN lmp != 0 THEN congestion / NULLIF(lmp, 0)
        ELSE 0
    END AS congestion_factor,
    CASE
        WHEN ABS(congestion / NULLIF(lmp, 0)) > 0.3 THEN 'high'
        WHEN ABS(congestion / NULLIF(lmp, 0)) > 0.1 THEN 'medium'
        ELSE 'low'
    END AS congestion_severity,
    CURRENT_TIMESTAMP AS calculated_at
FROM {{ source('curated', 'rt_lmp_5m') }}
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '7' DAY
    AND lmp IS NOT NULL
    AND congestion IS NOT NULL
