{{ config(
    materialized='table',
    tags=['kpi', 'demand', 'governed']
) }}

-- Canonical Load Factor Metric
-- Governed KPI: Ratio of average load to peak load over time period
-- Owner: demand-analytics
-- SLA: 60 minutes

WITH hourly_load AS (
    SELECT
        date_trunc('hour', timestamp) AS metric_timestamp,
        iso,
        zone,
        AVG(load_mw) AS avg_load_mw,
        MAX(load_mw) AS peak_load_mw
    FROM {{ source('curated', 'load_hourly') }}
    WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '7' DAY
    GROUP BY date_trunc('hour', timestamp), iso, zone
)

SELECT
    metric_timestamp,
    iso,
    zone,
    avg_load_mw,
    peak_load_mw,
    CASE 
        WHEN peak_load_mw > 0 THEN avg_load_mw / peak_load_mw
        ELSE 0
    END AS load_factor,
    CASE
        WHEN avg_load_mw / NULLIF(peak_load_mw, 0) > 0.8 THEN 'high'
        WHEN avg_load_mw / NULLIF(peak_load_mw, 0) > 0.6 THEN 'medium'
        ELSE 'low'
    END AS utilization_category,
    CURRENT_TIMESTAMP AS calculated_at
FROM hourly_load
WHERE peak_load_mw > 0
