{{ config(
    materialized='table',
    tags=['kpi', 'data-platform', 'governed']
) }}

-- Canonical Data Freshness % Metric
-- Governed KPI: Percentage of datasets delivered within SLA
-- Owner: data-platform
-- SLA: 15 minutes

WITH dataset_deliveries AS (
    SELECT
        date_trunc('hour', completed_at) AS metric_timestamp,
        dataset,
        pipeline,
        completed_at,
        expected_at,
        sla_minutes,
        CASE 
            WHEN completed_at <= expected_at + INTERVAL '1' MINUTE * sla_minutes THEN 1
            ELSE 0
        END AS on_time
    FROM {{ source('curated', 'pipeline_runs') }}
    WHERE completed_at >= CURRENT_TIMESTAMP - INTERVAL '7' DAY
        AND status = 'success'
)

SELECT
    metric_timestamp,
    dataset,
    pipeline,
    COUNT(*) AS total_runs,
    SUM(on_time) AS on_time_runs,
    (SUM(on_time)::FLOAT / NULLIF(COUNT(*), 0)) * 100 AS freshness_pct,
    CASE
        WHEN (SUM(on_time)::FLOAT / NULLIF(COUNT(*), 0)) * 100 >= 95 THEN 'excellent'
        WHEN (SUM(on_time)::FLOAT / NULLIF(COUNT(*), 0)) * 100 >= 90 THEN 'good'
        WHEN (SUM(on_time)::FLOAT / NULLIF(COUNT(*), 0)) * 100 >= 80 THEN 'needs_improvement'
        ELSE 'critical'
    END AS sla_status,
    CURRENT_TIMESTAMP AS calculated_at
FROM dataset_deliveries
GROUP BY metric_timestamp, dataset, pipeline
