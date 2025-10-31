{{ config(
    materialized='table',
    tags=['kpi', 'weather', 'governed']
) }}

-- Canonical Degree Day Delta Metric
-- Governed KPI: Variance from historical average degree days
-- Owner: weather-analytics
-- SLA: 360 minutes

WITH current_degree_days AS (
    SELECT
        date_trunc('day', timestamp) AS metric_date,
        region,
        weather_station,
        SUM(CASE 
            WHEN avg_temp_f < 65 THEN 65 - avg_temp_f  -- Heating degree days
            ELSE 0 
        END) AS hdd,
        SUM(CASE 
            WHEN avg_temp_f > 65 THEN avg_temp_f - 65  -- Cooling degree days
            ELSE 0 
        END) AS cdd
    FROM {{ source('curated', 'weather_hourly') }}
    WHERE timestamp >= CURRENT_DATE - INTERVAL '30' DAY
    GROUP BY date_trunc('day', timestamp), region, weather_station
),

historical_avg AS (
    SELECT
        EXTRACT(MONTH FROM timestamp) AS month,
        EXTRACT(DAY FROM timestamp) AS day,
        region,
        weather_station,
        AVG(hdd) AS avg_hdd,
        AVG(cdd) AS avg_cdd
    FROM {{ source('curated', 'weather_historical_degree_days') }}
    GROUP BY EXTRACT(MONTH FROM timestamp), EXTRACT(DAY FROM timestamp), region, weather_station
)

SELECT
    c.metric_date AS metric_timestamp,
    c.region,
    c.weather_station,
    c.hdd AS current_hdd,
    c.cdd AS current_cdd,
    h.avg_hdd AS historical_avg_hdd,
    h.avg_cdd AS historical_avg_cdd,
    c.hdd - h.avg_hdd AS hdd_delta,
    c.cdd - h.avg_cdd AS cdd_delta,
    CURRENT_TIMESTAMP AS calculated_at
FROM current_degree_days c
LEFT JOIN historical_avg h
    ON EXTRACT(MONTH FROM c.metric_date) = h.month
    AND EXTRACT(DAY FROM c.metric_date) = h.day
    AND c.region = h.region
    AND c.weather_station = h.weather_station
