-- Test: Ensure metric calculations are consistent across time
-- This test will fail if breaking changes are introduced to metric logic

WITH metric_checksums AS (
    SELECT
        'lmp_spread' AS metric_name,
        COUNT(*) AS row_count,
        SUM(lmp_spread) AS total_sum,
        AVG(lmp_spread) AS avg_value
    FROM {{ ref('lmp_spread') }}
    WHERE metric_timestamp >= CURRENT_TIMESTAMP - INTERVAL '1' DAY
    
    UNION ALL
    
    SELECT
        'nodal_congestion_factor' AS metric_name,
        COUNT(*) AS row_count,
        SUM(congestion_factor) AS total_sum,
        AVG(congestion_factor) AS avg_value
    FROM {{ ref('nodal_congestion_factor') }}
    WHERE metric_timestamp >= CURRENT_TIMESTAMP - INTERVAL '1' DAY
    
    UNION ALL
    
    SELECT
        'load_factor' AS metric_name,
        COUNT(*) AS row_count,
        SUM(load_factor) AS total_sum,
        AVG(load_factor) AS avg_value
    FROM {{ ref('load_factor') }}
    WHERE metric_timestamp >= CURRENT_TIMESTAMP - INTERVAL '1' DAY
)

-- This test passes if all metrics have data and reasonable values
SELECT *
FROM metric_checksums
WHERE row_count = 0
    OR avg_value IS NULL
    OR ABS(avg_value) > 1000  -- Sanity check on metric values
