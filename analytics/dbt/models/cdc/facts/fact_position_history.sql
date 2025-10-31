{{ config(
    materialized='incremental',
    unique_key='position_history_sk',
    tags=['cdc', 'fact', 'position-history']
) }}

/*
 * Position History Fact Table with CDC
 * Tracks position changes over time for risk management
 * Source: CDC stream from positions table
 */

WITH cdc_raw AS (
    SELECT
        position_id,
        position_date,
        book_id,
        trader_id,
        product_type,
        delivery_start,
        delivery_end,
        net_position_mwh,
        marked_to_market_value,
        var_95,  -- Value at Risk 95%
        position_status,  -- OPEN, CLOSED, HEDGED
        cdc_operation,
        cdc_timestamp,
        cdc_lsn
    FROM {{ source('cdc', 'positions_cdc') }}
    {% if is_incremental() %}
    WHERE cdc_timestamp > (SELECT MAX(cdc_timestamp) FROM {{ this }})
    {% endif %}
),

position_changes AS (
    SELECT
        {{ dbt_utils.generate_surrogate_key(['position_id', 'cdc_timestamp']) }} AS position_history_sk,
        position_id,
        position_date,
        book_id,
        trader_id,
        product_type,
        delivery_start,
        delivery_end,
        net_position_mwh,
        marked_to_market_value,
        var_95,
        position_status,
        cdc_operation,
        cdc_timestamp,
        cdc_lsn,
        LAG(net_position_mwh) OVER (PARTITION BY position_id ORDER BY cdc_timestamp) AS previous_position_mwh,
        LAG(marked_to_market_value) OVER (PARTITION BY position_id ORDER BY cdc_timestamp) AS previous_mtm_value,
        LAG(var_95) OVER (PARTITION BY position_id ORDER BY cdc_timestamp) AS previous_var,
        ROW_NUMBER() OVER (PARTITION BY position_id ORDER BY cdc_timestamp) AS change_sequence
    FROM cdc_raw
)

SELECT
    position_history_sk,
    position_id,
    position_date,
    book_id,
    trader_id,
    product_type,
    delivery_start,
    delivery_end,
    net_position_mwh,
    previous_position_mwh,
    marked_to_market_value,
    previous_mtm_value,
    var_95,
    previous_var,
    position_status,
    cdc_operation,
    cdc_timestamp,
    change_sequence,
    -- Calculated fields
    CASE 
        WHEN previous_position_mwh IS NOT NULL
        THEN net_position_mwh - previous_position_mwh
        ELSE net_position_mwh
    END AS position_delta_mwh,
    CASE 
        WHEN previous_mtm_value IS NOT NULL
        THEN marked_to_market_value - previous_mtm_value
        ELSE 0
    END AS pnl_change,
    CURRENT_TIMESTAMP AS dbt_updated_at
FROM position_changes
