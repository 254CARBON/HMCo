{{ config(
    materialized='incremental',
    unique_key='trade_history_sk',
    tags=['cdc', 'fact', 'trade-history']
) }}

/*
 * Trade History Fact Table with CDC
 * Captures complete trade lifecycle (submitted, executed, settled, cancelled)
 * Source: CDC stream from trades table
 */

WITH cdc_raw AS (
    SELECT
        trade_id,
        trade_timestamp,
        trade_status,  -- SUBMITTED, EXECUTED, SETTLED, CANCELLED
        trader_id,
        counterparty_id,
        product_type,
        delivery_start,
        delivery_end,
        quantity_mwh,
        price_per_mwh,
        total_value,
        settlement_date,
        settlement_status,
        cdc_operation,
        cdc_timestamp,
        cdc_lsn
    FROM {{ source('cdc', 'trades_cdc') }}
    {% if is_incremental() %}
    WHERE cdc_timestamp > (SELECT MAX(cdc_timestamp) FROM {{ this }})
    {% endif %}
),

trade_events AS (
    SELECT
        {{ dbt_utils.generate_surrogate_key(['trade_id', 'cdc_timestamp']) }} AS trade_history_sk,
        trade_id,
        trade_timestamp,
        trade_status,
        trader_id,
        counterparty_id,
        product_type,
        delivery_start,
        delivery_end,
        quantity_mwh,
        price_per_mwh,
        total_value,
        settlement_date,
        settlement_status,
        cdc_operation,
        cdc_timestamp,
        cdc_lsn,
        LAG(trade_status) OVER (PARTITION BY trade_id ORDER BY cdc_timestamp) AS previous_status,
        LAG(price_per_mwh) OVER (PARTITION BY trade_id ORDER BY cdc_timestamp) AS previous_price,
        ROW_NUMBER() OVER (PARTITION BY trade_id ORDER BY cdc_timestamp) AS event_sequence
    FROM cdc_raw
)

SELECT
    trade_history_sk,
    trade_id,
    trade_timestamp,
    trade_status,
    previous_status,
    trader_id,
    counterparty_id,
    product_type,
    delivery_start,
    delivery_end,
    quantity_mwh,
    price_per_mwh,
    previous_price,
    total_value,
    settlement_date,
    settlement_status,
    cdc_operation,
    cdc_timestamp,
    event_sequence,
    -- Calculated fields
    CASE 
        WHEN previous_price IS NOT NULL AND price_per_mwh != previous_price
        THEN price_per_mwh - previous_price
        ELSE 0
    END AS price_change,
    DATEDIFF('second', LAG(cdc_timestamp) OVER (PARTITION BY trade_id ORDER BY cdc_timestamp), cdc_timestamp) AS seconds_since_last_change,
    CURRENT_TIMESTAMP AS dbt_updated_at
FROM trade_events
