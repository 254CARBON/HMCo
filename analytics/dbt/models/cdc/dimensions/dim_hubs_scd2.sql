{{ config(
    materialized='incremental',
    unique_key='hub_sk',
    tags=['cdc', 'dimension', 'scd2', 'reference-data']
) }}

/*
 * SCD Type 2 Dimension for Power Trading Hubs
 * Maintains full history of hub attribute changes
 * Source: CDC stream from hub_master table
 */

WITH cdc_raw AS (
    SELECT
        hub_id,
        hub_name,
        iso,
        hub_type,  -- Load zone, Interface, Hub
        settlement_location,
        is_active,
        cdc_operation,
        cdc_timestamp,
        cdc_lsn
    FROM {{ source('cdc', 'hub_master_cdc') }}
    {% if is_incremental() %}
    WHERE cdc_timestamp > (SELECT MAX(valid_from) FROM {{ this }})
    {% endif %}
),

changes AS (
    SELECT
        hub_id,
        hub_name,
        iso,
        hub_type,
        settlement_location,
        is_active,
        cdc_timestamp AS valid_from,
        LEAD(cdc_timestamp) OVER (PARTITION BY hub_id ORDER BY cdc_timestamp) AS valid_to,
        CASE 
            WHEN LEAD(cdc_timestamp) OVER (PARTITION BY hub_id ORDER BY cdc_timestamp) IS NULL 
            THEN TRUE 
            ELSE FALSE 
        END AS is_current,
        cdc_lsn,
        ROW_NUMBER() OVER (PARTITION BY hub_id ORDER BY cdc_timestamp) AS version_number
    FROM cdc_raw
    WHERE cdc_operation != 'DELETE'
)

SELECT
    {{ dbt_utils.generate_surrogate_key(['hub_id', 'valid_from']) }} AS hub_sk,
    hub_id,
    hub_name,
    iso,
    hub_type,
    settlement_location,
    is_active,
    valid_from,
    COALESCE(valid_to, '9999-12-31'::timestamp) AS valid_to,
    is_current,
    version_number,
    cdc_lsn,
    CURRENT_TIMESTAMP AS dbt_updated_at
FROM changes
