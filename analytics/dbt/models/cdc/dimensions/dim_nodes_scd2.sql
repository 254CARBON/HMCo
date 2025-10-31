{{ config(
    materialized='incremental',
    unique_key='node_sk',
    tags=['cdc', 'dimension', 'scd2', 'reference-data']
) }}

/*
 * SCD Type 2 Dimension for Power Grid Nodes
 * Maintains full history of node attribute changes
 * Source: CDC stream from node_master table
 */

WITH cdc_raw AS (
    SELECT
        node_id,
        node_name,
        iso,
        zone,
        hub,
        pnode_type,
        latitude,
        longitude,
        capacity_mw,
        is_active,
        cdc_operation,  -- INSERT, UPDATE, DELETE
        cdc_timestamp,
        cdc_lsn  -- Log sequence number for ordering
    FROM {{ source('cdc', 'node_master_cdc') }}
    {% if is_incremental() %}
    WHERE cdc_timestamp > (SELECT MAX(valid_from) FROM {{ this }})
    {% endif %}
),

changes AS (
    SELECT
        node_id,
        node_name,
        iso,
        zone,
        hub,
        pnode_type,
        latitude,
        longitude,
        capacity_mw,
        is_active,
        cdc_operation,
        cdc_timestamp AS valid_from,
        LEAD(cdc_timestamp) OVER (PARTITION BY node_id ORDER BY cdc_timestamp) AS valid_to,
        CASE 
            WHEN LEAD(cdc_timestamp) OVER (PARTITION BY node_id ORDER BY cdc_timestamp) IS NULL 
            THEN TRUE 
            ELSE FALSE 
        END AS is_current,
        cdc_lsn,
        ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY cdc_timestamp) AS version_number
    FROM cdc_raw
    WHERE cdc_operation != 'DELETE'  -- Exclude deletes (mark as inactive instead)
)

SELECT
    {{ dbt_utils.generate_surrogate_key(['node_id', 'valid_from']) }} AS node_sk,
    node_id,
    node_name,
    iso,
    zone,
    hub,
    pnode_type,
    latitude,
    longitude,
    capacity_mw,
    is_active,
    valid_from,
    COALESCE(valid_to, '9999-12-31'::timestamp) AS valid_to,
    is_current,
    version_number,
    cdc_lsn,
    CURRENT_TIMESTAMP AS dbt_updated_at
FROM changes
