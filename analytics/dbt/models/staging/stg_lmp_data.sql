-- Staging model for LMP (Locational Marginal Pricing) data
-- Cleans and standardizes raw LMP data from various ISOs

{{ config(
    materialized='view',
    tags=['staging', 'lmp']
) }}

with source as (
    select * from {{ source('raw', 'lmp_prices') }}
),

cleaned as (
    select
        location_id,
        location_name,
        market_run_id,
        interval_start_time,
        interval_end_time,
        lmp_price,
        energy_component,
        congestion_component,
        loss_component,
        iso_name,
        created_at,
        updated_at
    from source
    where 
        lmp_price is not null
        and interval_start_time is not null
)

select * from cleaned
