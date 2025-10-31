-- Hourly LMP summary by location and ISO
-- Aggregates real-time and day-ahead LMP data

{{ config(
    materialized='table',
    tags=['marts', 'lmp'],
    unique_key=['location_id', 'hour']
) }}

with lmp_data as (
    select * from {{ ref('stg_lmp_data') }}
),

hourly_agg as (
    select
        location_id,
        location_name,
        iso_name,
        date_trunc('hour', interval_start_time) as hour,
        avg(lmp_price) as avg_lmp,
        min(lmp_price) as min_lmp,
        max(lmp_price) as max_lmp,
        stddev(lmp_price) as stddev_lmp,
        avg(energy_component) as avg_energy,
        avg(congestion_component) as avg_congestion,
        avg(loss_component) as avg_loss,
        count(*) as interval_count
    from lmp_data
    group by 1, 2, 3, 4
)

select * from hourly_agg
