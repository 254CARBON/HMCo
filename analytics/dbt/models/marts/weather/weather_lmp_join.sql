-- Join weather features with LMP data
-- Enable weather-driven trading strategies

{{ config(
    materialized='table',
    tags=['marts', 'weather', 'lmp']
) }}

with lmp as (
    select * from {{ ref('lmp_hourly_summary') }}
),

weather as (
    select * from {{ source('curated', 'weather_features') }}
),

joined as (
    select
        l.location_id,
        l.location_name,
        l.iso_name,
        l.hour,
        l.avg_lmp,
        l.min_lmp,
        l.max_lmp,
        w.temperature,
        w.humidity,
        w.wind_speed,
        w.solar_irradiance,
        w.cloud_cover
    from lmp l
    left join weather w
        on l.location_id = w.location_id
        and l.hour = w.observation_time
)

select * from joined
