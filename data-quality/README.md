# Data Quality Framework

This directory contains Great Expectations suites and runners for data quality validation.

## Structure

- `ge_expectations/`: Expectations suites for each dataset
- `runner.py`: Main runner script that executes suites against live data

## Usage

```bash
# Run EIA daily quality checks
python runner.py --suite eia_daily --table hub_curated.eia_daily_fuel

# Run NOAA hourly quality checks
python runner.py --suite noaa_hourly --table hub_curated.noaa_hourly

# Run RT LMP quality checks (ClickHouse)
python runner.py --suite rt_lmp --table default.rt_lmp --clickhouse
```

## Adding New Suites

1. Create a new file in `ge_expectations/` (e.g., `my_dataset.py`)
2. Define a `build_suite(df)` function that returns validation results
3. Add the suite to the appropriate workflow in `/workflows/`

## Integration

Quality checks are integrated into DolphinScheduler workflows as gated steps:
- If checks pass, pipeline continues to "publish" step
- If checks fail, pipeline stops and alerts are sent
