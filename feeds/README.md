# Feed Ingestion Infrastructure

This directory contains the infrastructure for feed ingestion and processing across the HMCo data platform.

## Overview

The feed infrastructure supports 10 core ingestion tasks designed for advanced analytics:

1. **ISO Real-Time LMP** - CAISO, MISO, SPP real-time pricing with 1-min partitions
2. **ISO Day-Ahead Awards** - Day-ahead market schedules and clearing prices
3. **Outage & Constraint Events** - Power plant and transmission outages/constraints
4. **Topology & Reference Data** - Node, hub, and constraint reference tables (SCD2)
5. **Weather H3 Features** - NOAA HRRR/GFS with H3 spatial tiling
6. **EIA/FRED/Census Macro** - Macro fundamentals and economic indicators
7. **LNG Send-out** - AIS vessel tracking to regas terminal send-out
8. **Renewables & Load Telemetry** - Real-time ISO telemetry
9. **Curve Snapshots** - Power/gas futures and options surfaces
10. **Feed Ledger** - Idempotent ingestion with watermarks and replay

## Directory Structure

```
feeds/
├── README.md                    # This file
├── schemas/                     # Avro/Protobuf schema contracts
│   ├── iso_rt_lmp.avsc
│   ├── iso_da_schedule.avsc
│   ├── outage_event.avsc
│   └── market_curve.avsc
└── vendor_adapters/             # Pluggable vendor integrations
    ├── README.md
    ├── base_adapter.py          # Abstract base classes
    └── __init__.py              # Adapter registry
```

## Streaming Topics

All topics are defined in `/streaming/topics.yaml`:

### ISO Real-Time LMP
- `iso.rt.lmp.caiso` - CAISO 5-minute LMP
- `iso.rt.lmp.miso` - MISO 5-minute LMP
- `iso.rt.lmp.spp` - SPP 5-minute LMP

### ISO Day-Ahead
- `iso.da.schedule.caiso` - CAISO DA awards
- `iso.da.schedule.miso` - MISO DA awards
- `iso.da.schedule.spp` - SPP DA awards

### Events
- `iso.events.outage.{caiso|miso|spp}` - Outage notifications
- `iso.events.constraint.{caiso|miso|spp}` - Constraint events

### Telemetry
- `iso.rt.telemetry.{caiso|miso|spp}` - Solar/wind/load actuals

### LNG/Marine
- `marine.ais.raw` - Raw AIS vessel positions
- `lng.port_events` - LNG terminal events (ETA/ETD)

### Market Data
- `market.curves.power` - Power futures curves
- `market.curves.gas` - Gas futures curves
- `market.options.power` - Power options surface

## Workflows

Workflows are defined in `/workflows/`:

- `da_awards_{caiso|miso|spp}.json` - Day-ahead market processing
- `iso_reference_*.json` - Reference data pipelines
- `weather_h3_hourly.json` - Weather H3 tiling
- `{eia|fred|census}_macro_*.json` - Macro fundamentals
- `lng_sendout_pipeline.json` - LNG send-out estimation

## Schema Contracts

All feed schemas are version-controlled Avro schemas in `schemas/`. These are registered in the Schema Registry with compatibility mode settings:

- **BACKWARD** for most feeds (can read old data with new schema)
- **FULL** for critical feeds (both forward and backward compatible)

### Schema Evolution

When updating schemas:
1. Add new fields as optional (with defaults)
2. Never remove existing fields
3. Test compatibility with `avro-tools` before deploying
4. Update schema version in registry

## Feed Ledger

The feed ledger tracks watermarks and state for idempotent ingestion:

```python
from services.feed_ledger import FeedLedger, FeedLedgerEntry, FeedState

ledger = FeedLedger(backend="postgres")

# Write entry
entry = FeedLedgerEntry(
    feed_id="iso.rt.lmp.caiso",
    partition="2025-01-15",
    watermark_ts=datetime.now(),
    checksum="abc123...",
    state=FeedState.COMPLETED
)
ledger.write_entry(entry)

# Get latest watermark
watermark = ledger.get_latest_watermark("iso.rt.lmp.caiso", "2025-01-15")
```

### CLI Usage

```bash
# Replay feed data
feeds replay --feed iso.rt.lmp.caiso --from 2025-07-01T00:00Z --to now

# Check feed status
feeds status --feed iso.rt.lmp.caiso

# List recent entries
feeds list --feed iso.rt.lmp.caiso --limit 100

# Get watermark
feeds watermark --feed iso.rt.lmp.caiso --partition 2025-01-15
```

## Vendor Adapters

Vendor adapters provide pluggable integrations for proprietary data sources:

```python
from feeds.vendor_adapters import get_adapter

# Get ICE adapter
adapter = get_adapter('ice', config={
    'api_key': os.getenv('ICE_API_KEY'),
    'endpoint': 'https://api.ice.com/v1'
})

# Fetch curve snapshot
snapshot = adapter.fetch_curve_snapshot('PJM_WH', datetime(2025, 1, 15))
```

See `vendor_adapters/README.md` for implementation details.

## Shared Utilities

### Units & Conversions

```python
from sdk.shared import UnitConverter

# Energy conversions
mmbtu = UnitConverter.convert_energy(100, "MWh", "MMBtu")  # 341.2142 MMBtu

# Power conversions
kw = UnitConverter.convert_power(50, "MW", "kW")  # 50000 kW

# Temperature conversions
celsius = UnitConverter.convert_temperature(70, "F", "C")  # 21.11 C

# Standardize ISO prices
price = UnitConverter.standardize_iso_price(45.5, "CAISO")  # $/MWh
```

### Calendar & DST Handling

```python
from sdk.shared import CalendarUtils

# Get ISO timezone
tz = CalendarUtils.get_iso_timezone("CAISO")

# Check DST
is_dst = CalendarUtils.is_dst(datetime.now(), "CAISO")

# Get DST transitions for year
transitions = CalendarUtils.get_dst_transitions(2025, "CAISO")

# Get settlement periods (handles DST)
periods = CalendarUtils.get_settlement_periods(date(2025, 3, 9), "CAISO")  # 23 (spring forward)

# Check NERC holidays
is_holiday = CalendarUtils.is_nerc_holiday(date(2025, 7, 4))  # True

# Align event time (handle clock skew)
aligned = CalendarUtils.align_event_time(dt, "CAISO", round_to_minutes=5)
```

## SLO Monitoring

Feed SLOs are monitored via Prometheus alerts defined in:
```
helm/charts/data-platform/charts/dolphinscheduler/templates/prometheus-alerts.yaml
```

### SLO Metrics

Each feed tracks:
- **Freshness** - Time since last watermark
- **Completeness** - Percentage of expected records received
- **Duplicate Rate** - Percentage of duplicate records
- **Late Data Rate** - Percentage of records arriving late

### Alert Thresholds

| Feed Type | Freshness SLA | Completeness | Dup Rate |
|-----------|--------------|--------------|----------|
| ISO RT LMP | 60 seconds | 95% | <1% |
| ISO DA Awards | 60 minutes | 98% | <0.1% |
| Outage Events | 5 minutes | 100% | 0% |
| Weather H3 | 90 minutes | 95% | <1% |
| LNG Send-out | 15 minutes | 90% | <5% |
| Macro Data | 30 minutes | 99% | <0.1% |
| Curve Snapshots | 15 minutes | 99% | <0.1% |

## Security

### Credentials Management

All feed credentials are managed via ExternalSecrets and Vault:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: feed-api-keys
spec:
  secretStoreRef:
    name: vault-backend
  target:
    name: feed-credentials
  data:
    - secretKey: CAISO_API_KEY
      remoteRef:
        key: feeds/iso/caiso
        property: api_key
```

### Key Rotation

- Feed API keys are rotated **quarterly**
- Rotation is tracked in Vault
- Old keys are kept for 30 days for rollback

## Testing

Run feed infrastructure tests:

```bash
# Unit tests
pytest tests/unit/feeds/

# Integration tests
pytest tests/integration/feeds/ --requires-kafka --requires-db

# E2E tests
pytest tests/e2e/feeds/ --slow
```

## Troubleshooting

### Feed Not Updating

1. Check feed ledger status: `feeds status --feed <feed_id>`
2. Check Prometheus alerts for SLA violations
3. Check DolphinScheduler workflow status
4. Review feed logs in Loki

### Duplicate Records

1. Check feed ledger for multiple entries with same watermark
2. Verify exactly-once processing in Flink/Spark
3. Check for concurrent runs in DolphinScheduler

### Late Data

1. Check event-time watermarks in Flink
2. Review late-data side output metrics
3. Adjust watermark delay if needed

## Performance Tuning

### Kafka Partitions

- ISO RT LMP: 12 partitions (high throughput)
- DA Awards: 8 partitions (moderate throughput)
- Events: 6 partitions (lower throughput)

### Batch Sizes

- Flink: 5000-10000 records
- Spark: 10000-100000 records (depending on data size)

### Checkpointing

- Flink: Every 60 seconds
- Spark: Every 5 minutes

## References

- [UIS Specification](../sdk/uis/README.md)
- [DolphinScheduler Workflows](../workflows/README.md)
- [Streaming Topics](../streaming/topics.yaml)
- [Prometheus Alerts](../helm/charts/data-platform/charts/dolphinscheduler/templates/prometheus-alerts.yaml)
