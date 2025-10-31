# Streaming Semantics Hardening

Production-grade streaming with watermarks, exactly-once semantics, and idempotence.

## Why

**Real-time without correctness is noise.** This hardening ensures:
- Event-time processing with bounded out-of-order handling
- Exactly-once guarantees (no duplicates in curated data)
- Late data handling (quarantine instead of silent drop)
- Idempotent writes (replay-safe)

## Features

### 1. Event-Time Watermarks

**Bounded out-of-order processing** for ISO feeds and vendor ticks:

```python
{
    "type": "watermark",
    "name": "add_event_time_watermark",
    "timestamp_column": "event_timestamp",
    "max_out_of_orderness_ms": 600000,  # 10 min for ISO feeds
    "idle_source_timeout_ms": 60000,
    "strategy": "bounded_out_of_orderness"
}
```

**Guarantees:**
- Events within 10 minutes of watermark are processed correctly
- Idle sources don't block watermark advancement
- Late data (>10 min) routed to quarantine

### 2. Keyed Deduplication

**Exactly-once via deduplication**:

```python
{
    "type": "keyed_dedup",
    "name": "dedup_by_message_id",
    "key_fields": ["iso", "node", "interval_start", "message_id"],
    "time_window_ms": 600000,  # 10-minute dedup window
    "strategy": "first",  # Keep first occurrence
    "state_ttl_ms": 3600000  # 1-hour state TTL
}
```

**Guarantees:**
- Duplicate events within 10-minute window are dropped
- First occurrence is kept
- State cleaned up after 1 hour (memory bound)

### 3. Exactly-Once Sinks

**Iceberg EOS:**
```python
{
    "type": "iceberg",
    "exactly_once": True,
    "checkpoint_enabled": True,
    "options": {
        "write.wap.enabled": "true"  # Write-Audit-Publish for atomicity
    }
}
```

**ClickHouse Idempotent Writes:**
```python
{
    "type": "clickhouse",
    "exactly_once": True,
    "idempotency": {
        "enabled": True,
        "key_columns": ["iso", "node", "interval_start"],
        "engine": "ReplacingMergeTree",
        "version_column": "event_version"
    }
}
```

**Guarantees:**
- Checkpoint failures don't cause duplicates
- Replays produce identical results
- ClickHouse automatically deduplicates via ReplacingMergeTree

### 4. Late Data Handling

**Quarantine instead of drop:**

```python
{
    "type": "late_data_side_output",
    "name": "quarantine_late_data",
    "output_tag": "late-iso-data",
    "destination": {
        "type": "iceberg",
        "table": "quarantine.late_iso_lmp",
        "partition_by": ["processing_date", "iso"]
    },
    "metadata": {
        "include_watermark": True,
        "include_lateness_ms": True,
        "include_original_timestamp": True
    }
}
```

**Guarantees:**
- Late data is not silently dropped
- Stored in quarantine for investigation
- Metadata tracks lateness for debugging

## Use Cases

### ISO RT Feeds (5-minute intervals)

**Watermark:** 10 minutes (bounded out-of-order)  
**Dedup:** By `[iso, node, interval_start, message_id]`  
**EOS:** Enabled for Iceberg + ClickHouse  
**Late Data:** Quarantined if >10 min late

```python
from flink.templates import FlinkTemplates

template = FlinkTemplates.get_hardened_iso_streaming_template()
```

### Vendor Ticks (high frequency)

**Watermark:** 30 seconds (bounded out-of-order)  
**Dedup:** By `[vendor, symbol, tick_id]`  
**EOS:** Enabled  
**Late Data:** Quarantined if >30s late

## DoD (Definition of Done)

✅ **Replay test produces identical results**  
✅ **Late data path measurable** (count of quarantined records)  
✅ **No duplicates in curated** (verified via dedup checks)  
✅ **Checkpoint recovery works** (no data loss or duplication)

## Testing

### Replay Test

```python
# 1. Run job and checkpoint
# 2. Stop job
# 3. Restart from checkpoint
# 4. Compare results

def test_replay_idempotence():
    # Run 1
    results1 = run_flink_job(checkpoint_dir='/tmp/ckpt1')
    
    # Run 2 (replay from checkpoint)
    results2 = run_flink_job(checkpoint_dir='/tmp/ckpt1')
    
    # Assert identical
    assert results1 == results2
```

### Late Data Test

```python
def test_late_data_quarantine():
    # Inject late event (>10 min old)
    late_event = {
        "event_timestamp": now() - timedelta(minutes=15),
        "iso": "CAISO",
        "node": "DLAP",
        "lmp": 45.5
    }
    
    # Verify it's quarantined
    quarantine_count = count_quarantine_records()
    assert quarantine_count > 0
```

### Dedup Test

```python
def test_deduplication():
    # Send duplicate events
    send_event({"message_id": "abc123", "iso": "MISO", "lmp": 30.0})
    send_event({"message_id": "abc123", "iso": "MISO", "lmp": 30.0})  # Duplicate
    
    # Verify only one record in curated
    count = query_curated("SELECT COUNT(*) FROM rt_lmp_5m WHERE message_id = 'abc123'")
    assert count == 1
```

## Monitoring

Metrics exposed:
- `flink_watermark_lag_ms`: Watermark lag (should be <10 min)
- `flink_late_records_total`: Count of late data
- `flink_dedup_duplicates_dropped`: Count of duplicates dropped
- `flink_checkpoint_duration_ms`: Checkpoint duration

Alerts:
- Watermark lag >15 minutes → page
- Late data >1000/min → investigate
- Checkpoint failures → page

## Configuration

```yaml
# Standard operators for 5-min ISO feeds
watermark_max_out_of_order_ms: 600000  # 10 min
dedup_window_ms: 600000  # 10 min
state_ttl_ms: 3600000  # 1 hour

# For high-frequency vendor ticks
watermark_max_out_of_order_ms: 30000  # 30 sec
dedup_window_ms: 60000  # 1 min
state_ttl_ms: 300000  # 5 min
```

## Migration

Existing jobs can be upgraded incrementally:

1. **Add Watermarks**: No breaking change
2. **Add Deduplication**: Requires reprocessing to remove existing duplicates
3. **Enable EOS Sinks**: Requires checkpoint migration
4. **Add Late Data Handling**: No breaking change

## Contact

**Owner:** data-platform@254carbon.com  
**Slack:** #streaming-hardening
