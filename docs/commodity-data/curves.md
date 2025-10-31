# Forward Curves & Risk Factors

**Version**: 1.0.0  
**Last Updated**: October 31, 2025  
**Owner**: Trading Analytics Team

## Quick Reference

- **Full Documentation**: See [`/curves/README.md`](../../curves/README.md)
- **ClickHouse Schema**: See [`/clickhouse/ddl/curves.sql`](../../clickhouse/ddl/curves.sql)
- **EOD Workflow**: See [`/workflows/10-curves-eod.json`](../../workflows/10-curves-eod.json)

## Purpose

Forward curves and risk factors provide standardized pricing and risk metrics for:
- **Trading**: Mark-to-market positions, pricing new deals
- **Risk Management**: VaR, stress testing, hedging strategies
- **Backtesting**: Historical curve reconstruction via Iceberg snapshots
- **Analytics**: Trend analysis, arbitrage identification

## Key Features

1. **EOD Snapshots**: Every curve snapshotted with Iceberg snapshot ID
2. **Reproducibility**: Any past EOD can be perfectly reconstructed
3. **Standardized Buckets**: 5x16, 7x8, 2x16, HLH, LLH
4. **Fast Access**: Latest curves materialized in ClickHouse
5. **Risk Factors**: PCA, basis spreads, correlations

## Common Queries

### Get Latest Prompt Month Price

```sql
SELECT commodity, region, bucket, price
FROM curves_latest
WHERE curve_date = (SELECT max(curve_date) FROM curves_latest)
  AND term = 'MONTHLY'
  AND delivery_start = toStartOfMonth(addMonths(today(), 1));
```

### Historical Curve Evolution

```sql
SELECT curve_date, avg(price) AS avg_price
FROM curves_eod
WHERE curve_id = 'CAISO_SP15_5x16'
  AND term = 'MONTHLY'
  AND delivery_start = '2025-06-01'
  AND curve_date >= '2025-01-01'
GROUP BY curve_date
ORDER BY curve_date;
```

### Day-over-Day Changes

```sql
SELECT * FROM v_curves_dod_changes
WHERE curve_date = today()
  AND abs(price_change_pct) > 5
ORDER BY abs(price_change_pct) DESC
LIMIT 20;
```

## SLA Guarantees

- **EOD Complete by**: 02:00 UTC
- **Snapshot Write**: < 5 minutes
- **Query Latency**: < 500ms (p95)
- **Data Freshness**: Prior business day

## Support

For issues with curves or factors:
- Slack: `#trading-ops`
- Email: `trading@254carbon.com`
- On-call: PagerDuty escalation policy `trading-analytics`
