# UIS to Spark Compiler (Micro-Batch)

This module provides compilation from Unified Ingestion Spec (UIS) specifications to Apache Spark job configurations for micro-batch processing.

## Overview

The Spark compiler generates configurations for executing micro-batch data processing pipelines using Apache Spark. It supports various data sources (REST APIs, files, Kafka) and sinks (Iceberg, ClickHouse, Kafka) with comprehensive transformation capabilities.

## Features

- **Micro-Batch Processing**: Optimized for periodic batch processing (minutes to hours)
- **Multiple Sources**: REST APIs, file systems, Kafka streaming
- **Multiple Sinks**: Parquet landing zones (MinIO/S3), Iceberg tables, ClickHouse, Kafka topics
- **Data Transforms**: Field mapping, schema validation, custom SQL transforms
- **Adaptive Query Execution**: Automatic query optimization and performance tuning
- **Kubernetes Integration**: Native Kubernetes deployment support
- **Monitoring & Observability**: Built-in metrics and alerting

## Supported Provider Types

- `rest_api`: REST API endpoints with JSON responses
- `file_ftp`: File-based data sources (CSV, JSON, XML, Parquet)
- `kafka`: Kafka streaming sources and sinks

## Supported Sink Types

- `iceberg`: Apache Iceberg tables with ACID transactions
- `clickhouse`: ClickHouse analytical database
- `parquet`: Object storage landing zone (e.g., MinIO/S3) using Parquet format
- `kafka`: Kafka topics for event streaming

## Usage

### Basic Compilation

```python
import json

from uis.spec import (
    UnifiedIngestionSpec,
    ProviderConfig,
    EndpointConfig,
    ProviderType,
    SinkType,
    IngestionMode,
    AuthType
)
from uis.compilers.spark import SparkCompiler

# Create micro-batch UIS specification
spec = UnifiedIngestionSpec(
    version="1.1",
    name="market-data-ingestion",
    provider=ProviderConfig(
        name="polygon_api",
        display_name="Polygon Market Data API",
        provider_type=ProviderType.REST_API,
        base_url="https://api.polygon.io",
        config={
            "executor_memory": "8g",
            "executor_cores": "4",
            "s3_endpoint": "http://minio:9000"
        },
        tenant_id="trading-platform",
        owner="trading-team@company.com",
        mode=IngestionMode.MICRO_BATCH,
        parallelism=4,
        endpoints=[
            EndpointConfig(
                name="stock_tickers",
                path="/v3/reference/tickers",
                method="GET",
                auth=AuthType.API_KEY,
                query_params={"market": "stocks", "limit": "1000"},
                pagination="cursor",
                response_path="$.results",
                field_mapping={
                    "ticker": "symbol",
                    "name": "company_name",
                    "market_cap": "market_capitalization"
                },
                rate_limit_per_second=20
            )
        ],
        sinks=[
            {
                "type": SinkType.PARQUET,
                "config": {
                    "bucket": "landing-zone",
                    "path_prefix": "market-data-ingestion",
                    "compression": "snappy"
                }
            },
            {
                "type": SinkType.ICEBERG,
                "table_name": "market_data.stock_tickers",
                "partition_by": ["exchange", "date"],
                "config": {
                    "catalog": "hive_prod",
                    "namespace": "market_data",
                    "warehouse": "s3://prod-warehouse/"
                }
            }
        ]
    ),
    created_by="data-engineer"
)

# Compile to Spark configuration and job artifacts
compiler = SparkCompiler()
artifacts = compiler.compile_to_job_artifacts(spec, job_config_path="/configs/market-data.json")

# Submit arguments and Spark session config
spark_args = artifacts.spark_submit_args
spark_session_conf = artifacts.spark_session_conf

# Persist job configuration for spark-submit
with open(artifacts.job_config_path, 'w') as f:
    json.dump(artifacts.job_config, f, indent=2)

print("Spark arguments:", spark_args)
print("Spark session config:", spark_session_conf)
```

### Spark Job Execution

The generated configurations can be executed using the provided Spark job runner:

```bash
# Compile UIS spec to Spark config
python -c "
from uis.spec import UnifiedIngestionSpec, ProviderConfig, EndpointConfig, ProviderType, SinkType, IngestionMode, AuthType
from uis.compilers.spark import SparkCompiler
import json

# Create and compile spec (as above)
compiler = SparkCompiler()
config = compiler.compile(spec)

# Save config for execution
with open('microbatch-config.json', 'w') as f:
    json.dump(config, f, indent=2)
"

# Execute Spark job
spark-submit \\
  --class com.hmco.dataplatform.SparkMicroBatchJob \\
  --master k8s://https://kubernetes.default.svc.cluster.local:443 \\
  --deploy-mode cluster \\
  --conf spark.kubernetes.namespace=trading-platform \\
  --conf spark.kubernetes.container.image=hmco/spark-runner:latest \\
  --conf spark.executor.memory=8g \\
  --conf spark.executor.cores=4 \\
  --job-config /configs/market-data.json \\
  spark_job.py
```

## Generated Configuration Structure

```json
{
  "job_type": "micro_batch",
  "spark_config": {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.streaming.trigger.processingTime": "5 minutes",
    "spark.executor.memory": "8g",
    "spark.sql.warehouse.dir": "s3://warehouse/"
  },
  "sources": [
    {
      "type": "rest_api",
      "name": "rest_stock_tickers",
      "url": "https://api.polygon.io/v3/reference/tickers",
      "method": "GET",
      "headers": {"Authorization": "Bearer {{api_key}}"},
      "params": {"market": "stocks", "limit": "1000"},
      "format": "json",
      "response_path": "$.results",
      "rate_limit": 20
    }
  ],
  "transforms": [
    {
      "type": "field_mapping",
      "name": "map_stock_tickers",
      "mapping": {
        "ticker": "symbol",
        "name": "company_name"
      }
    }
  ],
  "sinks": [
    {
      "type": "parquet",
      "name": "parquet_market-data-ingestion",
      "format": "parquet",
      "path": "s3://landing-zone/market-data-ingestion",
      "mode": "append",
      "options": {
        "compression": "snappy"
      }
    },
    {
      "type": "iceberg",
      "name": "iceberg_market_data.stock_tickers",
      "table": "market_data.stock_tickers",
      "catalog": "hive_prod",
      "namespace": "market_data",
      "mode": "append",
      "options": {
        "partitionBy": "exchange,date",
        "write.wap.enabled": "true"
      }
    }
  ],
  "schedule": {
    "trigger_type": "processing_time",
    "trigger_interval": "5 minutes"
  },
  "monitoring": {
    "metrics_enabled": true,
    "metrics_prefix": "uis.market-data-ingestion"
  }
}
```

## Spark Configuration

The compiler generates optimized Spark configurations for micro-batch processing:

### Performance Optimizations
- **Adaptive Query Execution**: Automatic query optimization and partition coalescing
- **Dynamic Partition Pruning**: Efficient partition filtering
- **Speculative Execution**: Straggler mitigation for long-running tasks

### Memory Management
- **Unified Memory Management**: Optimized memory allocation
- **Broadcast Join Optimization**: Automatic broadcast threshold tuning
- **Shuffle Service**: External shuffle service for large datasets

### Storage Integration
- **S3/MinIO**: Native object storage integration
- **Iceberg**: ACID transactions and time travel
- **Hive Metastore**: Catalog integration for table metadata

## Kubernetes Integration

The compiler generates Kubernetes-native Spark configurations:

```python
# Generated Kubernetes configuration
spark_args = [
    "--master", "k8s://https://kubernetes.default.svc.cluster.local:443",
    "--deploy-mode", "cluster",
    "--conf", "spark.kubernetes.namespace=trading-platform",
    "--conf", "spark.kubernetes.container.image=hmco/spark-runner:latest",
    "--conf", "spark.kubernetes.authenticate.driver.serviceAccountName=spark",
    "--conf", "spark.kubernetes.authenticate.executor.serviceAccountName=spark"
]
```

## Monitoring and Observability

Built-in monitoring with structured metrics:

```json
{
  "monitoring": {
    "metrics_enabled": true,
    "metrics_prefix": "uis.market-data-ingestion",
    "streaming_metrics": {
      "input_rows_per_second": "inputRate",
      "processed_rows_per_second": "processingRate",
      "backlog_rows": "inputRowsPerSecond"
    },
    "alerts": {
      "processing_delay_threshold_ms": 300000,
      "error_rate_threshold": 0.05
    }
  }
}
```

## Development

### Running Tests

```bash
cd sdk/uis/compilers/spark
python test_compiler.py
```

### Adding New Source Types

1. Add source type to `_build_sources_config` method
2. Implement specific source builder (e.g., `_build_rest_source`)
3. Add validation in `validate_config`
4. Update tests and documentation

### Adding New Transform Types

1. Add transform type to `_build_transforms_config` method
2. Implement transform logic in `apply_transforms` method
3. Add validation and error handling
4. Update tests and documentation

## Integration with DolphinScheduler

Generated Spark configurations integrate with DolphinScheduler:

1. **Task Definition**: Create Spark task in DolphinScheduler
2. **Resource Upload**: Upload compiled configuration as task resource
3. **Parameter Configuration**: Set API keys and other secrets as task parameters
4. **Scheduling**: Configure cron schedules for periodic execution
5. **Monitoring**: Monitor execution through DolphinScheduler UI and integrated metrics

## Performance Tuning

### Memory Configuration
```python
# For large datasets
config = {
    "executor_memory": "16g",
    "executor_cores": "8",
    "driver_memory": "4g",
    "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB"
}
```

### Parallelism Tuning
```python
# For high-throughput scenarios
parallelism = 8  # Number of executor cores
spark.sql.shuffle.partitions = 200  # Default shuffle partitions
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase executor memory or reduce batch size
2. **Rate Limiting**: Adjust API rate limits in endpoint configuration
3. **Schema Evolution**: Update schema contracts in UIS specification
4. **Checkpoint Issues**: Verify S3/MinIO connectivity and permissions

### Debugging

Enable debug logging:
```python
spark_config = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.eventLog.enabled": "true",
    "spark.eventLog.dir": "s3://spark-logs/"
}
```

## License

This compiler is part of the HMCo data platform SDK.


