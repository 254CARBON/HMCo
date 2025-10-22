# Real-time Streaming Infrastructure

**Platform**: 254Carbon Real-time Analytics  
**Components**: Kafka Connect + Apache Flink + Apache Doris  
**Updated**: October 22, 2025

## Overview

This directory contains the real-time streaming infrastructure for the 254Carbon platform, enabling:

- **Kafka Connect**: Real-time data ingestion from external sources
- **Apache Flink**: Stream processing and transformations
- **Apache Doris**: Real-time OLAP analytics with sub-second queries

## Architecture

```
External APIs → Kafka Connect → Kafka → Flink → Doris/Iceberg → Dashboards
                     ↓                      ↓         ↓
                 Connectors            Processing   Real-time
                                       & Alerts     Analytics
```

## Components

### Kafka Connect (`kafka-connect/`)
- Distributed workers for high availability
- Source connectors (HTTP, WebSocket, CDC)
- Sink connectors (Iceberg, Doris, Elasticsearch)
- Monitoring and metrics

### Apache Flink (`flink/`)
- Flink Kubernetes Operator
- Stream processing applications
- Flink SQL jobs
- State management

### Apache Doris (`doris/`)
- Frontend (FE) nodes for query coordination
- Backend (BE) nodes for data storage and computation
- Materialized views for pre-aggregation
- Real-time analytics queries

### Connectors (`connectors/`)
- Connector configurations
- Custom connector plugins
- Schema definitions

## Deployment

Deploy components in order:

```bash
# 1. Scale Kafka infrastructure
kubectl apply -f k8s/shared/kafka/kafka-production.yaml

# 2. Deploy Flink Operator
kubectl apply -f k8s/streaming/flink/flink-operator.yaml

# 3. Deploy Kafka Connect
kubectl apply -f k8s/streaming/kafka-connect/

# 4. Deploy Apache Doris
kubectl apply -f k8s/streaming/doris/

# 5. Deploy connectors
kubectl apply -f k8s/streaming/connectors/
```

## Monitoring

Access streaming metrics:
- Kafka Connect: http://kafka-connect-service:8083/metrics
- Flink Dashboard: http://flink-jobmanager:8081
- Doris: http://doris-fe:8030

## Documentation

- [Kafka Connect Guide](kafka-connect/README.md)
- [Flink Applications](flink/README.md)
- [Doris Configuration](doris/README.md)


