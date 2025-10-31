"""
Cost Attribution Service - Track and report data platform usage and costs
Collects metrics from Trino, ClickHouse, MinIO, and Spark
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import yaml
import pandas as pd
from clickhouse_driver import Client as ClickHouseClient
from trino.dbapi import connect as trino_connect
import boto3
from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)

# Metrics
cost_by_user = Gauge('data_platform_cost_by_user', 'Cost by user', ['user', 'service'])
cost_by_dataset = Gauge('data_platform_cost_by_dataset', 'Cost by dataset', ['dataset', 'service'])
cost_by_team = Gauge('data_platform_cost_by_team', 'Cost by team', ['team', 'service'])
queries_executed = Counter('data_platform_queries_total', 'Total queries executed', ['service', 'user'])
bytes_scanned = Counter('data_platform_bytes_scanned_total', 'Total bytes scanned', ['service', 'dataset'])


@dataclass
class UsageMetric:
    """Usage metric for a time period"""
    service: str
    user: str
    team: Optional[str]
    dataset: Optional[str]
    query_count: int
    bytes_scanned: int
    execution_time_ms: float
    cost_usd: float
    timestamp: datetime


@dataclass
class CostConfig:
    """Cost configuration per service"""
    # Costs per unit
    trino_query_cost: float = 0.001  # per query
    trino_tb_scanned_cost: float = 5.00  # per TB scanned
    clickhouse_query_cost: float = 0.0005  # per query
    clickhouse_gb_stored_cost: float = 0.02  # per GB per month
    minio_gb_stored_cost: float = 0.01  # per GB per month
    minio_gb_transferred_cost: float = 0.05  # per GB transferred
    spark_vcpu_hour_cost: float = 0.10  # per vCPU hour


class CostAttributionCollector:
    """Collect and attribute costs across data platform"""
    
    def __init__(self, config_path: str = "config/cost-config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.cost_config = CostConfig(**config.get('costs', {}))
        self.team_mapping = config.get('team_mapping', {})
        
        # Initialize clients
        self.ch_client = ClickHouseClient(host='clickhouse-service')
        self.trino_conn = trino_connect(
            host='trino-coordinator',
            port=8080,
            user='admin'
        )
        self.s3_client = boto3.client('s3', endpoint_url='http://minio-service:9000')
        
        logger.info("Cost Attribution Collector initialized")
    
    def collect_trino_usage(self, start_time: datetime, end_time: datetime) -> List[UsageMetric]:
        """Collect Trino query usage and costs"""
        query = """
        SELECT
            user,
            source,
            query_type,
            state,
            query,
            data_scanned_bytes,
            execution_time_ms,
            created,
            catalog,
            schema
        FROM system.runtime.queries
        WHERE created >= timestamp '{start_time}'
        AND created < timestamp '{end_time}'
        AND state = 'FINISHED'
        """
        
        cursor = self.trino_conn.cursor()
        cursor.execute(query.format(
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat()
        ))
        
        metrics = []
        for row in cursor.fetchall():
            user = row[0]
            bytes_scanned = row[5] or 0
            execution_time = row[6] or 0
            catalog = row[8]
            schema = row[9]
            
            # Calculate cost
            query_cost = self.cost_config.trino_query_cost
            tb_scanned = bytes_scanned / (1024 ** 4)
            scan_cost = tb_scanned * self.cost_config.trino_tb_scanned_cost
            total_cost = query_cost + scan_cost
            
            metric = UsageMetric(
                service='trino',
                user=user,
                team=self.team_mapping.get(user),
                dataset=f"{catalog}.{schema}",
                query_count=1,
                bytes_scanned=bytes_scanned,
                execution_time_ms=execution_time,
                cost_usd=total_cost,
                timestamp=datetime.utcnow()
            )
            metrics.append(metric)
            
            # Update Prometheus metrics
            queries_executed.labels(service='trino', user=user).inc()
            bytes_scanned.labels(service='trino', dataset=f"{catalog}.{schema}").inc(bytes_scanned)
        
        logger.info(f"Collected {len(metrics)} Trino usage metrics")
        return metrics
    
    def collect_clickhouse_usage(self, start_time: datetime, end_time: datetime) -> List[UsageMetric]:
        """Collect ClickHouse query usage and costs"""
        query = """
        SELECT
            user,
            query_kind,
            databases,
            tables,
            read_bytes,
            query_duration_ms,
            event_time
        FROM system.query_log
        WHERE 
            event_time >= toDateTime('{start_time}')
            AND event_time < toDateTime('{end_time}')
            AND type = 'QueryFinish'
            AND query_kind != 'System'
        """
        
        results = self.ch_client.execute(query.format(
            start_time=start_time.strftime('%Y-%m-%d %H:%M:%S'),
            end_time=end_time.strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        metrics = []
        for row in results:
            user = row[0]
            databases = row[2]
            tables = row[3]
            read_bytes = row[4]
            duration_ms = row[5]
            
            dataset = f"{databases[0]}.{tables[0]}" if databases and tables else "unknown"
            
            # Calculate cost
            cost = self.cost_config.clickhouse_query_cost
            
            metric = UsageMetric(
                service='clickhouse',
                user=user,
                team=self.team_mapping.get(user),
                dataset=dataset,
                query_count=1,
                bytes_scanned=read_bytes,
                execution_time_ms=duration_ms,
                cost_usd=cost,
                timestamp=datetime.utcnow()
            )
            metrics.append(metric)
            
            # Update Prometheus metrics
            queries_executed.labels(service='clickhouse', user=user).inc()
            bytes_scanned.labels(service='clickhouse', dataset=dataset).inc(read_bytes)
        
        logger.info(f"Collected {len(metrics)} ClickHouse usage metrics")
        return metrics
    
    def collect_storage_costs(self) -> List[UsageMetric]:
        """Collect storage costs from MinIO/S3"""
        metrics = []
        
        # Get bucket sizes
        buckets = self.s3_client.list_buckets()['Buckets']
        
        for bucket in buckets:
            bucket_name = bucket['Name']
            
            # Calculate total size
            total_bytes = 0
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name):
                if 'Contents' in page:
                    total_bytes += sum(obj['Size'] for obj in page['Contents'])
            
            gb_stored = total_bytes / (1024 ** 3)
            cost = gb_stored * self.cost_config.minio_gb_stored_cost / 30  # Daily cost
            
            metric = UsageMetric(
                service='minio',
                user='system',
                team=None,
                dataset=bucket_name,
                query_count=0,
                bytes_scanned=total_bytes,
                execution_time_ms=0,
                cost_usd=cost,
                timestamp=datetime.utcnow()
            )
            metrics.append(metric)
        
        logger.info(f"Collected storage costs for {len(buckets)} buckets")
        return metrics
    
    def aggregate_by_user(self, metrics: List[UsageMetric]) -> pd.DataFrame:
        """Aggregate metrics by user"""
        df = pd.DataFrame([vars(m) for m in metrics])
        
        agg = df.groupby(['service', 'user']).agg({
            'query_count': 'sum',
            'bytes_scanned': 'sum',
            'execution_time_ms': 'sum',
            'cost_usd': 'sum'
        }).reset_index()
        
        # Update Prometheus metrics
        for _, row in agg.iterrows():
            cost_by_user.labels(
                user=row['user'],
                service=row['service']
            ).set(row['cost_usd'])
        
        return agg
    
    def aggregate_by_dataset(self, metrics: List[UsageMetric]) -> pd.DataFrame:
        """Aggregate metrics by dataset"""
        df = pd.DataFrame([vars(m) for m in metrics])
        df = df[df['dataset'].notna()]
        
        agg = df.groupby(['service', 'dataset']).agg({
            'query_count': 'sum',
            'bytes_scanned': 'sum',
            'cost_usd': 'sum'
        }).reset_index()
        
        # Update Prometheus metrics
        for _, row in agg.iterrows():
            cost_by_dataset.labels(
                dataset=row['dataset'],
                service=row['service']
            ).set(row['cost_usd'])
        
        return agg
    
    def aggregate_by_team(self, metrics: List[UsageMetric]) -> pd.DataFrame:
        """Aggregate metrics by team"""
        df = pd.DataFrame([vars(m) for m in metrics])
        df = df[df['team'].notna()]
        
        agg = df.groupby(['service', 'team']).agg({
            'query_count': 'sum',
            'bytes_scanned': 'sum',
            'cost_usd': 'sum'
        }).reset_index()
        
        # Update Prometheus metrics
        for _, row in agg.iterrows():
            cost_by_team.labels(
                team=row['team'],
                service=row['service']
            ).set(row['cost_usd'])
        
        return agg
    
    def generate_report(self, start_time: datetime, end_time: datetime) -> Dict:
        """Generate comprehensive cost report"""
        logger.info(f"Generating cost report for {start_time} to {end_time}")
        
        # Collect metrics from all services
        all_metrics = []
        all_metrics.extend(self.collect_trino_usage(start_time, end_time))
        all_metrics.extend(self.collect_clickhouse_usage(start_time, end_time))
        all_metrics.extend(self.collect_storage_costs())
        
        # Generate aggregations
        by_user = self.aggregate_by_user(all_metrics)
        by_dataset = self.aggregate_by_dataset(all_metrics)
        by_team = self.aggregate_by_team(all_metrics)
        
        total_cost = sum(m.cost_usd for m in all_metrics)
        
        report = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'summary': {
                'total_cost_usd': total_cost,
                'total_queries': sum(m.query_count for m in all_metrics),
                'total_bytes_scanned': sum(m.bytes_scanned for m in all_metrics)
            },
            'by_user': by_user.to_dict('records'),
            'by_dataset': by_dataset.to_dict('records'),
            'by_team': by_team.to_dict('records'),
            'top_expensive_users': by_user.nlargest(10, 'cost_usd').to_dict('records'),
            'top_expensive_datasets': by_dataset.nlargest(10, 'cost_usd').to_dict('records')
        }
        
        logger.info(f"Generated report: ${total_cost:.2f} total cost")
        return report


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    
    collector = CostAttributionCollector()
    
    # Generate daily report
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)
    
    report = collector.generate_report(start_time, end_time)
    
    print(json.dumps(report, indent=2))
