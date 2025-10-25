"""
254Carbon Event Producer Library
Simplified event production for all services
"""

import json
import uuid
import time
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import logging

try:
    from confluent_kafka import Producer
    from confluent_kafka.schema_registry import SchemaRegistryClient
    from confluent_kafka.schema_registry.avro import AvroSerializer
    from confluent_kafka.serialization import StringSerializer, SerializationContext, MessageField
except ImportError:
    print("Warning: confluent_kafka not installed. Install with: pip install confluent-kafka[avro]")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types"""
    DATA_INGESTION = "DataIngestionEvent"
    DATA_QUALITY = "DataQualityEvent"
    DATA_LINEAGE = "DataLineageEvent"
    DATA_TRANSFORMATION = "DataTransformationEvent"
    SERVICE_HEALTH = "ServiceHealthEvent"
    DEPLOYMENT = "DeploymentEvent"
    CONFIG_CHANGE = "ConfigChangeEvent"
    SECURITY = "SecurityEvent"
    USER_ACTION = "AuditUserActionEvent"
    API_CALL = "AuditAPICallEvent"
    DATA_ACCESS = "AuditDataAccessEvent"
    ADMIN_OPERATION = "AuditAdminOperationEvent"


class Topic(Enum):
    """Kafka topics"""
    DATA_INGESTION = "data-ingestion"
    DATA_QUALITY = "data-quality"
    DATA_LINEAGE = "data-lineage"
    DATA_TRANSFORMATION = "data-transformation"
    SYSTEM_HEALTH = "system-health"
    DEPLOYMENT_EVENTS = "deployment-events"
    CONFIG_CHANGES = "config-changes"
    SECURITY_EVENTS = "security-events"
    AUDIT_USER_ACTIONS = "audit-user-actions"
    AUDIT_API_CALLS = "audit-api-calls"
    AUDIT_DATA_ACCESS = "audit-data-access"
    AUDIT_ADMIN_OPS = "audit-admin-operations"


@dataclass
class BaseEvent:
    """Base event structure"""
    event_id: str
    event_type: str
    timestamp: int
    source: str
    version: str = "1.0.0"
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DataIngestionEvent(BaseEvent):
    """Data ingestion event"""
    dataset_name: str
    record_count: int
    size_bytes: int
    format: str
    location: str
    status: str  # SUCCESS, FAILURE, PARTIAL
    metadata: Dict[str, str] = None


@dataclass
class ServiceHealthEvent(BaseEvent):
    """Service health event"""
    service_name: str
    namespace: str
    health_status: str  # HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN
    latency_ms: int
    error_rate: float
    message: Optional[str] = None


@dataclass
class AuditAPICallEvent(BaseEvent):
    """API call audit event"""
    service: str
    endpoint: str
    method: str
    status_code: int
    latency_ms: int
    request_size_bytes: int
    response_size_bytes: int
    user_id: Optional[str] = None
    api_key: Optional[str] = None
    ip_address: str = ""
    error_message: Optional[str] = None


class EventProducer:
    """
    254Carbon Event Producer
    
    Simplifies event production for all services with:
    - Automatic event ID generation
    - Timestamp handling
    - Error handling and retries
    - Delivery callbacks
    - Metrics tracking
    """
    
    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        schema_registry_url: Optional[str] = None,
        source_service: str = "unknown-service",
        security_protocol: Optional[str] = None,
        ssl_ca_location: Optional[str] = None,
        ssl_certificate_location: Optional[str] = None,
        ssl_key_location: Optional[str] = None,
        ssl_key_password: Optional[str] = None,
        additional_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize event producer
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            schema_registry_url: Schema Registry URL (optional)
            source_service: Name of the service producing events
            security_protocol: Kafka security protocol (defaults to env or SSL)
            ssl_ca_location: Path to CA certificate file
            ssl_certificate_location: Path to client certificate file
            ssl_key_location: Path to client key file
            ssl_key_password: Password for client key (if encrypted)
            additional_config: Extra configuration overrides for the producer
        """
        self.source_service = source_service
        self.delivery_count = 0
        self.error_count = 0

        resolved_bootstrap = bootstrap_servers or os.getenv(
            'KAFKA_BOOTSTRAP_SERVERS',
            'kafka-service.data-platform.svc.cluster.local:9093'
        )
        
        # Kafka producer configuration
        conf = {
            'bootstrap.servers': resolved_bootstrap,
            'client.id': f'{source_service}-producer',
            'compression.type': 'lz4',
            'linger.ms': 10,
            'batch.size': 16384,
            'acks': 'all',
            'retries': 3,
            'max.in.flight.requests.per.connection': 5,
            'enable.idempotence': True
        }

        resolved_security_protocol = (security_protocol or os.getenv(
            'KAFKA_SECURITY_PROTOCOL',
            'SSL'
        )).upper()

        if resolved_security_protocol:
            conf['security.protocol'] = resolved_security_protocol

        resolved_ssl_ca = ssl_ca_location or os.getenv('KAFKA_SSL_CA_LOCATION')
        resolved_ssl_cert = ssl_certificate_location or os.getenv('KAFKA_SSL_CERTIFICATE_LOCATION')
        resolved_ssl_key = ssl_key_location or os.getenv('KAFKA_SSL_KEY_LOCATION')
        resolved_ssl_key_password = ssl_key_password or os.getenv('KAFKA_SSL_KEY_PASSWORD')

        resolved_ssl_keystore = os.getenv('KAFKA_SSL_KEYSTORE_LOCATION')
        resolved_ssl_keystore_password = os.getenv('KAFKA_SSL_KEYSTORE_PASSWORD')

        if resolved_security_protocol in {'SSL', 'SASL_SSL'}:
            if resolved_ssl_ca:
                conf['ssl.ca.location'] = resolved_ssl_ca
            if resolved_ssl_cert:
                conf['ssl.certificate.location'] = resolved_ssl_cert
            if resolved_ssl_key:
                conf['ssl.key.location'] = resolved_ssl_key
            if resolved_ssl_key_password:
                conf['ssl.key.password'] = resolved_ssl_key_password
            if resolved_ssl_keystore:
                conf['ssl.keystore.location'] = resolved_ssl_keystore
            if resolved_ssl_keystore_password:
                conf['ssl.keystore.password'] = resolved_ssl_keystore_password

        if additional_config:
            conf.update(additional_config)
        
        self.producer = Producer(conf)
        self.string_serializer = StringSerializer('utf_8')
        
        # Schema Registry (optional)
        if schema_registry_url:
            self.schema_registry = SchemaRegistryClient({'url': schema_registry_url})
        else:
            self.schema_registry = None
        
        logger.info(
            "EventProducer initialized for service %s (bootstrap=%s, protocol=%s)",
            source_service,
            resolved_bootstrap,
            conf.get('security.protocol', 'PLAINTEXT')
        )
    
    def _delivery_callback(self, err, msg):
        """Delivery report callback"""
        if err:
            self.error_count += 1
            logger.error(f'Message delivery failed: {err}')
        else:
            self.delivery_count += 1
            logger.debug(f'Message delivered to {msg.topic()} [{msg.partition()}] @ offset {msg.offset()}')
    
    def _create_base_event(self, event_type: EventType, **kwargs) -> Dict[str, Any]:
        """Create base event structure"""
        return {
            'eventId': str(uuid.uuid4()),
            'eventType': event_type.value,
            'timestamp': int(time.time() * 1000),
            'source': self.source_service,
            'version': '1.0.0',
            **kwargs
        }
    
    def produce_data_ingestion_event(
        self,
        dataset_name: str,
        record_count: int,
        size_bytes: int,
        location: str,
        status: str = "SUCCESS",
        format: str = "parquet",
        metadata: Optional[Dict[str, str]] = None
    ):
        """
        Produce data ingestion event
        
        Example:
            producer.produce_data_ingestion_event(
                dataset_name="commodity_prices",
                record_count=10000,
                size_bytes=1024000,
                location="s3://bucket/data/commodity_prices.parquet",
                status="SUCCESS"
            )
        """
        event = self._create_base_event(
            EventType.DATA_INGESTION,
            datasetName=dataset_name,
            recordCount=record_count,
            sizeBytes=size_bytes,
            format=format,
            location=location,
            status=status,
            metadata=metadata or {}
        )
        
        self._produce(Topic.DATA_INGESTION.value, dataset_name, event)
    
    def produce_service_health_event(
        self,
        service_name: str,
        namespace: str,
        health_status: str,
        latency_ms: int,
        error_rate: float,
        message: Optional[str] = None
    ):
        """
        Produce service health event
        
        Example:
            producer.produce_service_health_event(
                service_name="datahub-gms",
                namespace="data-platform",
                health_status="HEALTHY",
                latency_ms=45,
                error_rate=0.001
            )
        """
        event = self._create_base_event(
            EventType.SERVICE_HEALTH,
            serviceName=service_name,
            namespace=namespace,
            healthStatus=health_status,
            latencyMs=latency_ms,
            errorRate=error_rate,
            message=message
        )
        
        self._produce(Topic.SYSTEM_HEALTH.value, service_name, event)
    
    def produce_api_call_event(
        self,
        service: str,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: int,
        request_size: int,
        response_size: int,
        user_id: Optional[str] = None,
        ip_address: str = "unknown",
        error_message: Optional[str] = None
    ):
        """
        Produce API call audit event
        
        Example:
            producer.produce_api_call_event(
                service="datahub-gms",
                endpoint="/api/v2/entity",
                method="GET",
                status_code=200,
                latency_ms=123,
                request_size=256,
                response_size=4096,
                user_id="user-123",
                ip_address="10.0.1.45"
            )
        """
        event = self._create_base_event(
            EventType.API_CALL,
            service=service,
            endpoint=endpoint,
            method=method,
            statusCode=status_code,
            latencyMs=latency_ms,
            requestSizeBytes=request_size,
            responseSizeBytes=response_size,
            userId=user_id,
            ipAddress=ip_address,
            errorMessage=error_message
        )
        
        self._produce(Topic.AUDIT_API_CALLS.value, service, event)
    
    def produce_data_quality_event(
        self,
        dataset_name: str,
        check_type: str,
        check_name: str,
        result: str,
        score: float,
        failed_records: int,
        total_records: int,
        message: Optional[str] = None
    ):
        """
        Produce data quality event
        
        Example:
            producer.produce_data_quality_event(
                dataset_name="commodity_prices",
                check_type="completeness",
                check_name="null_check",
                result="PASS",
                score=0.99,
                failed_records=10,
                total_records=10000
            )
        """
        event = self._create_base_event(
            EventType.DATA_QUALITY,
            datasetName=dataset_name,
            checkType=check_type,
            checkName=check_name,
            result=result,
            score=score,
            failedRecords=failed_records,
            totalRecords=total_records,
            message=message
        )
        
        self._produce(Topic.DATA_QUALITY.value, dataset_name, event)
    
    def produce_custom_event(
        self,
        topic: str,
        key: str,
        event: Dict[str, Any]
    ):
        """
        Produce custom event
        
        Args:
            topic: Kafka topic name
            key: Message key (for partitioning)
            event: Event payload (dict)
        """
        self._produce(topic, key, event)
    
    def _produce(self, topic: str, key: str, event: Dict[str, Any]):
        """Internal method to produce event"""
        try:
            self.producer.produce(
                topic=topic,
                key=self.string_serializer(key),
                value=json.dumps(event).encode('utf-8'),
                on_delivery=self._delivery_callback
            )
            self.producer.poll(0)  # Trigger delivery callbacks
            
        except Exception as e:
            logger.error(f"Error producing event to {topic}: {e}")
            self.error_count += 1
            raise
    
    def flush(self, timeout: float = 10.0):
        """
        Flush pending events
        
        Args:
            timeout: Timeout in seconds
        """
        remaining = self.producer.flush(timeout)
        if remaining > 0:
            logger.warning(f"{remaining} messages were not delivered")
    
    def close(self):
        """Close producer and flush pending events"""
        self.flush()
        logger.info(f"EventProducer closed. Delivered: {self.delivery_count}, Errors: {self.error_count}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get producer statistics"""
        return {
            'delivered': self.delivery_count,
            'errors': self.error_count
        }
    
    def __enter__(self):
        """Context manager enter"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Convenience function for quick integration
def get_event_producer(service_name: str) -> EventProducer:
    """
    Get event producer instance
    
    Example:
        producer = get_event_producer("my-service")
        producer.produce_service_health_event(...)
        producer.close()
    """
    return EventProducer(source_service=service_name)


# Example usage
if __name__ == "__main__":
    # Example 1: Using context manager
    with get_event_producer("example-service") as producer:
        producer.produce_data_ingestion_event(
            dataset_name="test_dataset",
            record_count=1000,
            size_bytes=50000,
            location="s3://test-bucket/data.parquet",
            status="SUCCESS"
        )
    
    # Example 2: Manual management
    producer = get_event_producer("example-service")
    
    producer.produce_service_health_event(
        service_name="example-service",
        namespace="data-platform",
        health_status="HEALTHY",
        latency_ms=25,
        error_rate=0.0
    )
    
    producer.produce_api_call_event(
        service="example-service",
        endpoint="/api/data",
        method="GET",
        status_code=200,
        latency_ms=50,
        request_size=256,
        response_size=2048,
        user_id="user-123",
        ip_address="10.0.1.100"
    )
    
    producer.close()
    
    print(f"Stats: {producer.get_stats()}")

