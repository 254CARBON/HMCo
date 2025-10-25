/**
 * 254Carbon Event Producer Library (Node.js)
 * Simplified event production for all services
 */

const fs = require('fs');
const { Kafka } = require('kafkajs');
const { v4: uuidv4 } = require('uuid');

/**
 * Event types enum
 */
const EventType = {
  DATA_INGESTION: 'DataIngestionEvent',
  DATA_QUALITY: 'DataQualityEvent',
  DATA_LINEAGE: 'DataLineageEvent',
  DATA_TRANSFORMATION: 'DataTransformationEvent',
  SERVICE_HEALTH: 'ServiceHealthEvent',
  DEPLOYMENT: 'DeploymentEvent',
  CONFIG_CHANGE: 'ConfigChangeEvent',
  SECURITY: 'SecurityEvent',
  USER_ACTION: 'AuditUserActionEvent',
  API_CALL: 'AuditAPICallEvent',
  DATA_ACCESS: 'AuditDataAccessEvent',
  ADMIN_OPERATION: 'AuditAdminOperationEvent'
};

/**
 * Kafka topics enum
 */
const Topic = {
  DATA_INGESTION: 'data-ingestion',
  DATA_QUALITY: 'data-quality',
  DATA_LINEAGE: 'data-lineage',
  DATA_TRANSFORMATION: 'data-transformation',
  SYSTEM_HEALTH: 'system-health',
  DEPLOYMENT_EVENTS: 'deployment-events',
  CONFIG_CHANGES: 'config-changes',
  SECURITY_EVENTS: 'security-events',
  AUDIT_USER_ACTIONS: 'audit-user-actions',
  AUDIT_API_CALLS: 'audit-api-calls',
  AUDIT_DATA_ACCESS: 'audit-data-access',
  AUDIT_ADMIN_OPS: 'audit-admin-operations'
};

function readFileOrThrow(filePath, label) {
  if (!filePath) {
    throw new Error(`Missing TLS ${label} path. Set KAFKA_SSL_${label.toUpperCase()}_LOCATION.`);
  }
  try {
    return fs.readFileSync(filePath, 'utf-8');
  } catch (err) {
    throw new Error(`Failed to read TLS ${label} at ${filePath}: ${err.message}`);
  }
}

function buildDefaultTlsConfig(env = process.env) {
  const protocol = (env.KAFKA_SECURITY_PROTOCOL || 'SSL').toUpperCase();
  if (protocol !== 'SSL' && protocol !== 'SASL_SSL') {
    return null;
  }

  const caPath = env.KAFKA_SSL_CA_LOCATION;
  const certPath = env.KAFKA_SSL_CERTIFICATE_LOCATION;
  const keyPath = env.KAFKA_SSL_KEY_LOCATION;

  if (!caPath || !certPath || !keyPath) {
    return null;
  }

  const tlsConfig = {
    ca: [readFileOrThrow(caPath, 'CA')],
    cert: readFileOrThrow(certPath, 'CERTIFICATE'),
    key: readFileOrThrow(keyPath, 'KEY'),
  };

  if (env.KAFKA_SSL_KEY_PASSWORD) {
    tlsConfig.passphrase = env.KAFKA_SSL_KEY_PASSWORD;
  }

  const rejectUnauthorized = env.KAFKA_SSL_REJECT_UNAUTHORIZED;
  if (rejectUnauthorized && rejectUnauthorized.toLowerCase() === 'false') {
    tlsConfig.rejectUnauthorized = false;
  }

  return tlsConfig;
}

/**
 * 254Carbon Event Producer
 * 
 * Simplifies event production with automatic:
 * - Event ID generation
 * - Timestamp handling
 * - Error handling and retries
 * - Delivery tracking
 */
class EventProducer {
  constructor(options = {}) {
    const {
      bootstrapServers = process.env.KAFKA_BOOTSTRAP_SERVERS || 'kafka-service.data-platform.svc.cluster.local:9093',
      sourceService = 'unknown-service',
      clientId = `${sourceService}-producer`,
      tls = undefined
    } = options;

    this.sourceService = sourceService;
    this.deliveryCount = 0;
    this.errorCount = 0;

    const brokers = Array.isArray(bootstrapServers)
      ? bootstrapServers
      : String(bootstrapServers)
          .split(',')
          .map(broker => broker.trim())
          .filter(Boolean);

    const tlsConfig = tls !== undefined ? tls : buildDefaultTlsConfig();
    if (!tlsConfig && (process.env.KAFKA_SECURITY_PROTOCOL || 'SSL').toUpperCase() === 'SSL') {
      console.warn('[EventProducer] TLS is enabled but no client certificates were provided; relying on default trust store.');
    }

    // Initialize Kafka client
    this.kafka = new Kafka({
      clientId,
      brokers,
      compression: 'GZIP',
      retry: {
        initialRetryTime: 100,
        retries: 8
      },
      ssl: tlsConfig || undefined
    });

    this.producer = this.kafka.producer({
      maxInFlightRequests: 5,
      idempotent: true,
      transactionalId: `${sourceService}-txn`
    });

    this.connected = false;
    console.log(`EventProducer initialized for service: ${sourceService}`);
  }

  /**
   * Connect to Kafka
   */
  async connect() {
    if (!this.connected) {
      await this.producer.connect();
      this.connected = true;
      console.log('EventProducer connected to Kafka');
    }
  }

  /**
   * Disconnect from Kafka
   */
  async disconnect() {
    if (this.connected) {
      await this.producer.disconnect();
      this.connected = false;
      console.log(`EventProducer disconnected. Delivered: ${this.deliveryCount}, Errors: ${this.errorCount}`);
    }
  }

  /**
   * Create base event structure
   */
  _createBaseEvent(eventType, additionalFields = {}) {
    return {
      eventId: uuidv4(),
      eventType,
      timestamp: Date.now(),
      source: this.sourceService,
      version: '1.0.0',
      ...additionalFields
    };
  }

  /**
   * Internal method to produce event
   */
  async _produce(topic, key, event) {
    try {
      await this.producer.send({
        topic,
        messages: [
          {
            key,
            value: JSON.stringify(event),
          },
        ],
      });
      this.deliveryCount++;
    } catch (error) {
      this.errorCount++;
      console.error(`Error producing event to ${topic}:`, error);
      throw error;
    }
  }

  /**
   * Produce data ingestion event
   * 
   * @example
   * await producer.produceDataIngestionEvent({
   *   datasetName: 'commodity_prices',
   *   recordCount: 10000,
   *   sizeBytes: 1024000,
   *   location: 's3://bucket/data.parquet',
   *   status: 'SUCCESS'
   * });
   */
  async produceDataIngestionEvent({
    datasetName,
    recordCount,
    sizeBytes,
    location,
    status = 'SUCCESS',
    format = 'parquet',
    metadata = {}
  }) {
    const event = this._createBaseEvent(EventType.DATA_INGESTION, {
      datasetName,
      recordCount,
      sizeBytes,
      format,
      location,
      status,
      metadata
    });

    await this._produce(Topic.DATA_INGESTION, datasetName, event);
  }

  /**
   * Produce service health event
   * 
   * @example
   * await producer.produceServiceHealthEvent({
   *   serviceName: 'datahub-gms',
   *   namespace: 'data-platform',
   *   healthStatus: 'HEALTHY',
   *   latencyMs: 45,
   *   errorRate: 0.001
   * });
   */
  async produceServiceHealthEvent({
    serviceName,
    namespace,
    healthStatus,
    latencyMs,
    errorRate,
    message = null
  }) {
    const event = this._createBaseEvent(EventType.SERVICE_HEALTH, {
      serviceName,
      namespace,
      healthStatus,
      latencyMs,
      errorRate,
      message
    });

    await this._produce(Topic.SYSTEM_HEALTH, serviceName, event);
  }

  /**
   * Produce API call audit event
   * 
   * @example
   * await producer.produceAPICallEvent({
   *   service: 'datahub-gms',
   *   endpoint: '/api/v2/entity',
   *   method: 'GET',
   *   statusCode: 200,
   *   latencyMs: 123,
   *   requestSize: 256,
   *   responseSize: 4096,
   *   userId: 'user-123',
   *   ipAddress: '10.0.1.45'
   * });
   */
  async produceAPICallEvent({
    service,
    endpoint,
    method,
    statusCode,
    latencyMs,
    requestSize,
    responseSize,
    userId = null,
    apiKey = null,
    ipAddress = 'unknown',
    errorMessage = null
  }) {
    const event = this._createBaseEvent(EventType.API_CALL, {
      service,
      endpoint,
      method,
      statusCode,
      latencyMs,
      requestSizeBytes: requestSize,
      responseSizeBytes: responseSize,
      userId,
      apiKey,
      ipAddress,
      errorMessage
    });

    await this._produce(Topic.AUDIT_API_CALLS, service, event);
  }

  /**
   * Produce data quality event
   * 
   * @example
   * await producer.produceDataQualityEvent({
   *   datasetName: 'commodity_prices',
   *   checkType: 'completeness',
   *   checkName: 'null_check',
   *   result: 'PASS',
   *   score: 0.99,
   *   failedRecords: 10,
   *   totalRecords: 10000
   * });
   */
  async produceDataQualityEvent({
    datasetName,
    checkType,
    checkName,
    result,
    score,
    failedRecords,
    totalRecords,
    message = null
  }) {
    const event = this._createBaseEvent(EventType.DATA_QUALITY, {
      datasetName,
      checkType,
      checkName,
      result,
      score,
      failedRecords,
      totalRecords,
      message
    });

    await this._produce(Topic.DATA_QUALITY, datasetName, event);
  }

  /**
   * Produce custom event
   * 
   * @param {string} topic - Kafka topic
   * @param {string} key - Message key
   * @param {object} event - Event payload
   */
  async produceCustomEvent(topic, key, event) {
    await this._produce(topic, key, event);
  }

  /**
   * Get producer statistics
   */
  getStats() {
    return {
      delivered: this.deliveryCount,
      errors: this.errorCount
    };
  }
}

/**
 * Convenience function for quick integration
 * 
 * @example
 * const producer = getEventProducer('my-service');
 * await producer.connect();
 * await producer.produceServiceHealthEvent({...});
 * await producer.disconnect();
 */
function getEventProducer(serviceName, options = {}) {
  return new EventProducer({
    sourceService: serviceName,
    ...options
  });
}

module.exports = {
  EventProducer,
  getEventProducer,
  EventType,
  Topic
};

// Example usage
if (require.main === module) {
  (async () => {
    const producer = getEventProducer('example-service');
    
    try {
      await producer.connect();
      
      // Example 1: Data ingestion event
      await producer.produceDataIngestionEvent({
        datasetName: 'test_dataset',
        recordCount: 1000,
        sizeBytes: 50000,
        location: 's3://test-bucket/data.parquet',
        status: 'SUCCESS'
      });
      console.log('✅ Data ingestion event produced');
      
      // Example 2: Service health event
      await producer.produceServiceHealthEvent({
        serviceName: 'example-service',
        namespace: 'data-platform',
        healthStatus: 'HEALTHY',
        latencyMs: 25,
        errorRate: 0.0
      });
      console.log('✅ Service health event produced');
      
      // Example 3: API call event
      await producer.produceAPICallEvent({
        service: 'example-service',
        endpoint: '/api/data',
        method: 'GET',
        statusCode: 200,
        latencyMs: 50,
        requestSize: 256,
        responseSize: 2048,
        userId: 'user-123',
        ipAddress: '10.0.1.100'
      });
      console.log('✅ API call event produced');
      
      console.log(`\nStats: ${JSON.stringify(producer.getStats())}`);
      
    } finally {
      await producer.disconnect();
    }
  })();
}

