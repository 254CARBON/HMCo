# 254Carbon Platform SDKs

Client libraries for interacting with the 254Carbon data platform.

## Available SDKs

### Python SDK
**Location**: `sdk/python/`  
**PyPI Package**: `carbon254-platform`  
Documentation: See `sdk/python/` (docs coming soon)

```python
from carbon254 import PlatformClient

client = PlatformClient(api_url="https://portal.254carbon.com/api")

# Query commodity data
data = client.commodities.get_prices("crude_oil", days=30)

# Submit workflow
workflow = client.workflows.submit("daily-ingestion")

# Get ML predictions
prediction = client.ml.predict("price-predictor", features)
```

### Java SDK
**Location**: `sdk/java/`  
**Maven Package**: `com.carbon254:platform-sdk`  
Documentation: See `sdk/java/` (docs coming soon)

```java
import com.carbon254.PlatformClient;

PlatformClient client = new PlatformClient("https://portal.254carbon.com/api");

// Query data
CommodityData data = client.commodities().getPrices("crude_oil", 30);

// Submit workflow
Workflow workflow = client.workflows().submit("daily-ingestion");
```

### Node.js SDK
**Location**: `sdk/nodejs/`  
**NPM Package**: `@carbon254/platform-sdk`  
Documentation: See `sdk/nodejs/` (docs coming soon)

```javascript
const { PlatformClient } = require('@carbon254/platform-sdk');

const client = new PlatformClient('https://portal.254carbon.com/api');

// Query data
const data = await client.commodities.getPrices('crude_oil', { days: 30 });

// Submit workflow
const workflow = await client.workflows.submit('daily-ingestion');
```

## Features

All SDKs provide:
- ✅ Commodity data access (Trino queries)
- ✅ Workflow management (DolphinScheduler API)
- ✅ ML model serving (Ray Serve)
- ✅ Feature serving (Feast)
- ✅ Metadata search (DataHub)
- ✅ Metrics and monitoring
- ✅ Event production (Kafka)
- ✅ Authentication (API keys, JWT)

## Installation

### Python
```bash
pip install carbon254-platform
# or from source
cd sdk/python && pip install -e .
```

### Java
```xml
<dependency>
    <groupId>com.carbon254</groupId>
    <artifactId>platform-sdk</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Node.js
```bash
npm install @carbon254/platform-sdk
# or from source
cd sdk/nodejs && npm install
```

## Quick Start

### Configuration
```python
# Python
from carbon254 import PlatformClient

client = PlatformClient(
    api_url="https://portal.254carbon.com/api",
    api_key="your-api-key"
)
```

### Query Commodity Data
```python
# Get latest prices
prices = client.commodities.get_latest_prices(["crude_oil", "natural_gas"])

# Historical data
history = client.commodities.get_historical(
    commodity="crude_oil",
    start_date="2025-01-01",
    end_date="2025-10-22"
)

# Analytics
volatility = client.analytics.calculate_volatility("crude_oil", window=30)
```

### Run Workflows
```python
# List workflows
workflows = client.workflows.list()

# Submit workflow
run = client.workflows.submit(
    workflow_name="daily-market-ingestion",
    parameters={"date": "2025-10-22"}
)

# Check status
status = client.workflows.get_status(run.id)
```

### ML Predictions
```python
# Get prediction
features = {
    "price_ma_7": 85.0,
    "price_ma_30": 84.5,
    "volatility": 2.3,
    "day_of_week": 1
}

prediction = client.ml.predict(
    model="commodity-price-predictor",
    features=features
)
```

### Feature Serving
```python
# Get online features
features = client.features.get_online_features(
    feature_view="commodity_features",
    entities={"commodity": "crude_oil"}
)
```

## Authentication

### API Key
```python
client = PlatformClient(
    api_url="https://portal.254carbon.com/api",
    api_key="your-api-key"
)
```

### JWT Token
```python
client = PlatformClient(
    api_url="https://portal.254carbon.com/api",
    jwt_token="your-jwt-token"
)
```

### Service Account
```python
client = PlatformClient(
    api_url="https://portal.254carbon.com/api",
    service_account="sa-name",
    service_account_key_file="/path/to/key.json"
)
```

## Examples

See `examples/` directory for complete examples:
- `examples/commodity-analysis.py` - Data analysis workflow
- `examples/ml-training.py` - Model training pipeline
- `examples/realtime-streaming.py` - Real-time data processing
- `examples/dashboard-integration.py` - Dashboard data integration

## Development

### Building from Source

**Python**:
```bash
cd sdk/python
pip install -e ".[dev]"
pytest tests/
```

**Java**:
```bash
cd sdk/java
mvn clean install
mvn test
```

**Node.js**:
```bash
cd sdk/nodejs
npm install
npm test
```

## Documentation

Full API documentation:
- Python: https://docs.254carbon.com/sdk/python
- Java: https://docs.254carbon.com/sdk/java
- Node.js: https://docs.254carbon.com/sdk/nodejs

## Support

- GitHub Issues: https://github.com/254carbon/platform-sdk
- Documentation: https://docs.254carbon.com
- Email: support@254carbon.com

---

**Version**: 1.0.0  
**Last Updated**: October 22, 2025  
**License**: Internal Use - 254Carbon


