# MVP Implementation Roadmap

## Current Status

### ✅ Completed (Phase 1-2)
- **Infrastructure**: Fixed DataHub, configured Kafka SSL
- **Portal UI**: Created Next.js pages for provider management and run monitoring
- **API Structure**: Built Next.js proxy routes for authentication
- **Components**: Reusable provider cards, navigation, and filtering

### 🚀 Next Priority: Build Backend API (Phase 3)

## Phase 3: Backend API Development

### 3.1 Setup Express Backend

Create `/backend` directory with Node.js/Express structure:

```bash
mkdir -p backend/src/{routes,models,services,middleware,config}
cd backend && npm init -y
npm install express pg cors dotenv axios
npm install -D typescript @types/express @types/node ts-node
```

### 3.2 Database Schema (PostgreSQL)

Create migrations for:

```sql
-- Providers table
CREATE TABLE providers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) NOT NULL,
  type VARCHAR(50) NOT NULL, -- 'api', 'database', 'file', 'stream'
  status VARCHAR(50) DEFAULT 'inactive',
  uis TEXT NOT NULL, -- JSON UIS specification
  config JSONB DEFAULT '{}',
  schedule VARCHAR(255), -- Cron expression
  last_run_at TIMESTAMP,
  next_run_at TIMESTAMP,
  total_runs INT DEFAULT 0,
  success_rate FLOAT DEFAULT 100.0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Runs table
CREATE TABLE runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  provider_id UUID NOT NULL REFERENCES providers(id),
  status VARCHAR(50) NOT NULL, -- 'running', 'success', 'failed', 'cancelled'
  started_at TIMESTAMP DEFAULT NOW(),
  completed_at TIMESTAMP,
  records_ingested INT DEFAULT 0,
  records_failed INT DEFAULT 0,
  duration INT, -- milliseconds
  logs TEXT,
  error_message TEXT,
  parameters JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_runs_provider_id ON runs(provider_id);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_created_at ON runs(created_at DESC);
```

### 3.3 Backend Service Implementation

#### Providers Service

```typescript
// backend/src/services/ProvidersService.ts
import { Client } from 'pg';

export class ProvidersService {
  constructor(private db: Client) {}

  async listProviders(
    status?: string,
    limit: number = 50,
    offset: number = 0
  ) {
    const query = `
      SELECT * FROM providers
      ${status ? 'WHERE status = $1' : ''}
      ORDER BY created_at DESC
      LIMIT $${status ? 2 : 1} OFFSET $${status ? 3 : 2}
    `;
    
    const params = status
      ? [status, limit, offset]
      : [limit, offset];
    
    const result = await this.db.query(query, params);
    const countResult = await this.db.query(
      `SELECT COUNT(*) FROM providers ${status ? 'WHERE status = $1' : ''}`,
      status ? [status] : []
    );

    return {
      providers: result.rows,
      total: parseInt(countResult.rows[0].count),
    };
  }

  async createProvider(data: {
    name: string;
    type: string;
    uis: string;
    config?: Record<string, any>;
    schedule?: string;
  }) {
    const result = await this.db.query(
      `INSERT INTO providers (name, type, uis, config, schedule)
       VALUES ($1, $2, $3, $4, $5)
       RETURNING *`,
      [data.name, data.type, data.uis, JSON.stringify(data.config || {}), data.schedule]
    );
    return result.rows[0];
  }

  async getProvider(id: string) {
    const result = await this.db.query(
      'SELECT * FROM providers WHERE id = $1',
      [id]
    );
    if (result.rows.length === 0) {
      throw new Error('Provider not found');
    }
    return result.rows[0];
  }

  async updateProvider(id: string, data: Partial<any>) {
    const updates = Object.entries(data)
      .map(([key, value], idx) => `${key} = $${idx + 1}`)
      .join(', ');
    
    const result = await this.db.query(
      `UPDATE providers SET ${updates}, updated_at = NOW()
       WHERE id = $${Object.keys(data).length + 1}
       RETURNING *`,
      [...Object.values(data), id]
    );
    return result.rows[0];
  }

  async deleteProvider(id: string) {
    // Delete related runs first
    await this.db.query('DELETE FROM runs WHERE provider_id = $1', [id]);
    await this.db.query('DELETE FROM providers WHERE id = $1', [id]);
  }
}
```

#### Runs Service

```typescript
// backend/src/services/RunsService.ts
export class RunsService {
  constructor(private db: Client) {}

  async listRuns(
    providerId?: string,
    status?: string,
    sortBy: string = 'created_at',
    sortOrder: string = 'DESC',
    limit: number = 50
  ) {
    let query = 'SELECT r.*, p.name as provider_name FROM runs r JOIN providers p ON r.provider_id = p.id WHERE 1=1';
    const params = [];
    let paramIdx = 1;

    if (providerId) {
      query += ` AND r.provider_id = $${paramIdx++}`;
      params.push(providerId);
    }
    if (status) {
      query += ` AND r.status = $${paramIdx++}`;
      params.push(status);
    }

    query += ` ORDER BY r.${sortBy} ${sortOrder} LIMIT $${paramIdx}`;
    params.push(limit);

    const result = await this.db.query(query, params);
    return { runs: result.rows };
  }

  async createRun(providerId: string, parameters?: Record<string, any>) {
    const result = await this.db.query(
      `INSERT INTO runs (provider_id, status, parameters)
       VALUES ($1, $2, $3)
       RETURNING *`,
      [providerId, 'running', JSON.stringify(parameters || {})]
    );
    return result.rows[0];
  }

  async getRun(id: string) {
    const result = await this.db.query(
      `SELECT r.*, p.name as provider_name FROM runs r
       JOIN providers p ON r.provider_id = p.id
       WHERE r.id = $1`,
      [id]
    );
    if (result.rows.length === 0) {
      throw new Error('Run not found');
    }
    return result.rows[0];
  }

  async updateRun(
    id: string,
    status: string,
    data?: {
      recordsIngested?: number;
      recordsFailed?: number;
      duration?: number;
      logs?: string;
      errorMessage?: string;
    }
  ) {
    const result = await this.db.query(
      `UPDATE runs SET
       status = $1,
       records_ingested = COALESCE($2, records_ingested),
       records_failed = COALESCE($3, records_failed),
       duration = COALESCE($4, duration),
       logs = COALESCE($5, logs),
       error_message = COALESCE($6, error_message),
       completed_at = CASE WHEN $1 IN ('success', 'failed', 'cancelled') THEN NOW() ELSE completed_at END
       WHERE id = $7
       RETURNING *`,
      [
        status,
        data?.recordsIngested,
        data?.recordsFailed,
        data?.duration,
        data?.logs,
        data?.errorMessage,
        id,
      ]
    );
    return result.rows[0];
  }
}
```

### 3.4 Express Routes

```typescript
// backend/src/routes/providers.ts
import express from 'express';
import { ProvidersService } from '../services/ProvidersService';

export function createProvidersRouter(providersService: ProvidersService) {
  const router = express.Router();

  router.get('/', async (req, res) => {
    try {
      const { status, limit, offset } = req.query;
      const result = await providersService.listProviders(
        status as string,
        parseInt(limit as string) || 50,
        parseInt(offset as string) || 0
      );
      res.json(result);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  router.post('/', async (req, res) => {
    try {
      const { name, type, uis, config, schedule } = req.body;
      if (!name || !type || !uis) {
        return res.status(400).json({ error: 'Missing required fields' });
      }
      const provider = await providersService.createProvider({
        name,
        type,
        uis,
        config,
        schedule,
      });
      res.status(201).json(provider);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  router.get('/:id', async (req, res) => {
    try {
      const provider = await providersService.getProvider(req.params.id);
      res.json(provider);
    } catch (err) {
      res.status(404).json({ error: err.message });
    }
  });

  router.patch('/:id', async (req, res) => {
    try {
      const provider = await providersService.updateProvider(req.params.id, req.body);
      res.json(provider);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  router.delete('/:id', async (req, res) => {
    try {
      await providersService.deleteProvider(req.params.id);
      res.json({ success: true });
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  return router;
}
```

## Phase 4: UIS Integration & Runner

### 4.1 Create Provider Form

Frontend: `/portal/app/providers/new/page.tsx`

```typescript
// Features:
// - Provider name, type selection
// - UIS specification editor with validation
// - Schedule configuration (cron)
// - Test connection button
// - Submit to backend
```

### 4.2 Job Runner Integration

Connect `/sdk/runner/` to backend:

```typescript
// backend/src/services/JobExecutor.ts
import { Runner } from '../../sdk/runner';

export class JobExecutor {
  async executeRun(providerId: string, uis: string) {
    // 1. Parse UIS specification
    const uisSpec = JSON.parse(uis);
    
    // 2. Select compiler (Spark/Flink/SeaTunnel)
    const compiler = this.selectCompiler(uisSpec.runtime);
    
    // 3. Compile to job
    const jobConfig = compiler.compile(uisSpec);
    
    // 4. Execute job
    const runner = new Runner(jobConfig);
    const result = await runner.execute();
    
    // 5. Update run status
    return result;
  }
}
```

## Phase 5: End-to-End Polygon Provider

### 5.1 Create Polygon UIS Template

```yaml
# sdk/uis/templates/polygon-stocks.uis.yaml
name: polygon-stocks
version: 1.1
type: api
runtime: spark

source:
  type: http
  baseUrl: https://api.polygon.io
  endpoints:
    - path: /v3/snapshot/stocks
      method: GET
      params:
        - name: ticker
          source: config
        - name: apikey
          source: secret:POLYGON_API_KEY
  rateLimit:
    requestsPerSecond: 5
  authentication:
    type: api_key
    header: Authorization

transform:
  - name: normalize_fields
    type: python
    script: |
      df['ingested_at'] = now()
      df['data_date'] = to_date(df['day'])

sink:
  type: iceberg
  catalog: minio
  database: raw
  table: polygon_stocks
  partitionBy: [data_date]
  mode: incremental
  stateTable: polygon_stocks_state
```

### 5.2 Create Spark Job Template

```python
# jobs/polygon_ingestion.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp

def ingest_polygon_data(spark, ticker, api_key):
    # Fetch from Polygon API
    df = spark.read.format("http").option("url", 
        f"https://api.polygon.io/v3/snapshot/stocks?ticker={ticker}&apikey={api_key}"
    ).load()
    
    # Transform
    df = df.withColumn("ingested_at", current_timestamp())
    
    # Write to Iceberg
    df.write.format("iceberg") \
        .mode("append") \
        .option("table", "catalog.raw.polygon_stocks") \
        .save()
```

## Implementation Timeline

- **Week 1**: Backend API setup, providers service
- **Week 2**: Runs service, job execution framework
- **Week 3**: UIS integration, provider form UI
- **Week 4**: Polygon template, end-to-end testing
- **Week 5**: Documentation, deployment

## Key Files to Create

```
backend/
  ├── src/
  │   ├── server.ts              # Express app
  │   ├── config/
  │   │   └── database.ts        # PostgreSQL config
  │   ├── models/
  │   │   ├── Provider.ts
  │   │   └── Run.ts
  │   ├── services/
  │   │   ├── ProvidersService.ts
  │   │   ├── RunsService.ts
  │   │   └── JobExecutor.ts
  │   ├── routes/
  │   │   ├── providers.ts
  │   │   └── runs.ts
  │   └── middleware/
  │       └── auth.ts            # API key authentication
  ├── package.json
  └── tsconfig.json

sdk/
  └── uis/
      └── templates/
          ├── polygon-stocks.uis.yaml
          ├── freefx-rates.uis.yaml
          └── tiingo-equities.uis.yaml
```

## Success Criteria

1. ✅ Portal UI functional and deployed
2. ✅ Backend API serving provider and run data
3. ✅ UIS specifications parsing correctly
4. ✅ Polygon data ingesting successfully
5. ✅ Data visible in Trino and Superset
6. ✅ Monitoring and logging operational
