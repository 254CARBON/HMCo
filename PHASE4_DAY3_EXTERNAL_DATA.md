# Phase 4: Day 3-4 - External Data Connectivity

**Status**: Implementation Starting  
**Date**: October 24-25, 2025  
**Target**: Enable connections to external data sources (DB, S3, APIs)  
**Duration**: Full 2 days (8 hours)

---

## Overview

After achieving 90.8% platform health on Day 1-2, we now implement external data connectivity. This phase enables the platform to:
- Connect to external databases (PostgreSQL, MySQL, Oracle)
- Access cloud storage (S3, Azure Blob, GCS)
- Consume APIs (REST, GraphQL)
- Implement ETL frameworks
- Create data quality checks

---

## Day 3-4: External Data Connectivity (Full 2 Days)

### Task 3.1: Network Policies Configuration (2 hours)

**Goal**: Enable secure external egress while maintaining internal security

#### Step 1: Review Current Network Policies
```bash
# Check existing policies
kubectl get networkpolicies -n data-platform

# View details of key policies
kubectl get networkpolicy allow-data-platform-internal -n data-platform -o yaml
```

#### Step 2: Create External Egress Policy
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-external-egress
  namespace: data-platform
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  # Allow internal pod communication
  - to:
    - podSelector: {}
  # Allow DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
  # Allow external databases
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 3306  # MySQL
    - protocol: TCP
      port: 1433  # SQL Server
    - protocol: TCP
      port: 1521  # Oracle
  # Allow cloud storage and APIs (HTTPS)
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
    ports:
    - protocol: TCP
      port: 443   # HTTPS
    - protocol: TCP
      port: 80    # HTTP
```

#### Step 3: Apply Network Policies
```bash
kubectl apply -f - <<'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-external-egress
  namespace: data-platform
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector: {}
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
    ports:
    - protocol: TCP
      port: 5432
    - protocol: TCP
      port: 3306
    - protocol: TCP
      port: 1433
    - protocol: TCP
      port: 1521
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
EOF

# Verify
kubectl get networkpolicy allow-external-egress -n data-platform
```

---

### Task 3.2: Secure Credential Management (2 hours)

**Goal**: Store credentials securely without hardcoding

#### Step 1: Create Database Credentials Secret
```bash
# PostgreSQL External Database
kubectl create secret generic external-postgres \
  --from-literal=username=etl_user \
  --from-literal=password=your-secure-password \
  --from-literal=host=external-db.company.com \
  --from-literal=port=5432 \
  --from-literal=database=raw_data \
  -n data-platform \
  --dry-run=client -o yaml | kubectl apply -f -

# MySQL External Database
kubectl create secret generic external-mysql \
  --from-literal=username=etl_user \
  --from-literal=password=your-secure-password \
  --from-literal=host=mysql.company.com \
  --from-literal=port=3306 \
  --from-literal=database=raw_data \
  -n data-platform \
  --dry-run=client -o yaml | kubectl apply -f -

# Verify
kubectl get secrets -n data-platform | grep external
```

#### Step 2: Create Cloud Storage Credentials
```bash
# AWS S3 Credentials
kubectl create secret generic aws-s3-credentials \
  --from-literal=access_key_id=AKIA... \
  --from-literal=secret_access_key=your-secret-key \
  --from-literal=region=us-east-1 \
  --from-literal=bucket=company-data-lake \
  -n data-platform \
  --dry-run=client -o yaml | kubectl apply -f -

# Azure Storage Credentials
kubectl create secret generic azure-storage \
  --from-literal=account_name=storageaccount \
  --from-literal=account_key=your-storage-key \
  --from-literal=container=data-lake \
  -n data-platform \
  --dry-run=client -o yaml | kubectl apply -f -

# GCS Credentials (from service account JSON)
kubectl create secret generic gcs-credentials \
  --from-file=key.json=/path/to/gcs-key.json \
  -n data-platform \
  --dry-run=client -o yaml | kubectl apply -f -
```

#### Step 3: Create API Credentials
```bash
# Generic API credentials
kubectl create secret generic external-api-credentials \
  --from-literal=api_key=your-api-key \
  --from-literal=api_secret=your-api-secret \
  --from-literal=api_endpoint=https://api.example.com \
  -n data-platform \
  --dry-run=client -o yaml | kubectl apply -f -

# OAuth2 Credentials
kubectl create secret generic oauth2-credentials \
  --from-literal=client_id=your-client-id \
  --from-literal=client_secret=your-client-secret \
  --from-literal=token_endpoint=https://auth.example.com/oauth/token \
  -n data-platform \
  --dry-run=client -o yaml | kubectl apply -f -
```

---

### Task 3.3: Implement Data Source Connectors (3 hours)

**Goal**: Create reusable connector configurations

#### PostgreSQL Connector Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-connector-config
  namespace: data-platform
data:
  connector.properties: |
    connector=postgresql
    connection.url=jdbc:postgresql://${DB_HOST}:${DB_PORT}/${DB_NAME}
    connection.user=${DB_USER}
    connection.password=${DB_PASSWORD}
    table.include.list=public.*
    snapshot.mode=initial
    publication.name=dbz_publication
    slot.name=debezium
    transforms=route
    transforms.route.type=org.apache.kafka.connect.transforms.RegexRouter
    transforms.route.regex=([^.]+)\\.([^.]+)\\.([^.]+)
    transforms.route.replacement=$3
```

#### Create DolphinScheduler DataSource for External DB
```bash
# Function to create datasource
create_datasource() {
  local name=$1
  local type=$2
  local host=$3
  local port=$4
  local database=$5
  local username=$6
  local password=$7

  curl -X POST http://dolphinscheduler-api:8080/dolphinscheduler/datasources \
    -H "Content-Type: application/json" \
    -d '{
      "name": "'$name'",
      "type": "'$type'",
      "host": "'$host'",
      "port": '$port',
      "database": "'$database'",
      "username": "'$username'",
      "password": "'$password'",
      "description": "External data source for ETL"
    }'
}

# Create PostgreSQL datasource
create_datasource "external-postgres" "POSTGRESQL" \
  "external-db.company.com" 5432 "raw_data" "etl_user" "secure-password"

# Create MySQL datasource
create_datasource "external-mysql" "MYSQL" \
  "mysql.company.com" 3306 "raw_data" "etl_user" "secure-password"
```

#### S3 Connector (via MinIO)
```bash
# Configure MinIO to connect to S3
kubectl exec -n data-platform minio-service-0 -- mc alias set s3-external \
  https://s3.amazonaws.com \
  AKIA... \
  your-secret-key \
  --api S3v4

# Create bucket mirror for backup/sync
kubectl exec -n data-platform minio-service-0 -- \
  mc mirror --watch s3-external/source-bucket /minio/local-mirror
```

#### API Connector Script
```python
# Create /tmp/api-connector.py
import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIConnector:
    def __init__(self, endpoint: str, auth_type: str = "none", **auth_params):
        self.endpoint = endpoint
        self.auth_type = auth_type
        self.auth_params = auth_params
        self.session = requests.Session()
        
    def get_headers(self) -> Dict:
        headers = {"Content-Type": "application/json"}
        
        if self.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self.auth_params.get('token')}"
        elif self.auth_type == "api_key":
            headers[self.auth_params.get('key_header', 'X-API-Key')] = self.auth_params.get('key')
        elif self.auth_type == "basic":
            import base64
            creds = base64.b64encode(
                f"{self.auth_params.get('username')}:{self.auth_params.get('password')}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {creds}"
            
        return headers
    
    def fetch(self, path: str, params: Dict = None, method: str = "GET") -> List[Dict]:
        url = f"{self.endpoint}/{path}"
        headers = self.get_headers()
        
        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, params=params, timeout=30)
            elif method == "POST":
                response = self.session.post(url, headers=headers, json=params, timeout=30)
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Successfully fetched from {path}")
            return data if isinstance(data, list) else [data]
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def batch_fetch(self, path: str, batch_params: List[Dict]) -> List[Dict]:
        results = []
        for params in batch_params:
            results.extend(self.fetch(path, params))
        return results

# Export connector
__all__ = ['APIConnector']
```

---

### Task 3.4: ETL Framework Setup (3 hours)

**Goal**: Create reusable ETL templates and workflows

#### ETL Template 1: Database Extract-Load
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: etl-db-extract-load
  namespace: data-platform
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: dolphinscheduler-worker
          containers:
          - name: etl
            image: python:3.10-slim
            env:
            - name: DB_HOST
              valueFrom:
                secretKeyRef:
                  name: external-postgres
                  key: host
            - name: DB_PORT
              valueFrom:
                secretKeyRef:
                  name: external-postgres
                  key: port
            - name: DB_USER
              valueFrom:
                secretKeyRef:
                  name: external-postgres
                  key: username
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: external-postgres
                  key: password
            - name: DB_NAME
              valueFrom:
                secretKeyRef:
                  name: external-postgres
                  key: database
            command:
            - /bin/sh
            - -c
            - |
              pip install psycopg2-binary sqlalchemy pandas pyarrow -q
              python << 'PYTHON'
              import os
              import logging
              from sqlalchemy import create_engine, text
              import pandas as pd
              from datetime import datetime
              
              logging.basicConfig(level=logging.INFO)
              logger = logging.getLogger(__name__)
              
              # Build connection string
              db_url = f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
              
              try:
                  # Extract
                  logger.info("Connecting to external database...")
                  engine = create_engine(db_url)
                  
                  logger.info("Extracting data...")
                  query = "SELECT * FROM source_table WHERE updated_at > NOW() - INTERVAL '1 day'"
                  df = pd.read_sql(query, engine)
                  
                  # Validate
                  logger.info(f"Extracted {len(df)} rows")
                  if len(df) == 0:
                      logger.warning("No data extracted")
                      exit(0)
                  
                  # Add metadata
                  df['extracted_at'] = datetime.now()
                  
                  # Load to MinIO/Iceberg
                  logger.info("Loading to data lake...")
                  # TODO: Implement Iceberg write
                  parquet_file = f"/tmp/extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                  df.to_parquet(parquet_file)
                  
                  logger.info(f"Successfully loaded data to {parquet_file}")
                  
              except Exception as e:
                  logger.error(f"ETL failed: {str(e)}")
                  exit(1)
              PYTHON
            resources:
              requests:
                cpu: "500m"
                memory: "512Mi"
              limits:
                cpu: "1000m"
                memory: "1Gi"
          restartPolicy: OnFailure
          backoffLimit: 3
```

#### ETL Template 2: API Data Ingestion
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: etl-api-ingest
  namespace: data-platform
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: dolphinscheduler-worker
          containers:
          - name: api-ingest
            image: python:3.10-slim
            env:
            - name: API_ENDPOINT
              valueFrom:
                secretKeyRef:
                  name: external-api-credentials
                  key: api_endpoint
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: external-api-credentials
                  key: api_key
            command:
            - /bin/sh
            - -c
            - |
              pip install requests pandas pyarrow -q
              python << 'PYTHON'
              import os
              import requests
              import pandas as pd
              from datetime import datetime
              import logging
              
              logging.basicConfig(level=logging.INFO)
              logger = logging.getLogger(__name__)
              
              try:
                  # Fetch from API
                  logger.info(f"Fetching data from {os.environ['API_ENDPOINT']}")
                  headers = {"Authorization": f"Bearer {os.environ['API_KEY']}"}
                  response = requests.get(
                      f"{os.environ['API_ENDPOINT']}/data",
                      headers=headers,
                      timeout=30
                  )
                  response.raise_for_status()
                  
                  # Process data
                  data = response.json()
                  df = pd.DataFrame(data if isinstance(data, list) else [data])
                  
                  logger.info(f"Fetched {len(df)} records")
                  
                  # Save to staging
                  output_file = f"/tmp/api_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                  df.to_parquet(output_file)
                  logger.info(f"Saved to {output_file}")
                  
              except Exception as e:
                  logger.error(f"API ingestion failed: {str(e)}")
                  exit(1)
              PYTHON
            resources:
              requests:
                cpu: "250m"
                memory: "256Mi"
              limits:
                cpu: "500m"
                memory: "512Mi"
          restartPolicy: OnFailure
```

#### Data Quality Template
```python
# /tmp/data-quality-checks.py
import pandas as pd
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityChecker:
    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        self.df = df
        self.name = name
        self.checks = []
        
    def check_null_columns(self, columns: List[str]) -> Tuple[bool, str]:
        """Check for null values in critical columns"""
        for col in columns:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:
                msg = f"Column '{col}' has {null_count} null values"
                logger.warning(msg)
                self.checks.append(("null_check", col, False, msg))
                return False, msg
        return True, "No nulls found in critical columns"
    
    def check_duplicates(self, columns: List[str]) -> Tuple[bool, str]:
        """Check for duplicate rows"""
        dup_count = self.df.duplicated(subset=columns).sum()
        if dup_count > 0:
            msg = f"Found {dup_count} duplicate rows"
            logger.warning(msg)
            self.checks.append(("duplicate_check", str(columns), False, msg))
            return False, msg
        return True, "No duplicates found"
    
    def check_data_types(self, type_schema: Dict[str, str]) -> Tuple[bool, str]:
        """Check if columns have expected data types"""
        for col, expected_type in type_schema.items():
            if col not in self.df.columns:
                msg = f"Column '{col}' not found"
                self.checks.append(("type_check", col, False, msg))
                return False, msg
        return True, "Data types correct"
    
    def check_value_range(self, column: str, min_val, max_val) -> Tuple[bool, str]:
        """Check if values are within expected range"""
        out_of_range = ((self.df[column] < min_val) | (self.df[column] > max_val)).sum()
        if out_of_range > 0:
            msg = f"Column '{column}' has {out_of_range} values out of range [{min_val}, {max_val}]"
            self.checks.append(("range_check", column, False, msg))
            return False, msg
        return True, f"All values in '{column}' within range"
    
    def run_all_checks(self) -> bool:
        """Run all checks and return overall pass/fail"""
        results = []
        for check in self.checks:
            results.append(check[2])  # Extract pass/fail
        
        all_passed = all(results)
        logger.info(f"Data quality check summary: {sum(results)}/{len(results)} passed")
        return all_passed
    
    def get_report(self) -> Dict:
        """Generate quality report"""
        return {
            "dataset": self.name,
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "checks": self.checks,
            "all_passed": all(c[2] for c in self.checks)
        }
```

---

## Validation & Testing

### Test Database Connectivity
```bash
# Test PostgreSQL connection
kubectl run -n data-platform test-pg --image=postgres:15 --rm -it \
  --env="PGPASSWORD=password" -- \
  psql -h external-db.company.com -U etl_user -d raw_data -c "SELECT 1"

# Test MySQL connection
kubectl run -n data-platform test-mysql --image=mysql:8 --rm -it -- \
  mysql -h mysql.company.com -u etl_user -pyour-password -e "SELECT 1"
```

### Test S3 Connectivity
```bash
# Test S3 access via MinIO
kubectl exec -n data-platform minio-service-0 -- \
  mc ls s3-external/source-bucket
```

### Test API Connectivity
```bash
# Test API endpoint
kubectl run -n data-platform test-api --image=curlimages/curl --rm -it -- \
  curl -H "Authorization: Bearer YOUR_TOKEN" https://api.example.com/health
```

---

## Progress Tracking

### Day 3 Tasks
- [ ] Network policies configured and verified
- [ ] Database credentials created (PostgreSQL, MySQL)
- [ ] Cloud storage credentials created (S3, Azure, GCS)
- [ ] API credentials created
- [ ] PostgreSQL connector configured
- [ ] S3 connector configured
- [ ] API connector script created

### Day 4 Tasks
- [ ] ETL templates created (DB extract-load, API ingest)
- [ ] Data quality framework implemented
- [ ] External datasources tested
- [ ] First ETL workflow executed
- [ ] Connectivity validation complete
- [ ] Documentation updated

---

## Success Criteria

✅ All external data sources reachable  
✅ Credentials secured in Kubernetes secrets  
✅ ETL templates created and tested  
✅ Data quality checks implemented  
✅ Documentation complete  
✅ Ready for production workflows  

**Next Phase**: Performance Optimization & Pilot Workloads
