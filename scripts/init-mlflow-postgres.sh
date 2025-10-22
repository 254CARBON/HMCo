#!/bin/bash
# Initialize PostgreSQL Database for MLflow
# Creates mlflow user and database with proper permissions

set -e

NAMESPACE="data-platform"
POSTGRES_POD=""
DB_NAME="mlflow"
DB_USER="mlflow"
DB_PASSWORD="mlflow-secure-password-change-me"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MLflow PostgreSQL Initialization${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Find PostgreSQL pod
echo -e "${BLUE}Finding PostgreSQL pod...${NC}"
POSTGRES_POD=$(kubectl get pods -n "$NAMESPACE" -l app=postgres-shared --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POSTGRES_POD" ]; then
    echo -e "${RED}✗ PostgreSQL pod not found${NC}"
    echo "Please ensure PostgreSQL is running in the $NAMESPACE namespace"
    exit 1
fi

echo -e "${GREEN}✓ Found PostgreSQL pod: $POSTGRES_POD${NC}"
echo ""

# Check if mlflow database already exists
echo -e "${BLUE}Checking if database exists...${NC}"
DB_EXISTS=$(kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- psql -U datahub -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" 2>/dev/null || echo "")

if [ "$DB_EXISTS" = "1" ]; then
    echo -e "${YELLOW}⚠ Database '$DB_NAME' already exists${NC}"
    read -p "Do you want to recreate it? (y/N): " confirm
    if [[ "$confirm" == "y" ]] || [[ "$confirm" == "Y" ]]; then
        echo -e "${YELLOW}Dropping existing database...${NC}"
        kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- psql -U datahub -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>/dev/null || true
        kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- psql -U datahub -d postgres -c "DROP USER IF EXISTS $DB_USER;" 2>/dev/null || true
    else
        echo "Keeping existing database. Exiting."
        exit 0
    fi
fi

echo ""
echo -e "${BLUE}Creating MLflow database and user...${NC}"

# Create user
kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- psql -U datahub -d postgres -c "
DO \$\$ BEGIN
  CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
EXCEPTION WHEN DUPLICATE_OBJECT THEN
  RAISE NOTICE 'User $DB_USER already exists';
END \$\$;
" 2>&1 | grep -v "^NOTICE" || true

echo -e "${GREEN}✓ User '$DB_USER' created${NC}"

# Create database
kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- psql -U datahub -d postgres -c "
CREATE DATABASE $DB_NAME OWNER $DB_USER;
" 2>&1 || echo -e "${YELLOW}⚠ Database may already exist${NC}"

echo -e "${GREEN}✓ Database '$DB_NAME' created${NC}"

# Grant privileges
kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- psql -U datahub -d postgres -c "
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
" 2>&1 || true

echo -e "${GREEN}✓ Privileges granted${NC}"

# Setup schema
kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- psql -U $DB_USER -d $DB_NAME -c "
CREATE SCHEMA IF NOT EXISTS mlflow AUTHORIZATION $DB_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA mlflow GRANT ALL ON TABLES TO $DB_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA mlflow GRANT ALL ON SEQUENCES TO $DB_USER;
GRANT ALL PRIVILEGES ON SCHEMA mlflow TO $DB_USER;
" 2>&1 || true

echo -e "${GREEN}✓ Schema created${NC}"

# Verify setup
echo ""
echo -e "${BLUE}Verifying setup...${NC}"
VERIFY=$(kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- psql -U datahub -d postgres -tAc "
SELECT datname FROM pg_database WHERE datname='$DB_NAME';
" 2>/dev/null)

if [ "$VERIFY" = "$DB_NAME" ]; then
    echo -e "${GREEN}✓ Database verified successfully${NC}"
else
    echo -e "${RED}✗ Database verification failed${NC}"
    exit 1
fi

# Test connection
echo -e "${BLUE}Testing connection...${NC}"
kubectl exec -n "$NAMESPACE" "$POSTGRES_POD" -- psql -U $DB_USER -d $DB_NAME -c "SELECT version();" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Connection test successful${NC}"
else
    echo -e "${RED}✗ Connection test failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  PostgreSQL Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Database Details:"
echo "  - Database: $DB_NAME"
echo "  - User: $DB_USER"
echo "  - Host: postgres-shared-service.data-platform.svc.cluster.local"
echo "  - Port: 5432"
echo ""
echo "Connection String:"
echo "  postgresql://$DB_USER:$DB_PASSWORD@postgres-shared-service.data-platform.svc.cluster.local:5432/$DB_NAME"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Create MinIO bucket: ./scripts/create-mlflow-minio-bucket.sh"
echo "  2. Deploy MLflow: kubectl apply -f k8s/compute/mlflow/"
echo ""

