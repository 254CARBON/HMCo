# Portal Backend Services

This directory contains the backend services for the data ingestion platform portal.

## Database Setup

### Prerequisites
- PostgreSQL database
- Python 3.8+

### Environment Variables
Set the following environment variables:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/portal"
```

### Setup Database
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create tables:
```bash
python setup_db.py
```

3. Run tests:
```bash
python test_db.py
```

### Database Schema

#### External Providers
Core configuration for external data sources including:
- Provider metadata (name, type, status)
- Authentication and credentials (Vault references)
- Scheduling and target sink configuration
- Quality gates and SLO configurations

#### Provider Endpoints
Individual API endpoints or data sources within a provider:
- Endpoint configuration (path, method, headers)
- Pagination and response handling
- Field mapping and validation rules

#### Provider Runs
Execution records and metrics:
- Run lifecycle (status, timing, performance)
- Data quality metrics and validation results
- Resource usage and cost tracking
- Lineage and tracing information

## Migration Management

To create new migrations:
```bash
cd migrations
alembic revision -m "description of changes"
alembic upgrade head
```

To rollback:
```bash
alembic downgrade -1
```