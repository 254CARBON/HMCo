# Automated Data Classifier

Automatically scans ingested data for PII, sensitive information, and license-restricted content, then attaches appropriate masking/deny policies.

## Purpose

**You'll unknowingly ingest restricted stuff—make that impossible to expose.**

This service:
- Scans data after ingest for PII, PHI, financial data, etc.
- Uses regex + ML heuristics (Presidio) for detection
- Tags Iceberg/ClickHouse columns with classification
- Attaches masking/deny policies via OpenMetadata, Trino RLS, and ClickHouse row policies
- Routes high-risk findings through approval workflow

## Features

- **Content Scanning**: Regex and ML-based PII detection (Presidio)
- **Risk Scoring**: Dataset-level risk assessment
- **Auto-Policy Attach**: Automatic policy enforcement for high-confidence detections
- **Approval Workflow**: Human review for ambiguous cases
- **Multi-System**: Applies policies to Trino, ClickHouse, and OpenMetadata

## Architecture

```
services/data-classifier/
├── src/
│   ├── scanners/
│   │   └── content_scanner.py       # PII/sensitive data detection
│   └── policies/
│       └── policy_engine.py         # Policy attachment logic
└── requirements.txt
```

## Supported Classifications

### Sensitivity Levels
- **PUBLIC**: Safe for open access
- **INTERNAL**: Requires authentication
- **CONFIDENTIAL**: Requires role-based access + partial masking
- **RESTRICTED**: Full masking or deny access

### Data Categories
- **PII**: Personally identifiable information (SSN, email, phone)
- **PHI**: Protected health information
- **FINANCIAL**: Credit cards, account numbers
- **PROPRIETARY**: API keys, trade secrets
- **LOCATION**: GPS coordinates, addresses

### Policy Types
- **mask_all**: Full redaction (e.g., `***REDACTED***`)
- **mask_partial**: Partial masking (e.g., show last 4 digits)
- **deny_access**: No access except for admins
- **role_based_access**: Conditional access by role
- **open_access**: No restrictions

## Usage

### Scan a Table

```python
from data_classifier.scanners import DatasetScanner
from trino.dbapi import connect

# Connect to data source
conn = connect(host='trino', port=8080, catalog='iceberg')

# Scan table
scanner = DatasetScanner()
classifications = scanner.scan_table(
    connection=conn,
    table_name='curated.customer_data',
    sample_size=1000
)

# Review results
for classification in classifications:
    print(f"Column: {classification.column_name}")
    print(f"Sensitivity: {classification.sensitivity_level.value}")
    print(f"Categories: {[c.value for c in classification.data_categories]}")
    print(f"Policy: {classification.recommended_policy}")
    print(f"Confidence: {classification.confidence_score:.2f}")
    print(f"Reasoning: {classification.reasoning}")
    print()

# Calculate risk score
risk = scanner.calculate_risk_score(classifications)
print(f"Risk Score: {risk['risk_score']:.1f}/100")
print(f"Risk Level: {risk['risk_level']}")
print(f"Sensitive Columns: {risk['sensitive_columns']}/{risk['total_columns']}")
```

### Attach Policies

```python
from data_classifier.policies import PolicyEngine

# Initialize policy engine
engine = PolicyEngine(
    openmetadata_endpoint='http://openmetadata.data-platform:8585',
    trino_endpoint='http://trino.data-platform:8080',
    clickhouse_endpoint='http://clickhouse.data-platform:8123'
)

# Create and attach policies
for classification in classifications:
    if classification.sensitivity_level.value != 'public':
        # Create policy
        policy = engine.create_policy_from_classification(classification)
        
        # Attach to systems
        if policy.auto_approved:
            engine.attach_policy_to_openmetadata(policy)
            engine.attach_policy_to_trino(policy)
            engine.attach_policy_to_clickhouse(policy)
        else:
            # Requires approval
            workflow = engine.create_approval_workflow(policy)
            print(f"Approval required: {workflow}")
```

### Ingest Hook

```python
# Post-ingest hook (triggered automatically)
def on_table_ingested(table_name: str):
    """Auto-classify new tables after ingest"""
    
    # Scan table
    classifications = scanner.scan_table(conn, table_name)
    
    # Attach policies
    for classification in classifications:
        policy = engine.create_policy_from_classification(classification)
        
        if policy.auto_approved:
            # Auto-attach
            engine.attach_policy_to_openmetadata(policy)
            engine.attach_policy_to_trino(policy)
            engine.attach_policy_to_clickhouse(policy)
            
            logger.info(f"✓ Policy auto-applied: {policy.policy_name}")
        else:
            # Create approval workflow
            workflow = engine.create_approval_workflow(policy)
            
            logger.info(f"⚠ Manual review required: {policy.policy_name}")
```

## Detected Patterns

### PII
- Social Security Numbers (SSN)
- Email addresses
- Phone numbers
- Names (via NLP)
- IP addresses

### Financial
- Credit card numbers
- Account numbers (10-20 digits)

### Proprietary
- API keys (32+ character alphanumeric)
- Access tokens

## Policy Examples

### Full Masking (RESTRICTED)
```sql
-- Trino
CREATE ROW FILTER ssn_mask ON customer_data
AS (
  CASE 
    WHEN current_user IN ('data_admin')
    THEN ssn
    ELSE '***REDACTED***'
  END
);

-- ClickHouse
CREATE ROW POLICY ssn_policy ON customer_data
FOR SELECT
USING currentUser() IN ('data_admin')
TO ALL;
```

### Partial Masking (CONFIDENTIAL)
```sql
-- Show last 4 digits only
CONCAT(REPEAT('*', LENGTH(account_number) - 4), SUBSTR(account_number, -4))
-- Result: ************1234
```

### Role-Based Access (INTERNAL)
```sql
-- Only data_analyst and data_admin can see column
WHERE current_user IN ('data_analyst', 'data_admin')
```

## DoD (Definition of Done)

✅ **"Leak test" sample trips classifier** → masked/blocked downstream within minutes  
✅ Classification runs automatically after ingest  
✅ Policies enforced in Trino, ClickHouse, and OpenMetadata  
✅ Approval workflow for ambiguous cases  
✅ Policy expiry and re-review after 90 days

## Configuration

Environment variables:
- `OPENMETADATA_ENDPOINT`: OpenMetadata API endpoint
- `TRINO_ENDPOINT`: Trino coordinator endpoint
- `CLICKHOUSE_ENDPOINT`: ClickHouse endpoint
- `AUTO_APPROVE_THRESHOLD`: Confidence threshold for auto-approval (default: 0.9)
- `SCAN_SAMPLE_SIZE`: Number of rows to sample (default: 1000)

## Deployment

Runs as:
1. **Post-Ingest Hook**: Triggered after each table ingest
2. **Scheduled Job**: Daily scan of all tables
3. **On-Demand**: Via API/CLI

## Monitoring

Metrics:
- `classifier_scans_total`: Total tables scanned
- `classifier_sensitive_columns_found`: Sensitive columns detected
- `classifier_policies_attached`: Policies auto-attached
- `classifier_approvals_pending`: Policies awaiting review

Alerts:
- RESTRICTED data found → notify security team
- Policy attachment failed → page on-call
- >10 pending approvals → notify governance team

## Override Trail

All policy overrides are tracked:
- Override reason
- Approver
- Expiry date (90 days default)
- Re-review reminder

```python
# Example override
engine.override_policy(
    policy_name='auto_customer_data_ssn',
    reason='Data is already hashed in production',
    approver='compliance_officer',
    expiry_days=90
)
```

## Contact

**Owner:** data-platform@254carbon.com  
**Slack:** #data-classification
