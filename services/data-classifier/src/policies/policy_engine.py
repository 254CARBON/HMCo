"""
Policy engine for attaching masking/deny policies to classified columns.
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import requests
import json

logger = logging.getLogger(__name__)


@dataclass
class DataPolicy:
    """Data access/masking policy"""
    policy_name: str
    policy_type: str  # 'masking', 'deny', 'rbac'
    target_table: str
    target_column: str
    rules: Dict[str, Any]
    justification: str
    auto_approved: bool
    requires_review: bool


class PolicyEngine:
    """Manages data policies for classified columns"""
    
    def __init__(self,
                 openmetadata_endpoint: str,
                 trino_endpoint: str,
                 clickhouse_endpoint: str):
        self.openmetadata_endpoint = openmetadata_endpoint
        self.trino_endpoint = trino_endpoint
        self.clickhouse_endpoint = clickhouse_endpoint
        self.policy_cache = {}
    
    def create_policy_from_classification(self, classification) -> DataPolicy:
        """Create data policy from classification result"""
        
        policy_type = self._map_recommendation_to_type(classification.recommended_policy)
        rules = self._generate_policy_rules(classification)
        
        # Determine if auto-approval is allowed
        auto_approved = self._can_auto_approve(classification)
        
        return DataPolicy(
            policy_name=f"auto_{classification.table_name}_{classification.column_name}",
            policy_type=policy_type,
            target_table=classification.table_name,
            target_column=classification.column_name,
            rules=rules,
            justification=classification.reasoning,
            auto_approved=auto_approved,
            requires_review=not auto_approved
        )
    
    def _map_recommendation_to_type(self, recommendation: str) -> str:
        """Map policy recommendation to policy type"""
        mapping = {
            'mask_all': 'masking',
            'mask_partial': 'masking',
            'deny_access': 'deny',
            'role_based_access': 'rbac',
            'authenticated_access': 'rbac',
            'open_access': 'none'
        }
        return mapping.get(recommendation, 'rbac')
    
    def _generate_policy_rules(self, classification) -> Dict[str, Any]:
        """Generate specific policy rules based on classification"""
        
        if classification.recommended_policy == 'mask_all':
            return {
                'mask_type': 'full',
                'mask_value': '***REDACTED***',
                'allowed_roles': ['data_admin']
            }
        
        elif classification.recommended_policy == 'mask_partial':
            return {
                'mask_type': 'partial',
                'mask_pattern': 'last_4',  # Show last 4 chars
                'mask_char': '*',
                'allowed_roles': ['data_analyst', 'data_admin']
            }
        
        elif classification.recommended_policy == 'deny_access':
            return {
                'access': 'deny',
                'allowed_roles': ['data_admin', 'compliance_officer']
            }
        
        elif classification.recommended_policy == 'role_based_access':
            return {
                'access': 'conditional',
                'allowed_roles': ['data_analyst', 'data_scientist', 'data_admin']
            }
        
        else:  # open_access
            return {
                'access': 'allow',
                'allowed_roles': ['*']
            }
    
    def _can_auto_approve(self, classification) -> bool:
        """Determine if policy can be auto-approved"""
        
        # Auto-approve if confidence is high and sensitivity is not RESTRICTED
        if classification.confidence_score >= 0.9:
            if classification.sensitivity_level.value != 'restricted':
                return True
        
        return False
    
    def attach_policy_to_openmetadata(self, policy: DataPolicy) -> bool:
        """Attach policy to column in OpenMetadata"""
        
        logger.info(f"Attaching policy to OpenMetadata: {policy.policy_name}")
        
        # Construct OpenMetadata policy
        om_policy = {
            "name": policy.policy_name,
            "displayName": policy.policy_name.replace('_', ' ').title(),
            "description": policy.justification,
            "policyType": "DataPolicy",
            "enabled": True,
            "rules": [
                {
                    "name": f"rule_{policy.target_column}",
                    "effect": "Allow" if policy.policy_type != 'deny' else "Deny",
                    "resources": [
                        f"table/{policy.target_table}/columns/{policy.target_column}"
                    ],
                    "condition": self._generate_om_condition(policy)
                }
            ]
        }
        
        try:
            response = requests.post(
                f"{self.openmetadata_endpoint}/api/v1/policies",
                json=om_policy,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info(f"✓ Policy attached to OpenMetadata: {policy.policy_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to attach policy to OpenMetadata: {e}")
            return False
    
    def _generate_om_condition(self, policy: DataPolicy) -> str:
        """Generate OpenMetadata policy condition"""
        
        if policy.rules.get('allowed_roles'):
            roles = policy.rules['allowed_roles']
            if roles == ['*']:
                return "true"
            else:
                return f"role in {json.dumps(roles)}"
        
        return "true"
    
    def attach_policy_to_trino(self, policy: DataPolicy) -> bool:
        """Attach row-level security policy to Trino"""
        
        if policy.policy_type == 'none':
            return True
        
        logger.info(f"Attaching policy to Trino: {policy.policy_name}")
        
        # Generate Trino policy SQL
        if policy.policy_type == 'masking':
            sql = self._generate_trino_masking_sql(policy)
        elif policy.policy_type == 'deny':
            sql = self._generate_trino_deny_sql(policy)
        elif policy.policy_type == 'rbac':
            sql = self._generate_trino_rbac_sql(policy)
        else:
            return True
        
        try:
            # Execute via Trino coordinator API
            # In production, use proper Trino client
            logger.info(f"Would execute Trino SQL: {sql}")
            return True
        except Exception as e:
            logger.error(f"Failed to attach policy to Trino: {e}")
            return False
    
    def _generate_trino_masking_sql(self, policy: DataPolicy) -> str:
        """Generate Trino column masking SQL"""
        
        table_parts = policy.target_table.split('.')
        catalog = table_parts[0] if len(table_parts) > 1 else 'iceberg'
        schema_table = '.'.join(table_parts[1:]) if len(table_parts) > 1 else policy.target_table
        
        if policy.rules.get('mask_type') == 'full':
            mask_expr = f"'***REDACTED***'"
        else:  # partial
            mask_expr = f"CONCAT(REPEAT('*', LENGTH({policy.target_column}) - 4), SUBSTR({policy.target_column}, -4))"
        
        allowed_roles = policy.rules.get('allowed_roles', [])
        
        return f"""
        CREATE OR REPLACE ROW FILTER {policy.policy_name}_filter
        ON {catalog}.{schema_table}
        AS (
          CASE 
            WHEN current_user IN ({','.join(f"'{r}'" for r in allowed_roles)})
            THEN {policy.target_column}
            ELSE {mask_expr}
          END
        );
        """
    
    def _generate_trino_deny_sql(self, policy: DataPolicy) -> str:
        """Generate Trino deny access SQL"""
        
        table_parts = policy.target_table.split('.')
        catalog = table_parts[0] if len(table_parts) > 1 else 'iceberg'
        schema_table = '.'.join(table_parts[1:]) if len(table_parts) > 1 else policy.target_table
        
        allowed_roles = policy.rules.get('allowed_roles', [])
        
        return f"""
        CREATE OR REPLACE ROW FILTER {policy.policy_name}_filter
        ON {catalog}.{schema_table}
        AS (current_user IN ({','.join(f"'{r}'" for r in allowed_roles)}));
        """
    
    def _generate_trino_rbac_sql(self, policy: DataPolicy) -> str:
        """Generate Trino RBAC SQL"""
        
        # Trino RBAC is typically handled at catalog/schema level
        # Column-level RBAC uses masking
        return self._generate_trino_masking_sql(policy)
    
    def attach_policy_to_clickhouse(self, policy: DataPolicy) -> bool:
        """Attach row-level security policy to ClickHouse"""
        
        if policy.policy_type == 'none':
            return True
        
        logger.info(f"Attaching policy to ClickHouse: {policy.policy_name}")
        
        # ClickHouse uses row policies
        sql = self._generate_ch_policy_sql(policy)
        
        try:
            # Execute via ClickHouse client
            logger.info(f"Would execute ClickHouse SQL: {sql}")
            return True
        except Exception as e:
            logger.error(f"Failed to attach policy to ClickHouse: {e}")
            return False
    
    def _generate_ch_policy_sql(self, policy: DataPolicy) -> str:
        """Generate ClickHouse row policy SQL"""
        
        allowed_roles = policy.rules.get('allowed_roles', [])
        
        if policy.policy_type == 'deny':
            # Deny access unless in allowed roles
            return f"""
            CREATE ROW POLICY {policy.policy_name} ON {policy.target_table}
            FOR SELECT
            USING currentUser() IN ({','.join(f"'{r}'" for r in allowed_roles)})
            TO ALL;
            """
        else:
            # Allow with masking (ClickHouse doesn't have native column masking)
            # Use view with CASE expression instead
            return f"""
            CREATE OR REPLACE VIEW {policy.target_table}_masked AS
            SELECT
              *,
              CASE 
                WHEN currentUser() IN ({','.join(f"'{r}'" for r in allowed_roles)})
                THEN {policy.target_column}
                ELSE '***REDACTED***'
              END AS {policy.target_column}_masked
            FROM {policy.target_table};
            """
    
    def create_approval_workflow(self, policy: DataPolicy) -> Dict[str, Any]:
        """Create approval workflow for policies requiring review"""
        
        if not policy.requires_review:
            return {'status': 'auto_approved'}
        
        logger.info(f"Creating approval workflow for policy: {policy.policy_name}")
        
        workflow = {
            'policy_name': policy.policy_name,
            'status': 'pending_review',
            'reviewers': ['compliance_team', 'data_governance'],
            'justification': policy.justification,
            'policy_details': {
                'table': policy.target_table,
                'column': policy.target_column,
                'type': policy.policy_type,
                'rules': policy.rules
            },
            'expiry_days': 90  # Policies require re-review after 90 days
        }
        
        # In production: Create Jira ticket, Slack notification, etc.
        logger.info(f"Approval workflow created: {workflow}")
        
        return workflow
