"""
UIS Validator for additional validation logic.
"""

import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

from spec import UnifiedIngestionSpec, ProviderType, SinkType, IngestionMode


class UISValidator:
    """Advanced validator for UIS specifications."""

    def __init__(self):
        self.cron_pattern = re.compile(r'^(\*|([0-9]|[1-5][0-9])|\*\/[0-9]+)\s+(\*|([0-9]|[1-5][0-9])|\*\/[0-9]+)\s+(\*|([0-9]|[1-3][0-9])|\*\/[0-9]+)\s+(\*|([1-9]|[1-2][0-9]|3[0-1])|\*\/[0-9]+)\s+(\*|([0-9]|1[0-2])|\*\/[0-9]+)$')

    def validate_completeness(self, spec: UnifiedIngestionSpec) -> List[str]:
        """Validate that the spec is complete and production-ready."""
        errors = []

        # Required fields for production
        if not spec.provider.credentials_ref:
            errors.append("credentials_ref is required for production deployment")

        if not spec.provider.owner:
            errors.append("owner field is required")

        if not spec.provider.tenant_id:
            errors.append("tenant_id is required")

        # Endpoint completeness
        for endpoint in spec.provider.endpoints:
            if endpoint.auth and not endpoint.auth_config:
                errors.append(f"Endpoint {endpoint.name} has auth type but missing auth_config")

        # Transform completeness
        for transform in spec.provider.transforms:
            if transform.type == 'wasm' and not transform.wasm_module:
                errors.append(f"Transform {transform.name} is WASM type but missing wasm_module")

        return errors

    def validate_cron_expression(self, cron_expr: str) -> bool:
        """Validate cron expression format."""
        return bool(self.cron_pattern.match(cron_expr))

    def validate_field_mappings(self, spec: UnifiedIngestionSpec) -> List[str]:
        """Validate field mappings are consistent."""
        errors = []

        # Check for duplicate mappings
        all_mappings = {}
        for endpoint in spec.provider.endpoints:
            if endpoint.field_mapping:
                for source_field, target_field in endpoint.field_mapping.items():
                    if target_field in all_mappings:
                        if all_mappings[target_field] != source_field:
                            errors.append(f"Conflicting field mapping for {target_field}: {all_mappings[target_field]} vs {source_field}")
                    else:
                        all_mappings[target_field] = source_field

        return errors

    def validate_rate_limits(self, spec: UnifiedIngestionSpec) -> List[str]:
        """Validate rate limiting configuration."""
        errors = []

        # Check for missing rate limits on high-frequency endpoints
        for endpoint in spec.provider.endpoints:
            if ('realtime' in endpoint.name.lower() or
                'stream' in endpoint.name.lower() or
                spec.provider.mode in ['streaming', 'websocket']):
                if not endpoint.rate_limit_per_second:
                    errors.append(f"High-frequency endpoint {endpoint.name} should have rate limiting configured")

        # Check rate limit groups
        rate_limit_groups = {}
        for endpoint in spec.provider.endpoints:
            if endpoint.rate_limit_group:
                if endpoint.rate_limit_group not in rate_limit_groups:
                    rate_limit_groups[endpoint.rate_limit_group] = []
                rate_limit_groups[endpoint.rate_limit_group].append(endpoint.name)

        # Validate group consistency
        for group, endpoints in rate_limit_groups.items():
            rates = [e.rate_limit_per_second for e in spec.provider.endpoints if e.rate_limit_group == group and e.rate_limit_per_second]
            if len(set(rates)) > 1:
                errors.append(f"Rate limit group {group} has inconsistent rate limits: {rates}")

        return errors

    def validate_sink_configuration(self, spec: UnifiedIngestionSpec) -> List[str]:
        """Validate sink configurations."""
        errors = []

        for sink in spec.provider.sinks:
            if sink.type == 'iceberg':
                if not sink.table_name:
                    errors.append("Iceberg sink requires table_name")
                if not sink.config.get('warehouse'):
                    errors.append("Iceberg sink requires warehouse configuration")

            elif sink.type == 'clickhouse':
                if not sink.clickhouse_table:
                    errors.append("ClickHouse sink requires clickhouse_table")

            elif sink.type == 'kafka':
                if not sink.kafka_topic:
                    errors.append("Kafka sink requires kafka_topic")

        return errors

    def validate_performance(self, spec: UnifiedIngestionSpec) -> List[str]:
        """Validate performance considerations."""
        warnings = []

        # Check parallelism vs endpoints
        if spec.provider.parallelism > 1 and len(spec.provider.endpoints) == 1:
            warnings.append("High parallelism with single endpoint may not improve performance")

        # Check for potential hotspots
        endpoint_counts = {}
        for endpoint in spec.provider.endpoints:
            base_path = endpoint.path.split('/')[1] if '/' in endpoint.path else endpoint.path
            endpoint_counts[base_path] = endpoint_counts.get(base_path, 0) + 1

        for base_path, count in endpoint_counts.items():
            if count > 5:
                warnings.append(f"High number of endpoints under {base_path} ({count}) - consider consolidation")

        # Check transform complexity
        for transform in spec.provider.transforms:
            if transform.type in ['spark', 'flink'] and len(spec.provider.transforms) > 3:
                warnings.append(f"Complex transform pipeline ({transform.type}) with {len(spec.provider.transforms)} transforms may impact performance")

        return warnings

    def validate_security(self, spec: UnifiedIngestionSpec) -> List[str]:
        """Validate security considerations."""
        warnings = []

        # Check for hardcoded secrets (basic check)
        json_str = spec.json()
        if 'password' in json_str.lower() or 'secret' in json_str.lower() or 'token' in json_str.lower():
            warnings.append("Potential hardcoded secrets detected - ensure credentials_ref is used")

        # Check for HTTP endpoints
        if spec.provider.base_url and spec.provider.base_url.startswith('http://'):
            warnings.append("HTTP endpoint detected - consider using HTTPS for production")

        # Check for missing authentication
        auth_endpoints = [e for e in spec.provider.endpoints if e.auth]
        if len(auth_endpoints) == 0 and spec.provider.provider_type in ['rest_api', 'graphql']:
            warnings.append("No authentication configured for API endpoints")

        return warnings

    def comprehensive_validation(self, spec: UnifiedIngestionSpec) -> Dict[str, List[str]]:
        """Run comprehensive validation and return all issues."""
        return {
            'completeness_errors': self.validate_completeness(spec),
            'field_mapping_errors': self.validate_field_mappings(spec),
            'rate_limit_errors': self.validate_rate_limits(spec),
            'sink_errors': self.validate_sink_configuration(spec),
            'performance_warnings': self.validate_performance(spec),
            'security_warnings': self.validate_security(spec)
        }

    def is_production_ready(self, spec: UnifiedIngestionSpec) -> tuple[bool, List[str]]:
        """Check if spec is ready for production deployment."""
        issues = self.comprehensive_validation(spec)

        # Production readiness criteria
        blocking_issues = (
            issues['completeness_errors'] +
            issues['field_mapping_errors'] +
            issues['rate_limit_errors'] +
            issues['sink_errors']
        )

        is_ready = len(blocking_issues) == 0

        return is_ready, blocking_issues
