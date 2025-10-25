"""
UIS 1.1 Parser and Validator.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jsonschema import Draft7Validator, exceptions as jsonschema_exceptions
from pydantic import ValidationError

from spec import EndpointConfig, ProviderConfig, UnifiedIngestionSpec


class UISParseError(Exception):
    """Exception raised when UIS parsing fails."""
    pass


class UISValidationError(Exception):
    """Exception raised when UIS validation fails."""
    pass


class UISParser:
    """Parser for Unified Ingestion Specification files."""

    def __init__(self, schema_path: Optional[str] = None):
        """Initialize parser with optional schema validation."""
        self.schema_path = schema_path or self._get_default_schema_path()
        self._schema_validator: Optional[Draft7Validator] = None
        self._schema = None

    def _get_default_schema_path(self) -> Optional[str]:
        """Get the default JSON schema path."""
        current_dir = Path(__file__).parent
        schema_file = current_dir / "schema" / "uis-1.1.json"
        return str(schema_file) if schema_file.exists() else None

    def load_schema(self) -> Dict[str, Any]:
        """Load the JSON schema for validation."""
        if not self.schema_path:
            return None

        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise UISParseError(f"Failed to load schema from {self.schema_path}: {e}")

    @property
    def schema(self) -> Optional[Dict[str, Any]]:
        """Get the JSON schema."""
        if self._schema is None:
            self._schema = self.load_schema()
        return self._schema

    def _get_schema_validator(self) -> Optional[Draft7Validator]:
        """Compile and cache the JSON schema validator."""
        if not self.schema:
            return None

        if self._schema_validator is None:
            try:
                self._schema_validator = Draft7Validator(self.schema)
            except jsonschema_exceptions.SchemaError as e:
                raise UISParseError(f"Invalid UIS schema definition: {e}") from e

        return self._schema_validator

    def _validate_with_schema(self, data: Dict[str, Any]) -> None:
        """Validate raw data against the UIS JSON schema."""
        validator = self._get_schema_validator()
        if not validator:
            return

        errors = sorted(validator.iter_errors(data), key=lambda err: list(err.path))
        if errors:
            formatted_errors = [
                f"{'.'.join(str(p) for p in error.path) or '<root>'}: {error.message}"
                for error in errors
            ]
            raise UISValidationError("Schema validation failed: " + "; ".join(formatted_errors))

    def _collect_schema_errors(self, spec: UnifiedIngestionSpec) -> List[str]:
        """Return schema validation errors for a validated spec."""
        validator = self._get_schema_validator()
        if not validator:
            return []

        try:
            payload = spec.model_dump(mode='json')  # type: ignore[attr-defined]
        except AttributeError:
            payload = spec.dict()

        errors = []
        for error in validator.iter_errors(payload):
            location = '.'.join(str(p) for p in error.path) or '<root>'
            errors.append(f"{location}: {error.message}")
        return errors

    def parse_file(self, file_path: str) -> UnifiedIngestionSpec:
        """Parse a UIS file (YAML or JSON)."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise UISParseError(f"File not found: {file_path}")

        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise UISParseError(f"Failed to read file {file_path}: {e}")

        # Parse based on file extension
        if file_path.suffix.lower() in ['.yml', '.yaml']:
            return self.parse_yaml(content)
        elif file_path.suffix.lower() == '.json':
            return self.parse_json(content)
        else:
            # Try YAML first, then JSON
            try:
                return self.parse_yaml(content)
            except Exception:
                return self.parse_json(content)

    def parse_yaml(self, yaml_content: str) -> UnifiedIngestionSpec:
        """Parse YAML content into UIS spec."""
        try:
            # Parse YAML
            data = yaml.safe_load(yaml_content)
            if data is None:
                raise UISParseError("Empty YAML content")
            if not isinstance(data, dict):
                raise UISParseError("Top-level YAML content must be a mapping/object")

            # Schema validation
            self._validate_with_schema(data)

            # Validate with Pydantic model
            return UnifiedIngestionSpec(**data)

        except yaml.YAMLError as e:
            raise UISParseError(f"YAML parsing error: {e}")
        except UISValidationError:
            raise
        except ValidationError as e:
            raise UISValidationError(f"UIS validation error: {e}")

    def parse_json(self, json_content: str) -> UnifiedIngestionSpec:
        """Parse JSON content into UIS spec."""
        try:
            # Parse JSON
            data = json.loads(json_content)

            if data is None:
                raise UISParseError("Empty JSON content")
            if not isinstance(data, dict):
                raise UISParseError("Top-level JSON content must be an object")

            # Schema validation
            self._validate_with_schema(data)

            # Validate with Pydantic model
            return UnifiedIngestionSpec(**data)

        except json.JSONDecodeError as e:
            raise UISParseError(f"JSON parsing error: {e}")
        except UISValidationError:
            raise
        except ValidationError as e:
            raise UISValidationError(f"UIS validation error: {e}")

    def parse_string(self, content: str, format: str = 'auto') -> UnifiedIngestionSpec:
        """Parse string content (auto-detect format or specify)."""
        if format == 'yaml' or (format == 'auto' and ('\n' in content or ':' in content)):
            return self.parse_yaml(content)
        else:
            return self.parse_json(content)

    def validate_spec(self, spec: UnifiedIngestionSpec) -> List[str]:
        """Validate a parsed UIS spec and return validation errors."""
        errors = []

        # Basic structure validation
        if not spec.provider:
            errors.append("Provider configuration is required")

        if not spec.provider.sinks:
            errors.append("At least one sink configuration is required")

        # Provider-specific validation
        if spec.provider.provider_type in ['rest_api', 'graphql'] and not spec.provider.base_url:
            errors.append("base_url is required for REST API and GraphQL providers")

        # Endpoint validation
        endpoint_names = set()
        for endpoint in spec.provider.endpoints:
            if endpoint.name in endpoint_names:
                errors.append(f"Duplicate endpoint name: {endpoint.name}")
            endpoint_names.add(endpoint.name)

            # Pagination validation
            if endpoint.pagination != 'none' and not endpoint.pagination_config:
                errors.append(f"Pagination config required for endpoint {endpoint.name}")

        # Transform validation
        transform_names = set()
        for transform in spec.provider.transforms:
            if transform.name in transform_names:
                errors.append(f"Duplicate transform name: {transform.name}")
            transform_names.add(transform.name)

        # Schema validation against JSON schema if available
        schema_errors = self._collect_schema_errors(spec)
        errors.extend(schema_errors)

        return errors

    def lint_spec(self, spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Lint a UIS spec and provide suggestions."""
        suggestions = {
            'warnings': [],
            'recommendations': [],
            'best_practices': []
        }

        # Check for missing descriptions
        if not spec.description:
            suggestions['recommendations'].append("Add a description to explain the purpose of this spec")

        if not spec.provider.description:
            suggestions['recommendations'].append("Add a description to the provider configuration")

        # Check for missing SLOs
        if not spec.provider.slos:
            suggestions['recommendations'].append("Consider adding SLO configuration for quality gates")

        # Check endpoint configurations
        for endpoint in spec.provider.endpoints:
            if not endpoint.response_path:
                suggestions['warnings'].append(f"Endpoint {endpoint.name} missing response_path - may need manual extraction")

            if endpoint.pagination == 'none' and 'list' in endpoint.name.lower():
                suggestions['recommendations'].append(f"Endpoint {endpoint.name} appears to be a list - consider pagination")

        # Check sink configurations
        for sink in spec.provider.sinks:
            if sink.type == 'iceberg' and not sink.table_name:
                suggestions['warnings'].append("Iceberg sink missing table_name")

        # Best practices
        if len(spec.provider.endpoints) > 10:
            suggestions['best_practices'].append("Consider splitting large specs with many endpoints")

        if spec.provider.mode == 'streaming' and len(spec.provider.transforms) > 3:
            suggestions['best_practices'].append("Streaming mode with many transforms may impact performance")

        return suggestions


def parse_uis_file(file_path: str) -> UnifiedIngestionSpec:
    """Convenience function to parse a UIS file."""
    parser = UISParser()
    return parser.parse_file(file_path)


def parse_uis_yaml(yaml_content: str) -> UnifiedIngestionSpec:
    """Convenience function to parse UIS YAML content."""
    parser = UISParser()
    return parser.parse_yaml(yaml_content)


def parse_uis_json(json_content: str) -> UnifiedIngestionSpec:
    """Convenience function to parse UIS JSON content."""
    parser = UISParser()
    return parser.parse_json(json_content)
