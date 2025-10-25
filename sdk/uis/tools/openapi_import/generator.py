"""
UIS specification generator from OpenAPI specs.
"""

import json
import uuid
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from parser import OpenAPIParser
from spec import (
    UnifiedIngestionSpec,
    ProviderConfig,
    EndpointConfig,
    ProviderType,
    SinkType,
    IngestionMode,
    AuthType,
    PaginationType,
)


class UISGenerationError(Exception):
    """Exception raised when UIS generation fails."""
    pass


class UISGenerator:
    """Generates UIS specifications from OpenAPI specs."""

    def __init__(self, default_tenant: str = "default", default_owner: str = "api-importer"):
        """Initialize generator with defaults."""
        self.default_tenant = default_tenant
        self.default_owner = default_owner

    def generate_from_openapi(self, openapi_spec: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> UnifiedIngestionSpec:
        """Generate UIS specification from OpenAPI spec."""
        options = options or {}

        # Parse OpenAPI spec
        parser = OpenAPIParser()
        validation_errors = parser.validate_spec(openapi_spec)
        if validation_errors:
            raise UISGenerationError(f"OpenAPI validation failed: {validation_errors}")

        # Extract basic info
        info = openapi_spec.get("info", {})
        title = info.get("title", "API Import")
        description = info.get("description", "")
        version = info.get("version", "1.0")

        # Generate provider configuration
        provider = self._generate_provider_config(openapi_spec, options)

        # Generate endpoints
        endpoints = parser.extract_endpoints(openapi_spec)

        # Filter endpoints based on options
        if not options.get("include_deprecated", False):
            endpoints = [e for e in endpoints if not e.get("deprecated", False)]

        if options.get("methods"):
            allowed_methods = [m.upper() for m in options["methods"]]
            endpoints = [e for e in endpoints if e["method"] in allowed_methods]

        filter_tags = options.get("filter_tags")
        if filter_tags:
            allowed_tags = set(filter_tags)
            endpoints = [e for e in endpoints if any(tag in e.get("tags", []) for tag in allowed_tags)]

        # Convert endpoints to UIS format
        uis_endpoints = [self._convert_endpoint(e, openapi_spec) for e in endpoints]

        # Update provider with endpoints
        provider.endpoints = uis_endpoints

        # Generate complete UIS spec
        uis_spec = UnifiedIngestionSpec(
            version="1.1",
            name=f"{title.lower().replace(' ', '-')}-api",
            description=f"Generated from {title} API v{version}. {description}",
            provider=provider,
            created_by=self.default_owner
        )

        return uis_spec

    def _generate_provider_config(self, openapi_spec: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> ProviderConfig:
        """Generate provider configuration."""
        options = options or {}
        info = openapi_spec.get("info", {})
        servers = openapi_spec.get("servers", [])

        # Determine base URL
        base_url = options.get("base_url")
        if not base_url and servers:
            server = servers[0]
            base_url = server.get("url")
            if not base_url.startswith("http"):
                base_url = f"https://{base_url}"

        # Determine provider type
        provider_type = self._infer_provider_type(openapi_spec, options)

        # Generate authentication config
        parser = OpenAPIParser()
        endpoints = parser.extract_endpoints(openapi_spec)
        auth_config = parser.infer_authentication(openapi_spec, endpoints)

        # Generate rate limits
        rate_limits = parser.extract_rate_limits(openapi_spec)

        # Generate schema contract from common responses
        schema_contract = self._generate_schema_contract(openapi_spec)

        provider_tags = options.get("provider_tags") or options.get("tags") or [info.get("title", "API")]
        provider_config = {
            "api_version": info.get("version"),
            **options.get("provider_config", {})
        }
        if rate_limits:
            provider_config["rate_limits"] = rate_limits

        return ProviderConfig(
            name=f"{info.get('title', 'API').lower().replace(' ', '-')}",
            display_name=info.get("title", "API Provider"),
            description=info.get("description", ""),
            provider_type=provider_type,
            base_url=base_url,
            config=provider_config,
            credentials_ref=options.get("credentials_ref"),
            tenant_id=options.get("tenant_id", self.default_tenant),
            owner=options.get("owner", self.default_owner),
            tags=provider_tags,
            mode=self._infer_ingestion_mode(openapi_spec, options),
            sinks=self._generate_default_sinks(options),
            schema_contract=schema_contract,
            slos=self._generate_default_slos(options)
        )

    def _infer_provider_type(self, openapi_spec: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> ProviderType:
        """Infer provider type from OpenAPI spec."""
        # Check for explicit type in options
        if options and options.get("provider_type"):
            return ProviderType(options["provider_type"])

        # Analyze endpoints to infer type
        parser = OpenAPIParser()
        endpoints = parser.extract_endpoints(openapi_spec)

        # Check for streaming/websocket patterns
        for endpoint in endpoints:
            path = endpoint.get("path", "").lower()
            if "stream" in path or "websocket" in path or "realtime" in path:
                return ProviderType.WEBSOCKET

        # Check for webhook patterns
        if any("webhook" in ep.get("path", "").lower() for ep in endpoints):
            return ProviderType.WEBHOOK

        # Default to REST API
        return ProviderType.REST_API

    def _infer_ingestion_mode(self, openapi_spec: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> IngestionMode:
        """Infer ingestion mode from OpenAPI spec."""
        if options and options.get("mode"):
            return IngestionMode(options["mode"])

        # Check for streaming indicators
        description = openapi_spec.get("info", {}).get("description", "").lower()
        if any(word in description for word in ["streaming", "realtime", "live"]):
            return IngestionMode.STREAMING

        # Check for webhook patterns
        paths = openapi_spec.get("paths", {})
        if any("webhook" in path.lower() for path in paths.keys()):
            return IngestionMode.WEBHOOK

        # Default to batch
        return IngestionMode.BATCH

    def _generate_default_sinks(self, options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate default sink configurations."""
        sinks = []

        # Iceberg sink (default for analytical data)
        sinks.append({
            "type": SinkType.ICEBERG,
            "table_name": options.get("default_table", "api_data"),
            "config": {
                "catalog": "hive_prod",
                "namespace": options.get("default_namespace", "default"),
                "warehouse": options.get("warehouse", "s3://warehouse/")
            }
        })

        # Add ClickHouse sink if specified
        if options.get("enable_clickhouse", False):
            sinks.append({
                "type": SinkType.CLICKHOUSE,
                "clickhouse_table": options.get("default_table", "api_data"),
                "config": {
                    "host": options.get("clickhouse_host", "clickhouse"),
                    "database": options.get("clickhouse_database", "default")
                }
            })

        return sinks

    def _generate_schema_contract(self, openapi_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate JSON schema contract from OpenAPI responses."""
        parser = OpenAPIParser()
        schemas = parser.extract_schemas(openapi_spec)

        if not schemas:
            return None

        # Use the first schema as the base contract
        first_schema_name = next(iter(schemas.keys()))
        schema = schemas[first_schema_name]

        # Convert OpenAPI schema to JSON Schema format
        return self._convert_openapi_schema_to_json_schema(schema)

    def _convert_openapi_schema_to_json_schema(self, openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAPI schema to JSON Schema format."""
        json_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        if "properties" in openapi_schema:
            json_schema["properties"] = {}
            for prop_name, prop_schema in openapi_schema["properties"].items():
                json_schema["properties"][prop_name] = self._convert_property_schema(prop_schema)

        if "required" in openapi_schema:
            json_schema["required"] = openapi_schema["required"]

        return json_schema

    def _convert_property_schema(self, prop_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAPI property schema to JSON Schema."""
        json_prop = {}

        # Map OpenAPI types to JSON Schema types
        type_mapping = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object"
        }

        if "type" in prop_schema:
            json_prop["type"] = type_mapping.get(prop_schema["type"], "string")

        if "format" in prop_schema:
            json_prop["format"] = prop_schema["format"]

        if "description" in prop_schema:
            json_prop["description"] = prop_schema["description"]

        if "enum" in prop_schema:
            json_prop["enum"] = prop_schema["enum"]

        if "minimum" in prop_schema:
            json_prop["minimum"] = prop_schema["minimum"]

        if "maximum" in prop_schema:
            json_prop["maximum"] = prop_schema["maximum"]

        if "minLength" in prop_schema:
            json_prop["minLength"] = prop_schema["minLength"]

        if "maxLength" in prop_schema:
            json_prop["maxLength"] = prop_schema["maxLength"]

        # Handle arrays
        if prop_schema.get("type") == "array":
            if "items" in prop_schema:
                json_prop["items"] = self._convert_property_schema(prop_schema["items"])

        # Handle objects
        if prop_schema.get("type") == "object":
            if "properties" in prop_schema:
                json_prop["properties"] = {
                    name: self._convert_property_schema(sub_prop)
                    for name, sub_prop in prop_schema["properties"].items()
                }

        return json_prop

    def _convert_endpoint(self, endpoint: Dict[str, Any], openapi_spec: Dict[str, Any]) -> EndpointConfig:
        """Convert OpenAPI endpoint to UIS endpoint."""
        # Determine authentication
        auth_type = self._infer_endpoint_auth(endpoint, openapi_spec)

        # Generate field mapping from response schema
        response_schema, response_path, mapping_key_path = self._resolve_response_schema_and_path(endpoint)
        field_mapping = self._generate_field_mapping_from_schema(response_schema, openapi_spec, mapping_key_path)

        # Extract pagination
        pagination = self._extract_pagination_from_endpoint(endpoint)
        pagination_type = PaginationType.NONE
        pagination_config = None
        if pagination:
            pagination_type_value = pagination.get("type", "none")
            try:
                pagination_type = PaginationType(pagination_type_value)
            except ValueError:
                pagination_type = PaginationType.NONE
            pagination_config = pagination.get("config")

        # Extract rate limits
        rate_limit = self._extract_rate_limit_from_endpoint(endpoint)

        return EndpointConfig(
            name=endpoint["name"],
            path=endpoint["path"],
            method=endpoint["method"],
            auth=auth_type,
            auth_config=self._generate_auth_config(endpoint, openapi_spec),
            headers=self._extract_headers_from_endpoint(endpoint),
            query_params=self._extract_query_params_from_endpoint(endpoint),
            pagination=pagination_type,
            pagination_config=pagination_config,
            response_path=response_path,
            field_mapping=field_mapping,
            rate_limit_per_second=rate_limit,
            validation_rules=self._generate_validation_rules(response_schema)
        )

    def _infer_endpoint_auth(self, endpoint: Dict[str, Any], openapi_spec: Dict[str, Any]) -> Optional[AuthType]:
        """Infer authentication type for endpoint."""
        # Check endpoint security
        security = endpoint.get("security", [])

        if not security:
            # Check global security
            security = openapi_spec.get("security", [])

        if not security:
            return None

        # Extract security schemes
        components = openapi_spec.get("components", {})
        security_schemes = components.get("securitySchemes", {})

        # Map to UIS auth types
        for security_req in security:
            for scheme_name in security_req.keys():
                if scheme_name in security_schemes:
                    scheme = security_schemes[scheme_name]
                    scheme_type = scheme.get("type")

                    if scheme_type == "apiKey":
                        return AuthType.API_KEY
                    elif scheme_type == "http":
                        return AuthType.BASIC if "basic" in scheme.get("scheme", "").lower() else AuthType.BEARER
                    elif scheme_type == "oauth2":
                        return AuthType.OAUTH2

        return None

    def _generate_auth_config(self, endpoint: Dict[str, Any], openapi_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate authentication configuration."""
        security = endpoint.get("security", []) or openapi_spec.get("security", [])
        components = openapi_spec.get("components", {})
        security_schemes = components.get("securitySchemes", {})

        for security_req in security:
            for scheme_name in security_req.keys():
                if scheme_name in security_schemes:
                    scheme = security_schemes[scheme_name]

                    if scheme.get("type") == "apiKey":
                        return {
                            "header_name": scheme.get("name", "Authorization"),
                            "location": scheme.get("in", "header")
                        }
                    elif scheme.get("type") == "oauth2":
                        return {
                            "flow": scheme.get("flow", "client_credentials"),
                            "token_url": scheme.get("tokenUrl"),
                            "scopes": scheme.get("scopes", [])
                        }

        return None

    def _extract_headers_from_endpoint(self, endpoint: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract headers from endpoint."""
        # For now, return basic headers
        # In a full implementation, this would extract from OpenAPI parameters
        return None

    def _extract_query_params_from_endpoint(self, endpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract query parameters from endpoint."""
        parameters = endpoint.get("parameters", [])
        query_params = {}

        for param in parameters:
            if param.get("in") == "query":
                param_name = param.get("name")
                if param_name:
                    # Use default value if available, otherwise use parameter name
                    default_value = param.get("schema", {}).get("default")
                    query_params[param_name] = default_value if default_value is not None else f"{{{param_name}}}"

        return query_params if query_params else None

    def _extract_response_path(self, endpoint: Dict[str, Any]) -> Optional[str]:
        """Extract response path for data extraction."""
        _, response_path, _ = self._resolve_response_schema_and_path(endpoint)
        return response_path

    def _extract_response_schema(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract response schema from endpoint."""
        schema, _, _ = self._resolve_response_schema_and_path(endpoint)
        return schema

    def _resolve_response_schema_and_path(self, endpoint: Dict[str, Any]) -> (Dict[str, Any], str, str):
        """Resolve the response schema and JSON path for an endpoint."""
        responses = endpoint.get("responses", {})

        for status_code in ["200", "201", "202"]:
            response = responses.get(status_code)
            if not response:
                continue

            schema = response.get("schema", {})
            path, mapping_schema, key_path = self._unwrap_response_schema(schema)
            return mapping_schema, path, key_path

        return {}, "$", ""

    def _unwrap_response_schema(self, schema: Dict[str, Any]) -> (str, Dict[str, Any], str):
        """Determine JSON path and mapping schema by unwrapping response containers."""
        if not isinstance(schema, dict):
            return "$", {}, ""

        path_parts: List[str] = []
        current_schema = schema
        last_object_schema = schema if schema.get("type") == "object" and schema.get("properties") else schema
        path_stack: List[tuple[str, str, Dict[str, Any]]] = []

        wrapper_priority = [
            "response",
            "data",
            "results",
            "items",
            "series",
            "records",
            "category",
            "childseries",
            "value"
        ]

        while isinstance(current_schema, dict):
            properties = current_schema.get("properties")
            if not isinstance(properties, dict) or not properties:
                break

            wrapper_prop = next((prop for prop in wrapper_priority if prop in properties), None)
            if not wrapper_prop:
                break

            path_parts.append(wrapper_prop)
            next_schema = properties.get(wrapper_prop, {})

            schema_type = ""
            if isinstance(next_schema, dict):
                schema_type = next_schema.get("type") or ""
                if not schema_type and next_schema.get("properties"):
                    schema_type = "object"
                elif not schema_type and next_schema.get("items"):
                    schema_type = "array"
            path_stack.append((wrapper_prop, schema_type or "unknown", next_schema if isinstance(next_schema, dict) else {}))

            if isinstance(next_schema, dict) and next_schema.get("type") == "object" and next_schema.get("properties"):
                last_object_schema = next_schema
                current_schema = next_schema
                continue

            current_schema = next_schema
            break

        mapping_schema = current_schema if isinstance(current_schema, dict) else {}
        if isinstance(mapping_schema, dict) and mapping_schema.get("type") == "array":
            items_schema = mapping_schema.get("items")
            if isinstance(items_schema, dict):
                mapping_schema = items_schema
        if not (isinstance(mapping_schema, dict) and (mapping_schema.get("properties") or mapping_schema.get("items"))):
            mapping_schema = last_object_schema if isinstance(last_object_schema, dict) else {}

        # Build key path for field mapping
        key_segments: List[str] = []
        for name, schema_type, _ in path_stack:
            if schema_type == "array":
                key_segments.append(f"{name}[]")
            else:
                key_segments.append(name)

        key_depth = len(key_segments)
        if mapping_schema is last_object_schema:
            # Find the segment corresponding to last_object_schema
            for idx in range(len(path_stack) - 1, -1, -1):
                if path_stack[idx][2] is last_object_schema:
                    key_depth = idx + 1
                    break

        response_path = "$"
        effective_parts = path_parts[:key_depth] if key_depth else []
        if effective_parts:
            response_path = "$." + ".".join(effective_parts)

        key_path = ".".join(key_segments[:key_depth]) if key_depth else ""

        return response_path, mapping_schema, key_path

    def _generate_field_mapping_from_schema(
        self,
        response_schema: Dict[str, Any],
        openapi_spec: Dict[str, Any],
        key_path: str
    ) -> Dict[str, str]:
        """Generate field mapping from response schema."""
        if not response_schema:
            return {}

        # Use the parser's field mapping generation
        parser = OpenAPIParser()
        prefix = ""
        if key_path:
            last_segment = key_path.split(".")[-1]
            is_array = last_segment.endswith("[]")
            segment_name = last_segment[:-2] if is_array else last_segment
            if segment_name:
                snake_segment = parser._to_snake_case(segment_name)
                prefix = f"{snake_segment}_item" if is_array else snake_segment

        return parser.generate_field_mapping(response_schema, openapi_spec, prefix=prefix, key_path=key_path)

    def _extract_pagination_from_endpoint(self, endpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract pagination configuration from endpoint."""
        pagination = endpoint.get("pagination")
        if not pagination:
            return None

        config = {}
        for entry in (pagination.get("config") or {}).values():
            if not isinstance(entry, dict):
                continue
            param_name = entry.get("name")
            if not param_name:
                continue
            schema = entry.get("schema", {})
            config[param_name] = {
                "in": entry.get("in"),
                "required": entry.get("required", False),
                "description": entry.get("description"),
                "type": schema.get("type"),
                "format": schema.get("format"),
                "default": schema.get("default"),
            }

        return {
            "type": pagination.get("type", "none"),
            "config": config or None
        }

    def _extract_rate_limit_from_endpoint(self, endpoint: Dict[str, Any]) -> Optional[int]:
        """Extract rate limit from endpoint."""
        # This would extract from endpoint metadata or parameters
        # For now, return None
        return None

    def _generate_validation_rules(self, response_schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate validation rules from response schema."""
        if not response_schema:
            return None

        rules = {
            "required_fields": response_schema.get("required", []),
            "field_types": {}
        }

        # Extract field types from schema
        properties = response_schema.get("properties", {})
        for field_name, field_schema in properties.items():
            field_type = field_schema.get("type", "string")
            rules["field_types"][field_name] = field_type

        return rules

    def _generate_default_slos(self, options: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Generate default SLO configuration."""
        return {
            "freshness_target_minutes": options.get("freshness_target_minutes", 60),
            "accuracy_threshold": options.get("accuracy_threshold", 0.99),
            "completeness_threshold": options.get("completeness_threshold", 0.95),
            "availability_target": options.get("availability_target", 0.999),
            "block_on_schema_drift": options.get("block_on_schema_drift", True),
            "block_on_quality_drop": options.get("block_on_quality_drop", True)
        }

    def generate_from_url(self, url: str, options: Optional[Dict[str, Any]] = None) -> UnifiedIngestionSpec:
        """Generate UIS specification from OpenAPI URL."""
        parser = OpenAPIParser()
        openapi_spec = parser.parse_url(url)
        return self.generate_from_openapi(openapi_spec, options)

    def generate_to_yaml(self, uis_spec: UnifiedIngestionSpec) -> str:
        """Generate YAML representation of UIS spec."""
        return uis_spec.to_yaml()

    def generate_to_json(self, uis_spec: UnifiedIngestionSpec) -> str:
        """Generate JSON representation of UIS spec."""
        return uis_spec.to_json()
