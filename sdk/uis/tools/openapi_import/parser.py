"""
OpenAPI specification parser.
"""

import copy
import json
import yaml
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import requests
import re


class OpenAPIParseError(Exception):
    """Exception raised when OpenAPI parsing fails."""
    pass


class OpenAPIParser:
    """Parser for OpenAPI 3.0+ specifications."""

    def __init__(self):
        """Initialize parser."""
        self.supported_versions = ["3.0.0", "3.0.1", "3.0.2", "3.0.3", "3.1.0"]

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse OpenAPI specification from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise OpenAPIParseError(f"File not found: {file_path}")

        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise OpenAPIParseError(f"Failed to read file {file_path}: {e}")

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

    def parse_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """Parse YAML OpenAPI content."""
        try:
            return yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise OpenAPIParseError(f"YAML parsing error: {e}")

    def parse_json(self, json_content: str) -> Dict[str, Any]:
        """Parse JSON OpenAPI content."""
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            raise OpenAPIParseError(f"JSON parsing error: {e}")

    def parse_url(self, url: str) -> Dict[str, Any]:
        """Parse OpenAPI specification from URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            content = response.text
            return self.parse_yaml(content)  # Most APIs return YAML
        except Exception as e:
            raise OpenAPIParseError(f"Failed to fetch OpenAPI from {url}: {e}")

    def validate_spec(self, spec: Dict[str, Any]) -> List[str]:
        """Validate OpenAPI specification."""
        errors = []

        # Check OpenAPI version
        version = spec.get("openapi")
        if not version:
            errors.append("Missing 'openapi' field")
        elif version not in self.supported_versions:
            errors.append(f"Unsupported OpenAPI version: {version}. Supported: {self.supported_versions}")

        # Check required fields
        required_fields = ["info", "paths"]
        for field in required_fields:
            if field not in spec:
                errors.append(f"Missing required field: {field}")

        # Check info section
        if "info" in spec:
            info = spec["info"]
            required_info_fields = ["title", "version"]
            for field in required_info_fields:
                if field not in info:
                    errors.append(f"Missing required info field: {field}")

        return errors

    def extract_endpoints(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract API endpoints from OpenAPI spec."""
        endpoints = []

        paths = spec.get("paths", {})
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            path_parameters = path_item.get("parameters", [])

            # Extract methods from path item
            for method, operation in path_item.items():
                if method.upper() not in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']:
                    continue

                if not isinstance(operation, dict):
                    continue

                endpoint = self._extract_endpoint_info(
                    spec,
                    path,
                    method,
                    operation,
                    path_parameters=path_parameters
                )
                if endpoint:
                    endpoints.append(endpoint)

        return endpoints

    def _extract_endpoint_info(
        self,
        spec: Dict[str, Any],
        path: str,
        method: str,
        operation: Dict[str, Any],
        path_parameters: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Extract endpoint information from operation."""
        parameters = self._extract_parameters(spec, operation, path_parameters or [])
        pagination = self.extract_pagination_info(parameters)

        endpoint = {
            "path": path,
            "method": method.upper(),
            "name": self._generate_endpoint_name(path, method, operation.get("operationId")),
            "summary": operation.get("summary", ""),
            "description": operation.get("description", ""),
            "operation_id": operation.get("operationId"),
            "parameters": parameters,
            "request_body": self._extract_request_body(spec, operation),
            "responses": self._extract_responses(spec, operation),
            "security": operation.get("security", []),
            "tags": operation.get("tags", []),
            "deprecated": operation.get("deprecated", False),
            "pagination": pagination
        }

        return endpoint

    def _generate_endpoint_name(self, path: str, method: str, operation_id: Optional[str] = None) -> str:
        """Generate endpoint name from path and method."""
        if operation_id:
            return self._to_snake_case(operation_id)

        # Remove leading/trailing slashes and convert to camelCase
        path_clean = (
            path.strip('/')
            .replace('/', '_')
            .replace('-', '_')
            .replace('{', '')
            .replace('}', '')
            .replace('.', '_')
        )

        # Handle path parameters
        path_clean = re.sub(r'\{[^}]+\}', 'by_id', path_clean)

        # Convert to camelCase
        parts = path_clean.split('_')
        if parts:
            name = parts[0] + ''.join(word.capitalize() for word in parts[1:])
        else:
            name = "endpoint"

        return f"{method.lower()}_{name}"

    def _extract_parameters(
        self,
        spec: Dict[str, Any],
        operation: Dict[str, Any],
        path_parameters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract parameters from operation."""
        parameters: List[Dict[str, Any]] = []
        combined_params = list(path_parameters) + operation.get("parameters", [])

        for parameter in combined_params:
            resolved_param = self._dereference_object(spec, parameter)
            if not isinstance(resolved_param, dict):
                continue

            schema = resolved_param.get("schema")
            resolved_schema = self._resolve_schema(spec, schema) if schema else {}

            parameters.append({
                "name": resolved_param.get("name"),
                "in": resolved_param.get("in"),
                "required": resolved_param.get("required", False),
                "schema": resolved_schema,
                "description": resolved_param.get("description", "")
            })

        return parameters

    def _extract_request_body(self, spec: Dict[str, Any], operation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract request body from operation."""
        request_body = operation.get("requestBody")
        if not request_body:
            return None

        request_body = self._dereference_object(spec, request_body)
        if not isinstance(request_body, dict):
            return None

        content = request_body.get("content", {})
        if not content:
            return None

        # Prefer JSON content type
        for content_type in ["application/json", "application/x-www-form-urlencoded"]:
            if content_type in content:
                return {
                    "content_type": content_type,
                    "schema": self._resolve_schema(spec, content[content_type].get("schema", {})),
                    "required": request_body.get("required", False)
                }

        # Use first available content type
        first_content_type = next(iter(content.keys()))
        return {
            "content_type": first_content_type,
            "schema": self._resolve_schema(spec, content[first_content_type].get("schema", {})),
            "required": request_body.get("required", False)
        }

    def _extract_responses(self, spec: Dict[str, Any], operation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract responses from operation."""
        responses = operation.get("responses", {})

        result = {}
        for status_code, response in responses.items():
            resolved_response = self._dereference_object(spec, response)
            if not isinstance(resolved_response, dict):
                continue

            description = resolved_response.get("description")
            if not description:
                continue

            schema = {}
            content = resolved_response.get("content", {})
            if isinstance(content, dict):
                schema = self._resolve_response_content(spec, content)

            result[status_code] = {
                "description": description,
                "schema": schema or {},
                "headers": resolved_response.get("headers", {})
            }

        return result

    def _resolve_response_content(self, spec: Dict[str, Any], content: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve schema from a response content block."""
        for content_type, content_obj in content.items():
            if "application/json" not in content_type:
                continue

            schema_obj = content_obj.get("schema")
            if not schema_obj:
                continue

            resolved_schema = self._resolve_schema(spec, schema_obj)

            # Components can point to another response; unwrap recursively
            if isinstance(resolved_schema, dict) and "content" in resolved_schema:
                inner_content = resolved_schema.get("content")
                if isinstance(inner_content, dict):
                    return self._resolve_response_content(spec, inner_content)

            return resolved_schema or {}

        return {}

    def _dereference_object(
        self,
        spec: Dict[str, Any],
        obj: Any,
        seen_refs: Optional[Set[str]] = None
    ) -> Any:
        """Resolve $ref pointers within objects."""
        if not isinstance(obj, dict) or "$ref" not in obj:
            return obj

        seen_refs = seen_refs or set()
        ref = obj.get("$ref")
        if not ref or ref in seen_refs:
            return {k: v for k, v in obj.items() if k != "$ref"}

        resolved = self._resolve_ref(spec, ref)
        if not isinstance(resolved, dict):
            return {k: v for k, v in obj.items() if k != "$ref"}

        merged = self._deep_merge_dicts(
            copy.deepcopy(resolved),
            {k: v for k, v in obj.items() if k != "$ref"}
        )
        seen_refs.add(ref)
        return self._dereference_object(spec, merged, seen_refs)

    def _resolve_ref(self, spec: Dict[str, Any], ref: str) -> Any:
        """Resolve a JSON reference within the OpenAPI spec."""
        if not ref or not ref.startswith("#/"):
            return None

        parts = ref.lstrip("#/").split("/")
        current: Any = spec

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return copy.deepcopy(current)

    def _resolve_schema(
        self,
        spec: Dict[str, Any],
        schema: Optional[Dict[str, Any]],
        seen_refs: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Resolve schema including $ref and composed schemas."""
        if not schema:
            return {}

        if not isinstance(schema, dict):
            return {"type": schema}

        resolved = copy.deepcopy(schema)
        seen_refs = seen_refs or set()

        ref = resolved.pop("$ref", None)
        if ref:
            if ref in seen_refs:
                return {}
            seen_refs.add(ref)
            ref_schema = self._resolve_ref(spec, ref) or {}
            resolved = self._deep_merge_dicts(ref_schema, resolved)

        for key in ["allOf", "oneOf", "anyOf"]:
            if key in resolved:
                composed = resolved.pop(key) or []
                if not isinstance(composed, list):
                    continue
                merged: Dict[str, Any] = {}
                for sub_schema in composed:
                    merged = self._deep_merge_dicts(
                        merged,
                        self._resolve_schema(spec, sub_schema, seen_refs.copy())
                    )
                resolved = self._deep_merge_dicts(resolved, merged)

        properties = resolved.get("properties")
        if isinstance(properties, dict):
            new_properties = {}
            for prop_name, prop_schema in properties.items():
                new_properties[prop_name] = self._resolve_schema(spec, prop_schema, seen_refs.copy())
            resolved["properties"] = new_properties

        if "items" in resolved:
            resolved["items"] = self._resolve_schema(spec, resolved.get("items"), seen_refs.copy())

        return resolved

    def _deep_merge_dicts(self, base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Deep merge dictionaries."""
        if not base:
            base = {}
        if not override:
            return copy.deepcopy(base)

        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def extract_schemas(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Extract schema definitions from OpenAPI spec."""
        schemas = {}

        # Extract components schemas
        components = spec.get("components", {})
        if "schemas" in components:
            for name, schema in components["schemas"].items():
                schemas[name] = self._resolve_schema(spec, schema)

        return schemas

    def extract_security_schemes(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Extract security schemes from OpenAPI spec."""
        components = spec.get("components", {})
        schemes = components.get("securitySchemes", {})

        resolved: Dict[str, Any] = {}
        for name, scheme in schemes.items():
            resolved[name] = self._dereference_object(spec, scheme)

        return resolved

    def infer_authentication(self, spec: Dict[str, Any], endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer authentication configuration from OpenAPI spec."""
        security_schemes = self.extract_security_schemes(spec)

        if not security_schemes:
            return {"type": "none"}

        # Check global security requirements
        global_security = spec.get("security", [])

        # Analyze endpoint security
        auth_types = set()
        for endpoint in endpoints:
            endpoint_security = endpoint.get("security", [])
            for security_req in endpoint_security + global_security:
                for scheme_name in security_req.keys():
                    if scheme_name in security_schemes:
                        scheme = security_schemes[scheme_name]
                        auth_types.add(scheme.get("type"))

        # Map OpenAPI auth types to UIS auth types
        auth_mapping = {
            "apiKey": "api_key",
            "http": "basic" if "basic" in str(security_schemes).lower() else "bearer",
            "oauth2": "oauth2",
            "openIdConnect": "oauth2"
        }

        primary_auth = auth_types.pop() if auth_types else None

        if not primary_auth:
            return {"type": "none"}

        auth_config = {"type": auth_mapping.get(primary_auth, primary_auth)}

        # Extract API key configuration
        if primary_auth == "apiKey":
            for scheme_name, scheme in security_schemes.items():
                if scheme.get("type") == "apiKey":
                    auth_config.update({
                        "header_name": scheme.get("name", "Authorization"),
                        "location": scheme.get("in", "header")
                    })
                    break

        return auth_config

    def generate_field_mapping(
        self,
        response_schema: Dict[str, Any],
        spec: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        key_path: str = ""
    ) -> Dict[str, str]:
        """Generate field mapping from response schema."""
        mapping = {}

        schema = response_schema.get("schema", response_schema)
        if spec:
            schema = self._resolve_schema(spec, schema)

        properties = schema.get("properties", {})

        for field_name, field_schema in properties.items():
            # Convert field names to snake_case for database compatibility
            db_field_name = self._to_snake_case(field_name)

            if prefix:
                db_field_name = f"{prefix}_{db_field_name}"

            source_key = f"{key_path}.{field_name}" if key_path else field_name
            mapping[source_key] = db_field_name

            # Handle nested objects
            if field_schema.get("type") == "object" and field_schema.get("properties"):
                nested_mapping = self.generate_field_mapping(field_schema, spec, db_field_name, source_key)
                mapping.update(nested_mapping)
            # Handle arrays of objects
            elif field_schema.get("type") == "array":
                items_schema = field_schema.get("items", {})
                if spec:
                    items_schema = self._resolve_schema(spec, items_schema)
                if isinstance(items_schema, dict) and items_schema.get("type") == "object":
                    nested_key_path = f"{source_key}[]"
                    nested_mapping = self.generate_field_mapping(items_schema, spec, f"{db_field_name}_item", nested_key_path)
                    mapping.update(nested_mapping)

        return mapping

    def _to_snake_case(self, name: str) -> str:
        """Convert camelCase/PascalCase to snake_case."""
        if not name:
            return name

        name = re.sub(r'[\s\-.]+', '_', name)
        name = name.replace('/', '_')
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        snake = re.sub(r'__+', '_', snake)
        return snake.strip('_')

    def extract_pagination_info(self, parameters: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract pagination information from parameters."""
        if not parameters:
            return None

        pagination_params = {}
        for param in parameters:
            param_name = param.get("name", "").lower()
            if any(keyword in param_name for keyword in ["page", "offset", "cursor", "limit", "length", "token", "next"]):
                pagination_params[param_name] = {
                    "name": param.get("name"),
                    "in": param.get("in"),
                    "required": param.get("required", False),
                    "schema": param.get("schema", {})
                }

        # Determine pagination type based on parameters
        if "cursor" in pagination_params or "next" in pagination_params or "token" in pagination_params:
            return {"type": "cursor", "config": pagination_params}
        elif "offset" in pagination_params or "page" in pagination_params:
            return {"type": "offset", "config": pagination_params}
        else:
            return {"type": "custom", "config": pagination_params} if pagination_params else None

    def extract_rate_limits(self, spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract rate limiting information from OpenAPI spec."""
        # Look for rate limit info in headers or descriptions
        info = spec.get("info", {})
        description = info.get("description", "").lower()

        # Common rate limit patterns
        if "rate limit" in description or "requests per" in description:
            # Try to extract rate limit from description
            match = re.search(r'(\d+)\s*(?:requests?|calls?)\s*(?:per\s*)?(\w+)', description)
            if match:
                limit, period = match.groups()
                return {
                    "requests_per_second": int(limit) if period in ["second", "sec"] else None,
                    "requests_per_minute": int(limit) if period in ["minute", "min"] else None,
                    "requests_per_hour": int(limit) if period in ["hour", "hr"] else None
                }

        return None
