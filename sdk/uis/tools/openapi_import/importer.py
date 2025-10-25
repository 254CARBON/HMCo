"""
Main OpenAPI importer tool.
"""

import json
import yaml
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))
from parser import OpenAPIParser, OpenAPIParseError
from generator import UISGenerator, UISGenerationError


class OpenAPIImporter:
    """Main OpenAPI to UIS importer."""

    def __init__(self):
        """Initialize importer."""
        self.parser = OpenAPIParser()
        self.generator = UISGenerator()

    def import_from_file(self, file_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Import OpenAPI spec from file and generate UIS."""
        try:
            # Parse OpenAPI spec
            openapi_spec = self.parser.parse_file(file_path)

            # Validate OpenAPI spec
            validation_errors = self.parser.validate_spec(openapi_spec)
            if validation_errors:
                return {
                    "success": False,
                    "errors": validation_errors,
                    "uis_spec": None
                }

            # Generate UIS spec
            uis_spec = self.generator.generate_from_openapi(openapi_spec, options)

            return {
                "success": True,
                "errors": [],
                "uis_spec": uis_spec,
                "metadata": {
                    "source": file_path,
                    "openapi_version": openapi_spec.get("openapi"),
                    "endpoints_count": len(uis_spec.provider.endpoints),
                    "provider_type": uis_spec.provider.provider_type.value
                }
            }

        except (OpenAPIParseError, UISGenerationError) as e:
            return {
                "success": False,
                "errors": [str(e)],
                "uis_spec": None
            }

    def import_from_url(self, url: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Import OpenAPI spec from URL and generate UIS."""
        try:
            # Parse OpenAPI spec from URL
            openapi_spec = self.parser.parse_url(url)

            # Validate OpenAPI spec
            validation_errors = self.parser.validate_spec(openapi_spec)
            if validation_errors:
                return {
                    "success": False,
                    "errors": validation_errors,
                    "uis_spec": None
                }

            # Generate UIS spec
            uis_spec = self.generator.generate_from_openapi(openapi_spec, options)

            return {
                "success": True,
                "errors": [],
                "uis_spec": uis_spec,
                "metadata": {
                    "source": url,
                    "openapi_version": openapi_spec.get("openapi"),
                    "endpoints_count": len(uis_spec.provider.endpoints),
                    "provider_type": uis_spec.provider.provider_type.value
                }
            }

        except (OpenAPIParseError, UISGenerationError) as e:
            return {
                "success": False,
                "errors": [str(e)],
                "uis_spec": None
            }

    def export_to_yaml(self, result: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """Export UIS spec to YAML file."""
        if not result["success"] or not result["uis_spec"]:
            return None

        yaml_content = result["uis_spec"].to_yaml()

        if output_path:
            with open(output_path, 'w') as f:
                f.write(yaml_content)

        return yaml_content

    def export_to_json(self, result: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
        """Export UIS spec to JSON file."""
        if not result["success"] or not result["uis_spec"]:
            return None

        json_content = result["uis_spec"].to_json()

        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_content)

        return json_content

    def get_prebuilt_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get prebuilt templates for common APIs."""
        return {
            "polygon": {
                "base_url": "https://api.polygon.io",
                "credentials_ref": "vault://polygon-api/credentials",
                "default_table": "market_data.polygon_data",
                "default_namespace": "market_data",
                "enable_clickhouse": True,
                "freshness_target_minutes": 5,
                "provider_tags": ["stocks", "market-data", "polygon"]
            },
            "eia": {
                "base_url": "https://api.eia.gov",
                "credentials_ref": "vault://eia-api/credentials",
                "default_table": "energy.eia_data",
                "default_namespace": "energy",
                "mode": "batch",
                "schedule_cron": "0 */6 * * *",  # Every 6 hours
                "provider_tags": ["energy", "eia", "government"]
            },
            "alpha_vantage": {
                "base_url": "https://www.alphavantage.co",
                "credentials_ref": "vault://alpha-vantage-api/credentials",
                "default_table": "market_data.alpha_vantage",
                "default_namespace": "market_data",
                "provider_config": {"rate_limits": {"requests_per_minute": 5}},  # Alpha Vantage free tier
                "provider_tags": ["stocks", "forex", "crypto"]
            },
            "weather": {
                "default_table": "weather.weather_data",
                "default_namespace": "weather",
                "enable_clickhouse": True,
                "freshness_target_minutes": 30,
                "provider_tags": ["weather", "meteorology"]
            }
        }


def main():
    """Command-line interface for OpenAPI importer."""
    parser = argparse.ArgumentParser(description="Import OpenAPI specs and generate UIS specifications")
    parser.add_argument("source", help="OpenAPI spec file path or URL")
    parser.add_argument("--output", "-o", help="Output UIS file path")
    parser.add_argument("--format", "-f", choices=["yaml", "json"], default="yaml", help="Output format")
    parser.add_argument("--template", "-t", choices=["polygon", "eia", "alpha_vantage", "weather"], help="Prebuilt template")
    parser.add_argument("--provider-type", choices=["rest_api", "graphql", "websocket"], help="Override provider type")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    parser.add_argument("--owner", default="api-importer", help="Data owner")
    parser.add_argument("--table", help="Default table name")
    parser.add_argument("--namespace", help="Default namespace")
    parser.add_argument("--clickhouse", action="store_true", help="Enable ClickHouse sink")
    parser.add_argument("--methods", nargs="+", help="Filter by HTTP methods")
    parser.add_argument("--tags", "--filter-tags", dest="filter_tags", nargs="+", help="Filter by API tags")
    parser.add_argument("--provider-tags", nargs="+", help="Override provider metadata tags")
    parser.add_argument("--include-deprecated", action="store_true", help="Include deprecated endpoints")

    args = parser.parse_args()

    # Build options
    options = {
        "tenant_id": args.tenant,
        "owner": args.owner,
        "provider_config": {}
    }

    if args.template:
        importer = OpenAPIImporter()
        templates = importer.get_prebuilt_templates()
        if args.template in templates:
            options.update(templates[args.template])
        else:
            print(f"Warning: Unknown template {args.template}")

    if args.provider_type:
        options["provider_type"] = args.provider_type

    if args.table:
        options["default_table"] = args.table

    if args.namespace:
        options["default_namespace"] = args.namespace

    if args.clickhouse:
        options["enable_clickhouse"] = True

    if args.methods:
        options["methods"] = args.methods

    if args.filter_tags:
        options["filter_tags"] = args.filter_tags

    if args.provider_tags:
        options["provider_tags"] = args.provider_tags

    if args.include_deprecated:
        options["include_deprecated"] = True

    # Import and generate
    importer = OpenAPIImporter()

    if args.source.startswith("http"):
        result = importer.import_from_url(args.source, options)
    else:
        result = importer.import_from_file(args.source, options)

    # Output results
    if result["success"]:
        print(f"✓ Successfully imported {result['metadata']['endpoints_count']} endpoints")
        print(f"  Provider type: {result['metadata']['provider_type']}")
        print(f"  OpenAPI version: {result['metadata']['openapi_version']}")

        if args.format == "yaml":
            output_content = importer.export_to_yaml(result, args.output)
        else:
            output_content = importer.export_to_json(result, args.output)

        if args.output:
            print(f"✓ Saved UIS spec to {args.output}")
        else:
            print("\nGenerated UIS specification:")
            print("=" * 50)
            print(output_content)

    else:
        print("❌ Import failed:")
        for error in result["errors"]:
            print(f"  - {error}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
