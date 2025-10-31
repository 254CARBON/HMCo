"""
Extract data lineage from UIS pipeline specifications.
Generates lineage edges for OpenMetadata/DataHub.
"""

import json
from typing import Dict, List, Set, Tuple
from pathlib import Path


class LineageExtractor:
    """Extract data lineage from UIS pipeline definitions."""
    
    def __init__(self):
        self.lineage_edges = []
        
    def extract_from_uis_spec(self, spec_path: str) -> List[Dict]:
        """
        Extract lineage edges from a UIS pipeline specification.
        
        Args:
            spec_path: Path to UIS JSON specification
            
        Returns:
            List of lineage edges (upstream -> downstream)
        """
        with open(spec_path, 'r') as f:
            spec = json.load(f)
        
        edges = []
        pipeline_name = spec.get('name', 'unknown')
        
        # Extract sources
        sources = []
        if 'source' in spec:
            source_type = spec['source'].get('type', 'unknown')
            source_id = self._extract_source_id(spec['source'])
            sources.append({
                'type': source_type,
                'id': source_id,
                'name': spec['source'].get('name', source_id)
            })
        
        # Extract sinks/targets
        targets = []
        if 'target' in spec:
            target_type = spec['target'].get('type', 'unknown')
            target_id = self._extract_target_id(spec['target'])
            targets.append({
                'type': target_type,
                'id': target_id,
                'name': spec['target'].get('name', target_id),
                'database': spec['target'].get('database'),
                'table': spec['target'].get('table')
            })
        
        # Create lineage edges
        for source in sources:
            for target in targets:
                edge = {
                    'pipeline': pipeline_name,
                    'upstream': source,
                    'downstream': target,
                    'transformation': self._extract_transformations(spec)
                }
                edges.append(edge)
        
        return edges
    
    def _extract_source_id(self, source_config: Dict) -> str:
        """Extract unique identifier for source."""
        source_type = source_config.get('type', 'unknown')
        
        if source_type == 'api':
            return source_config.get('base_url', 'unknown_api')
        elif source_type == 'database':
            db = source_config.get('database', 'unknown')
            table = source_config.get('table', 'unknown')
            return f"{db}.{table}"
        elif source_type == 's3':
            return source_config.get('path', 'unknown_path')
        else:
            return source_config.get('name', 'unknown')
    
    def _extract_target_id(self, target_config: Dict) -> str:
        """Extract unique identifier for target."""
        target_type = target_config.get('type', 'unknown')
        
        if target_type in ['iceberg', 'clickhouse', 'postgres']:
            db = target_config.get('database', 'unknown')
            table = target_config.get('table', 'unknown')
            return f"{db}.{table}"
        elif target_type == 's3':
            return target_config.get('path', 'unknown_path')
        else:
            return target_config.get('name', 'unknown')
    
    def _extract_transformations(self, spec: Dict) -> List[str]:
        """Extract transformation steps from spec."""
        transformations = []
        
        if 'transforms' in spec:
            for transform in spec['transforms']:
                transform_type = transform.get('type', 'unknown')
                transformations.append(transform_type)
        
        if 'schema_mapping' in spec:
            transformations.append('schema_mapping')
        
        if 'aggregations' in spec:
            transformations.append('aggregation')
        
        return transformations
    
    def extract_from_directory(self, specs_dir: str) -> List[Dict]:
        """
        Extract lineage from all UIS specs in a directory.
        
        Args:
            specs_dir: Directory containing UIS JSON specs
            
        Returns:
            List of all lineage edges
        """
        all_edges = []
        
        for spec_file in Path(specs_dir).glob('**/*.json'):
            try:
                edges = self.extract_from_uis_spec(str(spec_file))
                all_edges.extend(edges)
            except Exception as e:
                print(f"Error processing {spec_file}: {e}")
        
        return all_edges
    
    def export_to_openmetadata(self, edges: List[Dict], output_file: str):
        """
        Export lineage edges to OpenMetadata format.
        
        Args:
            edges: List of lineage edges
            output_file: Path to output JSON file
        """
        openmetadata_lineage = {
            "entities": [],
            "edges": []
        }
        
        # Track unique entities
        entities = {}
        
        for edge in edges:
            upstream_id = edge['upstream']['id']
            downstream_id = edge['downstream']['id']
            
            # Add upstream entity
            if upstream_id not in entities:
                entities[upstream_id] = {
                    "id": upstream_id,
                    "name": edge['upstream']['name'],
                    "type": edge['upstream']['type']
                }
            
            # Add downstream entity
            if downstream_id not in entities:
                entities[downstream_id] = {
                    "id": downstream_id,
                    "name": edge['downstream']['name'],
                    "type": edge['downstream']['type']
                }
                if 'database' in edge['downstream']:
                    entities[downstream_id]['database'] = edge['downstream']['database']
                if 'table' in edge['downstream']:
                    entities[downstream_id]['table'] = edge['downstream']['table']
            
            # Add edge
            openmetadata_lineage["edges"].append({
                "fromEntity": upstream_id,
                "toEntity": downstream_id,
                "pipeline": edge['pipeline'],
                "transformations": edge.get('transformation', [])
            })
        
        openmetadata_lineage["entities"] = list(entities.values())
        
        with open(output_file, 'w') as f:
            json.dump(openmetadata_lineage, f, indent=2)
    
    def export_to_datahub(self, edges: List[Dict], output_file: str):
        """
        Export lineage edges to DataHub format.
        
        Args:
            edges: List of lineage edges
            output_file: Path to output JSON file
        """
        datahub_lineage = []
        
        for edge in edges:
            upstream_urn = self._create_datahub_urn(edge['upstream'])
            downstream_urn = self._create_datahub_urn(edge['downstream'])
            
            lineage_edge = {
                "entityType": "dataJob",
                "entityUrn": f"urn:li:dataJob:(urn:li:dataFlow:(airflow,{edge['pipeline']},PROD),{edge['pipeline']})",
                "aspect": {
                    "upstreamLineage": {
                        "upstreams": [
                            {
                                "auditStamp": {
                                    "time": 0,
                                    "actor": "urn:li:corpuser:datapipeline"
                                },
                                "dataset": upstream_urn,
                                "type": "TRANSFORMED"
                            }
                        ]
                    },
                    "downstreamLineage": {
                        "downstreams": [
                            {
                                "auditStamp": {
                                    "time": 0,
                                    "actor": "urn:li:corpuser:datapipeline"
                                },
                                "dataset": downstream_urn,
                                "type": "TRANSFORMED"
                            }
                        ]
                    }
                }
            }
            datahub_lineage.append(lineage_edge)
        
        with open(output_file, 'w') as f:
            json.dump(datahub_lineage, f, indent=2)
    
    def _create_datahub_urn(self, entity: Dict) -> str:
        """Create DataHub URN for an entity."""
        entity_type = entity['type']
        entity_id = entity['id']
        
        if entity_type in ['iceberg', 'clickhouse', 'postgres']:
            platform = entity_type
            return f"urn:li:dataset:(urn:li:dataPlatform:{platform},{entity_id},PROD)"
        elif entity_type == 'api':
            return f"urn:li:dataset:(urn:li:dataPlatform:rest,{entity_id},PROD)"
        else:
            return f"urn:li:dataset:(urn:li:dataPlatform:custom,{entity_id},PROD)"


def main():
    """Main entry point for lineage extraction."""
    extractor = LineageExtractor()
    
    # Extract lineage from UIS specs
    edges = extractor.extract_from_directory('/app/sdk/uis/compilers/*/fixtures')
    
    # Export to both OpenMetadata and DataHub
    extractor.export_to_openmetadata(edges, '/tmp/lineage_openmetadata.json')
    extractor.export_to_datahub(edges, '/tmp/lineage_datahub.json')
    
    print(f"Extracted {len(edges)} lineage edges")


if __name__ == '__main__':
    main()
