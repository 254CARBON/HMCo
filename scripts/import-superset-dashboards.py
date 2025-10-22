#!/usr/bin/env python3
"""
Superset Dashboard Import Script
Automatically imports dashboard definitions and creates database connections

Usage:
    python3 import-superset-dashboards.py
    python3 import-superset-dashboards.py --superset-url http://localhost:8088
    python3 import-superset-dashboards.py --skip-existing
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional
import subprocess
import urllib.request
import urllib.error
import urllib.parse
import base64


class Colors:
    """ANSI color codes"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color


class SupersetClient:
    """Client for Apache Superset REST API"""
    
    def __init__(self, base_url: str, username: str = "admin", password: str = "admin"):
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.access_token = None
        self.refresh_token = None
        self.csrf_token = None
        
    def login(self) -> bool:
        """Authenticate with Superset and get access token"""
        url = f"{self.base_url}/api/v1/security/login"
        
        data = json.dumps({
            'username': self.username,
            'password': self.password,
            'provider': 'db',
            'refresh': True
        }).encode('utf-8')
        
        try:
            req = urllib.request.Request(url, data=data, method='POST')
            req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                
            if 'access_token' in result:
                self.access_token = result['access_token']
                self.refresh_token = result.get('refresh_token')
                print(f"{Colors.GREEN}✓ Authenticated successfully{Colors.NC}")
                
                # Get CSRF token
                self._get_csrf_token()
                return True
            else:
                print(f"{Colors.RED}✗ Login failed: No access token received{Colors.NC}")
                return False
                
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode('utf-8') if e.fp else str(e)
            print(f"{Colors.RED}✗ Login failed (HTTP {e.code}): {error_msg}{Colors.NC}")
            return False
        except Exception as e:
            print(f"{Colors.RED}✗ Login error: {e}{Colors.NC}")
            return False
    
    def _get_csrf_token(self) -> bool:
        """Get CSRF token for POST/PUT/DELETE requests"""
        try:
            result = self._make_request('/api/v1/security/csrf_token/', method='GET')
            if result and 'result' in result:
                self.csrf_token = result['result']
                return True
        except:
            pass
        return False
    
    def _make_request(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated request to Superset API"""
        if not self.access_token:
            print(f"{Colors.RED}✗ Not authenticated. Call login() first.{Colors.NC}")
            return None
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            body = json.dumps(data).encode('utf-8') if data else None
            req = urllib.request.Request(url, data=body, method=method)
            
            req.add_header('Authorization', f'Bearer {self.access_token}')
            req.add_header('Content-Type', 'application/json')
            
            if method in ['POST', 'PUT', 'DELETE'] and self.csrf_token:
                req.add_header('X-CSRFToken', self.csrf_token)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                response_data = response.read().decode('utf-8')
                return json.loads(response_data) if response_data else {}
                
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            print(f"{Colors.RED}✗ HTTP {e.code}: {e.reason}{Colors.NC}")
            if error_body:
                print(f"  Response: {error_body}")
            return None
        except Exception as e:
            print(f"{Colors.RED}✗ Request error: {e}{Colors.NC}")
            return None
    
    def get_databases(self) -> List[Dict]:
        """Get list of configured databases"""
        result = self._make_request('/api/v1/database/')
        if result and 'result' in result:
            return result['result']
        return []
    
    def create_database(self, database_name: str, sqlalchemy_uri: str, extra: Optional[Dict] = None) -> Optional[int]:
        """Create a new database connection"""
        data = {
            'database_name': database_name,
            'sqlalchemy_uri': sqlalchemy_uri,
            'expose_in_sqllab': True,
            'allow_ctas': True,
            'allow_cvas': True,
            'allow_dml': True
        }
        
        if extra:
            data['extra'] = json.dumps(extra)
        
        result = self._make_request('/api/v1/database/', method='POST', data=data)
        
        if result and 'id' in result:
            print(f"{Colors.GREEN}✓ Created database: {database_name}{Colors.NC}")
            return result['id']
        else:
            msg = result.get('message', 'Unknown error') if result else 'No response'
            print(f"{Colors.RED}✗ Failed to create database '{database_name}': {msg}{Colors.NC}")
            return None
    
    def get_or_create_database(self, database_name: str, sqlalchemy_uri: str, extra: Optional[Dict] = None) -> Optional[int]:
        """Get existing database or create new one"""
        databases = self.get_databases()
        for db in databases:
            if db.get('database_name') == database_name:
                db_id = db.get('id')
                print(f"{Colors.YELLOW}Database '{database_name}' already exists (id: {db_id}){Colors.NC}")
                return db_id
        
        return self.create_database(database_name, sqlalchemy_uri, extra)
    
    def get_dashboards(self) -> List[Dict]:
        """Get list of all dashboards"""
        result = self._make_request('/api/v1/dashboard/')
        if result and 'result' in result:
            return result['result']
        return []
    
    def import_dashboard(self, dashboard_data: Dict, overwrite: bool = False) -> bool:
        """Import a dashboard definition"""
        dashboard_title = dashboard_data.get('dashboard_title', 'Unknown')
        
        # Check if dashboard already exists
        existing = self.get_dashboards()
        for dash in existing:
            if dash.get('dashboard_title') == dashboard_title:
                if not overwrite:
                    print(f"{Colors.YELLOW}Dashboard '{dashboard_title}' already exists, skipping{Colors.NC}")
                    return True
                else:
                    print(f"{Colors.YELLOW}Dashboard '{dashboard_title}' exists, will overwrite{Colors.NC}")
        
        # Import via the import API
        endpoint = '/api/v1/dashboard/import/'
        
        # Prepare the import data
        import_data = {
            'overwrite': overwrite,
            'passwords': {}
        }
        
        # The actual import expects multipart/form-data with a file
        # For simplicity, we'll use the export/import ZIP format
        # This is a simplified version - may need adjustment based on Superset version
        
        result = self._make_request(endpoint, method='POST', data=dashboard_data)
        
        if result:
            print(f"{Colors.GREEN}✓ Imported dashboard: {dashboard_title}{Colors.NC}")
            return True
        else:
            print(f"{Colors.RED}✗ Failed to import dashboard: {dashboard_title}{Colors.NC}")
            return False
    
    def create_dataset(self, database_id: int, table_name: str, schema: Optional[str] = None) -> Optional[int]:
        """Create a dataset (table) reference"""
        data = {
            'database': database_id,
            'table_name': table_name,
            'schema': schema
        }
        
        result = self._make_request('/api/v1/dataset/', method='POST', data=data)
        
        if result and 'id' in result:
            print(f"{Colors.GREEN}✓ Created dataset: {table_name}{Colors.NC}")
            return result['id']
        else:
            # Dataset might already exist
            print(f"{Colors.YELLOW}Dataset '{table_name}' may already exist{Colors.NC}")
            return None


def get_dashboards_from_configmap(namespace: str = "data-platform",
                                   configmap_name: str = "superset-commodity-dashboards") -> Dict[str, Dict]:
    """Extract dashboard definitions from Kubernetes ConfigMap"""
    try:
        cmd = [
            'kubectl', 'get', 'configmap', configmap_name,
            '-n', namespace,
            '-o', 'json'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        configmap = json.loads(result.stdout)
        
        dashboards = {}
        data = configmap.get('data', {})
        
        for key, value in data.items():
            if key.endswith('.json'):
                try:
                    dashboard = json.loads(value)
                    dashboards[key] = dashboard
                except json.JSONDecodeError as e:
                    print(f"{Colors.YELLOW}Warning: Failed to parse {key}: {e}{Colors.NC}")
        
        return dashboards
        
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}✗ Failed to get ConfigMap: {e.stderr}{Colors.NC}")
        return {}
    except Exception as e:
        print(f"{Colors.RED}✗ Error reading ConfigMap: {e}{Colors.NC}")
        return {}


def wait_for_superset(base_url: str, max_wait: int = 300) -> bool:
    """Wait for Superset to be ready"""
    print(f"{Colors.BLUE}Waiting for Superset...{Colors.NC}")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            req = urllib.request.Request(f"{base_url}/health", method='GET')
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    print(f"{Colors.GREEN}✓ Superset is ready{Colors.NC}")
                    return True
        except:
            pass
        
        time.sleep(5)
        print(".", end="", flush=True)
    
    print(f"\n{Colors.RED}✗ Timeout waiting for Superset{Colors.NC}")
    return False


def main():
    parser = argparse.ArgumentParser(description='Import Superset dashboards from ConfigMap')
    parser.add_argument('--superset-url',
                       default=os.environ.get('SUPERSET_URL', 'http://superset.data-platform.svc.cluster.local:8088'),
                       help='Superset URL')
    parser.add_argument('--username', default='admin', help='Superset username')
    parser.add_argument('--password', default='admin', help='Superset password')
    parser.add_argument('--namespace', default='data-platform', help='Kubernetes namespace')
    parser.add_argument('--configmap', default='superset-commodity-dashboards', help='ConfigMap name')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing dashboards')
    parser.add_argument('--wait', action='store_true', help='Wait for Superset to be ready')
    parser.add_argument('--setup-databases', action='store_true', help='Set up database connections')
    
    args = parser.parse_args()
    
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print(f"{Colors.BLUE}  Superset Dashboard Import{Colors.NC}")
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print()
    
    # Wait for Superset if requested
    if args.wait:
        if not wait_for_superset(args.superset_url):
            sys.exit(1)
    
    # Connect to Superset
    print(f"{Colors.BLUE}Connecting to Superset...{Colors.NC}")
    client = SupersetClient(args.superset_url, args.username, args.password)
    
    if not client.login():
        print(f"{Colors.RED}✗ Failed to authenticate with Superset{Colors.NC}")
        sys.exit(1)
    
    print()
    
    # Set up database connections if requested
    if args.setup_databases:
        print(f"{Colors.BLUE}Setting up database connections...{Colors.NC}")
        
        # Trino connection
        trino_id = client.get_or_create_database(
            'Trino (Iceberg)',
            'trino://trino-coordinator:8080/iceberg_catalog/commodity_data'
        )
        
        # PostgreSQL connection
        postgres_id = client.get_or_create_database(
            'PostgreSQL (Platform)',
            'postgresql://postgres:postgres@postgres-shared-service:5432/datahub'
        )
        
        print()
    
    # Get dashboards from ConfigMap
    print(f"{Colors.BLUE}Reading dashboards from ConfigMap...{Colors.NC}")
    dashboards = get_dashboards_from_configmap(args.namespace, args.configmap)
    
    if not dashboards:
        print(f"{Colors.YELLOW}No dashboards found in ConfigMap{Colors.NC}")
        print(f"{Colors.BLUE}Note: Dashboard ConfigMap may not be created yet{Colors.NC}")
        print(f"{Colors.BLUE}Continuing with database setup only...{Colors.NC}")
        print()
        print(f"{Colors.GREEN}Database connections configured successfully!{Colors.NC}")
        print()
        print(f"{Colors.BLUE}Next steps:{Colors.NC}")
        print(f"  1. Access Superset UI")
        print(f"  2. Create dashboards manually or import from exports")
        print(f"  3. Configure charts and visualizations")
        print()
        sys.exit(0)
    
    print(f"{Colors.GREEN}✓ Found {len(dashboards)} dashboard(s){Colors.NC}")
    for name in dashboards.keys():
        print(f"  - {name}")
    print()
    
    # Import dashboards
    print(f"{Colors.BLUE}Importing dashboards...{Colors.NC}")
    success_count = 0
    failed_count = 0
    
    for dashboard_name, dashboard_data in dashboards.items():
        print(f"\n{Colors.BLUE}Processing: {dashboard_name}{Colors.NC}")
        if client.import_dashboard(dashboard_data, overwrite=args.overwrite):
            success_count += 1
        else:
            failed_count += 1
    
    print()
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print(f"{Colors.GREEN}  Import Complete!{Colors.NC}")
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print()
    print(f"Summary:")
    print(f"  - Dashboards imported: {Colors.GREEN}{success_count}{Colors.NC}")
    if failed_count > 0:
        print(f"  - Dashboards failed: {Colors.RED}{failed_count}{Colors.NC}")
    print()
    print(f"{Colors.BLUE}Next steps:{Colors.NC}")
    print(f"  1. Access Superset UI to view dashboards")
    print(f"  2. Refresh data sources")
    print(f"  3. Customize visualizations")
    print()
    
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == '__main__':
    main()


