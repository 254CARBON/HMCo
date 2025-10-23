#!/usr/bin/env python3
"""
DolphinScheduler Workflow Import Script (File-based)
Imports workflow definitions from local JSON files into DolphinScheduler

Usage:
    python3 import-workflows-from-files.py
    python3 import-workflows-from-files.py --workflow-dir /path/to/workflows
    python3 import-workflows-from-files.py --dolphin-url http://localhost:12345
    python3 import-workflows-from-files.py --skip-existing
"""

import argparse
import json
import os
import sys
import time
import subprocess
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path


class Colors:
    """ANSI color codes"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'


class DolphinSchedulerClient:
    """Client for DolphinScheduler REST API"""
    
    def __init__(self, base_url: str, username: str = "admin", password: str = "dolphinscheduler123"):
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None
        self.user_id = None
        
    def login(self) -> bool:
        """Authenticate with DolphinScheduler and get token"""
        url = f"{self.base_url}/dolphinscheduler/login"
        data = urllib.parse.urlencode({
            'userName': self.username,
            'userPassword': self.password
        }).encode('utf-8')
        
        try:
            req = urllib.request.Request(url, data=data, method='POST')
            req.add_header('Content-Type', 'application/x-www-form-urlencoded')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                
            if result.get('code') == 0 and result.get('data'):
                self.token = result['data'].get('token')
                self.user_id = result['data'].get('id')
                print(f"{Colors.GREEN}✓ Authenticated successfully{Colors.NC}")
                return True
            else:
                print(f"{Colors.RED}✗ Login failed: {result.get('msg', 'Unknown error')}{Colors.NC}")
                return False
                
        except urllib.error.URLError as e:
            print(f"{Colors.RED}✗ Connection error: {e}{Colors.NC}")
            return False
        except Exception as e:
            print(f"{Colors.RED}✗ Login error: {e}{Colors.NC}")
            return False
    
    def _make_request(self, endpoint: str, method: str = 'GET', 
                     data: Optional[bytes] = None, 
                     headers: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        """Make authenticated request to DolphinScheduler API"""
        if not self.token:
            print(f"{Colors.RED}✗ Not authenticated. Call login() first.{Colors.NC}")
            return None
        
        url = f"{self.base_url}{endpoint}"
        
        # Add token to URL
        separator = '&' if '?' in url else '?'
        url = f"{url}{separator}token={self.token}"
        
        try:
            req = urllib.request.Request(url, data=data, method=method)
            
            if headers:
                for key, value in headers.items():
                    req.add_header(key, value)
            
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode('utf-8'))
                
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ''
            print(f"{Colors.RED}✗ HTTP {e.code}: {e.reason}{Colors.NC}")
            if error_body:
                print(f"{Colors.RED}  {error_body}{Colors.NC}")
            return None
        except Exception as e:
            print(f"{Colors.RED}✗ Request error: {e}{Colors.NC}")
            return None
    
    def get_projects(self) -> List[Dict]:
        """Get list of all projects"""
        result = self._make_request('/dolphinscheduler/projects/list')
        if result and result.get('code') == 0:
            return result.get('data', [])
        return []
    
    def create_project(self, name: str, description: str = "") -> Optional[int]:
        """Create a new project"""
        endpoint = '/dolphinscheduler/projects/create'
        params = urllib.parse.urlencode({
            'projectName': name,
            'description': description
        })
        
        result = self._make_request(f"{endpoint}?{params}", method='POST')
        if result and result.get('code') == 0:
            print(f"{Colors.GREEN}✓ Created project: {name}{Colors.NC}")
            time.sleep(1)
            projects = self.get_projects()
            for proj in projects:
                if proj.get('name') == name:
                    return proj.get('code')
            return None
        else:
            msg = result.get('msg', 'Unknown error') if result else 'No response'
            print(f"{Colors.RED}✗ Failed to create project: {msg}{Colors.NC}")
            return None
    
    def get_or_create_project(self, name: str, description: str = "") -> Optional[int]:
        """Get existing project or create new one"""
        projects = self.get_projects()
        for project in projects:
            if project.get('name') == name:
                project_code = project.get('code')
                print(f"{Colors.YELLOW}Project '{name}' already exists (code: {project_code}){Colors.NC}")
                return project_code
        
        return self.create_project(name, description)
    
    def get_workflows(self, project_code: int) -> List[Dict]:
        """Get all workflows in a project"""
        endpoint = f'/dolphinscheduler/projects/{project_code}/process-definition'
        result = self._make_request(endpoint)
        if result and result.get('code') == 0:
            data = result.get('data', {})
            if isinstance(data, dict):
                return data.get('totalList', [])
            elif isinstance(data, list):
                return data
        return []
    
    def import_workflow_file(self, project_code: int, file_path: str, skip_existing: bool = True) -> Tuple[bool, str]:
        """Import a workflow from a JSON file using multipart upload"""
        workflow_name = Path(file_path).stem
        
        # Check if workflow already exists
        if skip_existing:
            existing = self.get_workflows(project_code)
            for wf in existing:
                if wf.get('name', '').lower() == workflow_name.lower():
                    return True, f"Workflow '{workflow_name}' already exists, skipped"
        
        # Read the file
        try:
            with open(file_path, 'r') as f:
                workflow_data = json.load(f)
        except Exception as e:
            return False, f"Failed to read file: {e}"
        
        # Validate JSON structure
        if 'name' not in workflow_data:
            return False, "Invalid workflow JSON: missing 'name' field"
        
        # Use import API endpoint
        endpoint = f'/dolphinscheduler/projects/{project_code}/process-definition/import'
        
        # Create multipart form data manually
        boundary = '----WebKitFormBoundary' + ''.join([str(i % 10) for i in range(16)])
        
        # Build multipart body
        body_parts = []
        
        # Add file part
        file_content = json.dumps(workflow_data, indent=2).encode('utf-8')
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(f'Content-Disposition: form-data; name="file"; filename="{Path(file_path).name}"'.encode())
        body_parts.append(b'Content-Type: application/json')
        body_parts.append(b'')
        body_parts.append(file_content)
        
        # End boundary
        body_parts.append(f'--{boundary}--'.encode())
        body_parts.append(b'')
        
        body = b'\r\n'.join(body_parts)
        
        headers = {
            'Content-Type': f'multipart/form-data; boundary={boundary}'
        }
        
        result = self._make_request(endpoint, method='POST', data=body, headers=headers)
        
        if result and result.get('code') == 0:
            return True, f"Successfully imported workflow: {workflow_name}"
        else:
            msg = result.get('msg', 'Unknown error') if result else 'No response'
            return False, f"Import failed: {msg}"


def find_workflow_files(directory: str) -> List[str]:
    """Find all workflow JSON files in directory"""
    workflow_dir = Path(directory)
    if not workflow_dir.exists():
        return []
    
    json_files = sorted(workflow_dir.glob('*.json'))
    return [str(f) for f in json_files]


def validate_workflow_json(file_path: str) -> Tuple[bool, str]:
    """Validate workflow JSON structure"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        required_fields = ['name']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        return True, "Valid"
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def setup_port_forward(namespace: str) -> Optional[subprocess.Popen]:
    """Setup kubectl port-forward for DolphinScheduler API"""
    print(f"{Colors.YELLOW}Setting up port-forward...{Colors.NC}")
    
    # Find API pod
    try:
        result = subprocess.run(
            ['kubectl', 'get', 'pods', '-n', namespace, 
             '-l', 'app.kubernetes.io/component=api', 
             '-o', 'jsonpath={.items[0].metadata.name}'],
            capture_output=True, text=True, check=True
        )
        pod_name = result.stdout.strip()
        
        if not pod_name:
            print(f"{Colors.RED}✗ No API pod found in namespace {namespace}{Colors.NC}")
            return None
        
        # Start port-forward
        proc = subprocess.Popen(
            ['kubectl', 'port-forward', '-n', namespace, pod_name, '12345:12345'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        time.sleep(3)  # Wait for port-forward to establish
        print(f"{Colors.GREEN}✓ Port-forward established (PID: {proc.pid}){Colors.NC}")
        return proc
        
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}✗ Failed to setup port-forward: {e}{Colors.NC}")
        return None
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.NC}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Import DolphinScheduler workflows from JSON files')
    parser.add_argument('--workflow-dir', 
                       default='/home/m/tff/254CARBON/HMCo/workflows',
                       help='Directory containing workflow JSON files')
    parser.add_argument('--dolphin-url', 
                       default='http://localhost:12345',
                       help='DolphinScheduler API URL')
    parser.add_argument('--username', default='admin', help='DolphinScheduler username')
    parser.add_argument('--password', default='dolphinscheduler123', help='DolphinScheduler password')
    parser.add_argument('--namespace', default='data-platform', help='Kubernetes namespace (for port-forward)')
    parser.add_argument('--project-name', default='Commodity Data Platform', help='Project name to create')
    parser.add_argument('--skip-existing', action='store_true', default=True, help='Skip existing workflows')
    parser.add_argument('--port-forward', action='store_true', help='Setup kubectl port-forward automatically')
    
    args = parser.parse_args()
    
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}  DolphinScheduler Workflow Import (File-based){Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}")
    print()
    
    # Setup port-forward if requested
    port_forward_proc = None
    if args.port_forward:
        port_forward_proc = setup_port_forward(args.namespace)
        if port_forward_proc:
            args.dolphin_url = 'http://localhost:12345'
    
    # Cleanup function
    def cleanup():
        if port_forward_proc:
            port_forward_proc.terminate()
            print(f"{Colors.YELLOW}Port-forward terminated{Colors.NC}")
    
    import atexit
    atexit.register(cleanup)
    
    # Find workflow files
    print(f"{Colors.BLUE}Scanning for workflow files...${Colors.NC}")
    print(f"Directory: {args.workflow_dir}")
    
    workflow_files = find_workflow_files(args.workflow_dir)
    
    if not workflow_files:
        print(f"{Colors.RED}✗ No workflow JSON files found${Colors.NC}")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓ Found {len(workflow_files)} workflow file(s)${Colors.NC}")
    for f in workflow_files:
        print(f"  - {Path(f).name}")
    print()
    
    # Validate workflow files
    print(f"{Colors.BLUE}Validating workflow files...${Colors.NC}")
    valid_files = []
    invalid_files = []
    
    for file_path in workflow_files:
        is_valid, message = validate_workflow_json(file_path)
        if is_valid:
            print(f"{Colors.GREEN}✓ {Path(file_path).name}: {message}{Colors.NC}")
            valid_files.append(file_path)
        else:
            print(f"{Colors.RED}✗ {Path(file_path).name}: {message}{Colors.NC}")
            invalid_files.append(file_path)
    
    if not valid_files:
        print(f"{Colors.RED}✗ No valid workflow files found${Colors.NC}")
        sys.exit(1)
    
    print()
    
    # Connect to DolphinScheduler
    print(f"{Colors.BLUE}Connecting to DolphinScheduler...${Colors.NC}")
    print(f"URL: {args.dolphin_url}")
    
    client = DolphinSchedulerClient(args.dolphin_url, args.username, args.password)
    
    if not client.login():
        print(f"{Colors.RED}✗ Failed to authenticate with DolphinScheduler${Colors.NC}")
        sys.exit(1)
    
    print()
    
    # Get or create project
    print(f"{Colors.BLUE}Setting up project...${Colors.NC}")
    project_code = client.get_or_create_project(
        args.project_name,
        "Automated commodity data ingestion and processing workflows"
    )
    
    if not project_code:
        print(f"{Colors.RED}✗ Failed to get or create project${Colors.NC}")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓ Using project: {args.project_name} (code: {project_code})${Colors.NC}")
    print()
    
    # Import workflows
    print(f"{Colors.BLUE}Importing workflows...${Colors.NC}")
    print()
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for file_path in valid_files:
        filename = Path(file_path).name
        print(f"{Colors.BLUE}Processing: {filename}${Colors.NC}")
        
        success, message = client.import_workflow_file(project_code, file_path, args.skip_existing)
        
        if success:
            if 'skipped' in message.lower():
                print(f"{Colors.YELLOW}⊘ {message}${Colors.NC}")
                skipped_count += 1
            else:
                print(f"{Colors.GREEN}✓ {message}${Colors.NC}")
                success_count += 1
        else:
            print(f"{Colors.RED}✗ {message}${Colors.NC}")
            failed_count += 1
        
        print()
    
    # Summary
    print(f"{Colors.BLUE}{'='*60}${Colors.NC}")
    print(f"{Colors.GREEN}  Import Complete!${Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}${Colors.NC}")
    print()
    print(f"Summary:")
    print(f"  - Workflows imported: {Colors.GREEN}{success_count}${Colors.NC}")
    if skipped_count > 0:
        print(f"  - Workflows skipped: {Colors.YELLOW}{skipped_count}${Colors.NC}")
    if failed_count > 0:
        print(f"  - Workflows failed: {Colors.RED}{failed_count}${Colors.NC}")
    if invalid_files:
        print(f"  - Invalid files: {Colors.RED}{len(invalid_files)}${Colors.NC}")
    print(f"  - Project: {args.project_name}")
    print()
    print(f"{Colors.BLUE}Next steps:${Colors.NC}")
    print(f"  1. Access DolphinScheduler UI: {args.dolphin_url.replace(':12345', '')}")
    print(f"  2. Verify workflows in project: {args.project_name}")
    print(f"  3. Run test execution: ./scripts/test-dolphinscheduler-workflows.sh")
    print()
    
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == '__main__':
    main()

