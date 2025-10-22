#!/usr/bin/env python3
"""
DolphinScheduler Workflow Import Script
Automatically imports workflow definitions from ConfigMap into DolphinScheduler

Usage:
    python3 import-dolphinscheduler-workflows.py
    python3 import-dolphinscheduler-workflows.py --dolphinscheduler-url http://localhost:12345
    python3 import-dolphinscheduler-workflows.py --skip-existing
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


class Colors:
    """ANSI color codes"""
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color


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
    
    def _make_request(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated request to DolphinScheduler API"""
        if not self.token:
            print(f"{Colors.RED}✗ Not authenticated. Call login() first.{Colors.NC}")
            return None
        
        url = f"{self.base_url}{endpoint}"
        
        # Add token to URL
        separator = '&' if '?' in url else '?'
        url = f"{url}{separator}token={self.token}"
        
        try:
            if method == 'GET':
                req = urllib.request.Request(url, method='GET')
            else:
                body = json.dumps(data).encode('utf-8') if data else None
                req = urllib.request.Request(url, data=body, method=method)
                req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
                
        except urllib.error.HTTPError as e:
            print(f"{Colors.RED}✗ HTTP {e.code}: {e.reason}{Colors.NC}")
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
            # Get the project to return its code
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
            return result.get('data', {}).get('totalList', [])
        return []
    
    def import_workflow(self, project_code: int, workflow_json: Dict) -> bool:
        """Import a workflow definition"""
        workflow_name = workflow_json.get('name', 'Unknown')
        
        # Check if workflow already exists
        existing = self.get_workflows(project_code)
        for wf in existing:
            if wf.get('name') == workflow_name:
                print(f"{Colors.YELLOW}Workflow '{workflow_name}' already exists, skipping{Colors.NC}")
                return True
        
        # Import workflow using the import API
        endpoint = f'/dolphinscheduler/projects/{project_code}/process-definition/import'
        
        # This is a simplified version - actual implementation may need file upload
        # For now, we'll use the create API as a fallback
        return self.create_workflow(project_code, workflow_json)
    
    def create_workflow(self, project_code: int, workflow_json: Dict) -> bool:
        """Create a new workflow definition"""
        workflow_name = workflow_json.get('name', 'Unknown')
        
        try:
            # Build the workflow creation payload
            endpoint = f'/dolphinscheduler/projects/{project_code}/process-definition'
            
            # Convert workflow JSON to DolphinScheduler format
            params = {
                'name': workflow_json.get('name'),
                'description': workflow_json.get('description', ''),
                'tenantCode': workflow_json.get('tenantCode', 'default'),
                'timeout': workflow_json.get('timeout', 0),
                'taskDefinitionJson': json.dumps(workflow_json.get('tasks', [])),
                'taskRelationJson': json.dumps(self._build_task_relations(workflow_json.get('tasks', []))),
                'locations': json.dumps({}),
                'executionType': 'PARALLEL'
            }
            
            params_encoded = urllib.parse.urlencode(params)
            result = self._make_request(f"{endpoint}?{params_encoded}", method='POST')
            
            if result and result.get('code') == 0:
                print(f"{Colors.GREEN}✓ Imported workflow: {workflow_name}{Colors.NC}")
                return True
            else:
                msg = result.get('msg', 'Unknown error') if result else 'No response'
                print(f"{Colors.RED}✗ Failed to import workflow '{workflow_name}': {msg}{Colors.NC}")
                return False
                
        except Exception as e:
            print(f"{Colors.RED}✗ Error importing workflow '{workflow_name}': {e}{Colors.NC}")
            return False
    
    def _build_task_relations(self, tasks: List[Dict]) -> List[Dict]:
        """Build task relation graph from task definitions"""
        relations = []
        task_codes = {task.get('name'): i for i, task in enumerate(tasks)}
        
        for task in tasks:
            task_name = task.get('name')
            pre_tasks = task.get('preTasks', [])
            
            if not pre_tasks:
                # Root task
                relations.append({
                    'name': task_name,
                    'preTaskCode': 0,
                    'preTaskVersion': 0,
                    'postTaskCode': task_codes[task_name],
                    'postTaskVersion': 1,
                    'conditionType': 'NONE',
                    'conditionParams': {}
                })
            else:
                for pre_task in pre_tasks:
                    relations.append({
                        'name': f"{pre_task}_to_{task_name}",
                        'preTaskCode': task_codes.get(pre_task, 0),
                        'preTaskVersion': 1,
                        'postTaskCode': task_codes[task_name],
                        'postTaskVersion': 1,
                        'conditionType': 'NONE',
                        'conditionParams': {}
                    })
        
        return relations


def get_workflows_from_configmap(namespace: str = "data-platform", 
                                  configmap_name: str = "dolphinscheduler-commodity-workflows") -> Dict[str, Dict]:
    """Extract workflow definitions from Kubernetes ConfigMap"""
    try:
        cmd = [
            'kubectl', 'get', 'configmap', configmap_name,
            '-n', namespace,
            '-o', 'json'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        configmap = json.loads(result.stdout)
        
        workflows = {}
        data = configmap.get('data', {})
        
        for key, value in data.items():
            if key.endswith('.json'):
                try:
                    workflow = json.loads(value)
                    workflows[key] = workflow
                except json.JSONDecodeError as e:
                    print(f"{Colors.YELLOW}Warning: Failed to parse {key}: {e}{Colors.NC}")
        
        return workflows
        
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}✗ Failed to get ConfigMap: {e.stderr}{Colors.NC}")
        return {}
    except Exception as e:
        print(f"{Colors.RED}✗ Error reading ConfigMap: {e}{Colors.NC}")
        return {}


def wait_for_dolphinscheduler(base_url: str, max_wait: int = 300) -> bool:
    """Wait for DolphinScheduler API to be ready"""
    print(f"{Colors.BLUE}Waiting for DolphinScheduler API...{Colors.NC}")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            req = urllib.request.Request(f"{base_url}/dolphinscheduler/ui/", method='GET')
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    print(f"{Colors.GREEN}✓ DolphinScheduler API is ready{Colors.NC}")
                    return True
        except:
            pass
        
        time.sleep(5)
        print(".", end="", flush=True)
    
    print(f"\n{Colors.RED}✗ Timeout waiting for DolphinScheduler API{Colors.NC}")
    return False


def main():
    parser = argparse.ArgumentParser(description='Import DolphinScheduler workflows from ConfigMap')
    parser.add_argument('--dolphinscheduler-url', 
                       default=os.environ.get('DOLPHINSCHEDULER_URL', 'http://dolphinscheduler-api-service.data-platform.svc.cluster.local:12345'),
                       help='DolphinScheduler API URL')
    parser.add_argument('--username', default='admin', help='DolphinScheduler username')
    parser.add_argument('--password', default='dolphinscheduler123', help='DolphinScheduler password')
    parser.add_argument('--namespace', default='data-platform', help='Kubernetes namespace')
    parser.add_argument('--configmap', default='dolphinscheduler-commodity-workflows', help='ConfigMap name')
    parser.add_argument('--project-name', default='Commodity Data Platform', help='Project name to create')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing workflows')
    parser.add_argument('--wait', action='store_true', help='Wait for DolphinScheduler to be ready')
    
    args = parser.parse_args()
    
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print(f"{Colors.BLUE}  DolphinScheduler Workflow Import{Colors.NC}")
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print()
    
    # Wait for DolphinScheduler if requested
    if args.wait:
        if not wait_for_dolphinscheduler(args.dolphinscheduler_url):
            sys.exit(1)
    
    # Get workflows from ConfigMap
    print(f"{Colors.BLUE}Reading workflows from ConfigMap...{Colors.NC}")
    workflows = get_workflows_from_configmap(args.namespace, args.configmap)
    
    if not workflows:
        print(f"{Colors.RED}✗ No workflows found in ConfigMap{Colors.NC}")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓ Found {len(workflows)} workflow(s){Colors.NC}")
    for name in workflows.keys():
        print(f"  - {name}")
    print()
    
    # Connect to DolphinScheduler
    print(f"{Colors.BLUE}Connecting to DolphinScheduler...{Colors.NC}")
    client = DolphinSchedulerClient(args.dolphinscheduler_url, args.username, args.password)
    
    if not client.login():
        print(f"{Colors.RED}✗ Failed to authenticate with DolphinScheduler{Colors.NC}")
        sys.exit(1)
    
    print()
    
    # Get or create project
    print(f"{Colors.BLUE}Setting up project...{Colors.NC}")
    project_code = client.get_or_create_project(
        args.project_name,
        "Automated commodity data ingestion and processing workflows"
    )
    
    if not project_code:
        print(f"{Colors.RED}✗ Failed to get or create project{Colors.NC}")
        sys.exit(1)
    
    print()
    
    # Import workflows
    print(f"{Colors.BLUE}Importing workflows...{Colors.NC}")
    success_count = 0
    failed_count = 0
    
    for workflow_name, workflow_data in workflows.items():
        print(f"\n{Colors.BLUE}Processing: {workflow_name}{Colors.NC}")
        if client.import_workflow(project_code, workflow_data):
            success_count += 1
        else:
            failed_count += 1
    
    print()
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print(f"{Colors.GREEN}  Import Complete!{Colors.NC}")
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print()
    print(f"Summary:")
    print(f"  - Workflows imported: {Colors.GREEN}{success_count}{Colors.NC}")
    if failed_count > 0:
        print(f"  - Workflows failed: {Colors.RED}{failed_count}{Colors.NC}")
    print(f"  - Project: {args.project_name}")
    print()
    print(f"{Colors.BLUE}Next steps:{Colors.NC}")
    print(f"  1. Access DolphinScheduler UI to view workflows")
    print(f"  2. Configure workflow schedules")
    print(f"  3. Run test executions")
    print()
    
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == '__main__':
    main()


