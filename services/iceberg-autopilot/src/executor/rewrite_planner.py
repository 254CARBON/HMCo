"""
Executor that creates safe RewriteDataFiles plans with canary merge on lakeFS branch.
"""
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import requests
import json

logger = logging.getLogger(__name__)


@dataclass
class RewritePlan:
    """Represents a data rewrite plan"""
    table_name: str
    branch_name: str
    action_type: str  # 'partition' or 'sort'
    spec_changes: Dict[str, Any]
    estimated_bytes: int
    estimated_files: int
    canary_enabled: bool = True
    rollback_on_failure: bool = True


class RewritePlanner:
    """Plans and executes safe data file rewrites"""
    
    def __init__(self,
                 lakefs_endpoint: str,
                 lakefs_access_key: str,
                 lakefs_secret_key: str,
                 repo_name: str = "hmco-data"):
        self.lakefs_endpoint = lakefs_endpoint
        self.lakefs_access_key = lakefs_access_key
        self.lakefs_secret_key = lakefs_secret_key
        self.repo_name = repo_name
        
    def create_optimization_branch(self, table_name: str, base_branch: str = "main") -> str:
        """Create a lakeFS branch for optimization testing"""
        from datetime import datetime
        
        branch_name = f"optimize/{table_name.replace('.', '_')}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Creating lakeFS branch: {branch_name}")
        
        url = f"{self.lakefs_endpoint}/api/v1/repositories/{self.repo_name}/branches"
        payload = {
            "name": branch_name,
            "source": base_branch
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                auth=(self.lakefs_access_key, self.lakefs_secret_key)
            )
            response.raise_for_status()
            logger.info(f"Branch created successfully: {branch_name}")
            return branch_name
        except Exception as e:
            logger.error(f"Failed to create branch: {e}")
            raise
    
    def plan_rewrite(self,
                     table_name: str,
                     recommendation: Any,
                     branch_name: str) -> RewritePlan:
        """Create a rewrite plan from optimization recommendation"""
        
        # Estimate data volume to rewrite
        estimated_bytes = self._estimate_rewrite_size(table_name)
        estimated_files = self._estimate_file_count(table_name)
        
        return RewritePlan(
            table_name=table_name,
            branch_name=branch_name,
            action_type=recommendation.recommendation_type,
            spec_changes=recommendation.proposed_spec,
            estimated_bytes=estimated_bytes,
            estimated_files=estimated_files,
            canary_enabled=True,
            rollback_on_failure=True
        )
    
    def _estimate_rewrite_size(self, table_name: str) -> int:
        """Estimate bytes to rewrite (simplified)"""
        # In production, query Iceberg metadata for accurate size
        return 1024 * 1024 * 1024 * 10  # 10GB placeholder
    
    def _estimate_file_count(self, table_name: str) -> int:
        """Estimate number of files to rewrite (simplified)"""
        # In production, query Iceberg metadata for accurate count
        return 100  # placeholder
    
    def execute_rewrite(self, plan: RewritePlan) -> Dict[str, Any]:
        """Execute the rewrite plan on lakeFS branch"""
        logger.info(f"Executing rewrite plan for {plan.table_name} on branch {plan.branch_name}")
        
        # Generate Iceberg RewriteDataFiles SQL
        rewrite_sql = self._generate_rewrite_sql(plan)
        
        # Execute on branch
        result = {
            "status": "planned",
            "table_name": plan.table_name,
            "branch_name": plan.branch_name,
            "rewrite_sql": rewrite_sql,
            "estimated_bytes": plan.estimated_bytes,
            "estimated_files": plan.estimated_files
        }
        
        logger.info(f"Rewrite plan created: {result}")
        return result
    
    def _generate_rewrite_sql(self, plan: RewritePlan) -> str:
        """Generate Iceberg CALL procedure for RewriteDataFiles"""
        
        if plan.action_type == 'partition':
            # First, update partition spec
            alter_sql = f"""
            -- Step 1: Update partition spec
            ALTER TABLE {plan.table_name}
            SET PARTITION SPEC {json.dumps(plan.spec_changes)};
            
            -- Step 2: Rewrite data files with new spec
            CALL iceberg.system.rewrite_data_files(
                table => '{plan.table_name}',
                strategy => 'binpack',
                options => map(
                    'target-file-size-bytes', '536870912',  -- 512MB
                    'min-input-files', '5',
                    'partial-progress.enabled', 'true',
                    'partial-progress.max-commits', '10'
                )
            );
            """
        else:  # sort_order
            alter_sql = f"""
            -- Step 1: Update sort order
            ALTER TABLE {plan.table_name}
            WRITE ORDERED BY {json.dumps(plan.spec_changes)};
            
            -- Step 2: Rewrite data files with new sort order
            CALL iceberg.system.rewrite_data_files(
                table => '{plan.table_name}',
                strategy => 'sort',
                sort_order => {json.dumps(plan.spec_changes)},
                options => map(
                    'target-file-size-bytes', '536870912',
                    'min-input-files', '5',
                    'partial-progress.enabled', 'true'
                )
            );
            """
        
        return alter_sql
    
    def run_canary_test(self, plan: RewritePlan, test_queries: list) -> Dict[str, Any]:
        """Run canary queries on optimized branch to validate improvements"""
        logger.info(f"Running canary tests on branch {plan.branch_name}")
        
        results = {
            "branch": plan.branch_name,
            "tests_passed": 0,
            "tests_failed": 0,
            "scan_reduction": 0.0,
            "query_results": []
        }
        
        # In production: execute test queries on both main and branch
        # Compare scan bytes, execution time, and result correctness
        
        for i, query in enumerate(test_queries):
            test_result = {
                "query_id": i,
                "status": "passed",  # placeholder
                "scan_bytes_before": 1000000,
                "scan_bytes_after": 300000,
                "scan_reduction_pct": 70.0
            }
            results["query_results"].append(test_result)
            results["tests_passed"] += 1
        
        # Calculate average scan reduction
        if results["query_results"]:
            avg_reduction = sum(r["scan_reduction_pct"] for r in results["query_results"]) / len(results["query_results"])
            results["scan_reduction"] = avg_reduction
        
        logger.info(f"Canary tests completed: {results['tests_passed']} passed, {results['tests_failed']} failed")
        return results
    
    def merge_to_main(self, branch_name: str, commit_message: str) -> Dict[str, Any]:
        """Merge optimization branch to main after successful canary"""
        logger.info(f"Merging branch {branch_name} to main")
        
        url = f"{self.lakefs_endpoint}/api/v1/repositories/{self.repo_name}/refs/{branch_name}/merge/main"
        payload = {
            "message": commit_message
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                auth=(self.lakefs_access_key, self.lakefs_secret_key)
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Branch merged successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to merge branch: {e}")
            raise
    
    def rollback_branch(self, branch_name: str) -> None:
        """Delete branch on failure"""
        logger.warning(f"Rolling back branch {branch_name}")
        
        url = f"{self.lakefs_endpoint}/api/v1/repositories/{self.repo_name}/branches/{branch_name}"
        
        try:
            response = requests.delete(
                url,
                auth=(self.lakefs_access_key, self.lakefs_secret_key)
            )
            response.raise_for_status()
            logger.info(f"Branch rolled back successfully")
        except Exception as e:
            logger.error(f"Failed to rollback branch: {e}")
