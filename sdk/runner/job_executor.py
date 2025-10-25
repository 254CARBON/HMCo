"""
Job execution engine for running compiled UIS jobs.
"""

import subprocess
import json
import time
import tempfile
import os
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import threading

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import ExecutionEngine, RunnerConfig
from metrics import MetricsCollector
from tracing import Tracer

logger = logging.getLogger(__name__)


class JobExecutionError(Exception):
    """Exception raised when job execution fails."""
    pass


class JobExecutor:
    """Executes compiled UIS jobs using appropriate engines."""

    def __init__(self, config: RunnerConfig, metrics: MetricsCollector, tracer: Tracer):
        """Initialize job executor."""
        self.config = config
        self.metrics = metrics
        self.tracer = tracer
        self._active_jobs: Dict[str, Optional[subprocess.Popen]] = {}
        self._job_lock = threading.Lock()
        self._engine_binaries = {
            ExecutionEngine.SEATUNNEL: "seatunnel",
            ExecutionEngine.SPARK: "spark-submit",
            ExecutionEngine.FLINK: "flink"
        }

    def execute_job(self, job_id: str, uis_spec: Any, compiler_output: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a compiled job."""
        with self.tracer.start_job_span(
            job_name=uis_spec.name,
            provider_type=uis_spec.provider.provider_type.value,
            tenant_id=uis_spec.provider.tenant_id,
            run_id=job_id
        ) as span:

            try:
                # Record job start
                self.metrics.record_job_start(
                    uis_spec.provider.provider_type.value,
                    uis_spec.provider.tenant_id
                )

                start_time = time.time()

                # Determine execution engine
                engine = self._determine_execution_engine(uis_spec)

                # Execute job based on engine or simulate if engine not available
                if self._should_simulate_engine(engine):
                    result = self._simulate_job_execution(job_id, uis_spec, compiler_output, engine)
                elif engine == ExecutionEngine.SEATUNNEL:
                    result = self._execute_seatunnel_job(job_id, uis_spec, compiler_output)
                elif engine == ExecutionEngine.SPARK:
                    result = self._execute_spark_job(job_id, uis_spec, compiler_output)
                elif engine == ExecutionEngine.FLINK:
                    result = self._execute_flink_job(job_id, uis_spec, compiler_output)
                else:
                    raise JobExecutionError(f"Unsupported execution engine: {engine}")

                # Record completion
                duration = time.time() - start_time
                self.metrics.record_job_completion("completed", uis_spec.provider.provider_type.value, uis_spec.provider.tenant_id)
                self.metrics.record_job_duration(duration, uis_spec.provider.provider_type.value, uis_spec.provider.tenant_id)

                if span:
                    span.set_attribute("job.duration_seconds", duration)
                    span.set_attribute("job.status", "completed")

                return result

            except Exception as e:
                # Record error
                self.metrics.record_job_completion("failed", uis_spec.provider.provider_type.value, uis_spec.provider.tenant_id)
                self.metrics.record_error("execution_error", uis_spec.provider.provider_type.value, uis_spec.provider.tenant_id)

                if span:
                    span.set_attribute("job.status", "failed")
                    span.set_attribute("job.error", str(e))

                logger.error(f"Job execution failed for {job_id}: {e}")
                raise

    def _determine_execution_engine(self, uis_spec: Any) -> ExecutionEngine:
        """Determine which execution engine to use."""
        # Check provider configuration for preferred engine
        if uis_spec.provider.config and uis_spec.provider.config.get("execution_engine"):
            engine_name = uis_spec.provider.config["execution_engine"]
            try:
                return ExecutionEngine(engine_name)
            except ValueError:
                logger.warning(f"Invalid execution engine: {engine_name}, using default")

        # Determine based on ingestion mode
        mode = uis_spec.provider.mode
        if mode.value in ["streaming", "websocket", "webhook"]:
            return ExecutionEngine.FLINK
        elif mode.value == "micro_batch":
            return ExecutionEngine.SPARK
        else:
            return ExecutionEngine.SEATUNNEL

    def _should_simulate_engine(self, engine: ExecutionEngine) -> bool:
        """Return True if we should simulate execution instead of launching a binary."""
        simulate_enabled = getattr(self.config, "simulate_missing_engines", False)
        if not simulate_enabled:
            return False

        binary_name = self._engine_binaries.get(engine)
        if not binary_name:
            return False

        binary_path = shutil.which(binary_name)
        if binary_path:
            return False

        logger.info(
            "Execution engine %s unavailable (binary '%s' not found); running simulated execution",
            engine.value,
            binary_name
        )
        return True

    def _simulate_job_execution(
        self,
        job_id: str,
        uis_spec: Any,
        compiler_output: Dict[str, Any],
        engine: ExecutionEngine
    ) -> Dict[str, Any]:
        """Simulate job execution when the real engine is not available."""
        # Track active job for observability endpoints
        with self._job_lock:
            self._active_jobs[job_id] = None

        try:
            delay_seconds = max(0.0, float(getattr(self.config, "simulation_delay_seconds", 0.0)))
            if delay_seconds:
                time.sleep(delay_seconds)

            endpoints = getattr(uis_spec.provider, "endpoints", []) or []
            records_per_endpoint = max(1, int(getattr(self.config, "simulation_records_per_endpoint", 100)))
            record_size_bytes = max(1, int(getattr(self.config, "simulation_avg_record_size_bytes", 512)))
            estimated_records = max(1, len(endpoints)) * records_per_endpoint
            estimated_bytes = estimated_records * record_size_bytes

            logger.info(
                "Simulated %s job %s for spec %s (records=%s, bytes=%s)",
                engine.value,
                job_id,
                getattr(uis_spec, "name", "unknown"),
                estimated_records,
                estimated_bytes
            )

            return {
                "success": True,
                "job_id": job_id,
                "engine": engine.value,
                "stdout": f"Simulated execution for {uis_spec.name} using {engine.value} engine",
                "stderr": "",
                "exit_code": 0,
                "records_ingested": estimated_records,
                "bytes_ingested": estimated_bytes,
                "simulated": True
            }
        finally:
            with self._job_lock:
                self._active_jobs.pop(job_id, None)

    def _execute_seatunnel_job(self, job_id: str, uis_spec: Any, compiler_output: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SeaTunnel job."""
        logger.info(f"Executing SeaTunnel job: {job_id}")

        # Save job configuration to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(compiler_output, f, indent=2)
            config_file = f.name

        try:
            # Build SeaTunnel command
            cmd = [
                "seatunnel",
                "--config",
                config_file,
                "--job-name",
                f"uis-{uis_spec.name}-{job_id}",
                "--master",
                "local[2]"  # Local mode for development
            ]

            # Execute command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/opt/seatunnel"
            )

            # Store process reference
            with self._job_lock:
                self._active_jobs[job_id] = process

            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.config.job_timeout_seconds)

                if process.returncode == 0:
                    result = {
                        "success": True,
                        "job_id": job_id,
                        "engine": "seatunnel",
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": process.returncode
                    }
                else:
                    result = {
                        "success": False,
                        "job_id": job_id,
                        "engine": "seatunnel",
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": process.returncode,
                        "error": f"SeaTunnel job failed with exit code {process.returncode}"
                    }

            except subprocess.TimeoutExpired:
                # Kill the process if it times out
                process.kill()
                result = {
                    "success": False,
                    "job_id": job_id,
                    "engine": "seatunnel",
                    "error": f"Job timed out after {self.config.job_timeout_seconds} seconds"
                }

        finally:
            # Clean up
            os.unlink(config_file)
            with self._job_lock:
                self._active_jobs.pop(job_id, None)

        result.setdefault("records_ingested", 0)
        result.setdefault("bytes_ingested", 0)
        result["simulated"] = result.get("simulated", False)
        return result

    def _execute_spark_job(self, job_id: str, uis_spec: Any, compiler_output: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Spark job."""
        logger.info(f"Executing Spark job: {job_id}")

        # Save job configuration to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(compiler_output, f, indent=2)
            config_file = f.name

        try:
            # Build Spark submit command from compiler output
            if "spark_submit_args" in compiler_output:
                cmd = ["spark-submit"] + compiler_output["spark_submit_args"]
            else:
                # Fallback to basic Spark configuration
                cmd = [
                    "spark-submit",
                    "--class", "com.hmco.dataplatform.SparkMicroBatchJob",
                    "--master", "local[2]",
                    "--job-name", f"uis-{uis_spec.name}-{job_id}",
                    "/opt/spark/jars/uis-spark-runner.jar",
                    "--job-config", config_file
                ]

            # Execute command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/opt/spark"
            )

            # Store process reference
            with self._job_lock:
                self._active_jobs[job_id] = process

            # Wait for completion
            try:
                stdout, stderr = process.communicate(timeout=self.config.job_timeout_seconds)

                if process.returncode == 0:
                    result = {
                        "success": True,
                        "job_id": job_id,
                        "engine": "spark",
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": process.returncode
                    }
                else:
                    result = {
                        "success": False,
                        "job_id": job_id,
                        "engine": "spark",
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": process.returncode,
                        "error": f"Spark job failed with exit code {process.returncode}"
                    }

            except subprocess.TimeoutExpired:
                process.kill()
                result = {
                    "success": False,
                    "job_id": job_id,
                    "engine": "spark",
                    "error": f"Job timed out after {self.config.job_timeout_seconds} seconds"
                }

        finally:
            # Clean up
            os.unlink(config_file)
            with self._job_lock:
                self._active_jobs.pop(job_id, None)

        result.setdefault("records_ingested", 0)
        result.setdefault("bytes_ingested", 0)
        result["simulated"] = result.get("simulated", False)
        return result

    def _execute_flink_job(self, job_id: str, uis_spec: Any, compiler_output: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Flink job."""
        logger.info(f"Executing Flink job: {job_id}")

        # Save job configuration to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(compiler_output, f, indent=2)
            config_file = f.name

        try:
            # Build Flink run command
            cmd = [
                "flink",
                "run",
                "--jobmanager", "flink-jobmanager:9081",
                "--parallelism", str(uis_spec.provider.parallelism),
                "--job-name", f"uis-{uis_spec.name}-{job_id}",
                "--job-config", config_file,
                "/opt/flink/jars/uis-flink-runner.jar"
            ]

            # Execute command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd="/opt/flink"
            )

            # Store process reference
            with self._job_lock:
                self._active_jobs[job_id] = process

            # Wait for completion
            try:
                stdout, stderr = process.communicate(timeout=self.config.job_timeout_seconds)

                if process.returncode == 0:
                    result = {
                        "success": True,
                        "job_id": job_id,
                        "engine": "flink",
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": process.returncode
                    }
                else:
                    result = {
                        "success": False,
                        "job_id": job_id,
                        "engine": "flink",
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": process.returncode,
                        "error": f"Flink job failed with exit code {process.returncode}"
                    }

            except subprocess.TimeoutExpired:
                process.kill()
                result = {
                    "success": False,
                    "job_id": job_id,
                    "engine": "flink",
                    "error": f"Job timed out after {self.config.job_timeout_seconds} seconds"
                }

        finally:
            # Clean up
            os.unlink(config_file)
            with self._job_lock:
                self._active_jobs.pop(job_id, None)

        result.setdefault("records_ingested", 0)
        result.setdefault("bytes_ingested", 0)
        result["simulated"] = result.get("simulated", False)
        return result

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        with self._job_lock:
            if job_id not in self._active_jobs:
                logger.warning(f"Job {job_id} not found in active jobs")
                return False

            process = self._active_jobs[job_id]

        try:
            if process is None:
                logger.info(f"Job {job_id} was simulated; marking as cancelled")
                with self._job_lock:
                    self._active_jobs.pop(job_id, None)
                return True

            process.terminate()

            # Wait a bit for graceful termination
            try:
                process.wait(timeout=10)
                logger.info(f"Job {job_id} terminated gracefully")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.info(f"Job {job_id} killed forcefully")

            with self._job_lock:
                self._active_jobs.pop(job_id, None)

            return True

        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def get_active_jobs(self) -> List[str]:
        """Get list of active job IDs."""
        with self._job_lock:
            return list(self._active_jobs.keys())

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job."""
        with self._job_lock:
            if job_id not in self._active_jobs:
                return None

            process = self._active_jobs[job_id]

        if process is None:
            return {
                "job_id": job_id,
                "running": False,
                "exit_code": 0,
                "return_code": 0,
                "simulated": True
            }

        poll_result = process.poll()
        return {
            "job_id": job_id,
            "running": poll_result is None,
            "exit_code": poll_result,
            "return_code": process.returncode if poll_result is not None else None,
            "simulated": False
        }
