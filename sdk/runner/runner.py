"""
Main ingestion runner application.
"""

import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

# Setup logger early
logger = logging.getLogger(__name__)

# Third-party imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import PlainTextResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Local imports
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import RunnerConfig, ExecutionEngine
from secret_manager import SecretManager, SecretManagerError
from metrics import MetricsCollector, MetricsConfig
from tracing import Tracer
from job_executor import JobExecutor, JobExecutionError

# Compiler imports
try:
    # Add UIS SDK path
    uis_path = Path(__file__).parent.parent / "uis"
    sys.path.insert(0, str(uis_path))

    from uis.compilers.seatunnel import SeaTunnelCompiler
    from uis.compilers.spark import SparkCompiler
    from uis.compilers.flink import FlinkCompiler
    from uis.spec import UnifiedIngestionSpec
    COMPILERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"UIS compilers not available: {e}")
    COMPILERS_AVAILABLE = False
    # Fallback type for when UIS is not available
    UnifiedIngestionSpec = Any


class IngestionRunner:
    """Main ingestion runner that executes UIS jobs."""

    def __init__(self, config: Optional[RunnerConfig] = None):
        """Initialize ingestion runner."""
        self.config = config or RunnerConfig.from_env()

        # Initialize components
        self.secret_manager = None
        self.metrics = None
        self.tracer = None
        self.job_executor = None
        self.compilers = {}

        self._setup_components()
        self._setup_compilers()
        self._setup_job_executor()

        logger.info(f"Ingestion runner initialized: {self.config.runner_id}")

    def _setup_components(self):
        """Setup runner components."""
        # Setup secret manager
        if self.config.vault_enabled:
            try:
                self.secret_manager = SecretManager(
                    vault_address=self.config.vault_address,
                    vault_token=self.config.vault_token,
                    role_id=self.config.vault_role_id,
                    secret_id=self.config.vault_secret_id,
                    mount_path=self.config.vault_mount_path
                )

                if self.secret_manager.is_enabled():
                    logger.info("Vault secret manager enabled")
                else:
                    logger.warning("Vault secret manager not available")

            except Exception as e:
                logger.error(f"Failed to setup secret manager: {e}")
                self.secret_manager = None

        # Setup metrics
        metrics_config = MetricsConfig(
            enabled=self.config.metrics_enabled,
            prefix="uis_runner"
        )
        self.metrics = MetricsCollector(metrics_config)

        # Setup tracing
        self.tracer = Tracer(
            service_name=self.config.tracing_service_name,
            endpoint=self.config.tracing_endpoint,
            enabled=self.config.tracing_enabled
        )

    def _setup_job_executor(self):
        """Setup job executor."""
        try:
            self.job_executor = JobExecutor(self.config, self.metrics, self.tracer)
            logger.info("Job executor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize job executor: {e}")
            self.job_executor = None

    def _setup_compilers(self):
        """Setup available compilers."""
        if not COMPILERS_AVAILABLE:
            logger.warning("UIS compilers not available")
            return

        # Setup SeaTunnel compiler
        if ExecutionEngine.SEATUNNEL in self.config.supported_engines:
            try:
                self.compilers[ExecutionEngine.SEATUNNEL] = SeaTunnelCompiler()
                logger.info("SeaTunnel compiler initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SeaTunnel compiler: {e}")

        # Setup Spark compiler
        if ExecutionEngine.SPARK in self.config.supported_engines:
            try:
                self.compilers[ExecutionEngine.SPARK] = SparkCompiler()
                logger.info("Spark compiler initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Spark compiler: {e}")

        # Setup Flink compiler
        if ExecutionEngine.FLINK in self.config.supported_engines:
            try:
                self.compilers[ExecutionEngine.FLINK] = FlinkCompiler()
                logger.info("Flink compiler initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Flink compiler: {e}")

    def load_uis_spec(self, spec_path: str) -> UnifiedIngestionSpec:
        """Load UIS specification from file."""
        try:
            # Import UIS parser
            from uis.parser import UISParser

            parser = UISParser()
            uis_spec = parser.parse_file(spec_path)

            logger.info(f"Loaded UIS spec: {uis_spec.name}")
            return uis_spec

        except Exception as e:
            logger.error(f"Failed to load UIS spec from {spec_path}: {e}")
            raise

    def compile_uis_spec(self, uis_spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Compile UIS specification using appropriate compiler."""
        try:
            # Determine execution engine
            engine = self._determine_execution_engine(uis_spec)

            if engine not in self.compilers:
                raise ValueError(f"Execution engine {engine} not available")

            compiler = self.compilers[engine]

            # Compile the specification
            compiled_config = compiler.compile(uis_spec)

            logger.info(f"Compiled UIS spec {uis_spec.name} for engine {engine}")
            return compiled_config

        except Exception as e:
            logger.error(f"Failed to compile UIS spec {uis_spec.name}: {e}")
            raise

    def _determine_execution_engine(self, uis_spec: UnifiedIngestionSpec) -> ExecutionEngine:
        """Determine execution engine for UIS spec."""
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

    def execute_uis_spec(self, uis_spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Execute UIS specification end-to-end."""
        run_id = str(uuid.uuid4())

        if not self.job_executor:
            raise RuntimeError("Job executor is not initialized")

        with self.tracer.start_job_span(
            job_name=uis_spec.name,
            provider_type=uis_spec.provider.provider_type.value,
            tenant_id=uis_spec.provider.tenant_id,
            run_id=run_id
        ) as span:

            try:
                logger.info(f"Starting execution of UIS spec: {uis_spec.name} (run: {run_id})")

                # Compile the specification
                with self.tracer.start_span("job.compile") as compile_span:
                    compiled_config = self.compile_uis_spec(uis_spec)
                    if compile_span:
                        try:
                            output_bytes = len(json.dumps(compiled_config))
                        except (TypeError, ValueError):
                            output_bytes = 0
                        compile_span.set_attribute("compiler.output_bytes", output_bytes)

                # Resolve secrets if needed
                if self.secret_manager:
                    with self.tracer.start_span("job.resolve_secrets"):
                        compiled_config = self._resolve_secrets(compiled_config, uis_spec)

                # Execute the job
                result = self.job_executor.execute_job(run_id, uis_spec, compiled_config)

                # Record metrics
                if result["success"]:
                    # Extract data metrics from result if available
                    records = result.get("records_ingested", 0)
                    bytes_size = result.get("bytes_ingested", 0)

                    self.metrics.record_data_ingested(
                        records=records,
                        bytes_size=bytes_size,
                        provider_type=uis_spec.provider.provider_type.value,
                        tenant_id=uis_spec.provider.tenant_id,
                        sink_type=",".join(
                            [
                                getattr(s.type, "value", str(getattr(s, "type", "unknown")))
                                for s in uis_spec.provider.sinks
                            ]
                        )
                    )

                if span:
                    span.set_attribute("job.engine", result.get("engine", "unknown"))
                    span.set_attribute("job.simulated", result.get("simulated", False))
                    span.set_attribute("job.records_ingested", result.get("records_ingested", 0))
                    span.set_attribute("job.bytes_ingested", result.get("bytes_ingested", 0))

                logger.info(f"Execution completed for {uis_spec.name}: {'success' if result['success'] else 'failed'}")
                return result

            except Exception as e:
                logger.error(f"Execution failed for {uis_spec.name}: {e}")
                raise

    def _resolve_secrets(self, compiled_config: Dict[str, Any], uis_spec: UnifiedIngestionSpec) -> Dict[str, Any]:
        """Resolve secrets in compiled configuration."""
        try:
            # Look for secret references in the configuration
            config_str = json.dumps(compiled_config)

            # Replace secret references with actual values
            if uis_spec.provider.credentials_ref:
                try:
                    secret_data = self.secret_manager.get_secret(uis_spec.provider.credentials_ref)

                    # Replace placeholder values
                    for key, value in secret_data.items():
                        placeholder = f"{{{{{key}}}}}"
                        config_str = config_str.replace(placeholder, str(value))

                except SecretManagerError as e:
                    logger.error(f"Failed to resolve secrets for {uis_spec.provider.credentials_ref}: {e}")
                    # Continue without secrets - some jobs might work without them

            return json.loads(config_str)

        except Exception as e:
            logger.error(f"Failed to resolve secrets: {e}")
            return compiled_config

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the runner."""
        health = {
            "status": "healthy",
            "runner_id": self.config.runner_id,
            "tenant_id": self.config.tenant_id,
            "timestamp": time.time(),
            "components": {}
        }

        # Check secret manager
        if self.secret_manager:
            vault_health = self.secret_manager.health_check()
            health["components"]["vault"] = vault_health
            if vault_health["status"] != "healthy":
                health["status"] = "degraded"

        # Check metrics
        metrics_status = self.metrics.get_metrics_json()
        health["components"]["metrics"] = metrics_status

        # Check active jobs
        active_jobs = self.job_executor.get_active_jobs()
        health["components"]["jobs"] = {
            "active_count": len(active_jobs),
            "active_jobs": active_jobs
        }

        return health

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return self.metrics.get_metrics_text()

    def create_fastapi_app(self) -> Optional[Any]:
        """Create FastAPI application for the runner."""
        if not FASTAPI_AVAILABLE:
            logger.warning("FastAPI not available, HTTP server disabled")
            return None

        app = FastAPI(
            title="UIS Ingestion Runner",
            description="Data ingestion runner for Unified Ingestion Spec (UIS)",
            version="1.0.0"
        )

        @app.get("/health")
        async def health_endpoint():
            """Health check endpoint."""
            return self.get_health_status()

        @app.get("/metrics")
        async def metrics_endpoint():
            """Prometheus metrics endpoint."""
            return PlainTextResponse(self.get_metrics(), media_type="text/plain")

        @app.post("/jobs/execute")
        async def execute_job_endpoint(spec_path: str, background_tasks: BackgroundTasks):
            """Execute UIS job endpoint."""
            try:
                # Load UIS spec
                uis_spec = self.load_uis_spec(spec_path)

                # Execute in background
                background_tasks.add_task(self.execute_uis_spec, uis_spec)

                return {
                    "status": "queued",
                    "job_id": str(uuid.uuid4()),
                    "spec_name": uis_spec.name
                }

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @app.get("/jobs/active")
        async def active_jobs_endpoint():
            """Get active jobs."""
            return {
                "active_jobs": self.job_executor.get_active_jobs()
            }

        @app.delete("/jobs/{job_id}")
        async def cancel_job_endpoint(job_id: str):
            """Cancel a job."""
            success = self.job_executor.cancel_job(job_id)
            if success:
                return {"status": "cancelled", "job_id": job_id}
            else:
                raise HTTPException(status_code=404, detail="Job not found")

        return app

    def run_http_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Run HTTP server for the runner."""
        app = self.create_fastapi_app()
        if not app:
            logger.error("Cannot start HTTP server - FastAPI not available")
            return

        logger.info(f"Starting HTTP server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

    def run_sample_job(self, spec_path: str) -> Dict[str, Any]:
        """Run a sample job for testing."""
        try:
            logger.info(f"Running sample job from: {spec_path}")

            # Load UIS spec
            uis_spec = self.load_uis_spec(spec_path)

            # Compile and execute
            result = self.execute_uis_spec(uis_spec)

            logger.info(f"Sample job completed: {'success' if result['success'] else 'failed'}")
            return result

        except Exception as e:
            logger.error(f"Sample job failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def shutdown(self):
        """Shutdown the runner gracefully."""
        logger.info("Shutting down ingestion runner...")

        # Cancel active jobs
        active_jobs = self.job_executor.get_active_jobs()
        for job_id in active_jobs:
            logger.info(f"Cancelling job: {job_id}")
            self.job_executor.cancel_job(job_id)

        # Shutdown tracing
        if self.tracer:
            self.tracer.shutdown()

        logger.info("Ingestion runner shutdown complete")
