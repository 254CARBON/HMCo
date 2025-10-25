"""
OpenTelemetry tracing integration.
"""

import time
from typing import Dict, Any, Optional
from contextlib import contextmanager
import logging

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace.status import Status, StatusCode
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # Fallback types for when OpenTelemetry is not available
    Status = None
    StatusCode = None
    OPENTELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)


class TracerError(Exception):
    """Exception raised when tracing fails."""
    pass


class Tracer:
    """OpenTelemetry tracing integration."""

    def __init__(self, service_name: str, endpoint: Optional[str] = None,
                 enabled: bool = True):
        """Initialize tracer."""
        self.service_name = service_name
        self.endpoint = endpoint
        self.enabled = enabled
        self.tracer = None
        self.span_processor = None

        if self.enabled and OPENTELEMETRY_AVAILABLE:
            self._setup_tracing()
        elif self.enabled and not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry requested but not available, tracing disabled")

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0"
            })

            # Create tracer provider
            tracer_provider = TracerProvider(resource=resource)
            self.tracer = tracer_provider.get_tracer(__name__)

            # Setup span processor based on endpoint
            if self.endpoint and "jaeger" in self.endpoint.lower():
                # Jaeger exporter
                jaeger_exporter = JaegerExporter(
                    agent_host_name=self.endpoint.split("://")[1].split(":")[0] if "://" in self.endpoint else "jaeger",
                    agent_port=6831,
                )
                self.span_processor = BatchSpanProcessor(jaeger_exporter)
            elif self.endpoint and "http" in self.endpoint.lower():
                # OTLP exporter
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.endpoint,
                    insecure=True
                )
                self.span_processor = BatchSpanProcessor(otlp_exporter)
            else:
                # Console exporter for development
                console_exporter = ConsoleSpanExporter()
                self.span_processor = BatchSpanProcessor(console_exporter)

            tracer_provider.add_span_processor(self.span_processor)

            # Set as global tracer provider
            trace.set_tracer_provider(tracer_provider)

            logger.info(f"Initialized OpenTelemetry tracing for service: {self.service_name}")

        except Exception as e:
            logger.error(f"Failed to setup tracing: {e}")
            self.enabled = False

    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self.enabled and OPENTELEMETRY_AVAILABLE and self.tracer is not None

    @contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager to start a tracing span."""
        if not self.is_enabled():
            yield None
            return

        span = self.tracer.start_span(name, attributes=attributes or {})

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            span.end()

    @contextmanager
    def start_job_span(self, job_name: str, provider_type: str, tenant_id: str, run_id: str):
        """Start a span for job execution."""
        attributes = {
            "job.name": job_name,
            "provider.type": provider_type,
            "tenant.id": tenant_id,
            "run.id": run_id,
            "component": "ingestion_runner"
        }

        with self.start_span(f"job.{job_name}", attributes) as span:
            if span:
                span.set_attribute("job.start_time", time.time())
            yield span

    def add_event(self, span_name: str, event_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current span."""
        if not self.is_enabled() or Status is None:
            return

        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(event_name, attributes=attributes or {})

    def set_span_attribute(self, span_name: str, key: str, value: Any):
        """Set attribute on current span."""
        if not self.is_enabled():
            return

        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute(key, value)

    def set_span_status(self, span_name: str, status_code: Any, description: str = ""):
        """Set status on current span."""
        if not self.is_enabled() or Status is None:
            return

        current_span = trace.get_current_span()
        if current_span:
            current_span.set_status(Status(status_code, description))

    def shutdown(self):
        """Shutdown tracing."""
        if self.span_processor:
            self.span_processor.shutdown()
            logger.info("Tracing shutdown complete")
