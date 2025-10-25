import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
import sys

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.orm import sessionmaker

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from portal.models import ExternalProvider, ProviderEndpoint, ProviderRun  # noqa: E402
from portal.models.provider import ProviderStatus, ProviderType  # noqa: E402
from portal.models.run import RunStatus  # noqa: E402


TEST_MIGRATION_TABLES = {
    "external_providers",
    "provider_endpoints",
    "provider_runs",
}


def _ensure_image_available(image: str) -> None:
    """Guard that the requested Docker image is available locally."""
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if inspect.returncode == 0:
        return

    pull = subprocess.run(
        ["docker", "pull", image],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if pull.returncode != 0:
        raise RuntimeError(pull.stderr or pull.stdout)


def _wait_for_postgres_ready(database_url: str, timeout: float = 60.0) -> None:
    """Wait until a Postgres instance is ready to accept connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        engine = create_engine(database_url, future=True)
        try:
            with engine.connect():
                return
        except Exception:
            time.sleep(0.5)
        finally:
            engine.dispose()
    raise TimeoutError("Postgres container did not become ready in time")


@pytest.fixture(scope="session")
def postgres_dsn():
    """Provide a dockerised Postgres DSN for integration tests."""
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is required for Postgres integration tests")

    image = os.getenv("PORTAL_TEST_POSTGRES_IMAGE", "postgres:15-alpine")
    container_name = f"portal-test-postgres-{uuid.uuid4().hex[:12]}"
    env = [
        "-e", "POSTGRES_PASSWORD=test",
        "-e", "POSTGRES_USER=test",
        "-e", "POSTGRES_DB=test",
    ]
    ports = ["-p", "0:5432"]

    try:
        _ensure_image_available(image)
    except RuntimeError as exc:
        pytest.skip(f"Unable to fetch Postgres image: {exc}")

    run = subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", container_name, *env, *ports, image],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if run.returncode != 0:
        pytest.skip(f"Unable to start Postgres container: {run.stderr or run.stdout}")

    try:
        try:
            port_result = subprocess.run(
                ["docker", "port", container_name, "5432/tcp"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            pytest.skip(f"Unable to determine Postgres port mapping: {exc.stderr or exc.stdout}")

        host_port = port_result.stdout.strip().rsplit(":", 1)[-1]
        database_url = f"postgresql://test:test@127.0.0.1:{host_port}/test"
        _wait_for_postgres_ready(database_url)
        yield database_url
    finally:
        subprocess.run(
            ["docker", "stop", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )


@pytest.fixture(scope="session")
def alembic_config(postgres_dsn):
    """Load Alembic configuration configured for the test Postgres instance."""
    migrations_dir = Path(__file__).resolve().parents[1] / "migrations"
    cfg = Config(str(migrations_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(migrations_dir))

    cfg.set_main_option("sqlalchemy.url", postgres_dsn)

    previous_database_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = postgres_dsn

    try:
        yield cfg
    finally:
        if previous_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_database_url


def _create_engine(dsn: str):
    return create_engine(dsn, future=True)


def test_migration_upgrade_and_downgrade(postgres_dsn, alembic_config):
    """Ensure the external provider schema can be migrated up and down cleanly."""
    engine = _create_engine(postgres_dsn)
    upgraded = False

    try:
        command.upgrade(alembic_config, "head")
        upgraded = True

        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assert TEST_MIGRATION_TABLES.issubset(tables)

        command.downgrade(alembic_config, "base")
        upgraded = False

        inspector = inspect(engine)
        tables_after = set(inspector.get_table_names())
        assert TEST_MIGRATION_TABLES.isdisjoint(tables_after)
    finally:
        if upgraded:
            command.downgrade(alembic_config, "base")
        engine.dispose()


def test_crud_round_trip(postgres_dsn, alembic_config):
    """Exercise a full CRUD round-trip across the external provider schema."""
    engine = _create_engine(postgres_dsn)
    SessionLocal = sessionmaker(bind=engine, future=True)
    upgraded = False

    try:
        command.upgrade(alembic_config, "head")
        upgraded = True

        session = None
        try:
            session = SessionLocal()
            provider = ExternalProvider(
                name="polygon",
                display_name="Polygon Market Data",
                description="Polygon REST API for market data.",
                provider_type=ProviderType.REST_API,
                status=ProviderStatus.ACTIVE,
                base_url="https://api.polygon.io",
                config={"auth": "api_key"},
                credentials_ref="secret/polygon",
                rate_limits={"requests_per_minute": 1000},
                tenant_id="tenant-123",
                owner="data-platform",
                tags=["market-data", "stocks"],
                schedule_cron="0 * * * *",
                schedule_timezone="UTC",
                sink_type="iceberg",
                sink_config={"catalog": "analytics"},
                schema_contract={"type": "object"},
                slo_config={"freshness_minutes": 60},
                created_by="integration-test",
                updated_by="integration-test",
            )

            session.add(provider)
            session.commit()
            session.refresh(provider)
            assert provider.id is not None

            endpoint = ProviderEndpoint(
                provider=provider,
                name="reference_tickers",
                path="/v3/reference/tickers",
                method="GET",
                headers={"Accept": "application/json"},
                query_params={"limit": 200, "type": "CS"},
                pagination_type="cursor",
                pagination_config={"cursor_param": "cursor", "page_size": 200},
                response_path="$.results",
                field_mapping={"symbol": "ticker"},
                rate_limit_group="default",
                sample_size=200,
                validation_rules={"symbol": {"required": True}},
                is_active=True,
            )

            provider_run = ProviderRun(
                provider=provider,
                run_id=str(uuid.uuid4()),
                run_mode="batch",
                triggered_by="manual",
                status=RunStatus.COMPLETED,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                records_ingested=1500,
                bytes_ingested=512000,
                duration_ms=45000,
                throughput_records_sec=33,
                throughput_bytes_sec=11377,
                latency_p50_ms=120,
                latency_p95_ms=450,
                latency_p99_ms=800,
                schema_drift_detected=False,
                data_quality_score=98,
                validation_errors=None,
                cpu_seconds=12,
                memory_mb_peak=256,
                network_bytes=2048,
                uis_spec={"job": "ingest"},
                compiler_output={"plan": "full"},
                error_message=None,
                error_stack_trace=None,
                retry_count=0,
                trace_id="trace-123",
                estimated_cost_usd=42,
                created_by="integration-test",
                updated_by="integration-test",
            )

            session.add_all([endpoint, provider_run])
            session.commit()

            fetched = session.execute(
                select(ExternalProvider).where(ExternalProvider.name == "polygon")
            ).scalar_one()
            assert fetched.endpoints[0].name == "reference_tickers"
            assert fetched.runs[0].status == RunStatus.COMPLETED

            fetched.runs[0].status = RunStatus.FAILED
            fetched.runs[0].error_message = "Simulated failure"
            session.commit()
            session.refresh(fetched.runs[0])
            assert fetched.runs[0].status == RunStatus.FAILED

            # Deleting the provider should cascade to endpoints and runs
            session.delete(fetched)
            session.commit()
            provider_count = session.execute(
                select(func.count()).select_from(ExternalProvider)
            ).scalar_one()
            endpoint_count = session.execute(
                select(func.count()).select_from(ProviderEndpoint)
            ).scalar_one()
            run_count = session.execute(
                select(func.count()).select_from(ProviderRun)
            ).scalar_one()

            assert provider_count == 0
            assert endpoint_count == 0
            assert run_count == 0
        finally:
            if session is not None:
                session.close()
    finally:
        if upgraded:
            command.downgrade(alembic_config, "base")
        engine.dispose()
