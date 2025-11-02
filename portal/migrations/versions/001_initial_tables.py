"""Initial tables

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

provider_type_enum = postgresql.ENUM(
    'rest_api', 'graphql', 'websocket', 'webhook', 'file_ftp', 'database', name='provider_type'
)
provider_status_enum = postgresql.ENUM(
    'active', 'inactive', 'error', 'maintenance', name='provider_status'
)
run_status_enum = postgresql.ENUM(
    'pending', 'running', 'completed', 'failed', 'cancelled', 'retrying', name='run_status'
)


def upgrade() -> None:
    """Create initial tables."""
    # Create external_providers table
    op.create_table(
        'external_providers',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('display_name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('provider_type', provider_type_enum, nullable=False),
        sa.Column('status', provider_status_enum, server_default=sa.text("'inactive'::provider_status"), nullable=False),
        sa.Column('base_url', sa.String(length=1024), nullable=True),
        sa.Column('config', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('credentials_ref', sa.String(length=255), nullable=True),
        sa.Column('rate_limits', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('tenant_id', sa.String(length=255), nullable=False),
        sa.Column('owner', sa.String(length=255), nullable=False),
        sa.Column('tags', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('schedule_cron', sa.String(length=100), nullable=True),
        sa.Column('schedule_timezone', sa.String(length=50), server_default=sa.text("'UTC'"), nullable=False),
        sa.Column('sink_type', sa.String(length=50), nullable=True),
        sa.Column('sink_config', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('schema_contract', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('slo_config', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=False),
        sa.Column('updated_by', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('id', name='pk_external_providers'),
        sa.UniqueConstraint('tenant_id', 'name', name='uq_external_providers_tenant_name'),
    )

    # Create indexes for external_providers
    op.create_index('ix_external_providers_id', 'external_providers', ['id'], unique=False)
    op.create_index('ix_external_providers_name', 'external_providers', ['name'], unique=False)
    op.create_index('ix_external_providers_tenant_id', 'external_providers', ['tenant_id'], unique=False)

    # Create provider_endpoints table
    op.create_table(
        'provider_endpoints',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('provider_id', sa.Integer(), sa.ForeignKey('external_providers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('path', sa.String(length=1024), nullable=False),
        sa.Column('method', sa.String(length=10), server_default=sa.text("'GET'"), nullable=False),
        sa.Column('headers', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('query_params', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('body_template', sa.Text(), nullable=True),
        sa.Column('pagination_type', sa.String(length=50), nullable=True),
        sa.Column('pagination_config', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('response_path', sa.String(length=255), nullable=True),
        sa.Column('field_mapping', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('rate_limit_group', sa.String(length=100), nullable=True),
        sa.Column('sample_size', sa.Integer(), server_default=sa.text('100'), nullable=False),
        sa.Column('validation_rules', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_provider_endpoints'),
        sa.UniqueConstraint('provider_id', 'name', name='uq_provider_endpoints_provider_name'),
    )

    # Create indexes for provider_endpoints
    op.create_index('ix_provider_endpoints_id', 'provider_endpoints', ['id'], unique=False)
    op.create_index('ix_provider_endpoints_provider_id', 'provider_endpoints', ['provider_id'], unique=False)
    op.create_index('ix_provider_endpoints_name', 'provider_endpoints', ['name'], unique=False)

    # Create provider_runs table
    op.create_table(
        'provider_runs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('provider_id', sa.Integer(), sa.ForeignKey('external_providers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('run_id', sa.String(length=255), nullable=False),
        sa.Column('run_mode', sa.String(length=50), server_default=sa.text("'batch'"), nullable=False),
        sa.Column('triggered_by', sa.String(length=255), nullable=True),
        sa.Column('status', run_status_enum, server_default=sa.text("'pending'::run_status"), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('records_ingested', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('bytes_ingested', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('throughput_records_sec', sa.Integer(), nullable=True),
        sa.Column('throughput_bytes_sec', sa.Integer(), nullable=True),
        sa.Column('latency_p50_ms', sa.Integer(), nullable=True),
        sa.Column('latency_p95_ms', sa.Integer(), nullable=True),
        sa.Column('latency_p99_ms', sa.Integer(), nullable=True),
        sa.Column('schema_drift_detected', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('data_quality_score', sa.Integer(), nullable=True),
        sa.Column('validation_errors', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('cpu_seconds', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('memory_mb_peak', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('network_bytes', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('uis_spec', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('compiler_output', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_stack_trace', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('trace_id', sa.String(length=255), nullable=True),
        sa.Column('parent_run_id', sa.Integer(), sa.ForeignKey('provider_runs.id', ondelete='SET NULL'), nullable=True),
        sa.Column('estimated_cost_usd', sa.Integer(), nullable=True),
        sa.Column('created_by', sa.String(length=255), nullable=False),
        sa.Column('updated_by', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('id', name='pk_provider_runs'),
        sa.UniqueConstraint('run_id', name='uq_provider_runs_run_id'),
    )

    # Create indexes for provider_runs
    op.create_index('ix_provider_runs_id', 'provider_runs', ['id'], unique=False)
    op.create_index('ix_provider_runs_provider_id', 'provider_runs', ['provider_id'], unique=False)
    op.create_index('ix_provider_runs_parent_run_id', 'provider_runs', ['parent_run_id'], unique=False)
    op.create_index('ix_provider_runs_status', 'provider_runs', ['status'], unique=False)
    op.create_index('ix_provider_runs_run_id', 'provider_runs', ['run_id'], unique=True)


def downgrade() -> None:
    """Drop initial tables."""
    bind = op.get_bind()

    # Drop indexes
    op.drop_index('ix_provider_runs_run_id', table_name='provider_runs')
    op.drop_index('ix_provider_runs_status', table_name='provider_runs')
    op.drop_index('ix_provider_runs_parent_run_id', table_name='provider_runs')
    op.drop_index('ix_provider_runs_provider_id', table_name='provider_runs')
    op.drop_index('ix_provider_runs_id', table_name='provider_runs')

    op.drop_index('ix_provider_endpoints_name', table_name='provider_endpoints')
    op.drop_index('ix_provider_endpoints_provider_id', table_name='provider_endpoints')
    op.drop_index('ix_provider_endpoints_id', table_name='provider_endpoints')

    op.drop_index('ix_external_providers_tenant_id', table_name='external_providers')
    op.drop_index('ix_external_providers_name', table_name='external_providers')
    op.drop_index('ix_external_providers_id', table_name='external_providers')

    # Drop tables
    op.drop_table('provider_runs')
    op.drop_table('provider_endpoints')
    op.drop_table('external_providers')

    # Drop enum types
    run_status_enum.drop(bind, checkfirst=True)
    provider_status_enum.drop(bind, checkfirst=True)
    provider_type_enum.drop(bind, checkfirst=True)


