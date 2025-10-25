"""
External provider model.
"""

from sqlalchemy import Column, Integer, String, Text, JSON, Enum as SQLEnum, UniqueConstraint
from enum import Enum

from .base import BaseModel
from sqlalchemy.orm import relationship


class ProviderType(Enum):
    """Types of external data providers."""
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"
    FILE_FTP = "file_ftp"
    DATABASE = "database"


class ProviderStatus(Enum):
    """Status of a provider."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


def _enum_values(enum_cls):
    """Return the value payload for SQLAlchemy enum mappings."""
    return [member.value for member in enum_cls]


class ExternalProvider(BaseModel):
    """External data provider configuration."""

    __tablename__ = "external_providers"
    __table_args__ = (
        UniqueConstraint('tenant_id', 'name', name='uq_external_providers_tenant_name'),
    )

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)

    # Provider details
    provider_type = Column(
        SQLEnum(ProviderType, values_callable=_enum_values, name="provider_type"),
        nullable=False,
    )
    status = Column(
        SQLEnum(ProviderStatus, values_callable=_enum_values, name="provider_status"),
        default=ProviderStatus.INACTIVE,
        nullable=False,
    )

    # Configuration
    base_url = Column(String(1024))  # For REST/GraphQL APIs
    config = Column(JSON)  # Provider-specific configuration
    credentials_ref = Column(String(255))  # Vault secret reference
    rate_limits = Column(JSON)  # Rate limiting configuration

    # Data governance
    tenant_id = Column(String(255), nullable=False, index=True)
    owner = Column(String(255), nullable=False)
    tags = Column(JSON)  # List of tags for categorization

    # Scheduling
    schedule_cron = Column(String(100))  # Cron expression for scheduled runs
    schedule_timezone = Column(String(50), default="UTC", nullable=False)

    # Target configuration
    sink_type = Column(String(50))  # iceberg, clickhouse, kafka
    sink_config = Column(JSON)  # Sink-specific configuration

    # Quality gates
    schema_contract = Column(JSON)  # JSON Schema for data validation
    slo_config = Column(JSON)  # SLO targets (freshness, accuracy, etc.)

    # Metadata
    created_by = Column(String(255), nullable=False)
    updated_by = Column(String(255))

    # Relationships
    endpoints = relationship(
        "ProviderEndpoint",
        back_populates="provider",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    runs = relationship(
        "ProviderRun",
        back_populates="provider",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self):
        return f"<ExternalProvider(id={self.id}, name='{self.name}', type={self.provider_type})>"
