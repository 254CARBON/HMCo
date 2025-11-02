"""
Provider endpoint model.
"""

from sqlalchemy import Column, Integer, String, Text, Boolean, JSON, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from .base import BaseModel


class ProviderEndpoint(BaseModel):
    """Individual endpoint configuration for a provider."""

    __tablename__ = "provider_endpoints"
    __table_args__ = (
        UniqueConstraint('provider_id', 'name', name='uq_provider_endpoints_provider_name'),
    )

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(
        Integer,
        ForeignKey("external_providers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Endpoint details
    name = Column(String(255), nullable=False)
    path = Column(String(1024), nullable=False)  # API path or endpoint identifier
    method = Column(String(10), default="GET", nullable=False)  # HTTP method for REST APIs

    # Configuration
    headers = Column(JSON)  # Static headers
    query_params = Column(JSON)  # Static query parameters
    body_template = Column(Text)  # Template for request body

    # Pagination
    pagination_type = Column(String(50))  # cursor, offset, page, none
    pagination_config = Column(JSON)  # Pagination-specific settings

    # Response handling
    response_path = Column(String(255))  # JSONPath to extract data
    field_mapping = Column(JSON)  # Map provider fields to internal schema

    # Rate limiting
    rate_limit_group = Column(String(100))  # Group for shared rate limiting

    # Quality checks
    sample_size = Column(Integer, default=100, nullable=False)  # Sample size for validation
    validation_rules = Column(JSON)  # Field validation rules

    # Status
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    provider = relationship("ExternalProvider", back_populates="endpoints", passive_deletes=True)

    def __repr__(self):
        return f"<ProviderEndpoint(id={self.id}, provider_id={self.provider_id}, name='{self.name}')>"

