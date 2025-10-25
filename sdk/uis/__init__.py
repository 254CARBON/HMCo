"""
Unified Ingestion Spec (UIS) SDK.
"""

from .spec import UnifiedIngestionSpec, ProviderConfig, EndpointConfig, TransformConfig
from .parser import UISParser
from .validator import UISValidator

__version__ = "1.1.0"

__all__ = [
    'UnifiedIngestionSpec',
    'ProviderConfig',
    'EndpointConfig',
    'TransformConfig',
    'UISParser',
    'UISValidator'
]
