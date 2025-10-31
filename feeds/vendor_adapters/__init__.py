"""
Vendor adapter registry and factory.
"""

from typing import Dict, Any, Type
from .base_adapter import VendorAdapter, CurveAdapter, AISAdapter

# Adapter registry
_ADAPTER_REGISTRY: Dict[str, Type[VendorAdapter]] = {}


def register_adapter(name: str, adapter_class: Type[VendorAdapter]):
    """
    Register an adapter.
    
    Args:
        name: Adapter name
        adapter_class: Adapter class
    """
    _ADAPTER_REGISTRY[name.lower()] = adapter_class


def get_adapter(name: str, config: Dict[str, Any]) -> VendorAdapter:
    """
    Get adapter instance.
    
    Args:
        name: Adapter name
        config: Configuration dict
        
    Returns:
        Adapter instance
        
    Raises:
        ValueError: If adapter not found
    """
    adapter_class = _ADAPTER_REGISTRY.get(name.lower())
    if not adapter_class:
        raise ValueError(f"Adapter '{name}' not found. Available: {list(_ADAPTER_REGISTRY.keys())}")
    
    return adapter_class(config)


def list_adapters() -> list:
    """
    List registered adapters.
    
    Returns:
        List of adapter names
    """
    return list(_ADAPTER_REGISTRY.keys())


# Example adapter implementations would be imported and registered here
# from .ice_adapter import ICEAdapter
# register_adapter('ice', ICEAdapter)
# from .cme_adapter import CMEAdapter
# register_adapter('cme', CMEAdapter)

__all__ = [
    'VendorAdapter',
    'CurveAdapter',
    'AISAdapter',
    'register_adapter',
    'get_adapter',
    'list_adapters',
]
