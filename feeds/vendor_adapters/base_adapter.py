"""
Base adapter interface for vendor feed integrations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass


@dataclass
class CurveSnapshot:
    """Curve snapshot data structure."""
    snapshot_id: str
    hub: str
    timestamp: datetime
    vendor: str
    commodity: str
    curve_points: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AISPosition:
    """AIS position data structure."""
    mmsi: str
    timestamp: datetime
    lat: float
    lon: float
    speed_knots: float
    course: float
    vessel_type: str
    metadata: Optional[Dict[str, Any]] = None


class VendorAdapter(ABC):
    """Base adapter interface for vendor integrations."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adapter.
        
        Args:
            config: Configuration dictionary with credentials, endpoints, etc.
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self):
        """Validate configuration. Raise ValueError if invalid."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test connectivity to vendor API.
        
        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def get_rate_limit(self) -> Dict[str, int]:
        """
        Get rate limit information.
        
        Returns:
            Dict with 'limit' and 'remaining' keys
        """
        pass


class CurveAdapter(VendorAdapter):
    """Base adapter for curve data providers."""

    @abstractmethod
    def fetch_curve_snapshot(
        self,
        hub: str,
        snapshot_date: Optional[datetime] = None
    ) -> CurveSnapshot:
        """
        Fetch curve snapshot for hub.
        
        Args:
            hub: Hub identifier
            snapshot_date: Snapshot date (defaults to latest)
            
        Returns:
            Curve snapshot
        """
        pass

    @abstractmethod
    def list_available_hubs(self) -> List[str]:
        """
        List available hubs from vendor.
        
        Returns:
            List of hub identifiers
        """
        pass

    @abstractmethod
    def get_options_surface(
        self,
        hub: str,
        snapshot_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get options surface for hub.
        
        Args:
            hub: Hub identifier
            snapshot_date: Snapshot date
            
        Returns:
            Options surface data
        """
        pass


class AISAdapter(VendorAdapter):
    """Base adapter for AIS/marine data providers."""

    @abstractmethod
    def fetch_positions(
        self,
        vessel_ids: Optional[List[str]] = None,
        bounding_box: Optional[Dict[str, float]] = None
    ) -> List[AISPosition]:
        """
        Fetch AIS positions.
        
        Args:
            vessel_ids: Optional list of vessel MMSIs
            bounding_box: Optional geographic bounding box
            
        Returns:
            List of AIS positions
        """
        pass

    @abstractmethod
    def get_vessel_info(self, mmsi: str) -> Dict[str, Any]:
        """
        Get vessel information.
        
        Args:
            mmsi: Vessel MMSI
            
        Returns:
            Vessel information
        """
        pass

    @abstractmethod
    def track_vessel(
        self,
        mmsi: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[AISPosition]:
        """
        Get historical track for vessel.
        
        Args:
            mmsi: Vessel MMSI
            start_time: Start of tracking period
            end_time: End of tracking period
            
        Returns:
            List of positions
        """
        pass
