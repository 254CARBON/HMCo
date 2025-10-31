"""
PTDF (Power Transfer Distribution Factor) estimation
Shift factors for DC power flow approximation
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PTDFEstimator:
    """
    Estimates PTDF/shift factors for transmission network
    Used in physics-aware LMP modeling
    """
    
    def __init__(self):
        self.ptdf_cache = {}
        
    def compute_ptdf(
        self,
        topology: Dict,
        reference_bus: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute PTDF matrix from network topology
        
        Args:
            topology: Network topology dict
            reference_bus: Reference bus index (default: first node)
            
        Returns:
            PTDF matrix [num_nodes, num_nodes]
        """
        cache_key = topology['iso']
        if cache_key in self.ptdf_cache:
            return self.ptdf_cache[cache_key]
        
        logger.info(f"Computing PTDF for {topology['iso']}")
        
        num_nodes = topology['num_nodes']
        
        # Get admittance matrix from topology
        Y = self._build_admittance_matrix(topology)
        
        # Set reference bus
        if reference_bus is None:
            reference_bus = 0
        
        # Compute PTDF using DC power flow approximation
        ptdf = self._dc_ptdf(Y, reference_bus)
        
        self.ptdf_cache[cache_key] = ptdf
        logger.info(f"PTDF matrix computed: {ptdf.shape}")
        
        return ptdf
    
    def _build_admittance_matrix(self, topology: Dict) -> np.ndarray:
        """Build network admittance matrix"""
        num_nodes = topology['num_nodes']
        Y = np.zeros((num_nodes, num_nodes))
        
        # Add transmission line admittances
        for i, j in topology['edges']:
            # Default susceptance (simplified)
            susceptance = 1.0 / 0.1  # 1 / reactance
            
            Y[i, i] += susceptance
            Y[j, j] += susceptance
            Y[i, j] -= susceptance
            Y[j, i] -= susceptance
        
        return Y
    
    def _dc_ptdf(
        self,
        Y: np.ndarray,
        ref_bus: int
    ) -> np.ndarray:
        """
        DC power flow PTDF calculation
        
        PTDF[l,k] = change in flow on line l due to injection at bus k
        """
        n = Y.shape[0]
        
        # Remove reference bus row and column
        Y_reduced = np.delete(np.delete(Y, ref_bus, axis=0), ref_bus, axis=1)
        
        try:
            # Compute inverse (pseudo-inverse for stability)
            Y_inv = np.linalg.pinv(Y_reduced)
            
            # Reconstruct full matrix with reference bus
            Y_full_inv = np.zeros((n, n))
            
            # Fill in non-reference entries
            mask = np.ones(n, dtype=bool)
            mask[ref_bus] = False
            
            Y_full_inv[np.ix_(mask, mask)] = Y_inv
            
            # PTDF approximation
            # For simplified model: PTDF ≈ sensitivity matrix
            ptdf = Y_full_inv
            
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix, using identity PTDF")
            ptdf = np.eye(n)
        
        return ptdf
    
    def get_shift_factors(
        self,
        topology: Dict,
        line_from: int,
        line_to: int
    ) -> np.ndarray:
        """
        Get shift factors for a specific transmission line
        
        Args:
            topology: Network topology
            line_from: Source bus of line
            line_to: Sink bus of line
            
        Returns:
            Shift factors for the line [num_nodes]
        """
        ptdf = self.compute_ptdf(topology)
        
        # Shift factor for line (from -> to) with injection at each bus
        shift_factors = ptdf[line_to, :] - ptdf[line_from, :]
        
        return shift_factors
