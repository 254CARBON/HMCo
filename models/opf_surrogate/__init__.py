"""
OPF Surrogate Model - Fast approximation of DC-OPF for constraint analysis
"""

from .surrogate_model import OPFSurrogate
from .trainer import OPFSurrogateTrainer

__all__ = ['OPFSurrogate', 'OPFSurrogateTrainer']
