"""Sequence detection and clustering analysis tools.

This package provides utilities for:
- Loading and preprocessing neuronal sequence data
- Clustering sequences using various methods (hierarchical, DBSCAN, Leiden)
- Analyzing sequence similarity and structure
- Visualizing results
- Simulating synthetic sequence data
"""

__version__ = "0.1.0"

# Make key modules easily accessible
from . import analysis
from . import clustering
from . import config
from . import data
from . import data_utils
from . import simulation
from . import visualization

# Convenience import for nrm data
from .data_utils import load_nrm, get_nrm

__all__ = [
    "analysis",
    "clustering",
    "config",
    "data",
    "data_utils",
    "simulation",
    "visualization",
    "load_nrm",
    "get_nrm",
]
