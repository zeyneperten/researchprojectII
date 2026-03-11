"""Data utilities for loading shared resources.

This module provides utilities for loading data files that are shared
across multiple scripts, with proper path handling and caching.
"""

# Standard library imports
import functools
from pathlib import Path

# Third-party imports
import numpy as np

#TODO: load_data, load_results, load_mat_dict


@functools.lru_cache(maxsize=1)
def load_nrm():
    """Load normalization parameters from nrm.npy file.
    
    This function loads the normalization parameters used for sequence analysis.
    The result is cached, so the file is only read once even if called multiple times.
    
    Returns
    -------
    np.ndarray
        Normalization parameters array from nrm.npy file.
        
    Raises
    ------
    FileNotFoundError
        If nrm.npy cannot be found in the expected location.
        
    Examples
    --------
    >>> from scripts.data_utils import load_nrm
    >>> nrm = load_nrm()
    >>> # nrm is now available for use
    
    Notes
    -----
    The nrm.npy file is expected to be in the scripts directory.
    The function uses caching, so subsequent calls return the same
    array without re-reading the file.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    nrm_path = script_dir / 'nrm.npy'
    
    if not nrm_path.exists():
        raise FileNotFoundError(
            f"nrm.npy not found at expected location: {nrm_path}\n"
            f"Please ensure nrm.npy is in the scripts directory."
        )
    
    return np.load(nrm_path, allow_pickle=True)


def get_nrm():
    """Convenience alias for load_nrm().
    
    Returns
    -------
    np.ndarray
        Normalization parameters array.
        
    See Also
    --------
    load_nrm : The main function for loading normalization parameters.
    """
    return load_nrm()
