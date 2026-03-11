# Standard library imports
import copy
from collections import OrderedDict

# Third-party imports
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import rc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex
from scipy.cluster.hierarchy import dendrogram

# Local imports
from ..clustering import core as seq_clustering
from .style import set_plot_style, PublicationStandard

# Apply project-wide matplotlib configuration
set_plot_style()


def average_sequence_times(bursts, clist, method="center_of_mass"):
    """
    Compute per-neuron template times (seconds) for one cluster, plus the neuron order (earliest→latest).
    bursts: list of bursts; each burst is a list of per-neuron spike-time arrays.
    clist: indices of bursts in the cluster.
    method: "center_of_mass" (mean within burst) or "first_spike" (min within burst).
    """
    n_neurons = len(bursts[0])
    time_values = [[] for _ in range(n_neurons)]

    for i in np.asarray(clist, dtype=int):
        burst = bursts[i]
        for j, spikes in enumerate(burst):
            if len(spikes) == 0:
                continue
            if method == "center_of_mass":
                val = float(np.mean(spikes))
            elif method == "first_spike":
                val = float(np.min(spikes))
            else:
                raise ValueError(f"Unknown method: {method}")
            time_values[j].append(val)

    if method == "center_of_mass":
        template_times = np.array([np.nanmean(v) if v else np.nan for v in time_values], dtype=float)
    else:
        template_times = np.array([np.nanmedian(v) if v else np.nan for v in time_values], dtype=float)

    # Order neurons by template time (NaNs pushed to end by using nan_to_num)
    isort = np.argsort(np.nan_to_num(template_times, nan=np.inf))
    return template_times, isort

def summarize_burst_times(burst, method="center_of_mass"):
    """Summarize one burst to a 1D array of times per neuron (NaN if silent)."""
    x = np.full(len(burst), np.nan, float)
    for j, spikes in enumerate(burst):
        if len(spikes) == 0:
            continue
        if method == "center_of_mass":
            x[j] = float(np.mean(spikes))
        elif method == "first_spike":
            x[j] = float(np.min(spikes))
        else:
            raise ValueError(f"Unknown method: {method}")
    return x

def _finite_span(x):
    """Return (x_min, x_span) over finite elements; (0,0) if none."""
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return 0.0, 0.0
    xmin = xf.min()
    xspan = xf.max() - xmin
    return float(xmin), float(xspan)

def _tile_block(ax, x, y, nshift, color, s=10, pad=0.05, min_span=1e-6):
    """
    Plot one block: shift times to start at 0, place at current nshift,
    return updated nshift (block_span + pad added).
    """
    # Shift to start at 0 in its own local axis
    xmin, xspan = _finite_span(x)
    # guard zero-span blocks
    if xspan < min_span:
        xspan = min_span
    x_rel = x - xmin
    ax.scatter(nshift + x_rel, y, s=s, c=color, edgecolor='none')
    # advance by block width + padding
    return nshift + xspan + pad