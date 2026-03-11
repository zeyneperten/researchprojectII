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

# =============================================================================
# Raw data plots
# =============================================================================

def plot_raster_and_population_rate(filtered_spikes, poprate, time, 
                                    col_interval, phase_intervals=None, 
                                    title="Raster and Population Rate"):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   figsize=(15, 6), height_ratios=[2, 1],
                                   gridspec_kw={'hspace': 0.05})
    
    # Raster Plot 
    for i, unit_spikes in enumerate(filtered_spikes):
        ax1.vlines(unit_spikes, i + 0.5, i + 1.5, color='black', linewidth=0.5)

    ax1.set_ylabel("Unit #")
    ax1.set_title(title)
    ax1.set_ylim(0.5, len(filtered_spikes) + 0.5)
    ax1.invert_yaxis()  # Optional: unit 0 at top
    ax1.grid(False)

    # Population Rate Plot
    ax2.plot(time, poprate, color='blue', linewidth=1.2, alpha=0.4)
    ax2.set_ylabel("Firing rate (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(time[0], time[-1])
    ax2.grid(True, linestyle=':', linewidth=0.5)

    for start, end in col_interval:
        ax1.axvspan(start, end, color='grey', alpha=0.4)
        ax2.axvspan(start, end, color='grey', alpha=0.4)
    colors = cm.get_cmap('tab20b', len(phase_intervals))
    if phase_intervals:
        for i, (start, end) in enumerate(phase_intervals):
            ax1.axvspan(start, end, color=colors(i), alpha=0.1)

    plt.show()

def plot_spike_locations(xyt, spikes, unit_index, title=None):

    # Extract x, y, t (in seconds)
    x = xyt[0]
    y = xyt[1]
    t = xyt[2] / 1e6  # timestamps in seconds

    unit_spikes = spikes[unit_index]

    # Match spike times to nearest position timestamps
    spike_idx = np.searchsorted(t, unit_spikes)

    # Remove out-of-bounds indices (e.g., spikes beyond xyt time)
    spike_idx = spike_idx[spike_idx < len(x)]

    # Get spike locations
    x_spikes = x[spike_idx]
    y_spikes = y[spike_idx]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=1, color='lightgray', label='Trajectory', alpha=0.5)
    plt.scatter(x_spikes, y_spikes, s=8, color='red', label='Spikes')
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(title or f"Spike Locations - Unit {unit_index}")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
