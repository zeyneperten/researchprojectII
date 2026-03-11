"""Matplotlib style configuration for figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt

_STYLE_APPLIED = False


def set_plot_style():
    """
    Configure Matplotlib settings for publication-quality figures
    with Computer Modern fonts and transparent backgrounds.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    mpl.rcParams.update({
        # Font and text rendering
        'pdf.fonttype': 42,           # Keep text as text in vector outputs
        'ps.fonttype': 42,
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'], #CMU Serif Computer Modern
        'font.sans-serif': ['DejaVu Sans'], #CMU Sans Serif
        'font.size': 12,
        'text.usetex': False,       # keep off unless you’ve verified TeX toolchain

        # Line and layout
        'lines.linewidth': 1,
        'savefig.transparent': True,  # Transparent backgrounds
        'savefig.bbox': 'tight',      # Trim margins automatically

        # Axes and tick formatting
        'axes.titlesize': 11,
        'axes.labelsize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 10,
    })

    _STYLE_APPLIED = True

def PublicationStandard():
    set_plot_style()

    # do all the publication-standard decorations
    fig = plt.gcf()

    # get original figure size
    #siz = np.array(fig.get_size_inches())
    #siz *= 1.5 / siz[1]   # shrink vertically to 1.5 inches
    #fig.set_size_inches(siz)
    #fig.set_dpi(300)

    # loop over all axes in the figure
    for ax in fig.get_axes():
        # remove top/right spines
        ax.spines[['top', 'right']].set_visible(False)

        # legend cleanup
        legend = ax.get_legend()
        if legend:
            legend.set_frame_on(False)
            legend.set_bbox_to_anchor([1, 1])
            legend.set_loc('upper left')
