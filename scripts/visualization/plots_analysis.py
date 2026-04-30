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
# Analysis plots
# =============================================================================

def plot_cluster_scores(
    ids_clust,
    clust_scores,
    score_keys=("ratio", "within_clust"),
    sort_by="count",                 # "count" or the name of a score in clust_scores
    sort_desc=True,                  # descending order
    threshold=None,                  # e.g., 0.5 or {"ratio":0.5, "within_clust":0.6}
    figsize=(6, 3),
    bar_alpha=0.7,
    score_style="points",            # "points" or "line"
    colors=None,                     # dict like {"ratio": "C1", "within_clust": "C2"}
    markers=None                     # dict like {"ratio": ".", "within_clust": "x"}
):
    """
    Parameters
    ----------
    ids_clust : array-like of int
        Cluster id per sequence (e.g., result_dict["ids_clust"]).
    clust_scores : dict of {score_name: array-like}
        Each entry is indexable by cluster id (i.e., clust_scores[name][cluster_id] -> float).
    score_keys : sequence of str
        Which scores to plot on the secondary axis.
    sort_by : str
        "count" or one of `score_keys` to define sorting.
    sort_desc : bool
        Sort descending (True) or ascending (False).
    threshold : None, float, or dict
        If float, draws one horizontal line for all scores; if dict, can specify per-score threshold.
    figsize : tuple
        Figure size.
    score_style : str
        "points" for scatter points, "line" for connected lines.
    colors : dict or None
        Optional mapping from score name to matplotlib color.
    markers : dict or None
        Optional mapping from score name to marker style (only used for "points").
    """
    ids_clust = np.asarray(ids_clust)
    unique_ids = np.unique(ids_clust)

    # --- counts per cluster ---
    counts = np.array([(ids_clust == u).sum() for u in unique_ids])

    # --- align all requested scores to cluster id order ---
    aligned_scores = {}
    for key in score_keys:
        if key not in clust_scores:
            raise KeyError(f"score '{key}' not found in clust_scores")
        # assume clust_scores[key] is indexable by cluster id
        aligned_scores[key] = np.array([clust_scores[key][u] for u in unique_ids])

    # --- choose sorting key ---
    if sort_by == "count":
        sort_values = counts
    else:
        if sort_by not in aligned_scores:
            raise ValueError(f"sort_by='{sort_by}' not in scores {list(aligned_scores.keys())} or 'count'")
        sort_values = aligned_scores[sort_by]

    sort_idx = np.argsort(sort_values)
    if sort_desc:
        sort_idx = sort_idx[::-1]

    unique_ids = unique_ids[sort_idx]
    counts = counts[sort_idx]
    for key in aligned_scores:
        aligned_scores[key] = aligned_scores[key][sort_idx]

    # --- plotting ---
    fig, ax1 = plt.subplots(figsize=figsize)

    # Bars: counts
    bars = ax1.bar(np.arange(len(unique_ids)), counts, alpha=bar_alpha, color='k')
    ax1.set_ylabel("# Sequences")
    ax1.set_xlabel("Cluster (sorted by {})".format(sort_by))
    ax1.set_xticks([])

    # Secondary axis: scores
    ax2 = ax1.twinx()
    x = np.arange(len(unique_ids))

    # defaults if not provided
    if colors is None:
        colors = {k: None for k in score_keys}
    if markers is None:
        markers = {k: "." for k in score_keys}

    for key in score_keys:
        y = aligned_scores[key]
        if score_style == "line":
            ax2.plot(x, y, label=key, linewidth=1)
        else:
            ax2.plot(x, y, linestyle="", marker=markers.get(key, "."), label=key, alpha=0.5)

    # thresholds
    if threshold is not None:
        if isinstance(threshold, dict):
            for key in score_keys:
                if key in threshold:
                    ax2.axhline(float(threshold[key]), linestyle="--", linewidth=1)
        else:
            ax2.axhline(float(threshold), linestyle="--", linewidth=1)

    ax2.set_ylabel("Scores")
    ax2.legend(frameon=False)
    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_cluster_scores_comparison(
    rd_real,
    out_shuf,
    kept,
    score_key="within_clust",
    descending=True,
    figsize=(5, 3),
    bar_width=0.5,
    show_zero_line=True,
    shuf_label="shuffled mean ± std",
    real_label="real",
):
    """
    Plot real cluster scores as bars, with shuffled mean ± std overlaid
    as a red dot + vertical error bar on the same x position.

    Assumptions
    -----------
    rd_real["clust_scores"][score_key]:
        one score per cluster, ordered like np.unique(rd_real["ids_clust"])

    out_shuf["mean"], out_shuf["std"]:
        one shuffled mean/std per cluster, ordered like np.unique(rd_real["ids_clust"])
        out_shuf output from within_clust_shuffle in shuffling.py
    """
    kept = np.asarray(list(kept), dtype=int)

    # cluster labels from real result
    uniq_real = np.unique(np.asarray(rd_real["ids_clust"]))
    kept_set = set(kept.tolist())
    clusters = np.array([c for c in uniq_real if int(c) in kept_set], dtype=int)

    # --- real scores ---
    score_real_all = np.asarray(rd_real["clust_scores"][score_key], dtype=float)
    if len(score_real_all) != len(uniq_real):
        raise ValueError(
            f"rd_real['clust_scores']['{score_key}'] must have one value per cluster "
            f"(expected {len(uniq_real)}, got {len(score_real_all)})."
        )

    real_map = {int(c): float(s) for c, s in zip(uniq_real, score_real_all)}
    score_real = np.array([real_map[int(c)] for c in clusters], dtype=float)

    # --- shuffled mean/std ---
    shuf_mean_all = np.asarray(out_shuf["mean"], dtype=float)
    shuf_std_all = np.asarray(out_shuf["std"], dtype=float)

    if len(shuf_mean_all) != len(uniq_real) or len(shuf_std_all) != len(uniq_real):
        raise ValueError(
            "out_shuf['mean'] and out_shuf['std'] must each contain one value per cluster, "
            f"ordered like np.unique(rd_real['ids_clust']) = {len(uniq_real)} clusters.\n"
            f"Got len(mean)={len(shuf_mean_all)}, len(std)={len(shuf_std_all)}."
        )

    shuf_mean_map = {int(c): float(m) for c, m in zip(uniq_real, shuf_mean_all)}
    shuf_std_map = {int(c): float(s) for c, s in zip(uniq_real, shuf_std_all)}

    score_shuf_mean = np.array([shuf_mean_map[int(c)] for c in clusters], dtype=float)
    score_shuf_std = np.array([shuf_std_map[int(c)] for c in clusters], dtype=float)

    # --- sort by real score ---
    order = np.argsort(score_real)
    if descending:
        order = order[::-1]

    clusters = clusters[order]
    score_real = score_real[order]
    score_shuf_mean = score_shuf_mean[order]
    score_shuf_std = score_shuf_std[order]

    # --- plot ---
    x = np.arange(len(clusters))

    fig, ax = plt.subplots(figsize=figsize)

    # real bars
    ax.bar(
        x,
        score_real,
        width=bar_width,
        color="tab:blue",
        alpha=0.8,
        edgecolor="none",
        label=real_label,
        zorder=1,
    )

    # shuffled mean ± std overlaid on bar center
    ax.errorbar(
        x,
        score_shuf_mean,
        yerr=score_shuf_std,
        fmt="o",
        color="tab:red",
        capsize=3,
        elinewidth=1.5,
        markersize=5,
        label=shuf_label,
        zorder=3,
    )

    if show_zero_line:
        ax.axhline(0, color="0.2", lw=1, alpha=0.6, zorder=0)

    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Intra-cluster score")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in clusters], rotation=90)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    plt.tight_layout()
    return fig, ax


def plot_survival_scores(
    scores,
    color_data,
    c_label,
    x_thresh=None,
    y_thresh=None,
    x_key="survival_freq",
    y_key="pairwise_jaccard_cond_survival",
    x_label="survival frequency",
    y_label="pairwise Jaccard",
    cmap="viridis",
    figsize=(3,2.5),
    s=8,
    alpha=0.6,
):
    """
    Scatter plot from a score dictionary.

    Parameters
    ----------
    scores : dict
        Dictionary containing arrays for x, y, and color values.
    color_data : array-like
        Data for point colors, e.g., mean cluster size or sequence labels.
    c_label : str
        Color bar label.
    x_key, y_key, color_key : str
        Keys in `scores` used for x-axis, y-axis, and point color.
    x_label, y_label : str
        Axis labels.
    x_thresh, y_thresh, y_thresh_light : float or None
        Optional reference lines.
    cmap : str
        Matplotlib colormap.
    figsize : tuple
        Figure size.
    s : float
        Marker size.
    alpha : float
        Marker transparency.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    x = np.asarray(scores[x_key], dtype=float)
    y = np.asarray(scores[y_key], dtype=float)
    c = np.asarray(color_data, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)

    fig, ax = plt.subplots(figsize=figsize)

    scat = ax.scatter(
        x[m],
        y[m],
        c=c[m],
        s=s,
        alpha=alpha,
        cmap=cmap,
    )

    if x_thresh is not None:
        ax.axvline(x_thresh, color="k", linestyle="--", lw=1)
    if y_thresh is not None:
        ax.axhline(y_thresh, color="k", linestyle="--", lw=1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(scat, ax=ax, location="right", fraction=0.2, pad=0.02, shrink=0.8)
    cbar.set_label(c_label)

    plt.tight_layout()
    return fig, ax


# =============================================================================
# Cell contributions plots
# =============================================================================

def plot_cell_contributions(
    contrib,
    clusters,
    cell_ids=None,
    top_k_cells=None,
    max_cols=8,
    figsize_per_cell=(2.2, 1.6),
    sharey=True,
    title=None,
    xtick_rotation=90,
    sort_bars="none",  # "none" | "desc" | "asc"
):
    """
    Small barplot per cell: x = clusters, y = contribution.
    """
    contrib = np.asarray(contrib)
    clusters = np.asarray(clusters)
    n_cells, n_clust = contrib.shape

    if cell_ids is None:
        cell_ids = np.arange(n_cells)

    # pick subset if requested
    if top_k_cells is not None:
        score = contrib.max(axis=1)
        idx = np.argsort(score)[::-1][:top_k_cells]
        cell_ids = np.asarray(cell_ids)[idx]
        contrib = contrib[idx, :]

    n_plot = len(cell_ids)
    n_cols = min(max_cols, n_plot)
    n_rows = int(np.ceil(n_plot / n_cols))

    fig_w = figsize_per_cell[0] * n_cols
    fig_h = figsize_per_cell[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=sharey)
    axes = np.array(axes).reshape(-1)

    for ax_i, (ax, cid) in enumerate(zip(axes, cell_ids)):
        y = contrib[ax_i]

        # per-cell sorting
        if sort_bars == "desc":
            order = np.argsort(-y)
        elif sort_bars == "asc":
            order = np.argsort(y)
        else:
            order = np.arange(n_clust)

        y_plot = y[order]
        cl_plot = clusters[order]
        x = np.arange(n_clust)

        ax.bar(x, y_plot)
        ax.set_title(f"cell {cid}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cl_plot], rotation=xtick_rotation, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)

    # hide unused axes
    for ax in axes[n_plot:]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    return fig


def plot_cluster_dominance(
    contrib_cell_given_cluster,   # shape (n_cells, n_clusters), columns sum to 1
    clusters,                     # shape (n_clusters,) cluster labels
    cell_ids=None,                # length n_cells; default 0..n_cells-1
    top_k=15,                     # show top-k cells per cluster
    max_cols=6,
    figsize_per_panel=(2.8, 1.9),
    sharey=True,
    title="p(cell | cluster) — dominance by cells",
    annotate_top=True,
    xtick_rotation=90,
):
    """
    Plot one subplot per cluster. Within each subplot: barplot over cells of p(cell | cluster).
    Good for spotting a single dominating cell.
    """
    P = np.asarray(contrib_cell_given_cluster, dtype=float)
    clusters = np.asarray(clusters)

    n_cells, n_clust = P.shape
    if cell_ids is None:
        cell_ids = np.arange(n_cells)
    cell_ids = np.asarray(cell_ids)

    # Make sure we interpret columns as clusters
    # (Optional sanity) columns sum to 1 for non-empty clusters.
    # colsum = P.sum(axis=0)

    n_plot = n_clust
    n_cols = min(max_cols, n_plot)
    n_rows = int(np.ceil(n_plot / n_cols))

    fig_w = figsize_per_panel[0] * n_cols
    fig_h = figsize_per_panel[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=sharey)
    axes = np.array(axes).reshape(-1)

    for i, (ax, clab) in enumerate(zip(axes, clusters)):
        y = P[:, i]

        if top_k is None or top_k >= n_cells:
            idx = np.argsort(-y)
        else:
            idx = np.argpartition(y, -top_k)[-top_k:]
            idx = idx[np.argsort(-y[idx])]  # sort those top_k desc

        y_top = y[idx]
        cells_top = cell_ids[idx]
        x = np.arange(y_top.size)

        ax.bar(x, y_top)
        ax.set_title(f"cluster {clab}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cells_top], rotation=xtick_rotation, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)

        if annotate_top and y_top.size > 0:
            ax.text(
                0.02, 0.98,
                f"top: cell {cells_top[0]} = {y_top[0]:.2f}",
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=7
            )

    for ax in axes[n_plot:]:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig