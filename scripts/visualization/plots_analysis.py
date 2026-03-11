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
    rd_shuf,
    kept,
    score_key="within_clust",
    descending=True,
    figsize=(12, 3.5),
    bar_width=0.72,          # wider since we overlap
    alpha_real=0.55,
    alpha_shuf=0.55,
    show_zero_line=True,
    check_label="shuffled"
):
    """
    Overlapped barplot (real vs shuffled) for clusters in `kept` (from real filtering).

    Assumptions:
      - rd_*["ids_clust"] is per-sequence labels (used only to define cluster set/order)
      - rd_*["clust_scores"][score_key] is one score per cluster, ordered like:
            np.unique(rd_*["ids_clust"])
      - `kept` is iterable of cluster labels to include (real case).
    """
    kept = np.asarray(list(kept), dtype=int)

    # --- cluster universe (from real labels) ---
    ids_real = np.asarray(rd_real["ids_clust"])
    clusters_all = np.unique(ids_real)

    kept_set = set(int(x) for x in kept.tolist())
    clusters = np.array([c for c in clusters_all if int(c) in kept_set], dtype=int)

    # --- helper: label -> score (one per cluster) ---
    def label_to_score_array(rd, clusters):
        uniq = np.unique(np.asarray(rd["ids_clust"]))
        scores = np.asarray(rd["clust_scores"][score_key], float)

        if scores.shape[0] != uniq.shape[0]:
            raise ValueError(
                f"Score length mismatch for '{score_key}': "
                f"len(scores)={len(scores)} but len(unique_clusters)={len(uniq)}. "
                "Ensure clust_scores[score_key] is one value per cluster and corresponds "
                "to np.unique(ids_clust)."
            )

        m = {int(c): float(s) for c, s in zip(uniq, scores)}
        return np.array([m.get(int(c), np.nan) for c in clusters], float)

    score_real = label_to_score_array(rd_real, clusters)
    score_shuf = label_to_score_array(rd_shuf, clusters)

    # --- ordering (descending by real) ---
    order = np.argsort(score_real)
    if descending:
        order = order[::-1]
    clusters   = clusters[order]
    score_real = score_real[order]
    score_shuf = score_shuf[order]

    # --- plot (overlapped bars) ---
    x = np.arange(len(clusters))
    fig, ax = plt.subplots(figsize=figsize)

    # draw the larger bars first so the smaller one remains visible
    # (keeps overlap readable when values differ a lot)
    draw_real_first = np.nan_to_num(np.abs(score_real), nan=-np.inf) >= np.nan_to_num(np.abs(score_shuf), nan=-np.inf)

    # base: draw all real then all shuf (simple and consistent)
    ax.bar(x, score_real, width=bar_width, color="tab:blue", alpha=alpha_real, label="Cluster scores", edgecolor="none")
    ax.bar(x, score_shuf, width=bar_width, color="tab:red",  alpha=alpha_shuf, label=check_label, edgecolor="none")

    if show_zero_line:
        ax.axhline(0, linewidth=1, color="0.2", alpha=0.6)

    # cosmetics
    ax.set_xlabel("Kept clusters (ordered by real score)")
    ax.set_ylabel(score_key)
    ax.legend(frameon=False, ncol=2, loc="best")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in clusters], rotation=90)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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