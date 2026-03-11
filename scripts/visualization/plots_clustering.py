# Standard library imports
import copy
from collections import OrderedDict

# Third-party imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex
from scipy.cluster.hierarchy import dendrogram

# Local imports
from ..clustering import core as seq_clustering
from .style import set_plot_style
from .plots_helpers import average_sequence_times, summarize_burst_times, _tile_block

# Apply project-wide matplotlib configuration
set_plot_style()

# =============================================================================
# Cluster plots
# =============================================================================

def dendrogram_with_cluster_scores(
    Z,
    cluster_scores,
    clust_idx,
    labels=None,          # if None, auto-generate "0..n-1"
    cmap_name="viridis",
    figsize=(6, 3),
    dflt_col="#A0A0A0"
):
    n = Z.shape[0] + 1
    if labels is None:
        labels = [str(i) for i in range(n)]

    # default scores if none given
    uniq = np.unique(clust_idx)
    
    # map cluster -> color
    scores = np.array([cluster_scores[cid] for cid in uniq])
    valid = np.isfinite(scores)
    if np.any(valid):
        vmin = float(scores[valid].min())
        vmax = float(scores[valid].max()) if np.ptp(scores[valid]) > 0 else vmin + 1.0
    else:
        # fallback if all scores are NaN
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    cmap = mpl.colormaps[cmap_name] 
    cid2color = {cid: to_hex(cmap(norm(cluster_scores[cid]))) for cid in uniq}
    leaf_color = {i: cid2color[clust_idx[i]] for i in range(n)} # leaf colors

    # link colors (propagate until cross-cluster merge)
    link_cols = {}
    for i, (L, R) in enumerate(Z[:, :2].astype(int)):
        cL = link_cols.get(L, leaf_color.get(L, dflt_col))
        cR = link_cols.get(R, leaf_color.get(R, dflt_col))
        link_cols[i + n] = cL if cL == cR else dflt_col

    fig, ax = plt.subplots(figsize=figsize)
    D = dendrogram(
        Z,
        labels=None,                # no leaf labels
        color_threshold=None,
        link_color_func=lambda x: link_cols.get(x, dflt_col),
        leaf_font_size=9,
        leaf_rotation=45,
        p=len(uniq)*5, #10 15
        truncate_mode="lastp",
        ax=ax
    )

    # --- remove x-tick labels ---
    ax.set_xticklabels([])

    # colorbar for score scale
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Cluster score")
    
    plt.tight_layout()
    return D

def plot_mats(
    res,
    corrmat,
    cmap="coolwarm",
    figsize=(7, 3),
    cbar_fraction=0.02,
    cbar_pad=0.04,
    boundary_color="c",
    vmin=-1,
    vmax=1,
    lwall=1,
    replace_with=-1,
):
    idx_sorted = res["idx_sorted"]
    mask_kept_sorted = res.get("mask_kept_sorted", None)
    ids_filtered = res.get("ids_clust_filtered", None)

    # sorted matrix
    corrmat_sorted = corrmat[idx_sorted][:, idx_sorted]
    # kept-only matrix (kept mask is over sorted axis)
    corrmat_kept = corrmat_sorted[np.ix_(mask_kept_sorted, mask_kept_sorted)]

    fig, (ax_all, ax_kept_ax) = plt.subplots(1, 2, figsize=figsize)

    # ---- left: all (sorted) ----
    im_all = ax_all.imshow(corrmat_sorted, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_all.set_xticks([]); ax_all.set_yticks([])

    # boundaries between labels in the sorted view
    boundaries_all = np.flatnonzero(np.diff(ids_filtered.astype(int))) + 1

    for b in boundaries_all:
        ax_all.axhline(b - 0.5, color=boundary_color, linewidth=lwall)
        ax_all.axvline(b - 0.5, color=boundary_color, linewidth=lwall)

    # ---- right: kept only ----
    im_kept = ax_kept_ax.imshow(corrmat_kept, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar_kept = plt.colorbar(
        im_kept, ax=ax_kept_ax, orientation="vertical",
        fraction=cbar_fraction, pad=cbar_pad
    )
    cbar_kept.ax.tick_params(labelsize=8)
    ax_kept_ax.set_xticks([]); ax_kept_ax.set_yticks([])

    # kept labels in sorted order
    ids_kept = ids_filtered[mask_kept_sorted].astype(int)

    ids_kept_no_rep = ids_kept[ids_kept != replace_with]
    if ids_kept_no_rep.size > 1:
        boundaries_kept = np.flatnonzero(np.diff(ids_kept_no_rep)) + 1
        for b in boundaries_kept:
            ax_kept_ax.axhline(b - 0.5, color=boundary_color, linewidth=.5)
            ax_kept_ax.axvline(b - 0.5, color=boundary_color, linewidth=.5)

    fig.tight_layout()
    return fig, corrmat_sorted, corrmat_kept


def plot_cluster_confusion(
    labels_x,
    labels_y,
    xlabel,
    ylabel,
    normalize=None,        # None | "x" | "y"
    exclude_labels=None,   # e.g. {-1}
    cmap="viridis",
    ax=None,
):
    """
    Plot a confusion matrix between two cluster labelings.

    Parameters
    ----------
    labels_x : array-like, shape (n,)
        Cluster labels defining the x-axis (columns).
    labels_y : array-like, shape (n,)
        Cluster labels defining the y-axis (rows).
    normalize : None | "x" | "y", optional
        - None: raw counts
        - "x": normalize columns (each x-cluster sums to 1)
        - "y": normalize rows
    exclude_labels : set or list, optional
        Labels to exclude from both axes (e.g. {-1}).
    cmap : str
        Matplotlib colormap.
    ax : matplotlib Axes, optional

    Returns
    -------
    cm : ndarray
        Confusion matrix (rows=y clusters, cols=x clusters)
    x_labels : ndarray
        Unique x-axis cluster labels
    y_labels : ndarray
        Unique y-axis cluster labels
    """
    labels_x = np.asarray(labels_x)
    labels_y = np.asarray(labels_y)

    if labels_x.shape != labels_y.shape:
        raise ValueError("labels_x and labels_y must have the same shape")

    mask = np.ones(len(labels_x), dtype=bool)

    # exclude NaNs
    mask &= ~np.isnan(labels_x) & ~np.isnan(labels_y)

    # exclude specific labels (e.g. -1)
    if exclude_labels is not None:
        exclude_labels = set(exclude_labels)
        mask &= ~np.isin(labels_x, list(exclude_labels))
        mask &= ~np.isin(labels_y, list(exclude_labels))

    labels_x = labels_x[mask]
    labels_y = labels_y[mask]

    x_labels = np.unique(labels_x)
    y_labels = np.unique(labels_y)

    x_map = {c: i for i, c in enumerate(x_labels)}
    y_map = {c: i for i, c in enumerate(y_labels)}

    cm = np.zeros((len(y_labels), len(x_labels)), dtype=float)

    for lx, ly in zip(labels_x, labels_y):
        cm[y_map[ly], x_map[lx]] += 1



# =============================================================================
# Sequence plots
# =============================================================================

# ---------- templates ----------
def plot_templates_timebased(templates, isort, ax, fac=1.0, pad=0.05):
    """
    Plot all cluster templates side by side (colored), using adaptive spacing:
    each template is left-aligned (min shifted to 0) and placed after the previous
    by its (span + pad). This removes irregular large gaps.
    """
    cmap = plt.get_cmap('tab10')
    custom_colors = ['#984ea3', '#ff7f00', '#a65628', '#f781bf',
                     '#999999', '#66c2a5', '#fc8d62']

    ncells = len(next(iter(templates.values())))
    linsp = np.arange(1, ncells + 1)

    nshift = 0.0
    for idx, (ckey, tmpl) in enumerate(templates.items()):
        color = custom_colors[idx] if idx < len(custom_colors) else cmap.colors[np.mod(idx, 10)]
        x = (tmpl[isort]) * fac
        nshift = _tile_block(ax, x, linsp, nshift, color=color, s=20, pad=pad)

    ax.set_ylabel("Neuron")
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.15)

# ---------- example sequences ----------
def plot_examples_timebased(samples, isort, ax, fac=1.0, pad=0.05):
    """
    Plot example sequences from the current cluster (black), side by side with
    adaptive spacing: each sample starts at 0 and is placed after the previous
    by its (span + pad). This removes irregular large gaps.
    """
    if len(samples) == 0:
        ax.set_xticks([])
        return

    ncells = len(samples[0])
    linsp = np.arange(1, ncells + 1)

    nshift = 0.0
    for smp in samples:
        x = (smp[isort]) * fac
        nshift = _tile_block(ax, x, linsp, nshift, color='k', s=20, pad=pad)

    ax.set_xticks([])
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.15)

# ---------- full multi-row plot ----------
def plot_clusters(temp_info, bursts,
                  max_clusters_to_show=3,
                  allowed_idx = np.array([]),
                  method="center_of_mass",
                  max_samples_per_cluster=10,
                  fac=1.0,
                  pad_templates=8000,
                  pad_examples=0.4,
                  seed=42):
    """
    For each cluster, make one row with two subplots:
      - left: all templates (colored, adaptively tiled)
      - right: examples from that cluster (black, adaptively tiled)
    """
    rng = np.random.default_rng(seed=42)
    n_show = min(max_clusters_to_show, len(temp_info['clist']))
    source_idx = allowed_idx if allowed_idx.size > 0 else np.arange(len(temp_info['clist']))
    sampled_idx = rng.choice(source_idx, size=n_show, replace=False)
    #sampled_idx = source_idx
    
    templates = OrderedDict()
    tsamples = {}

    for nt in sampled_idx:
        clist = np.asarray(temp_info['clist'][nt], dtype=int)
        if clist.size == 0:
            n_neurons = len(bursts[0])
            templates[nt] = np.full(n_neurons, np.nan, float)
            tsamples[nt] = []
            continue

        tmpl_times, _ = average_sequence_times(bursts, clist, method=method)
        templates[nt] = tmpl_times

        cluster_bursts = [bursts[i] for i in clist[:max_samples_per_cluster]]
        tsamples[nt] = [summarize_burst_times(b, method=method) for b in cluster_bursts]

    fig, axes = plt.subplots(n_show, 2, figsize=(10, 1 * n_show),
                             sharey=True)

    if n_show == 1:
        axes = np.array([axes])  # unify shape

    for row, nt in enumerate(templates.keys()):
        tmpl_nt = templates[nt]
        # sort by THIS cluster's template (NaNs go to end via +inf)
        isort = np.argsort(np.nan_to_num(tmpl_nt, nan=np.inf))

        axL, axR = axes[row]
        plot_templates_timebased(templates, isort, axL, fac=fac, pad=pad_templates)
        plot_examples_timebased(tsamples[nt], isort, axR, fac=fac, pad=pad_examples)

        axL.set_title(f"Sorted by template {nt}", fontsize=11)
        axR.set_title(f"Example sequences (cluster {nt})", fontsize=11)
        # y-label on left only; right shares y
        axR.set_ylabel("")

    plt.tight_layout()
    plt.show()
    return fig


# =============================================================================
# Graph plots
# ===========================================================================

def plot_precedence_graph(res, with_labels=True, node_size=100):
    """
    Visualization of the directed precedence graph.

    Parameters
    ----------
    res : dict
        Output of compute_template_pairwise_precedence(...)
    with_labels : bool
        Whether to show neuron IDs.
    node_size : int
        Size of nodes in the plot.
    """
    G = nx.DiGraph()
    G.add_nodes_from(res["neurons"])
    G.add_edges_from(res["edges"])

    # layout: spring works well for small graphs
    pos = nx.spring_layout(G, seed=0)

    plt.figure(figsize=(12,12))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="lightgray")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10, width=1)
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=9)

    plt.title("Pairwise precedence graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def cluster_graph(temp_info, seq):

    seqAll=copy.copy(seq)
    for nc in range(len(temp_info['radius'])):    
        seqAll.append(temp_info['template'][nc])
        
    G, pos = graph(seqAll, temp_info)

    return G, pos
    
def graph(seqAll, temp_info=[], temp_infoD=[]):
    cmap=plt.get_cmap('Set3')
    options = {"alpha": 0.7}
    
    nclust=0
    if len(temp_info)>0:
        temp=temp_info['template']
        clist=temp_info['clist']
        radius=temp_info['radius']        

    nclust=len(radius)
    
    for nD in range(len(temp_infoD)):
        temp.extend(temp_infoD[nD]['template'])
        clist.extend(temp_infoD[nD]['clist'])
        radius.extend(temp_infoD[nD]['radius'])        


        
    lseqs=len(seqAll)-len(radius)
    mat_dict = seq_clustering.allmot(seqAll)
    bmat = mat_dict["bmat"]
    zmat = mat_dict["zmat"]
    #
    
    G=nx.Graph()
    for jm in range(bmat.shape[1]):
        for jn in range(bmat.shape[0]):
            if bmat[jn,jm]>0:
                G.add_edge(jn,jm,weight=zmat[jn,jm])

    pos = nx.spring_layout(G, seed=12647)  # positions for all nodes
    #
    allist=list(pos.keys())

    for nc in range(nclust):
        if np.sum(np.array(temp_info['exclude'])==nc):
            continue

        colrgb=np.array(cmap.colors[np.mod(nc,10)]).reshape(1,3)
        q=[]
        for ok in allist:
            if np.sum(clist[nc]==ok):
                q.append(ok)

        tmpnodelabel=lseqs+nc
        if np.sum(np.array(allist)==tmpnodelabel)>0:
            nx.draw_networkx_nodes(G, pos, nodelist=q, node_size=5*radius[nc],edgecolors=["black"],node_color=colrgb, **options)
            nx.draw_networkx_nodes(G, pos, nodelist=[tmpnodelabel], edgecolors=["red"], node_size=200,node_color=colrgb, **options)
            labels = {x: x-lseqs+1 for x in G.nodes if x >=lseqs}
            nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color='b')

    nc=nclust
    colD=["red","black"]
    for nD in range(len(temp_infoD)):
        for ncloc in range(len(temp_infoD[nD]['radius'])):
            
            if np.sum(np.array(temp_infoD[nD]['exclude'])==ncloc):
                nc=nc+1
                continue


            tmpnodelabel=lseqs+nc
            nx.draw_networkx_nodes(G, pos, nodelist=[tmpnodelabel], edgecolors=[colD[nD]], node_size=50,node_color=[colD[1-nD]], **options)
            nc=nc+1

            
        
    #nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.05)

    return G,pos


