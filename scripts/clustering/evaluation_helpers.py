# Third-party imports
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.metrics import silhouette_score

# Local imports
from .rank_correlation import rankseq, choose_nrm_param

# =============================================================================
# Cluster evaluation
# =============================================================================
    
def relabel_contiguous(labels, ignore=(-1,)):
    labels = np.asarray(labels)
    out = labels.copy()

    ignore = set(ignore)
    keep_mask = ~np.isin(out, list(ignore))

    uniq = np.unique(out[keep_mask])
    mapping = {old: new for new, old in enumerate(uniq)}

    out[keep_mask] = np.vectorize(mapping.get)(out[keep_mask])
    return out, mapping
    
def silhouette_ignore_singletons(dist_matrix, labels):
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    non_singleton_labels = unique[counts > 1]

    mask = np.isin(labels, non_singleton_labels)
    filtered_labels = labels[mask]
    filtered_dist = dist_matrix[np.ix_(mask, mask)]

    if len(np.unique(filtered_labels)) < 2:
        return np.nan  # Can't compute silhouette with <2 clusters

    return silhouette_score(filtered_dist, filtered_labels, metric='precomputed')

def compute_template(bursts, clist, method):
    """
    Construct a template for a set of bursts using the appropriate helper.

    Dispatches to:
      * get_template(clist, bursts, method) when `bursts` is a list of per-burst objects,
      * get_template_mat(bursts, clist) when `bursts` is a 2-D array (matrix of bursts).

    Parameters
    ----------
    bursts : list or array-like
        Burst data; either a list of per-burst structures or a 2-D array/matrix.
    clist : array-like
        Indices of bursts to include in the template.
    method : str
        Template construction method forwarded to `get_template` when applicable.

    Returns
    -------
    template
        Template object/array as produced by the selected helper function.

    Raises
    ------
    ValueError
        If `bursts` has an unsupported dimensionality.
    """
    if isinstance(bursts, list):
        return get_template(clist, bursts, method)
    elif np.array(bursts).ndim == 2:
        return get_template_mat(bursts, clist)
    else:
        raise ValueError("Unsupported burst data dimensionality.")
        
def evaluate_template(seqs, temp):
    """
    Score a template against sequences and produce a compact quality metric.

    Calls `check_template(seqs, temp)` and uses its second returned element
    (assumed to be a per-neuron or per-feature significance/flag array) as the
    primary quality vector. Computes a scalar "radius" as mean(significance) * 30.

    Parameters
    ----------
    seqs : list-like
        Sequences used for template scoring (passed to `check_template`).
    temp : array-like
        Template to evaluate.
    Returns
    -------
    sig_flags : ndarray
        The significance/flag array returned by `check_template(...)[1]`.
    radius : float
        Scalar quality metric computed as `np.mean(sig_flags) * 30`.
    """
    # Check the template against all sequences.
    zval, sig, sig2 = check_template(seqs, temp)
    zval = np.asarray(zval, dtype=np.float64)
    sig2 = np.asarray(sig2, dtype=np.float64)

    ##############################
    ## Turn zval into distance between 0 and 1
    #zvalmax = np.nanmax(zval)
    #zval_th = zval.copy()
    ## Replace NaN values with -inf to map non overlapping sequences to high distances.
    #zval_th = np.nan_to_num(zval_th, nan=-np.inf)
    ## Replace values above the corresponding threshold with zmax for low distances
    #zval_th[sig==1] = zvalmax
    #zval_th = invert_sigmoid_expit(zval_th, k=1.0, x0=0.5)
    zval_th = zval.copy()
    zval_th = np.nan_to_num(zval_th, nan=-10.0)   # non-overlap => very negative
    zval_th = norm.cdf(zval_th)
    zval_th = 1.0 - zval_th
    ##############################
    
    # Define the template's "radius" as the mean significance flag scaled by 30.
    radius = np.mean(sig) * 30
    adj = zval_th
    return adj, radius
    
def get_template(clist, bursts, method):
    """
    Compute a template sequence for a cluster of bursts.

    This function determines the activation order of neurons across a 
    set of bursts by computing the center-of-mass (mean spike time) for each 
    neuron in each burst, and averaging these across all bursts in the cluster.
    
    Parameters
    ----------
    clist : list of int
        Indices of bursts that belong to the cluster of interest.
    
    bursts : list of list of np.ndarray
        A list of bursts. Each burst is itself a list of arrays, one per neuron.
        Each array contains the spike times of that neuron during the burst.
        Neurons without spikes are represented by empty arrays.

    method : str
        Method for computing sequence. Options:
        - 'center_of_mass': Use mean spike time per neuron.
        - 'first_spike': Use first spike time per neuron.
        Default is 'center_of_mass'.

    Returns
    -------
    temp : np.ndarray
        A 1D array of neuron indices sorted by their average activation time 
        across the cluster. Neurons that never fired in any of the selected 
        bursts are excluded.
    """
    n_neurons = len(bursts[0])  # number of neurons (same for each burst)
    # Initialize list to hold center-of-mass per neuron across bursts
    time_values = [[] for _ in range(n_neurons)]  # one list per neuron

    for i in clist:
        burst = bursts[i]  # burst is a list of spike time arrays (one per neuron)
        for j, spikes in enumerate(burst):
            if len(spikes) > 0:
                val = np.mean(spikes) if method == 'center_of_mass' else \
                      np.min(spikes)  if method == 'first_spike' else None
                if val is None:
                    raise ValueError(f"Unknown method: {method}")
                time_values[j].append(val)
    
    if method == 'center_of_mass':
        # Compute average center-of-mass per neuron across bursts
        mns = np.array([np.nanmean(times) if times else np.nan for times in time_values])
    elif method == 'first_spike':
        # Copute median per neuron across bursts
        mns = np.array([np.nanmedian(times) if times else np.nan for times in time_values])
    
    # Sort neuron indices by average activation time
    tmp = np.argsort(mns)
    temp = tmp[~np.isnan(mns[tmp])]  # remove neurons with NaN (never active in cluster)

    return temp

def build_precedence_graph(
    seqs,
    clist,
    min_cooccur_frac=0.1,
    edge_thresh=0.7,
):
    """
    Build pairwise precedence graph from a cluster of sequences.

    Returns
    -------
    out : dict
        neurons, P, N, wins, wins_norm, edges
    """
    min_cooccur_abs = 3 # absolute minimum co-occurrence threshold (ignored if min_cooccur_frac is higher)
    clist = np.asarray(clist, dtype=int)
    cluster_size = len(clist)
    cluster_seqs = [np.asarray(seqs[i], dtype=int) for i in clist]

    neurons = np.unique(np.concatenate(cluster_seqs)) if cluster_seqs else np.array([], dtype=int)
    n = len(neurons)
    id2ix = {nid: k for k, nid in enumerate(neurons)}

    before = np.zeros((n, n), dtype=np.int32)
    co = np.zeros((n, n), dtype=np.int32)

    for s in cluster_seqs:
        pos = {}
        for p, nid in enumerate(s):
            if nid not in pos:
                pos[nid] = p
        items = list(pos.items())

        for i in range(len(items)):
            ni, pi = items[i]
            ix_i = id2ix[ni]

            for j in range(i + 1, len(items)):
                nj, pj = items[j]
                ix_j = id2ix[nj]

                co[ix_i, ix_j] += 1
                co[ix_j, ix_i] += 1

                if pi < pj:
                    before[ix_i, ix_j] += 1
                elif pj < pi:
                    before[ix_j, ix_i] += 1

    P = np.full((n, n), np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = co > 0
        P[mask] = before[mask] / co[mask]
    np.fill_diagonal(P, np.nan)

    wins = np.nan_to_num(P, nan=0.0).sum(axis=1)
    denom = np.sum(~np.isnan(P), axis=1)
    wins_norm = np.where(denom > 0, wins / denom, np.nan)

    min_co = max(min_cooccur_abs, int(np.ceil(min_cooccur_frac * cluster_size)))

    edges = []
    if n > 0:
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if co[a, b] >= min_co and P[a, b] >= edge_thresh:
                    edges.append((neurons[a], neurons[b]))

    return {
        "neurons": neurons,
        "P": P,
        "N": co,
        "wins": wins,
        "wins_norm": wins_norm,
        "edges": edges,
    }


def kahn_template_from_graph(neurons, edges, wins_norm=None):
    """
    Best-effort linearization of a precedence graph using Kahn's algorithm.

    Parameters
    ----------
    neurons : 1D array-like
        Node IDs.
    edges : list of tuple(int, int)
        Directed edges u -> v.
    wins_norm : 1D array-like or None
        Optional node scores for tie-breaking among zero-indegree nodes.
        Must be aligned with `neurons`.

    Returns
    -------
    template : np.ndarray
        Best-effort total order.
    components : list[list[int]]
        Weakly connected components, sorted by template order.
    """
    neurons = np.asarray(neurons, dtype=int)
    nodes = list(neurons.tolist())
    node_set = set(nodes)

    if wins_norm is None:
        score = {u: 0.0 for u in nodes}
    else:
        wins_norm = np.asarray(wins_norm, dtype=float)
        score = {
            neurons[i]: (wins_norm[i] if np.isfinite(wins_norm[i]) else -np.inf)
            for i in range(len(neurons))
        }

    adj = {u: set() for u in nodes}
    indeg = {u: 0 for u in nodes}
    for u, v in edges:
        if u in node_set and v in node_set and v not in adj[u]:
            adj[u].add(v)
            indeg[v] += 1

    und = {u: set() for u in nodes}
    for u, v in edges:
        if u in node_set and v in node_set:
            und[u].add(v)
            und[v].add(u)

    seen = set()
    components = []
    for u in nodes:
        if u in seen:
            continue
        stack = [u]
        comp = []
        seen.add(u)

        while stack:
            x = stack.pop()
            comp.append(x)
            for y in und[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)

        components.append(comp)

    zero = [u for u in nodes if indeg[u] == 0]
    zero.sort(key=lambda u: score[u], reverse=True)

    order = []
    indeg2 = indeg.copy()
    adj2 = {u: set(vs) for u, vs in adj.items()}

    while zero:
        u = zero.pop(0)
        order.append(u)

        for v in list(adj2[u]):
            indeg2[v] -= 1
            adj2[u].remove(v)
            if indeg2[v] == 0:
                zero.append(v)

        zero.sort(key=lambda u: score[u], reverse=True)

    remaining = [u for u in nodes if u not in set(order)]
    remaining.sort(key=lambda u: score[u], reverse=True)

    template = np.array(order + remaining, dtype=int)
    components = sort_components(template, components)

    return template, components


def compute_template_pairwise_precedence(
    seqs,
    clist,
    min_cooccur_frac=0.1, # relative minimum of cooccurrence as a fraction of cluster size
    edge_thresh=0.7,      # require p(i before j) >= edge_thresh to add edge i->j
    return_template=True, # if False, skip producing a total-order fallback
):
    """
    Build a template from sequences via pairwise precedence.
    Which neurons reliably activate before which other neurons—whenever they co-occur?
    handles missing/non-overlapping neurons
	handles variable burst length

    Parameters
    ----------
    seqs : list of 1D array-like
        Each entry is a burst as an ordered list of neuron indices (unique within burst).
    clist : 1D array-like (int)
        Indices into `seqs` that belong to the cluster.
    min_cooccur_frac : float
        Relative minimum co-occurrence threshold as a fraction of clutser size.
    edge_thresh : float in (0.5, 1]
        Threshold on p(i before j) to assert a directed edge i -> j.
    return_template : bool
        If True, also return a best-effort total order ("template") via topological sorting
        on the confident edges; disconnected/ambiguous nodes are appended by score.

    Returns
    -------
    out : dict with keys
        neurons : np.ndarray
            All neurons that appear in the cluster bursts.
        P : np.ndarray (float, shape [n,n])
            Pairwise precedence probabilities p(i before j); NaN if never co-occur.
        N : np.ndarray (int, shape [n,n])
            Co-occurrence counts (number of bursts where i and j both appear), symmetric.
        wins : np.ndarray (float, shape [n,])
            Score s_i = sum_j p(i before j) over defined pairs (NaNs ignored).
        edges : list of tuple(int,int)
            Directed edges (u,v) in neuron-ID space with sufficient evidence.
        template : np.ndarray (optional)
            Best-effort total order of neuron IDs (only if return_template=True).
        components : list[list[int]] (optional)
            Weakly-connected components in the confident-edge graph (IDs), if return_template=True.
    """
    out = build_precedence_graph(
        seqs=seqs,
        clist=clist,
        min_cooccur_frac=min_cooccur_frac,
        edge_thresh=edge_thresh,
    )

    if not return_template:
        return out

    template, components = kahn_template_from_graph(
        neurons=out["neurons"],
        edges=out["edges"],
        wins_norm=out["wins_norm"],
    )

    out["template"] = template
    out["components"] = components
    return out["template"], out["components"]

    
def sort_components(template, components):
    pos = {n: i for i, n in enumerate(template)}

    components_sorted = sorted(
        components,
        key=lambda comp: min(pos[n] for n in comp)
    )

    components_sorted = [
        sorted(comp, key=lambda n: pos[n])
        for comp in components_sorted
    ]
    return components_sorted
    
def get_template_mat(bursts, clist, method):
    """
    Compute a template sequence for a cluster of bursts.

    ...
    
    Parameters
    ----------
    clist : list of int
        Indices of bursts that belong to the cluster of interest.
    
    bursts : np.array
        A 2D array of burst sequences. If 2D, each row represents a burst; if 3D, bursts 
        may include additional dimensions (e.g., time bins x neurons).

    Returns
    -------
    temp : np.ndarray
        A 1D array of neuron indices sorted by their average activation time 
        across the cluster. Neurons that never fired in any of the selected 
        bursts are excluded.
    """
    mns = np.nanmean(np.array(bursts)[clist, :], axis=0)
    tmp = np.argsort(mns)
    # Remove any indices corresponding to NaN values.
    temp = tmp[~np.isnan(np.sort(mns))]
    return temp

def check_template(seqs, temp):
    """
    Evaluate a template against a set of sequences using rank correlation and compute z-scores.

    For each sequence in `seqs`, this function calculates the Spearman rank order correlation 
    with the template `temp` (using the previously defined `rankseq` function). Based on the 
    overlap length between the template and each sequence, appropriate normalization parameters 
        are selected from the cached `nrm` table. A z-score is then computed, and a binary
        significance flag is set if the z-score exceeds a threshold.

    Parameters
    ----------
    seqs : list or array_like
        A list of sequences (e.g., neuron spike orders) to compare with the template.
    temp : array_like
        The template sequence against which each sequence is evaluated.
    Returns
    -------
    zval : numpy.ndarray
        Array of z-scores for each sequence based on the rank correlation with the template.
    sig : numpy.ndarray
        Binary array (0 or 1) indicating whether each sequence's z-score exceeds the threshold.
    sig2 : numpy.ndarray
        Binary array (0 or 1) indicating whether the absolute value of each sequence's z-score 
        exceeds the threshold.
    
    """
    nseqs = len(seqs)
    # Flatten the template sequence for comparison.
    s1 = np.array(temp).flatten()
    
    sig = np.zeros(nseqs)
    sig2 = np.zeros(nseqs)
    zval = np.zeros(nseqs)
        
    for ns in range(nseqs):
        s2 = seqs[ns]
        rc, ln = rankseq(s1, s2)  # Compute rank correlation and overlap length
        
        # Choose normalization parameters based on the overlap length.
        mns = choose_nrm_param(ln)

        # Calculate the z-score using the reference mean and standard deviation.
        ztmp = (rc - mns[1]) / mns[2]
        # Set the significance flag if the z-score exceeds the threshold.
        sig[ns] = 1.0 * (ztmp > mns[3])
        sig2[ns] = 1.0 * (abs(ztmp) > mns[3])
        zval[ns] = ztmp

    return zval, sig, sig2

def within_across(adj, ids_clust):
    """
    Compute within-cluster and across-cluster mean adjustments.

    For each cluster, this function calculates the mean adjustment (e.g., significance flag) 
    for sequences within the cluster (within-cluster) and for those outside the cluster 
    (across-cluster).

    Parameters
    ----------
    adj : list of array_like
        A list where each element corresponds to a cluster and contains adjustment values.
    ids_clust : array_like
        An array of cluster IDs assigning each sequence (or burst) to a cluster.

    Returns
    -------
    ret : dict
        Dictionary with keys:
            - 'within': List of mean adjustment values computed for sequences within each cluster.
            - 'across': List of mean adjustment values computed for sequences outside each cluster.
    """
    ret = {'within': [], 'across': [], 'ratio': [], 'auc': [] }
    ids_clust = np.array(ids_clust)
    # Loop over each cluster's adjustment values.
    for nc in range(len(adj)):
        # Get indices of sequences inside and outside the current cluster.
        idin = np.where(ids_clust == nc)[0]
        idout = np.where(~(ids_clust == nc))[0]
        # Compute the mean adjustments.
        within = np.nanmean(adj[nc][idin])
        across = np.nanmean(adj[nc][idout])

        ratio = across / within # within / across
        eps = 1e-6  # small stabilizer
        # assuming adj is based on distances between 0 and 1
        log_ratio = np.log((np.mean(across) + eps) / (np.mean(within) + eps))

        y_true = np.concatenate([np.ones(len(adj[nc][idout])), np.zeros(len(adj[nc][idin]))])
        scores = np.concatenate([adj[nc][idout], adj[nc][idin]])
        #auc_score = roc_auc_score(y_true, scores)

        ret['within'].append(within)
        ret['across'].append(across)
        ret['ratio'].append(log_ratio)
        #ret['auc'].append(auc_score)
        
    # to arrays
    ret['within'] = np.array(ret['within'])
    ret['across'] = np.array(ret['across'])
    ret['ratio'] = np.array(ret['ratio'])
    #ret['auc'] = np.array(ret['auc'])

    return ret

def _k_permutations_sim_chunk(labels, spk_times, seqs, seeds):
    """
    Run len(seeds) permutations in one worker and aggregate results:
    returns dict: size -> list of abs(ratio) values
    """
    # Lazy import to avoid circular dependency
    from . import core as sc
    
    labels = np.asarray(labels)
    out = defaultdict(list)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = labels.copy()
        rng.shuffle(perm)

        result_dict_sh = sc.info_cluster(spk_times, seqs, perm, method='center_of_mass')
        sc.add_within_across_score(result_dict_sh, spk_times, seqs, permutation=False)
        pratios = np.asarray(result_dict_sh['clust_scores']['ratio'], dtype=float)
        pratios = np.abs(pratios)

        u_perm, counts_perm = np.unique(perm, return_counts=True)
        if pratios.shape[0] != u_perm.shape[0]:
            raise ValueError(
                f"info_cluster_sim returned {pratios.shape[0]} ratios, "
                f"but perm has {u_perm.shape[0]} unique labels."
            )

        # Append by size
        for cnt, val in zip(counts_perm, pratios):
            if np.isfinite(val):
                out[int(cnt)].append(float(val))

    return out

def within_across_permutation(
    ratios,
    labels,
    spk_times,
    seqs,
    reduction="mean",
    n_permute=200,
    random_state=None,
    n_jobs=-1,
    backend="loky",           # try "threading" too—benchmark!
    verbose=0,
    chunk_size=16,            # do 16 perms per worker call; tune 16–64
    max_nbytes="100M",        # memmap big numpy args to workers
    mmap_mode="r",
    inner_max_num_threads=1,  # avoid oversubscription
):



    # Inputs
    ratios = np.abs(np.asarray(ratios, dtype=float))
    labels = np.asarray(labels)
    eps_sd = 1e-12

    # Original cluster order & sizes
    u_orig, counts_orig = np.unique(labels, return_counts=True)
    size_by_label_orig = dict(zip(u_orig, counts_orig))

    # Seeds: one per permutation, then group into chunks
    ss = np.random.SeedSequence(random_state)
    all_seeds = ss.generate_state(n_permute).tolist()

    # Make chunks of seeds
    if chunk_size < 1:
        chunk_size = 1
    seed_chunks = [all_seeds[i:i+chunk_size] for i in range(0, n_permute, chunk_size)]

    # Run chunks in parallel
    # Note: max_nbytes & mmap_mode tell joblib to memmap large numpy arrays
    per_chunk_dicts = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        max_nbytes=max_nbytes,
        mmap_mode=mmap_mode,
        inner_max_num_threads=inner_max_num_threads,
        prefer="processes" if backend == "loky" else None
    )(
        delayed(_k_permutations_sim_chunk)(labels, spk_times, seqs, seeds)
        for seeds in seed_chunks
    )

    # Merge dicts
    null_by_size = defaultdict(list)
    for d in per_chunk_dicts:
        for sz, vals in d.items():
            null_by_size[sz].extend(vals)

    # Size-matched null stats
    null_mean = {sz: (float(np.mean(v)) if len(v) else np.nan)
                 for sz, v in null_by_size.items()}
    null_std  = {sz: (float(np.std(v, ddof=1)) if len(v) > 1 else np.nan)
                 for sz, v in null_by_size.items()}

    # Z and empirical p (greater-or-equal) in the order of u_orig
    zscore = np.full_like(ratios, np.nan, dtype=float)
    pval   = np.full_like(ratios, np.nan, dtype=float)

    for i, lab in enumerate(u_orig):
        r  = ratios[i]
        sz = size_by_label_orig[lab]
        mu = null_mean.get(sz, np.nan)
        sd = null_std.get(sz, np.nan)

        if np.isfinite(r) and np.isfinite(mu) and np.isfinite(sd):
            zscore[i] = (r - mu) / max(sd, eps_sd)

        null_vals = np.asarray(null_by_size.get(sz, []), dtype=float)
        null_vals = null_vals[np.isfinite(null_vals)]
        if null_vals.size > 0 and np.isfinite(r):
            more_extreme = np.sum(null_vals >= r)
            pval[i] = float((1 + more_extreme) / (1 + null_vals.size))

    return zscore, pval, null_mean, null_std

def get_mean_cluster_score(
    mat_dict,
    result_dict,
    ids_clust_merged=None,
    mat="zmat",
    ignore_labels=(-1,),
    return_labels=False,
):
    zmat_0 = mat_dict[mat].copy()
    # only consider significant zscores
    zmat_0[mat_dict["bmat"] == 0] = 0  # optional bmat_pm
    zmat_0 = np.nan_to_num(zmat_0, nan=0)

    if ids_clust_merged is None:
        ids_clust_merged = result_dict["ids_clust"]

    # derive clusters from the current ids_clust_merged
    unique_clusters = np.unique(ids_clust_merged)

    if ignore_labels is not None:
        ignore = set(ignore_labels)
        unique_clusters = np.array([c for c in unique_clusters if c not in ignore], dtype=unique_clusters.dtype)

    n_clust = unique_clusters.size
    block_means = np.full((n_clust, n_clust), np.nan, dtype=float)

    for a, cla in enumerate(unique_clusters):
        idx_a = np.where(ids_clust_merged == cla)[0]
        if idx_a.size == 0:
            continue

        for b, clb in enumerate(unique_clusters):
            idx_b = np.where(ids_clust_merged == clb)[0]
            if idx_b.size == 0:
                continue

            submat = zmat_0[np.ix_(idx_a, idx_b)]

            if a == b:
                # within-cluster: upper triangle, exclude diagonal
                if submat.shape[0] < 2:
                    continue
                iu = np.triu_indices_from(submat, k=1)
                vals = submat[iu]
            else:
                # between-cluster: all pairwise entries
                vals = submat.ravel()

            if vals.size > 0:
                block_means[a, b] = np.nanmean(vals)

    return (block_means, unique_clusters) if return_labels else block_means

