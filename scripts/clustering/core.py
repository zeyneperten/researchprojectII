# Third-party imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from scipy.cluster import hierarchy as sch
from scipy.stats import binom
from scipy.spatial.distance import squareform

# Local imports
from .. import data_utils 
from . import rank_correlation as sc_helpers
from . import evaluation_helpers as eval_helpers
from . import leiden as l

BINOM_P_VALUE = 0.05
# =============================================================================
# Main functions
# =============================================================================

def allmot(seqs, n_jobs=-1, block=256):
    nrm = data_utils.load_nrm()
    n = len(seqs)

    corr  = np.zeros((n, n), dtype=np.float64)
    zmat  = np.zeros((n, n), dtype=np.float64)
    bmat  = np.zeros((n, n), dtype=np.float64)
    bmat2 = np.zeros((n, n), dtype=np.float64)

    # set diagonals
    np.fill_diagonal(zmat, np.nan)
    np.fill_diagonal(bmat, np.nan)
    np.fill_diagonal(bmat2, np.nan) 

    results = Parallel(
        n_jobs=n_jobs, backend="loky", batch_size=1,
        pre_dispatch="all", prefer="processes"
    )(
        delayed(sc_helpers._compute_block_ret)(i0, i1, j0, j1, seqs, nrm)
        for (i0, i1, j0, j1) in sc_helpers._block_pairs_upper_full(n, bs=block)
    )

    for i0, i1, j0, j1, cblk, zblk, bblk, bblk2 in results:
        if i0 == j0:
            # diagonal block: write block, then reflect only strict upper ---
            corr[i0:i1, i0:i1] = cblk
            zmat[i0:i1, i0:i1] = zblk
            bmat[i0:i1, i0:i1] = bblk
            bmat2[i0:i1, i0:i1] = bblk2

            Bi = i1 - i0
            iu = np.triu_indices(Bi, k=1)  # strict upper indices within block
            # reflect to lower within the same square:
            corr[i0 + iu[1], i0 + iu[0]] = cblk[iu]
            zmat[i0 + iu[1], i0 + iu[0]] = zblk[iu]
            bmat[i0 + iu[1], i0 + iu[0]] = bblk[iu]
            bmat2[i0 + iu[1], i0 + iu[0]] = bblk2[iu]
        else:
            # off-diagonal block: full rectangle + symmetric mirror ---
            corr[i0:i1, j0:j1] = cblk
            zmat[i0:i1, j0:j1] = zblk
            bmat[i0:i1, j0:j1] = bblk
            bmat2[i0:i1, j0:j1] = bblk2

            corr[j0:j1, i0:i1] = cblk.T
            zmat[j0:j1, i0:i1] = zblk.T
            bmat[j0:j1, i0:i1] = bblk.T
            bmat2[j0:j1, i0:i1] = bblk2.T

    # restore diagonal NaNs
    np.fill_diagonal(zmat, np.nan)
    np.fill_diagonal(bmat, np.nan)
    np.fill_diagonal(bmat2, np.nan) 

    # row-wise counts & stats
    nsig = np.nansum(bmat, axis=1)
    pval = 1.0 - binom.cdf(nsig, n - 1, BINOM_P_VALUE)
    std_n = np.std(nsig)
    rep_index = nsig / std_n if std_n > 0 else np.zeros_like(nsig, dtype=float)

    mat_dict = {"repid": rep_index, "nsig": nsig, "pval": pval, "bmat": bmat, "zmat": zmat, "corrmat": corr,"bmat_pm": bmat2}

    return mat_dict
    
def seq_cluster(dist_mat, k=None, fac=None, method="ward"):
    """
    Cluster detected sequences based on a dissimilarity matrix.

    This function applies either Agglomerative Hierarchical Clustering.

      - A pairwise distance matrix is computed from the filtered clustering matrix.
      - As default Ward's linkage method is used to perform hierarchical clustering.
      - The clustering threshold is set as a fraction (`fac`) of the maximum in the linkage matrix.
      - Clusters are formed based on this threshold or number of clusters k.

    Parameters
    ----------
    dist_mat : array_like
        A pairwise distance matrix (condensed distance matrix as output from 
        scipy.spatial.distance.pdist or .squareform)
    k: int number of clusters
    fac: float between 0 and 1, cuts dendogram.

    Returns
    -------
    ids_clust : ndarray
        An array of cluster labels (starting at 0) assigned to each sequence.
    """

    # Perform hierarchical clustering using the specified method (e.g., 'ward', 'average', 'complete', 'weighted').
    linkage_matrix = sch.linkage(dist_mat, method=method) 

    if (fac is not None) and (k is not None):
        print("Set either 'fac' or 'k' to None to choose one method.")
        return None

    if (fac is None) and (k is None):
        print("Set either 'fac' or 'k' to a valid value to perform clustering.")
        return None

    # Apply clustering based on the selected criterion (distance threshold or number of clusters)
    if fac:
        # Determine a clustering threshold: a fraction of the maximum linkage.
        max_d = np.max(linkage_matrix[:, 2]) # Contains the linkage heights (merge distances)
        threshold = max_d * fac
        
        # Form clusters using the distance criterion.
        ids_clust = sch.fcluster(linkage_matrix, threshold, criterion='distance') - 1 # Subtract 1 so that cluster IDs start at 0.
    elif k:
        # Form clusters using a fixed number of clusters
        ids_clust = sch.fcluster(linkage_matrix, k, criterion='maxclust') - 1 # Subtract 1 so that cluster IDs start at 0.

    return ids_clust


def seq_cluster_leiden(mat_dict, excess_dict, excess, res, w_keep, min_comp_size=0, weighted=False, verbose=True):
    W = l.build_graph_from_edges(mat_dict['corrmat'].shape[0], excess_dict["eval_edges"][0], excess_dict["eval_edges"][1], excess, w_keep=w_keep)
    
    ids_clust = l.cluster_graph_components_cpm(
        W,
        min_comp_size=min_comp_size,
        resolution=res,
        n_reps=10,
        seed=0,
        weighted=weighted,
        verbose=verbose
    )
    return ids_clust

# =============================================================================
# Evaluation and scores
# =============================================================================

def info_cluster(bursts, seqs, ids_clust, method):
    """
    Build one template sequence per cluster and evaluate template quality.

    For each unique cluster id in `ids_clust` this function computes a
    representative template (via `compute_template`), scores it against the
    provided sequences using `evaluate_template`, and aggregates results.

    Parameters
    ----------
    bursts : array-like or list
        Burst data (per-burst structures expected by `compute_template`).
    seqs : list
        Sequences corresponding to bursts (used for evaluation).
    ids_clust : 1-D array-like of int
        Cluster id for each burst/sequence.
    method : str
        Template construction method (e.g. 'center_of_mass' or 'first_spike').

    Returns
    -------
    dict
        Keys include:
          - 'template': list of template sequences (one per cluster)
          - 'clist': list of arrays with burst indices for each cluster
          - 'radius': list of radius/quality values per template
          - 'adj': list of adjustment/score values used for further analysis
          - 'clust_scores': criterion combining within/across measures
          - 'exclude', 'seqs', 'ids_clust', 'bursts', 'seq_method' (metadata)
    """
    # Initialize the return dictionary.
    retval = {
        'adj': [], 'template': [], 'clist': [], 'radius': [],
        'seqs': seqs, 'ids_clust': np.array(ids_clust), 'bursts': bursts,
        'clust_scores': {}, 'exclude': [], 'seq_method': method, 'temp_components': []
    }

    components = None
    
    # Find the indices of bursts belonging to the current cluster.
    cluster_indices = [np.where(ids_clust == nc)[0] for nc in np.unique(ids_clust)]
    
    # Process each cluster based on its identifier.
    for clist in cluster_indices:
        # Generate template for current cluster.
        #temp = eval_helpers.compute_template(bursts, clist, method)
        temp, components = eval_helpers.compute_template_pairwise_precedence(seqs,clist)
        # Check template quality and compute radius.
        adj, radius = eval_helpers.evaluate_template(seqs, temp)
        
        # Store the computed template and related metrics.
        retval['temp_components'].append(components)
        retval['template'].append(temp)
        retval['clist'].append(clist)
        retval['radius'].append(radius)
        retval['adj'].append(adj)
 
    return retval

def add_within_across_score(retval, bursts, seqs, permutation=True):
    """
    Evaluate the quality of each cluster template using within- and across-cluster measures.

    This function computes the within- and across-cluster scores for each cluster template
    stored in `retval`. It uses the `within_clust` function to compute the mean within-cluster
    score and the `within_across` function to compute the ratio of within- to across-cluster scores.
    
    Parameters
    ----------
    retval : dict
        Result dictionary produced by `info_cluster()` containing at least 'adj', 'ids_clust', 'bursts', and 'seqs'.
    bursts : array-like
        Burst data corresponding to the sequences.
    seqs : list
        Sequences used for template scoring.
    permutation : bool, optional
        If True, performs a permutation test on the ratio scores (default is True).
    """
    # Evaluate the quality of each cluster template using within- and across-cluster measures.
    ids_clust = retval['ids_clust']
    crit = eval_helpers.within_across(retval['adj'], ids_clust)

    if permutation:
        #print('running ratio permutation test')
        zscore, pval, _, _ = eval_helpers.within_across_permutation(
        crit['ratio'], ids_clust, bursts, seqs, reduction='mean', n_permute=200, random_state=9)
        retval['clust_scores']['zscore'] = np.array(zscore)
        retval['clust_scores']['pval'] = np.array(pval)
    
    retval['clust_scores']['within'] = crit['within']
    retval['clust_scores']['across'] = crit['across']
    retval['clust_scores']['ratio'] = crit['ratio']
    #retval['clust_scores']['auc'] = crit['auc']
    

def add_within_clust_score(retval, mat_dict):
    """
    Evaluate clusters by computing mean within-cluster score.

    For each cluster, this function extracts the submatrix of the z-score 
    matrix corresponding to all members of that cluster and computes the mean 
    score (excluding the diagonal). The resulting values are added to the 
    `retval` dictionary.

    Parameters
    ----------
    retval : dict
        Result dictionary produced by `info_cluster()` containing at least 'ids_clust'.
    mat_dict : dict
        Dictionary containing the z-score and binary matrix.
    
    Returns
    -------
    mean_corr : list
        mean within-cluster z-score for each cluster.
    """
    
    ids_clust = retval['ids_clust']
    within_score = []
    within_var = []
    
    zmat_0 = mat_dict['zmat'].copy()
    zmat_0 = np.nan_to_num(zmat_0, nan=0)
    # 0 where bmat_pm is 0, i.e. scores are not significant
    zmat_0[mat_dict['bmat'] == 0] = 0 #bmat_pm
    
    for cl in np.unique(ids_clust):
        # indices of sequences belonging to this cluster
        idx = np.where(ids_clust == cl)[0]
        if len(idx) < 2:
            # skip singletons (undefined mean z-score)
            within_score.append(np.nan)
            within_var.append(np.nan)
            continue
        
        # extract submatrix for this cluster
        submat = zmat_0[np.ix_(idx, idx)]
        # upper triangle indices (exclude diagonal)
        iu = np.triu_indices_from(submat, k=1)
        # compute mean z-score
        mean_val = np.nanmean(submat[iu])
        variance_val = np.nanvar(submat[iu])
        
        within_score.append(mean_val)
        within_var.append(variance_val)

    retval['clust_scores']['within_clust'] = np.array(within_score)
    retval['clust_scores']['within_clust_var'] = np.array(within_var)


# =============================================================================
# Merging
# =============================================================================

def merge_clusters(mat_dict, result_dict, data, thr, verbose=True):
    merged_clusters = []

    ids_clust_merged = result_dict["ids_clust"].copy()

    # initial block means + label order (excluding -1)
    block_means_merged, labels = eval_helpers.get_mean_cluster_score(
        mat_dict, result_dict, ids_clust_merged=ids_clust_merged,
        return_labels=True, ignore_labels=(-1,)
    )

    while True:
        iu = np.triu_indices_from(block_means_merged, k=1)
        vals = block_means_merged[iu]
        mask = vals > thr
        if not np.any(mask):
            break

        pairs_idx = np.column_stack((iu[0][mask], iu[1][mask]))
        pairs_idx_remapped = [(int(labels[i]), int(labels[j])) for (i, j) in pairs_idx]
        if verbose:
            print(f"All pairs idx {[(int(a), int(b)) for a, b in pairs_idx_remapped]}")

        # build graph in index space
        G = nx.Graph()
        G.add_edges_from(map(tuple, pairs_idx))
        chosen_pairs_idx = []

        for comp in nx.connected_components(G):
            edges = list(G.subgraph(comp).edges())
            if not edges:
                continue
            w = np.array([block_means_merged[i, j] for (i, j) in edges], dtype=float)
            best_edge = edges[int(np.nanargmax(w))]
            chosen_pairs_idx.append(tuple(map(int, best_edge)))

        # map chosen index-pairs back to *cluster labels*
        chosen_pairs = [(int(labels[i]), int(labels[j])) for (i, j) in chosen_pairs_idx]
        if verbose:
            print(f"Chosen pairs labels {chosen_pairs}\n")

        if len(chosen_pairs) == 0:
            break

        prev = ids_clust_merged.copy()

        # merge using label space (and never touch -1)
        for a, b in chosen_pairs:
            if a == -1 or b == -1:
                continue

            merged_clusters.append((a, b))
            new_label = min(a, b)

            mask_merge = np.isin(ids_clust_merged, [a, b])
            ids_clust_merged[mask_merge] = new_label

        # safety: if nothing changed, break to avoid infinite loop
        if np.array_equal(prev, ids_clust_merged):
            print("No label changes in this iteration — breaking to avoid infinite loop.")
            break

        # recompute block means + labels from the updated ids (still excluding -1)
        block_means_merged, labels = eval_helpers.get_mean_cluster_score(
            mat_dict, result_dict, ids_clust_merged=ids_clust_merged,
            return_labels=True, ignore_labels=(-1,)
        )

    if verbose:
        print("recompute result dict")
    ids_clust_relab, mapping = eval_helpers.relabel_contiguous(ids_clust_merged, ignore=(-1,))
    result_dict_merged = info_cluster(data["bursts"], data["seqs"], ids_clust_relab, data["seq_method"])
    add_within_clust_score(result_dict_merged, mat_dict)
    add_within_across_score(result_dict_merged, data["bursts"], data["seqs"], permutation=False)
    result_dict_merged["merged_clusters_origidx"] = merged_clusters

    # should be True: each matrix row corresponds to one label
    assert len(labels) == block_means_merged.shape[0]
    # should be True: all labels appear in ids_clust_merged (except ignored)
    assert set(labels).issubset(set(np.unique(ids_clust_merged)))

    return result_dict_merged, ids_clust_merged, block_means_merged, merged_clusters


