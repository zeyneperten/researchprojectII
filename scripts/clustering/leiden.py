# Third-party imports
import igraph as ig
import leidenalg as la
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# Local imports
from . import rank_correlation as sc_helpers
from . import core as sc

# ----------------------------------
# Clustering Laiden
# ----------------------------------
def cluster_graph_components_cpm(
    W,
    min_comp_size,
    resolution,
    n_reps=10,
    seed=0,
    weighted=False,
    verbose=False
):
    n = W.shape[0]

    comp, sizes = connected_components_stats(W, verbose)

    labels = np.full(n, -1, dtype=int)
    next_label = 0

    # group nodes by component
    order = np.argsort(comp)
    splits = np.flatnonzero(np.diff(comp[order])) + 1
    groups = np.split(order, splits)

    weight = None # None if all edges should be treated equally (unweighted graph)
    if weighted:
        weight = "weight" # uses weights in g
    for nodes in groups:
        
        if nodes.size < min_comp_size:
            continue

        Wsub = W[nodes][:, nodes].tocsr()
        coo = sp.triu(Wsub, k=1).tocoo()
        edges = list(zip(coo.row.tolist(), coo.col.tolist()))
        if len(edges) == 0:
            continue

        g = ig.Graph(n=nodes.size, edges=edges, directed=False)
        g.es["weight"] = coo.data.tolist()

        best_part, best_q = None, -np.inf
        for r in range(max(1, n_reps)):
            part = la.find_partition(
                g,
                la.CPMVertexPartition,
                weights="weight",
                resolution_parameter=float(resolution),
                seed=int(seed + r),
            )
            q = part.quality()
            if q > best_q:
                best_q, best_part = q, part

        memb = np.asarray(best_part.membership, dtype=int)

        for c in np.unique(memb):
            idx = np.where(memb == c)[0]
            labels[nodes[idx]] = next_label
            next_label += 1

    return labels

"""def cluster_graph_components_cpm(
    W,
    min_comp_size=10,
    resolution=0.03,
    n_reps=10,
    seed=0,
    weighted=False,
):

    W = W.tocsr()
    W.sum_duplicates()
    W.eliminate_zeros()

    n = W.shape[0]
    comp, sizes = connected_components_stats(W)  # your function

    labels = np.full(n, -1, dtype=int)
    next_label = 0

    # group nodes by component id
    order = np.argsort(comp)
    splits = np.flatnonzero(np.diff(comp[order])) + 1
    groups = np.split(order, splits)

    weights_arg = "weight" if weighted else None

    for nodes in groups:
        if nodes.size < int(min_comp_size):
            continue

        Wsub = W[nodes][:, nodes].tocsr()
        Wsub.sum_duplicates()
        Wsub.eliminate_zeros()

        # Upper triangle edges only (undirected graph)
        coo = sp.triu(Wsub, k=1).tocoo()
        if coo.nnz == 0:
            continue
        coo.sum_duplicates()

        # Canonicalize edge ordering for reproducibility
        ord_e = np.lexsort((coo.col, coo.row))
        rows = coo.row[ord_e]
        cols = coo.col[ord_e]
        wts  = coo.data[ord_e]

        edges = list(zip(rows.tolist(), cols.tolist()))
        g = ig.Graph(n=nodes.size, edges=edges, directed=False)
        if weighted:
            g.es["weight"] = wts.tolist()

        best_part, best_q = None, -np.inf

        # optional: stable component-specific seed salt (doesn't hurt)
        comp_seed0 = int(seed + int(nodes.min()) * 1000003)

        for r in range(max(1, int(n_reps))):
            part = la.find_partition(
                g,
                la.CPMVertexPartition,
                weights=weights_arg,  # None => unweighted
                resolution_parameter=float(resolution),
                seed=int(comp_seed0 + r),
            )
            q = part.quality()
            if q > best_q:
                best_q, best_part = q, part

        if best_part is None:
            continue

        memb = np.asarray(best_part.membership, dtype=int)

        # Assign globally unique labels across components
        for c in np.unique(memb):
            idx = np.flatnonzero(memb == c)
            labels[nodes[idx]] = next_label
            next_label += 1

    return labels"""

# ----------------------------------
# Excess matrix
# ----------------------------------

def compute_order_excess(
    seqs,
    mat_dict_real,
    M_baseline=50,
    seed=0,
    eval_mask=None,
):
    """
    Compute baseline (mu, sd) on eval edges, and return REAL excess values on those edges.

    Returns
    -------
    out : dict with
        N : int
        corrmat : (N,N) float array (real corrmat)
        eval_edges : (ii, jj) int arrays (edge indices)
        eval_mask_used : bool array (from compute_eval_edges_corr)
        baseline_mu : float array (per-edge baseline mean)
        baseline_sd : float array (per-edge baseline sd)
        excess_real : float array (per-edge excess for real data)
    """
    corrmat = np.asarray(mat_dict_real["corrmat"], float)
    N = corrmat.shape[0]

    # which edges to evaluate
    ii, jj, eval_mask_used = compute_eval_edges_corr(mat_dict_real, eval_mask=eval_mask)

    # baseline on those edges
    mu, sd = baseline_mu_sd_on_edges_corr(
        seqs, mat_dict_real, ii, jj, M=M_baseline, seed=seed
    )

    # real excess on those edges
    ex_r = excess_on_edges(corrmat, ii, jj, mu, sd)

    return {
        "N": N,
        "corrmat": corrmat,
        "eval_edges": (ii, jj),
        "eval_mask_used": eval_mask_used,
        "baseline_mu": mu,
        "baseline_sd": sd,
        "excess": ex_r,
    }


def compare_clustering_real_vs_shuffle(
    seqs,
    excess_dict,
    T=50,
    seed=0,
    w_keep=1.0,
    min_comp_size=10,
    cpm_resolution=0.03,
    n_reps=10,
    cluster_fn="cpm",  # placeholder
):
    """
    Using precomputed baseline + eval edges (+ real excess), build W for REAL and for shuffled NULL,
    then cluster and summarize.

    Parameters
    ----------
    excess_dict : dict
        Output of compute_order_excess(). Must contain:
        N, eval_edges(ii,jj), baseline_mu, baseline_sd, excess_real

    Returns
    -------
    result : dict
        labels_real, W_real, real_stats, null, keep_idx_real, plus baseline + eval_edges passthrough
    """
    rng = np.random.default_rng(seed)

    N = int(excess_dict["N"])
    ii, jj = excess_dict["eval_edges"]
    mu = excess_dict["baseline_mu"]
    sd = excess_dict["baseline_sd"]
    ex_r = excess_dict["excess"]

    # ---------------- REAL: build W and cluster ----------------
    print("--- true graph ---")
    W_r = build_graph_from_edges(N, ii, jj, ex_r, w_keep=w_keep)
    
    if cluster_fn == "cpm":
        labels_r = cluster_graph_components_cpm(
            W_r,
            min_comp_size=min_comp_size,
            resolution=cpm_resolution,
            n_reps=n_reps,
            seed=seed,
            weighted=False,
            verbose=True
        )
        
    real_cs = cluster_size_stats(labels_r, min_size=min_comp_size)

    real_stats = {
        "edges_eval": int(ii.size),
        "edges_excess_kept": int(W_r.nnz // 2),
        "w_keep": float(w_keep),
        "cpm_resolution": float(cpm_resolution),
        "n_clusters": int(real_cs["n_clusters"]),
        "n_ge_min": int(real_cs["n_ge_min"]),
        "max_size": int(real_cs["max_size"]) if np.isfinite(real_cs["max_size"]) else 0,
        "median_size": float(real_cs["median_size"]) if np.isfinite(real_cs["median_size"]) else 0.0,
    }

    # ---------------- NULL: shuffled validation ----------------
    null = {
        "edges_excess_kept": np.empty(T, float),
        "largest_comp": np.empty(T, float),
        "cl_n_clusters": np.empty(T, float),
        "cl_n_ge_min": np.empty(T, float),
        "cl_max_size": np.empty(T, float),
        "cl_median_size": np.empty(T, float),
    }

    print("--- shuffeled graph ---")
    for t in range(T):
        s = int(rng.integers(0, 2**31 - 1))
        seqs_sh = sc_helpers.shuffle_sequences(seqs, seed=s)
        md_sh = sc.allmot(seqs_sh)

        ex_s = excess_on_edges(md_sh["corrmat"], ii, jj, mu, sd)
        W_s = build_graph_from_edges(N, ii, jj, ex_s, w_keep=w_keep)

        null["edges_excess_kept"][t] = int(W_s.nnz // 2)

        comp_s, sizes_s = connected_components_stats(W_s, verbose=False)
        null["largest_comp"][t] = int(np.max(sizes_s)) if sizes_s.size else 0

        if cluster_fn == "cpm":
            labels_s = cluster_graph_components_cpm(
                W_s,
                min_comp_size=min_comp_size,
                resolution=cpm_resolution,
                n_reps=n_reps,
                seed=s,
                weighted=False,
                verbose=True
            )

        cs = cluster_size_stats(labels_s, min_size=min_comp_size)
        null["cl_n_clusters"][t] = cs["n_clusters"]
        null["cl_n_ge_min"][t] = cs["n_ge_min"]
        null["cl_max_size"][t] = cs["max_size"]
        null["cl_median_size"][t] = cs["median_size"]

    # ---------------- reporting helpers ----------------
    def summarize(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return "empty"
        return f"mean={x.mean():.4g} sd={x.std():.4g} q05={np.quantile(x,0.05):.4g} q95={np.quantile(x,0.95):.4g}"

    print("\n=== REAL (cluster on excess graph) ===")
    for k_, v_ in real_stats.items():
        print(f"{k_:>22s}: {v_}")

    print("\n=== SHUFFLE NULL (evaluate same excess threshold) ===")
    for k_ in ["edges_excess_kept", "largest_comp", "cl_n_clusters", "cl_n_ge_min", "cl_max_size", "cl_median_size"]:
        print(f"{k_:>22s}: {summarize(null[k_])}")

    return {
        "labels_real": labels_r,
        "W_real": W_r,
        "baseline_mu": mu,
        "baseline_sd": sd,
        "eval_edges": (ii, jj),
        "real_stats": real_stats,
        "null": null,
        #"keep_idx_real": keep_idx,
        "excess_real_edges": ex_r,
    }

# ----------------------------------
# Helpers
# ----------------------------------
    
def connected_components_stats(W, verbose=True):
    W = W.tocsr()
    n_comp, comp = connected_components(W, directed=False, connection="weak")
    sizes = np.bincount(comp, minlength=n_comp)
    # print a small summary
    sizes_sorted = np.sort(sizes)[::-1]
    singletons = int(np.sum(sizes == 1))
    deg0 = int(np.sum(np.asarray(W.sum(axis=1)).ravel() == 0))
    if verbose:
        print(f"Connected components: {n_comp}")
        print(f"Largest 10 component sizes: {sizes_sorted[:10].tolist()}")
        print(f"Singletons: {singletons}")
        print(f"Nodes with degree 0: {deg0}")
    return comp, sizes


def cluster_size_stats(labels, min_size=10):
    labels = np.asarray(labels)
    uniq, cnt = np.unique(labels, return_counts=True)
    d = dict(zip(uniq.tolist(), cnt.tolist()))
    noise = d.get(-1, 0)
    cl_sizes = np.array([c for lab, c in d.items() if lab != -1], dtype=int)
    cl_sizes.sort()
    return {
        "n_clusters": int((uniq != -1).sum()),
        "n_noise": int(noise),
        "n_ge_min": int(np.sum(cl_sizes >= min_size)) if cl_sizes.size else 0,
        "max_size": int(cl_sizes.max()) if cl_sizes.size else 0,
        "median_size": float(np.median(cl_sizes)) if cl_sizes.size else 0.0,
    }


def build_graph_from_edges(N, ii, jj, w, w_keep=1.0):
    """Build symmetric sparse graph from edge list and weights, keeping w>=w_keep."""
    keep = np.isfinite(w) & (w >= float(w_keep))
    i = ii[keep]
    j = jj[keep]
    ww = w[keep].astype(float)

    W = sp.csr_matrix((ww, (i, j)), shape=(N, N), dtype=float)
    W = W + W.T
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W

    
def excess_on_edges(Z, ii, jj, mu, sd, eps=1e-6):
    Z = np.asarray(Z, float)
    z = np.nan_to_num(Z[ii, jj], nan=0.0).astype(np.float32)
    ex = (z - mu) / (sd + eps)
    #ex = (z - mu) #delta
    return ex


def compute_eval_edges_corr(mat_dict_real, eval_mask=None):
    Rr = np.asarray(mat_dict_real["corrmat"], float)

    if eval_mask is None:
        # valid correlation entries
        eval_mask = np.isfinite(Rr)
    else:
        eval_mask = np.asarray(eval_mask).astype(bool) & np.isfinite(Rr)

    # never include diagonal
    np.fill_diagonal(eval_mask, False)

    ii, jj = np.where(np.triu(eval_mask, 1))
    return ii.astype(np.int32), jj.astype(np.int32), eval_mask


def baseline_mu_sd_on_edges_corr(seqs, mat_dict_real, ii, jj, M=50, seed=0):
    rng = np.random.default_rng(seed)
    sumr = np.zeros(ii.size, dtype=np.float64)
    sumr2 = np.zeros(ii.size, dtype=np.float64)

    for m in range(M):
        s = int(rng.integers(0, 2**31 - 1))
        seqs_sh = sc_helpers.shuffle_sequences(seqs, seed=s)
        md_sh = sc.allmot(seqs_sh)
        Rs = np.asarray(md_sh["corrmat"], float)

        r = Rs[ii, jj]
        r = np.nan_to_num(r, nan=0.0)  # should rarely happen if eval_mask from real is valid
        sumr += r
        sumr2 += r * r

    mu = (sumr / M).astype(np.float32)
    var = (sumr2 / M) - (sumr / M) ** 2
    var = np.maximum(var, 0.0).astype(np.float32)
    sd = np.sqrt(var).astype(np.float32)
    return mu, sd


def keep_nodes_from_components(W, min_comp_size=10):
    n_comp, comp = connected_components(W, directed=False, connection="weak")
    sizes = np.bincount(comp, minlength=n_comp)
    keep = sizes[comp] >= int(min_comp_size)
    return np.flatnonzero(keep), comp, sizes


def filter_square(M, keep_idx):
    if sp.issparse(M):
        return M.tocsr()[keep_idx][:, keep_idx].tocsr()
    M = np.asarray(M)
    return M[np.ix_(keep_idx, keep_idx)]

def filter_list(x, keep_idx):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [x[i] for i in keep_idx]
    x = np.asarray(x)
    return x[keep_idx]

def W_to_nan_matrix(W, dtype=np.float32, symmetric=True):
    """
    Convert sparse adjacency W to dense matrix A with:
      - A[i,j] = weight if edge exists
      - A[i,j] = NaN if no edge
      - diag = 0

    Parameters
    ----------
    W : scipy.sparse matrix (N,N)
        Weighted adjacency.
    dtype : numpy dtype
    symmetric : bool
        If True, symmetrize by max(W, W.T) before densifying.

    Returns
    -------
    A : (N,N) ndarray
        Dense matrix with NaNs for missing edges.
    """
    if not sp.issparse(W):
        W = sp.csr_matrix(W)

    W = W.tocsr()
    if symmetric:
        W = W.maximum(W.T)

    n = W.shape[0]
    A = np.full((n, n), np.nan, dtype=dtype)

    coo = W.tocoo()
    A[coo.row, coo.col] = coo.data.astype(dtype)

    np.fill_diagonal(A, np.nan)
    return A
"""keep_idx = excess_dict["keep_idx_real"]
seqs944_new  = filter_list(seqs, keep_idx)
bursts944_new = filter_list(bursts, keep_idx)
newmat = W_to_nan_matrix(W_r)
newmat_reduced = W_to_nan_matrix(W_r_)
bmat_new = np.isfinite(newmat_reduced)
np.fill_diagonal(bmat_new, False)
mat_dict_new = {
    "zmat": newmat_reduced,
    "bmat": bmat_new,
    "corrmat": mat_dict["corrmat"][keep_idx][:, keep_idx],
    "repid": mat_dict["repid"][keep_idx],
    "nsig": mat_dict["nsig"][keep_idx],
    "pval": mat_dict["pval"][keep_idx],
}"""