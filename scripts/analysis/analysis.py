# Standard library imports
from collections import Counter

# Third-party imports
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

def high_firing_cells(seqs, thresh):
    """Returns a list of cell IDs that appear in more than `thresh` fraction of sequences (high firing cells)."""
    n_seqs = len(seqs)
    sequence_count = Counter()
    for s in seqs:
        for n in np.unique(s):
            sequence_count[n] += 1
    fracs = {n: c / n_seqs for n, c in sequence_count.items()}
    high = [n for n,c in fracs.items() if c>thresh]
    return np.sort(np.asarray(high, dtype=int))


def get_highly_active_cells(seqs, frac=0.20, return_counts=False, dtype=int):
    """
    Find cell IDs that appear in at least `frac` of sequences.

    Parameters
    ----------
    seqs : list of array-like
        Each element is a sequence of cell indices.
    frac : float
        Minimum fraction of sequences a cell must appear in (default 0.20).
    return_counts : bool
        If True, also return a dict {cell_id: n_sequences_containing_cell}.
    dtype : type
        dtype to use for returned cell ids.

    Returns
    -------
    active_cells : np.ndarray
        Sorted array of cell ids meeting the criterion.
    counts (optional) : dict
        cell_id -> number of sequences containing that cell.
    """
    n_seq = len(seqs)
    if n_seq == 0:
        return (np.array([], dtype=dtype), {}) if return_counts else np.array([], dtype=dtype)

    # count in how many sequences each cell appears (presence/absence per sequence)
    counts = {}
    for s in seqs:
        if s is None:
            continue
        u = np.unique(np.asarray(s))
        for cid in u:
            counts[cid] = counts.get(cid, 0) + 1

    thr = int(np.ceil(frac * n_seq))  # "at least frac" -> ceiling
    active = np.array([cid for cid, c in counts.items() if c >= thr], dtype=dtype)
    active.sort()

    return (active, counts) if return_counts else active

def _scores_aligned_to_labels(score_arr, score_labels, query_labels, *, fill_value=np.nan):
    """
    Align a per-label score array to `query_labels`.

    Assumes:
      - score_labels is sorted unique labels (e.g. np.unique(ids_clust)) and includes -1
      - score_arr has same length as score_labels, same order
    Returns:
      - aligned scores array, same shape as query_labels; fill_value for missing labels
    """
    score_arr = np.asarray(score_arr, dtype=float)
    score_labels = np.asarray(score_labels, dtype=int)
    q = np.asarray(query_labels, dtype=int)

    if score_arr.shape[0] != score_labels.shape[0]:
        raise ValueError("score_arr and score_labels must have the same length.")

    pos = np.searchsorted(score_labels, q)
    ok = (pos >= 0) & (pos < score_labels.size) & (score_labels[pos] == q)

    out = np.full(q.shape, fill_value, dtype=float)
    out[ok] = score_arr[pos[ok]]
    return out

def sort_and_filter_labels(
    ids_clust,
    clust_scores,
    sort_by="ratio",
    ascending=True,
    nan_policy="last",  # "last" (NaNs to end) or "first"
    exclude_labels=None,
    include_labels=None,
    min_score=None,  # e.g. {"ratio": 0.2}
    max_score=None,  # e.g. {"pval": 0.05}
    min_size=None,
    replace_with=-1,
    exclude_noise=True,  # NEW: whether to always exclude label -1 from "kept"
):
    """
    Correct version that aligns score arrays to labels via label->position mapping,
    not by indexing with label values.

    Assumption: for each score key k in clust_scores,
      clust_scores[k] is an array aligned to np.unique(ids_clust) INCLUDING -1.
    """
    ids_clust = np.asarray(ids_clust, dtype=int)
    labels_u = np.unique(ids_clust)  # sorted, includes -1 if present

    # --- normalize sort args ---
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    sort_by = list(sort_by)

    if isinstance(ascending, bool):
        ascending = [ascending] * len(sort_by)
    if len(ascending) != len(sort_by):
        raise ValueError("`ascending` must be a bool or same length as `sort_by`.")

    if nan_policy not in ("last", "first"):
        raise ValueError("nan_policy must be 'last' or 'first'.")

    # --- build lexsort keys over labels_u ---
    key_cols = []
    for i, k in enumerate(sort_by):
        if k == "size":
            v = np.array([np.sum(ids_clust == lbl) for lbl in labels_u], dtype=float)
        else:
            if k not in clust_scores:
                raise KeyError(f"Score '{k}' not found in clust_scores.")
            # scores are already aligned to labels_u by assumption
            v = np.asarray(clust_scores[k], dtype=float)
            if v.shape[0] != labels_u.shape[0]:
                raise ValueError(
                    f"clust_scores['{k}'] length {v.shape[0]} does not match "
                    f"np.unique(ids_clust) length {labels_u.shape[0]}."
                )

        # NaN placement for sorting
        v = np.where(np.isnan(v), np.inf if nan_policy == "last" else -np.inf, v)

        # direction
        if not ascending[i]:
            v = -v

        key_cols.append(v)

    sort_idx_labels = np.lexsort(key_cols[::-1])
    sorted_labels = labels_u[sort_idx_labels]
    order_map = {int(lbl): int(rank) for rank, lbl in enumerate(sorted_labels)}

    # --- sort ids by cluster rank (stable) ---
    sort_keys = np.array([order_map[int(lbl)] for lbl in ids_clust], dtype=int)
    sorted_indices = np.argsort(sort_keys, kind="mergesort")
    ids_sorted = ids_clust[sorted_indices]

    # --- build exclusion set (label space) ---
    all_labels = np.unique(ids_sorted)  # should match labels_u
    excl = set(exclude_labels or [])

    if exclude_noise:
        excl.add(-1)

    incl = set(all_labels) if include_labels is None else set(include_labels)
    if exclude_noise:
        incl.discard(-1)  # prevent re-adding noise if user included it by accident

    # score-threshold exclusions
    if min_score:
        for k, thr in min_score.items():
            if k not in clust_scores:
                raise KeyError(f"Score '{k}' not found in clust_scores.")
            vals = _scores_aligned_to_labels(clust_scores[k], labels_u, all_labels, fill_value=np.nan)
            excl |= set(all_labels[np.isnan(vals) | (vals < thr)])

    if max_score:
        for k, thr in max_score.items():
            if k not in clust_scores:
                raise KeyError(f"Score '{k}' not found in clust_scores.")
            vals = _scores_aligned_to_labels(clust_scores[k], labels_u, all_labels, fill_value=np.nan)
            excl |= set(all_labels[np.isnan(vals) | (vals > thr)])

    # size-based exclusions
    if min_size is not None:
        if min_size < 1:
            raise ValueError("min_size must be >= 1 or None.")
        lbls_cnt, counts = np.unique(ids_clust, return_counts=True)
        small = lbls_cnt[counts < min_size]
        excl |= set(map(int, small.tolist()))

    # exclude anything not explicitly included
    excl |= {int(lbl) for lbl in all_labels if int(lbl) not in incl}

    # --- apply on sorted ids ---
    is_excluded_sorted = np.isin(ids_sorted, list(excl))
    ids_filtered = ids_sorted.copy()
    ids_filtered[is_excluded_sorted] = replace_with

    kept_mask_sorted = ~is_excluded_sorted

    # --- apply on original ids (same exclusion set) ---
    is_excluded_unsorted = np.isin(ids_clust, list(excl))
    ids_replaced = ids_clust.copy()
    ids_replaced[is_excluded_unsorted] = replace_with

    # kept labels list in sorted order, and unique kept set in first-appearance order
    kept_labels_sorted = ids_filtered[kept_mask_sorted]
    _, first_idx = np.unique(kept_labels_sorted, return_index=True)
    kept = kept_labels_sorted[np.sort(first_idx)]

    return {
        "idx_sorted": sorted_indices,             # reindex per-sequence arrays to sorted axis
        "order_map": order_map,                   # label -> rank
        "labels_unique": labels_u,                # np.unique(ids_clust) used for score alignment
        "ids_clust_sorted": ids_sorted,           # ids sorted by rank
        "ids_clust_filtered": ids_filtered,       # sorted + excluded -> replace_with
        "ids_clust_replaced": ids_replaced,       # original order + excluded -> replace_with
        "mask_kept_sorted": kept_mask_sorted,     # mask over sorted axis
        "mask_kept": ~is_excluded_unsorted,       # mask over original axis
        "ids_clust_kept": kept_labels_sorted,     # ids_filtered[mask_kept_sorted]
        "excluded": np.array(sorted(excl), dtype=int),
        "kept": kept,
    }


def compute_cluster_phase_stats(res, rat_data):
    """
    Compute phase counts, rates and ratios per cluster.

    Parameters
    ----------
    res : dict
        Must contain key "ids_clust_replaced" (array-like of cluster labels, -1 for excluded).
    rat_data : dict
        Must contain keys:
            - 'burst_phases': array-like of phase labels per sequence
            - 'exp_phases': list/array, where rat_data['exp_phases'][1] is
              [(start_pre, end_pre), (start_delay, end_delay), (start_post, end_post)]
    Returns
    -------
    dict with keys:
        - df : DataFrame with columns ['cluster', 'phase']
        - phase_counts : DataFrame, counts per cluster x phase
        - phase_rate : DataFrame, rate per cluster x phase (counts / rest_time)
        - phase_rate_sorted : DataFrame, phase_rate sorted by pattern of active phases
        - phase_ratio : DataFrame, within-cluster phase ratios (rows possibly sorted by 'delay')
        - sort_order : list of cluster labels in the custom sort order
        - durations : dict of experiment phase durations computed from exp_phases
    """
    # phases and labels
    phases_all = np.asarray(rat_data['burst_phases'])
    assert len(res["ids_clust_replaced"]) == len(phases_all), "labels and phases must align"

    # experiment phase durations (from exp_phases, for reference)
    exp_phases = rat_data['exp_phases'][0]
    phase_names = rat_data['exp_phases'][1]
    durations = {name: float(e - s) for name, (s, e) in zip(phase_names, exp_phases)}

    # exclude -1 clusters
    mask = res["ids_clust_replaced"] != -1
    labels = res["ids_clust_replaced"][mask]
    phases = phases_all[mask]

    # remove unlabeled sequences (NaN or empty)
    valid_mask = pd.notna(phases) & (phases != "")
    labels = labels[valid_mask]
    phases = phases[valid_mask]

    # build DataFrame
    df = pd.DataFrame({"cluster": labels, "phase": phases})

    # compute absolute counts
    phase_counts = pd.crosstab(df["cluster"], df["phase"])
    phase_counts = phase_counts.drop(
        columns=[col for col in phase_counts.columns if pd.isna(col) or col == "unlabeled"],
        errors="ignore"
    )

    # rates normalized by immobility_durations
    phase_rate = phase_counts.div(pd.Series(rat_data['immobility_durations']), axis=1)

    # number of phases per cluster
    ph_active = (phase_rate > 0).sum(axis=1)

    def sort_key(row):
        active = set(row.index[row > 0])
        if len(active) == 1:
            return (1, 0)  # group 1: single-phase
        elif len(active) == 2:
            # group 2: two-phase → prefer (delay,pre) first, then (delay,post)
            if {"delay", "pre"} == active:
                return (2, 0)
            elif {"delay", "post"} == active:
                return (2, 1)
            else:
                return (2, 2)  # other combinations if any
        elif len(active) == 3:
            return (3, 0)
        else:
            return (4, 0)  # safety

    # apply sorting
    sort_order = sorted(phase_rate.index, key=lambda i: sort_key(phase_rate.loc[i]))
    phase_rate_sorted = phase_rate.loc[sort_order]

    # normalize to ratios for plotting
    phase_ratio = phase_counts.div(phase_counts.sum(axis=1), axis=0)

    # sort by number of 'delay' bursts if column exists
    if "delay" in phase_ratio.columns:
        phase_ratio = phase_ratio.sort_values(by="delay", ascending=False)

    return {
        "df": df,
        "phase_counts": phase_counts,
        "phase_rate": phase_rate,
        "phase_rate_sorted": phase_rate_sorted,
        "phase_ratio": phase_ratio,
        "sort_order": sort_order,
        "durations": durations,
        "ph_active": ph_active,
    }
    
def treves_rolls_sparseness(R, axis=1, eps=1e-12):
    """
    Treves–Rolls sparseness along `axis`.
    If R is (cells x clusters), use axis=1 to get one value per cell.
    Returns S in [0, 1] (approximately), with 1 = very sparse.
    """
    R = np.asarray(R, dtype=float)
    if np.any(R < -eps):
        raise ValueError("Treves-Rolls requires nonnegative responses R (>=0).")

    R = np.clip(R, 0, None)

    K = R.shape[axis]
    if K < 2:
        return np.full(R.shape[0 if axis == 1 else 1], np.nan)

    mean_r = np.mean(R, axis=axis)
    mean_r2 = np.mean(R * R, axis=axis)

    # a = (mean_r^2) / mean_r2
    a = (mean_r * mean_r) / (mean_r2 + eps)

    # S = (1 - a) / (1 - 1/K)
    S = (1.0 - a) / (1.0 - 1.0 / K)

    # If a cell has all zeros, mean_r=0 and mean_r2=0 -> define sparseness as 0 (or NaN if you prefer)
    all_zero = mean_r2 < eps
    if np.any(all_zero):
        S = np.asarray(S)
        S[all_zero] = 0.0

    return S

def cell_cluster_contribution(
    seqs,
    ids_clust,
    n_cells=None,
    normalize="by_total",   # "by_total" (sums to 1 per cell), or "by_cluster" (P(cell|cluster))
    exclude_labels=(-1,),
    return_counts=False,
):
    """
    Compute, for each cell, how its activity is distributed across clusters.

    Parameters
    ----------
    seqs : list
        List of sequences. Each element can be:
        - iterable of active cell indices (e.g. [3,10,11])
        - or boolean/binary 1D array of length n_cells
    ids_clust : array-like, shape (n_seqs,)
        Cluster label for each sequence.
    n_cells : int or None
        Total number of cells. If None, inferred from seqs (only works reliably for index-list seqs).
    normalize : {"by_total","by_cluster"}
        - "by_total": contribution[c,k] = count(c in seqs of cluster k) / total_count(c)
                      -> rows sum to 1 for cells with any activity
        - "by_cluster": contribution[c,k] = count(c in seqs of cluster k) / n_seqs_in_cluster_k
                        -> interpretable as P(cell active | cluster)
    exclude_labels : iterable
        Cluster labels to ignore (e.g. (-1,) for noise).
    min_total_occ : int
        Only keep cells with total_count(c) >= min_total_occ (others set to 0 / NaN depending on normalize).
    return_counts : bool
        If True, also return raw count matrix (cells x clusters).

    Returns
    -------
    contrib : ndarray, shape (n_cells, n_clusters_kept)
    clusters : ndarray, shape (n_clusters_kept,)
        The cluster labels corresponding to contrib columns.
    (optional) counts : ndarray, shape (n_cells, n_clusters_kept)
    """

    ids_clust = np.asarray(ids_clust)
    keep_seq = np.ones(ids_clust.shape[0], dtype=bool)
    if exclude_labels is not None:
        excl = set(exclude_labels)
        keep_seq &= np.array([lab not in excl for lab in ids_clust], dtype=bool)

    seqs_kept = [seqs[i] for i in np.flatnonzero(keep_seq)]
    labs_kept = ids_clust[keep_seq]

    clusters = np.unique(labs_kept)
    k_map = {lab: j for j, lab in enumerate(clusters)}
    n_seqs = len(seqs_kept)
    n_clust = clusters.size

    # Build sparse membership M: (n_seqs x n_cells), M[i,c]=1 if cell c active in seq i
    rows, cols = [], []
    for i, s in enumerate(seqs_kept):
        if isinstance(s, np.ndarray) and s.dtype == bool:
            active = np.flatnonzero(s)
        elif isinstance(s, np.ndarray) and s.ndim == 1 and s.size == n_cells and np.isin(s, [0, 1]).all():
            active = np.flatnonzero(s)
        else:
            active = np.asarray(list(s), dtype=int)
        if active.size:
            rows.append(np.full(active.size, i, dtype=int))
            cols.append(active)

    if len(rows) == 0:
        counts = np.zeros((n_cells, n_clust), dtype=float)
    else:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.ones(rows.size, dtype=np.float32)
        M = sp.csr_matrix((data, (rows, cols)), shape=(n_seqs, n_cells))

        # One-hot cluster indicator Z: (n_seqs x n_clust)
        z_rows = np.arange(n_seqs, dtype=int)
        z_cols = np.array([k_map[lab] for lab in labs_kept], dtype=int)
        Z = sp.csr_matrix((np.ones(n_seqs, dtype=np.float32), (z_rows, z_cols)), shape=(n_seqs, n_clust))

        # counts per cell per cluster: (n_cells x n_clust)
        counts = (M.T @ Z).toarray()

    # Apply min_total_occ mask
    total = counts.sum(axis=1)
    active_mask = total >= 0

    if normalize == "by_total":
        denom = np.maximum(total, 1.0)[:, None]
        contrib = counts / denom
        contrib[~active_mask, :] = 0.0
    elif normalize == "by_cluster":
        n_per_cluster = np.bincount(
            np.array([k_map[lab] for lab in labs_kept], dtype=int),
            minlength=n_clust
        ).astype(float)
        denom = np.maximum(n_per_cluster, 1.0)[None, :]
        contrib = counts / denom
        contrib[~active_mask, :] = 0.0
    else:
        raise ValueError("normalize must be 'by_total' or 'by_cluster'")

    # distribution over cells within each cluster (columns sum to 1)
    col_sums = counts.sum(axis=0, keepdims=True)
    contrib_cell_given_cluster = np.divide(
        counts, col_sums,
        out=np.zeros_like(counts, dtype=float),
        where=col_sums > 0
    )

    if return_counts:
        return contrib, clusters, counts
    return contrib, contrib_cell_given_cluster, clusters
