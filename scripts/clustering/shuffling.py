from __future__ import annotations

# Standard library imports
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Local imports
from ..analysis import analysis as sa
from ..clustering import distances as dist_helpers
from ..visualization.style import PublicationStandard
from . import core as sc


def shuffle_sequences(seqs, seed=None):
    """
    Shuffle each sequence independently, preserving format.

    Parameters
    ----------
    seqs : list of np.ndarray
        List of sequences to shuffle.
    seed : int or None
        Seed for reproducible shuffling.

    Returns
    -------
    shuffled_seqs : list of np.ndarray
        Same format as input, with each sequence shuffled.
    """
    rng = np.random.default_rng(seed)
    return [rng.permutation(x) for x in seqs]


# ==========================================
# Survival analysis for sequence clustering
# ==========================================

def survival_scores(data, k, thr, seeds):
    """
    Outputs (all per-sequence, length n_seqs):
      - survival_freq: fraction of seeds where seq i is in a surviving cluster
      - mean_cluster_size: average size of i's surviving cluster (including i) over seeds where i survives

    Optionally:
      - pair_probs: list of Counter objects mapping neighbor j -> p_ij
          (sparse per-i map; can be large)

    Notes:
      - Pairwise "co-cluster probability" p_ij is computed over ALL seeds, as:
            p_ij = (# seeds where i and j co-cluster in surviving clusters) / (# seeds total)
        so it naturally downweights pairs that only occur when i survives rarely.
    """
    seqs = data["seqs"]
    n_seqs = len(seqs)
    seeds = list(seeds)
    S = len(seeds)

    # Cache labels for each seed
    labels_by_seed = []
    survival_hits = np.zeros(n_seqs, dtype=int)
    clust_size_sum = np.zeros(n_seqs, dtype=int)
    clust_size_n = np.zeros(n_seqs, dtype=int)

    # Sparse per-sequence co-cluster counts:
    # co_counts[i][j] = number of seeds where i and j are in the same *surviving* cluster.
    co_counts = [Counter() for _ in range(n_seqs)]

    # --- run pipeline for each seed, accumulate survival + co-cluster counts ---
    for seed in seeds:
        seqs_sh = shuffle_sequences(seqs, seed=seed)
        mat_dict_sh = sc.allmot(seqs_sh)
        res_sh, _ = run_one(seqs_sh, data, mat_dict_sh, k, thr)
        ids = np.asarray(res_sh["ids_clust_replaced"], dtype=int)
        labels_by_seed.append(ids)

        surv = (ids != -1)
        survival_hits += surv.astype(np.int32)

        # group members by cluster label (excluding replace_with=-1)
        groups = defaultdict(list)
        for i, c in enumerate(ids):
            if c != -1:
                groups[int(c)].append(i)

        # update cluster-size stats and pairwise co-cluster counts
        for members in groups.values():
            m = len(members)
            if m <= 1:
                # singleton surviving clusters contribute size info but no pairs
                for i in members:
                    clust_size_sum[i] += 1
                    clust_size_n[i] += 1
                continue

            members = np.asarray(members, dtype=int)

            # size stats (for average surviving cluster size)
            clust_size_sum[members] += m
            clust_size_n[members] += 1

            # co-cluster counts [i,j] how many times did i co-cluster with j
            # For each i in cluster, add all other members as neighbors.
            for idx_i, i in enumerate(members):
                # update counter with all members
                co_counts[i].update(members.tolist())
                co_counts[i][int(i)] -= 1  # remove self i=j

    # --- Survival fraction
    survival_freq = (survival_hits / S).astype(float)
    # mean cluster size when surviving
    mean_cluster_size = np.zeros(n_seqs, dtype=float)
    mask_cs = clust_size_n > 0
    mean_cluster_size[mask_cs] = (clust_size_sum[mask_cs] / clust_size_n[mask_cs]).astype(float)

    # --- Pair probabilities
    pair_probs = []
    for i in range(n_seqs):
        # sparse mapping j -> p_ij
        pair_probs.append(Counter({j: cnt / S for j, cnt in co_counts[i].items() if j != i}))

    # --- Pairwise Jaccard neighbor stability 
    pj = pairwise_neighbor_jaccard(labels_by_seed, replace_with=-1)

    out = {
        "survival_freq": survival_freq,  # fraction in surviving cluster
        "pair_probs": pair_probs,
        "pairwise_jaccard_cond_survival": pj["pairwise_jaccard_cond_survival"],
        "n_pairs_used_cond_survival": pj["n_pairs_used_cond_survival"],
        "pairwise_jaccard_non_survival_empty": pj["pairwise_jaccard_non_survival_empty"],
        "n_pairs_used_non_survival_empty": pj["n_pairs_used_non_survival_empty"],
        "mean_cluster_size": mean_cluster_size,
        "labels_by_seed": [np.asarray(x, dtype=int).copy() for x in labels_by_seed]
    }
    return out


def run_one(seqs, data, mat_dict, k, thr, memmap_dir=None):
    """
    Run one clustering pipeline.
    
    Parameters
    ----------
    seqs : list
        Sequences to cluster.
    data : dict
        Data dictionary with 'bursts' and 'seqs' keys.
    mat_dict : dict
        Output from allmot(...).
    k : int
        Number of clusters.
    thr : float
        Threshold for merging/filtering.
    memmap_dir : str or Path, optional
        Directory for memmap file. If None, uses temporary directory (auto-cleaned).
        This allows flexibility: use temp dir for one-off analysis, or persistent
        dir if you need to reuse the memmap.
    
    Returns
    -------
    res, rd_merged : tuple
        Clustering results and merged result dict.
    """
    # Auto-create temp directory if not specified; will be cleaned up after use
    if memmap_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        memmap_path = Path(temp_dir.name) / "pdist.dat"
    else:
        memmap_dir = Path(memmap_dir)
        memmap_dir.mkdir(parents=True, exist_ok=True)
        memmap_path = memmap_dir / "pdist.dat"
    
    memmap_path_str = str(memmap_path)
    
    # Compute memmap
    y = dist_helpers.pdist_euclid_sparse_memmap(mat_dict, memmap_path_str, dtype=np.float64)
    # Force copy=True to load data into memory before temp dir cleanup
    pdist_euclid = np.memmap(memmap_path_str, mode='r', dtype=np.float64, shape=(y.size,)).astype(np.float64, copy=True)

    """cmat = dist_helpers.zmat_to_cmat(mat_dict)
    pdist_euclid = pdist(cmat, metric='euclidean')"""

    ids_clust = sc.seq_cluster(pdist_euclid, k=k, method="ward")

    rd = sc.info_cluster(data["bursts"], seqs, ids_clust, method="center_of_mass")
    sc.add_within_clust_score(rd, mat_dict)
    rd_merged, _, _, _ = sc.merge_clusters(mat_dict, rd, data, thr=thr, verbose=False)

    res = sa.sort_and_filter_labels(
        ids_clust=rd_merged["ids_clust"],
        clust_scores=rd_merged["clust_scores"],
        sort_by="within_clust",
        ascending=True,
        min_score={"within_clust": thr},
        min_size=1,
        replace_with=-1,
    )
    
    # Clean up temp dir if created
    if memmap_dir is None:
        temp_dir.cleanup()
    
    return res, rd_merged


def pairwise_neighbor_jaccard(labels_by_seed, replace_with = -1):
    """
    Compute per-sequence pairwise Jaccard neighbor stability across seeds in TWO modes:

      1) conditional on survival:
         compare only seed pairs where sequence i survives in both seeds

      2) non-survival as empty set:
         include all seed pairs, treating non-survival as neighbors = empty set

    Parameters
    ----------
    labels_by_seed : list of arrays, each shape (n_seqs,)
        Cluster labels per seed. `replace_with` means "not surviving".
    replace_with : int, default -1
        Label indicating non-survival / excluded sequence.

    Returns
    -------
    out : dict
        {
          "pairwise_jaccard_cond_survival": (n_seqs,) float array,
          "n_pairs_used_cond_survival": (n_seqs,) int array,
          "pairwise_jaccard_non_survival_empty": (n_seqs,) float array,
          "n_pairs_used_non_survival_empty": (n_seqs,) int array,
          "neighbor_sets_by_seed": list[list[set or None]],
        }

    Notes
    -----
    - neighbor_sets_by_seed[s][i] is:
        * set of neighbors if i survives in seed s
        * None if i does not survive
      (the "non-survival as empty" mode is derived by treating None as empty set during scoring)
    """
    if len(labels_by_seed) == 0:
        raise ValueError("labels_by_seed must be non-empty")

    labels_by_seed = [np.asarray(ids, dtype=int) for ids in labels_by_seed]
    n_seqs = labels_by_seed[0].shape[0]
    S = len(labels_by_seed)

    # --- Build neighbor sets per seed (None for non-survival) ---
    neighbor_sets_by_seed = []
    for ids in labels_by_seed:
        if ids.shape != (n_seqs,):
            raise ValueError(f"All ids must have shape {(n_seqs,)}, got {ids.shape}")

        neigh = [None] * n_seqs  # None = non-survival in this seed

        groups = defaultdict(list)
        for i, c in enumerate(ids):
            if c != replace_with:
                groups[int(c)].append(i)

        for members in groups.values():
            members = [int(x) for x in members]
            mem_set = set(members)
            for i in members:
                neigh[i] = mem_set - {i}  # empty set if singleton

        neighbor_sets_by_seed.append(neigh)

    # --- Accumulators for both modes ---
    jacc_sum_cond = np.zeros(n_seqs, dtype=float)
    n_pairs_cond = np.zeros(n_seqs, dtype=np.int32)

    jacc_sum_empty = np.zeros(n_seqs, dtype=float)
    n_pairs_empty = np.zeros(n_seqs, dtype=np.int32)

    # --- Compare all seed pairs ---
    for s in range(S):
        neigh_s = neighbor_sets_by_seed[s]
        for t in range(s + 1, S):
            neigh_t = neighbor_sets_by_seed[t]

            for i in range(n_seqs):
                A = neigh_s[i]  # set or None
                B = neigh_t[i]  # set or None

                # Mode 1: conditional on survival (skip if absent in either seed)
                if (A is not None) and (B is not None):
                    inter = len(A.intersection(B))
                    union = len(A) + len(B) - inter
                    jac = 1.0 if union == 0 else (inter / union)
                    jacc_sum_cond[i] += jac
                    n_pairs_cond[i] += 1

                # Mode 2: include all seed pairs, pair involving non-survival counts as 0
                # This means:
                #   - survive/survive  -> Jaccard(neighbor sets)
                #   - survive/noise    -> 0
                #   - noise/noise      -> 0
                if (A is None) or (B is None):
                    jac2 = 0.0
                else:
                    inter2 = len(A.intersection(B))
                    union2 = len(A) + len(B) - inter2
                    jac2 = 1.0 if union2 == 0 else (inter2 / union2)  # singleton/singleton -> 1 if both empty sets
                
                jacc_sum_empty[i] += jac2
                n_pairs_empty[i] += 1

    # --- Final averages ---
    pairwise_jaccard_cond_survival = np.zeros(n_seqs, dtype=float)
    m = n_pairs_cond > 0
    pairwise_jaccard_cond_survival[m] = jacc_sum_cond[m] / n_pairs_cond[m]

    pairwise_jaccard_non_survival_empty = np.zeros(n_seqs, dtype=float)
    m2 = n_pairs_empty > 0
    pairwise_jaccard_non_survival_empty[m2] = jacc_sum_empty[m2] / n_pairs_empty[m2]

    return {
        "pairwise_jaccard_cond_survival": pairwise_jaccard_cond_survival,
        "n_pairs_used_cond_survival": n_pairs_cond,
        "pairwise_jaccard_non_survival_empty": pairwise_jaccard_non_survival_empty,
        "n_pairs_used_non_survival_empty": n_pairs_empty,
        "neighbor_sets_by_seed": neighbor_sets_by_seed,
    }

    return out


# ==========================================
# Null model
# ==========================================

def survival_freq_from_labels(labels_by_seed, replace_with=-1):
    """Compute per-sequence survival frequency from cached labels_by_seed."""
    labels_by_seed = [np.asarray(ids, dtype=int) for ids in labels_by_seed]
    S = len(labels_by_seed)
    n_seqs = labels_by_seed[0].shape[0]

    hits = np.zeros(n_seqs, dtype=np.int32)
    for ids in labels_by_seed:
        hits += (ids != replace_with).astype(np.int32)
    return hits / float(S)


def permute_labels_within_seed_ids(labels_by_seed, rng):
    """
    Permute sequence identities across ALL entries (including noise=-1 positions).
    Preserves per-seed cluster sizes and total number of survivors, but randomizes
    which sequence identities are survivors/noise.
    """
    out = []
    for ids in labels_by_seed:
        ids = np.asarray(ids, dtype=int)
        perm = rng.permutation(ids.shape[0])
        out.append(ids[perm].copy())
    return out


def permute_labels_within_seed_ids_preserve_noise(labels_by_seed, rng, replace_with=-1):
    """
    Permute sequence identities only among surviving entries (ids != replace_with),
    leaving noise positions fixed.
    """
    out = []
    for ids in labels_by_seed:
        ids = np.asarray(ids, dtype=int).copy()
        surv_idx = np.flatnonzero(ids != replace_with)
        if surv_idx.size > 1:
            vals = ids[surv_idx].copy()
            ids[surv_idx] = vals[rng.permutation(surv_idx.size)]
        out.append(ids)
    return out


def permute_labels_by_mode(labels_by_seed, rng, *, preserve_noise: bool, replace_with=-1):
    """Wrapper to choose permutation mode."""
    if preserve_noise:
        return permute_labels_within_seed_ids_preserve_noise(
            labels_by_seed, rng=rng, replace_with=replace_with
        )
    return permute_labels_within_seed_ids(labels_by_seed, rng=rng)


def compute_scores_from_labels_only(labels_by_seed, replace_with=-1):
    """
    Compute key scores directly from cached labels (no reclustering):
      - survival_freq
      - pairwise_jaccard_cond_survival
      - pairwise_jaccard_non_survival_empty
    """
    sf = survival_freq_from_labels(labels_by_seed, replace_with=replace_with)
    pj = pairwise_neighbor_jaccard(labels_by_seed, replace_with=replace_with)

    return {
        "survival_freq": sf,
        "pairwise_jaccard_cond_survival": pj["pairwise_jaccard_cond_survival"],
        "n_pairs_used_cond_survival": pj["n_pairs_used_cond_survival"],
        "pairwise_jaccard_non_survival_empty": pj["pairwise_jaccard_non_survival_empty"],
        "n_pairs_used_non_survival_empty": pj["n_pairs_used_non_survival_empty"],
    }


def null_model_from_labels(
    labels_by_seed,
    n_null=100,
    seed=0,
    replace_with=-1,
    preserve_noise=False,
):
    """
    Build a null distribution by permuting sequence identities within each seed's
    label vector (post-clustering null).

    Returns
    -------
    null_out : dict
      - sf_null_all            (n_null, n_seqs)
      - pj_null_all            (n_null, n_seqs)   [conditional survival]
      - pj_empty_null_all      (n_null, n_seqs)   [non-survival=0 mode]
      - *_flat                 flattened marginals
    """
    rng = np.random.default_rng(seed)

    labels_by_seed = [np.asarray(ids, dtype=int) for ids in labels_by_seed]
    n_seqs = labels_by_seed[0].shape[0]

    sf_null_all = np.zeros((n_null, n_seqs), dtype=float)
    pj_null_all = np.zeros((n_null, n_seqs), dtype=float)
    pj_empty_null_all = np.zeros((n_null, n_seqs), dtype=float)
    pj_pairs_null_all = np.zeros((n_null, n_seqs), dtype=np.int32)
    pj_empty_pairs_null_all = np.zeros((n_null, n_seqs), dtype=np.int32)

    for b in range(n_null):
        labels_perm = permute_labels_by_mode(
            labels_by_seed,
            rng=rng,
            preserve_noise=preserve_noise,
            replace_with=replace_with,
        )
        sc_null = compute_scores_from_labels_only(labels_perm, replace_with=replace_with)

        sf_null_all[b] = sc_null["survival_freq"]
        pj_null_all[b] = sc_null["pairwise_jaccard_cond_survival"]
        pj_empty_null_all[b] = sc_null["pairwise_jaccard_non_survival_empty"]
        pj_pairs_null_all[b] = sc_null["n_pairs_used_cond_survival"]
        pj_empty_pairs_null_all[b] = sc_null["n_pairs_used_non_survival_empty"]

    return {
        "sf_null_all": sf_null_all,
        "pj_null_all": pj_null_all,
        "pj_empty_null_all": pj_empty_null_all,
        "pj_pairs_null_all": pj_pairs_null_all,
        "pj_empty_pairs_null_all": pj_empty_pairs_null_all,
        "sf_null_flat": sf_null_all.ravel(),
        "pj_null_flat": pj_null_all.ravel(),
        "pj_empty_null_flat": pj_empty_null_all.ravel(),
        "n_null": n_null,
        "n_seqs": n_seqs,
        "preserve_noise": bool(preserve_noise),
    }


def null_marginal_thresholds(null_out, q_sf=0.95, q_pj=0.95, q_pj_empty=0.95):
    """
    Separate thresholds from null marginals.
    """
    sf_thr = float(np.quantile(null_out["sf_null_flat"], q_sf))
    pj_thr = float(np.quantile(null_out["pj_null_flat"], q_pj))
    pj_empty_thr = float(np.quantile(null_out["pj_empty_null_flat"], q_pj_empty))

    return {
        "t_sf": sf_thr,
        "t_pj": pj_thr,
        "t_pj_empty": pj_empty_thr,
        "q_sf": q_sf,
        "q_pj": q_pj,
        "q_pj_empty": q_pj_empty,
        "preserve_noise": null_out.get("preserve_noise", None),
    }


def null_thresholds_split(
    labels_by_seed,
    *,
    n_null=100,
    seed_sf=123,
    seed_pj=456,
    seed_pj_empty=789,
    q_sf=0.95,
    q_pj=0.95,
    q_pj_empty=0.95,
    preserve_noise_for_sf=False,
    preserve_noise_for_pj=True,
    preserve_noise_for_pj_empty=False,
):
    """
    Compute thresholds using potentially different null modes:
      - sf threshold          (default preserve_noise=False)
      - pj threshold          (default preserve_noise=True)
      - pj_empty threshold    (default preserve_noise=False)

    Suggested defaults:
      sf      : preserve_noise=False   (identity-randomization incl noise)
      pj      : preserve_noise=True    (conditional-on-survival null)
      pj_empty: preserve_noise=False   (score includes noise, so let noise randomize too)
    """
    null_sf = null_model_from_labels(
        labels_by_seed,
        n_null=n_null,
        seed=seed_sf,
        replace_with=-1,
        preserve_noise=preserve_noise_for_sf,
    )
    null_pj = null_model_from_labels(
        labels_by_seed,
        n_null=n_null,
        seed=seed_pj,
        replace_with=-1,
        preserve_noise=preserve_noise_for_pj,
    )
    null_pj_empty = null_model_from_labels(
        labels_by_seed,
        n_null=n_null,
        seed=seed_pj_empty,
        replace_with=-1,
        preserve_noise=preserve_noise_for_pj_empty,
    )

    t_sf = float(np.quantile(null_sf["sf_null_flat"], q_sf))
    t_pj = float(np.quantile(null_pj["pj_null_flat"], q_pj))
    t_pj_empty = float(np.quantile(null_pj_empty["pj_empty_null_flat"], q_pj_empty))

    return {
        "t_sf": t_sf,
        "t_pj": t_pj,
        "t_pj_empty": t_pj_empty,
        "q_sf": q_sf,
        "q_pj": q_pj,
        "q_pj_empty": q_pj_empty,
        "preserve_noise_for_sf": preserve_noise_for_sf,
        "preserve_noise_for_pj": preserve_noise_for_pj,
        "preserve_noise_for_pj_empty": preserve_noise_for_pj_empty,
        "null_sf": null_sf,
        "null_pj": null_pj,
        "null_pj_empty": null_pj_empty,
    }


# ==========================================
# Summary plots
# ==========================================
def summarize_survival_scores(
    scores,
    *,
    freq_thresholds=(0.1, 0.25, 0.5, 0.75, 0.9),
    bins=50,
    scatter_s=10,
    scatter_alpha=0.35,
    cmap="viridis",
):
    """
    Print + plot summaries for survival/co-clustering metrics.

    Focused on the current metrics:
      - survival_freq
      - pairwise_jaccard_cond_survival
      - pairwise_jaccard_non_survival_empty
      - mean_cluster_size
      - n_pairs_used_cond_survival

    Optional legacy metric (if present):
      - neighbor_jaccard_consensus
      - consensus_neighbor_count
    """
    sf   = np.asarray(scores["survival_freq"], dtype=float)
    msz  = np.asarray(scores.get("mean_cluster_size", np.full_like(sf, np.nan)), dtype=float)

    pj       = np.asarray(scores.get("pairwise_jaccard_cond_survival", np.full_like(sf, np.nan)), dtype=float)
    pj_n     = np.asarray(scores.get("n_pairs_used_cond_survival", np.full_like(sf, np.nan)), dtype=float)
    pj_empty = np.asarray(scores.get("pairwise_jaccard_non_survival_empty", np.full_like(sf, np.nan)), dtype=float)
    pj_empty_n = np.asarray(scores.get("n_pairs_used_non_survival_empty", np.full_like(sf, np.nan)), dtype=float)

    # optional legacy metrics (kept only for printing / fallback)
    jac_cons = np.asarray(scores.get("neighbor_jaccard_consensus", np.full_like(sf, np.nan)), dtype=float)
    cn       = np.asarray(scores.get("consensus_neighbor_count", np.full_like(sf, np.nan)), dtype=float)

    n = sf.size
    eps = 1e-12

    def qstats(x, name):
        x = np.asarray(x, float)
        finite = np.isfinite(x)
        xf = x[finite]
        if xf.size == 0:
            print(f"{name}: no finite values")
            return
        qs = np.quantile(xf, [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        print(f"\n{name} (n_finite={xf.size}/{x.size})")
        print("  min  p01  p05  p10  p25  p50  p75  p90  p95  p99  max")
        print(" ", " ".join(f"{v:6.3f}" for v in qs))

    print(f"=== Survival score summary (n_seqs={n}) ===")
    qstats(sf, "survival_freq")
    if np.isfinite(pj).any():
        qstats(pj, "pairwise_jaccard_cond_survival")
    if np.isfinite(pj_empty).any():
        qstats(pj_empty, "pairwise_jaccard_non_survival_empty")
    if np.isfinite(jac_cons).any():
        qstats(jac_cons, "neighbor_jaccard_consensus")
    if np.isfinite(msz).any():
        qstats(msz, "mean_cluster_size")
    if np.isfinite(cn).any():
        qstats(cn, "consensus_neighbor_count")
    if np.isfinite(pj_n).any():
        qstats(pj_n, "n_pairs_used_cond_survival")
    if np.isfinite(pj_empty_n).any():
        qstats(pj_empty_n, "n_pairs_used_non_survival_empty")

    # Threshold counts for survival_freq
    print("\nSurvival freq threshold counts:")
    for t in freq_thresholds:
        print(f"  >= {t:>4.2f}: {int(np.sum(sf >= t))}  ({np.mean(sf >= t)*100:5.1f}%)")

    # Quick checks
    print("\nQuick checks:")
    print(f"  P(sf==0):  {np.mean(sf <= eps)*100:5.1f}%")
    print(f"  P(sf==1):  {np.mean(sf >= 1-eps)*100:5.1f}%")
    if np.isfinite(cn).any():
        print(f"  Median consensus neighbors: {np.nanmedian(cn):.1f}")
        print(f"  P(consensus_neighbor_count==0): {np.nanmean(cn <= eps)*100:5.1f}%")
    if np.isfinite(pj_n).any():
        print(f"  Median n_pairs_used_cond_survival: {np.nanmedian(pj_n):.1f}")
    if np.isfinite(pj_empty_n).any():
        print(f"  Median n_pairs_used_non_survival_empty: {np.nanmedian(pj_empty_n):.1f}")

    def _corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if np.sum(m) < 3:
            return np.nan
        aa = a[m]
        bb = b[m]
        if np.std(aa) < eps or np.std(bb) < eps:
            return np.nan
        return float(np.corrcoef(aa, bb)[0, 1])

    if np.isfinite(pj).any():
        print(f"\nCorr(sf, pairwise_jaccard_cond_survival): {_corr(sf, pj):.3f}")
    if np.isfinite(pj).any() and np.isfinite(pj_empty).any():
        print(f"Corr(pairwise_jaccard_cond_survival, pairwise_jaccard_non_survival_empty): {_corr(pj, pj_empty):.3f}")
    if np.isfinite(sf).any() and np.isfinite(pj_empty).any():
        print(f"Corr(sf, pairwise_jaccard_non_survival_empty): {_corr(sf, pj_empty):.3f}")
    if np.isfinite(msz).any() and np.isfinite(pj).any():
        print(f"Corr(mean_cluster_size, pairwise_jaccard_cond_survival): {_corr(msz, pj):.3f}")

    # ------------ plots ------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), dpi=120)

    # Row 1: histograms
    ax = axes[0, 0]
    ax.hist(sf[np.isfinite(sf)], bins=bins)
    ax.set_title("survival_freq")
    ax.set_xlabel("fraction of shuffles")
    ax.set_ylabel("count")
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[0, 1]
    if np.isfinite(pj).any():
        ax.hist(pj[np.isfinite(pj)], bins=bins)
    ax.set_title("pairwise_jaccard_cond_survival")
    ax.set_xlabel("mean pairwise Jaccard")
    ax.set_ylabel("count")
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[0, 2]
    if np.isfinite(pj_empty).any():
        ax.hist(pj_empty[np.isfinite(pj_empty)], bins=bins)
    ax.set_title("pairwise_jaccard_non_survival_empty")
    ax.set_xlabel("mean pairwise Jaccard")
    ax.set_ylabel("count")
    ax.spines[['top', 'right']].set_visible(False)

    # Row 2 col 1: sf vs pj, color = mean_cluster_size
    ax = axes[1, 0]
    m = np.isfinite(sf) & np.isfinite(pj) & np.isfinite(msz)
    if np.any(m):
        sc = ax.scatter(sf[m], pj[m], c=msz[m], s=scatter_s, alpha=scatter_alpha, cmap=cmap)
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("mean_cluster_size")
    ax.set_title("sf vs pairwise_jaccard (color=mean_cluster_size)")
    ax.set_xlabel("survival_freq")
    ax.set_ylabel("pairwise_jaccard_cond_survival")
    ax.spines[['top', 'right']].set_visible(False)

    # Row 2 col 2: sf vs pj, color = n_pairs_used_cond_survival
    ax = axes[1, 1]
    m = np.isfinite(sf) & np.isfinite(pj) & np.isfinite(pj_n)
    if np.any(m):
        sc = ax.scatter(sf[m], pj[m], c=pj_empty[m], s=scatter_s, alpha=scatter_alpha, cmap=cmap)
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("pairwise_jaccard_non_survival_empty")
    ax.set_title("sf vs pairwise_jaccard (color=n_pairs)")
    ax.set_xlabel("survival_freq")
    ax.set_ylabel("pairwise_jaccard_cond_survival")
    ax.spines[['top', 'right']].set_visible(False)

    # Row 2 col 3: pj vs pj_empty, color = survival_freq
    ax = axes[1, 2]
    m = np.isfinite(pj) & np.isfinite(pj_empty) & np.isfinite(sf)
    if np.any(m):
        sc = ax.scatter(pj[m], pj_empty[m], c=sf[m], s=scatter_s, alpha=scatter_alpha, cmap=cmap)
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("survival_freq")

        lo = np.nanmin(np.r_[pj[m], pj_empty[m]])
        hi = np.nanmax(np.r_[pj[m], pj_empty[m]])
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1)
    ax.set_title("pairwise_jaccard vs jaccard_empty (color=sf)")
    ax.set_xlabel("pairwise_jaccard_cond_survival")
    ax.set_ylabel("pairwise_jaccard_non_survival_empty")
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()

    # ----------- Top sequences for inspection --------------
    top_sf = np.argsort(-sf)[:20]
    print("\nTop 20 sequences by survival_freq:")
    for i in top_sf:
        print(
            f"  i={i:5d}  sf={sf[i]:.3f}"
            + (f"  pj={pj[i]:.3f}" if np.isfinite(pj[i]) else "")
            + (f"  pjE={pj_empty[i]:.3f}" if np.isfinite(pj_empty[i]) else "")
            + (f"  pairs={int(pj_n[i])}" if np.isfinite(pj_n[i]) else "")
            + (f"  msz={msz[i]:.2f}" if np.isfinite(msz[i]) else "")
        )

    if np.isfinite(pj).any():
        top_pj = np.argsort(-np.nan_to_num(pj, nan=-1))[:20]
        print("\nTop 20 sequences by pairwise_jaccard_cond_survival:")
        for i in top_pj:
            print(
                f"  i={i:5d}  pj={pj[i]:.3f}  sf={sf[i]:.3f}"
                + (f"  pjE={pj_empty[i]:.3f}" if np.isfinite(pj_empty[i]) else "")
                + (f"  pairs={int(pj_n[i])}" if np.isfinite(pj_n[i]) else "")
                + (f"  msz={msz[i]:.2f}" if np.isfinite(msz[i]) else "")
            )



# ==========================================
# GMM clustering of survival scores
# ==========================================

def gmm_bic_clustering(
    X,
    k_range=range(1, 11),
    covariance_type="full",
    n_init=10,
    seed=0,
    reg_covar=1e-6,
):
    """
    Fit GMMs for K in k_range and return BIC values (plus fitted models if you want to inspect them).

    Returns
    -------
    out : dict
      - "scaler": fitted StandardScaler
      - "Xz": standardized data
      - "Ks": np.ndarray of tested K
      - "bics": np.ndarray of BIC scores aligned with Ks
      - "models": list of fitted GaussianMixture objects aligned with Ks
    """
    X = np.asarray(X)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    Ks = np.array(list(k_range), dtype=int)
    bics = np.empty_like(Ks, dtype=float)
    models = []

    for i, K in enumerate(Ks):
        gmm = GaussianMixture(
            n_components=K,
            covariance_type=covariance_type,
            n_init=n_init,
            random_state=seed,
            reg_covar=reg_covar,
        )
        gmm.fit(Xz)
        models.append(gmm)
        bics[i] = gmm.bic(Xz)

    return {"scaler": scaler, "Xz": Xz, "Ks": Ks, "bics": bics, "models": models}


def plot_bic_sweep(sweep):
    Ks, bics = sweep["Ks"], sweep["bics"]
    fig, ax = plt.subplots(figsize=(3,2))
    ax.plot(Ks, bics, marker="o")
    ax.set_xlabel("K (#components)")
    ax.set_ylabel("BIC")
    ax.set_xticks(Ks)
    PublicationStandard()
    plt.show()