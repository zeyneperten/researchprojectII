import numpy as np

def add_within_clust_score(seqs_labels, zmat, bmat):
    """
    Evaluate clusters by computing mean within-cluster score.

    For each cluster, this function extracts the submatrix of the z-score 
    matrix corresponding to all members of that cluster and computes the mean 
    score (excluding the diagonal). The resulting values are added to the 
    `retval` dictionary.

    Parameters
    ----------
    seqs_labels : motif indeces
    zmat: z-score matrix returned by allmot
    bmat: binary matrix of significant correlations
    
    Returns
    -------
    mean_corr : list
        mean within-cluster z-score for each cluster.
    """
    
    ids_clust = seqs_labels
    within_score = []
    within_var = []
    
    zmat = np.nan_to_num(zmat, nan=0)
    # 0 where bmat_pm is 0, i.e. scores are not significant
    zmat[bmat == 0] = 0 #bmat_pm
    
    for cl in np.unique(ids_clust):
        # indices of sequences belonging to this cluster
        idx = np.where(ids_clust == cl)[0]
        if len(idx) < 2:
            # skip singletons (undefined mean z-score)
            within_score.append(np.nan)
            within_var.append(np.nan)
            continue
        
        # extract submatrix for this cluster
        submat = zmat[np.ix_(idx, idx)]
        # upper triangle indices (exclude diagonal)
        iu = np.triu_indices_from(submat, k=1)
        # compute mean z-score
        mean_val = np.nanmean(submat[iu])
        variance_val = np.nanvar(submat[iu])
        
        within_score.append(mean_val)
        within_score_arr = np.array(within_score)
        within_var.append(variance_val)
        within_var_arr = np.array(within_var)

    return within_score_arr, within_var_arr