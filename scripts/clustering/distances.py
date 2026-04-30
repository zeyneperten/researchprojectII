# Third-party imports
import numpy as np
from numba import njit, prange
from scipy.special import expit
from scipy.spatial.distance import squareform, pdist
from scipy.sparse import csr_matrix
#import editdistance

# =============================================================================
# Distance matrix computation
# =============================================================================

def seq_to_str(seq):
    s = ""
    for item in seq:
        s += str(item)
    return s

def edit_distance(seq1, seq2):
    "Compute the Levenshtein edit distance between two sequences."
    dist_edit = editdistance.eval(seq_to_str(seq1), seq_to_str(seq2))
    GLD = 2*dist_edit / (len(seq1) + len(seq2) + dist_edit)
    return GLD

def pairwise_edit_distance(seqs):
    "Compute pairwise edit distance matrix for a list of sequences."
    n = len(seqs)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            _, idx1, idx2 = np.intersect1d(seqs[i], seqs[j], return_indices=True, assume_unique=False)
            k = idx1.size
            
            # Only compute distance for pairs with min overlap k, distance is 1 otherwise.
            if k >= 5: # threshold for reliable correlation
                dist = edit_distance(seqs[i], seqs[j]) # idx1, idx2 seqs[i], seqs[j]
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # symmetric
            else:
                dist_matrix[i, j] = 1.0
                dist_matrix[j, i] = 1.0
    return squareform(dist_matrix)

def invert_sigmoid_expit(z, k=1.0, x0=0.0):
    "Invert sigmoid function (expit) to map similarity values to dissimilarity."
    z = np.array(z, dtype=float) 
    return expit(-k * (z - x0)) 


def zmat_to_dist(zmat, bmat):
    "Distance based on zmat."
    zmax = np.nanmax(zmat)
    zmat_th = zmat.copy()
    # Replace NaN values with -inf to map non overlapping sequences to high distances.
    zmat_th = np.nan_to_num(zmat_th, nan=-np.inf)
    # Replace values above the corresponding threshold with zmax for low distances
    zmat_th[bmat==1] = zmax
    zmat_th = invert_sigmoid_expit(zmat_th, k=1.0, x0=0.5)
    np.fill_diagonal(zmat_th, 0)
    return squareform(zmat_th)
    

def bmat_to_dist(zmat, bmat):
    "Distance based on bmat, originally used in method."
    # Create a clustering matrix (cmat) by keeping bmat values only where zmat is valid (not NaN)
    cmat = np.zeros_like(zmat)
    cmat[~np.isnan(zmat)] = bmat[~np.isnan(zmat)]
    # Compute the pairwise distance matrix from cmat.
    # pdist returns a condensed distance matrix.
    pdist_matrix = pdist(cmat)
    return pdist_matrix

def zmat_to_cmat(mat_dict):
    cmat = np.zeros_like(mat_dict["zmat"])
    m = ~np.isnan(mat_dict["zmat"])
    cmat[m] = mat_dict["bmat"][m]
    cmat = cmat.astype(bool, copy=False)
    return cmat

def bmat_to_dist(mat_dict, metric="euclidean"):
    cmat = zmat_to_cmat(mat_dict)
    pdist_matrix = pdist(cmat, metric=metric)
    return pdist_matrix

# condensed index for 0 <= i < j < n
@njit(inline='always')
def _condensed_index(n, i, j):
    return n*i - (i*(i+1))//2 + (j - i - 1)

# intersection size of two sorted int arrays
@njit(inline='always')
def _intersect_len(a, b):
    i = j = cnt = 0
    na, nb = a.size, b.size
    while i < na and j < nb:
        av, bv = a[i], b[j]
        if av == bv:
            cnt += 1; i += 1; j += 1
        elif av < bv:
            i += 1
        else:
            j += 1
    return cnt

@njit(parallel=True, fastmath=False) #fastmath=True
def _pdist_jaccard_csr_bool(n, indptr, indices, out_vec):
    for i in prange(n-1):
        ai = indices[indptr[i]:indptr[i+1]]
        ai_len = ai.size
        for j in range(i+1, n):
            bj = indices[indptr[j]:indptr[j+1]]
            inter = _intersect_len(ai, bj)
            if inter == 0:
                # union = ai_len + bj_len
                union = ai_len + (bj.size)
                # if both zero rows, define distance 0
                d = 0.0 if union == 0 else 1.0
            else:
                union = ai_len + (bj.size) - inter
                d = 1.0 - (inter / union)
            out_vec[_condensed_index(n, i, j)] = d

def pdist_jaccard_sparse_memmap(mat_dict, out_path, dtype=np.float32):
    # cmat_bool: (n,n) boolean or 0/1 array (rows = sets)
    cmat_bool = zmat_to_cmat(mat_dict)
    X = csr_matrix(cmat_bool.astype(np.bool_, copy=False))
    n = X.shape[0]
    m = n*(n-1)//2

    y = np.memmap(out_path, mode='w+', dtype=dtype, shape=(m,))
    _pdist_jaccard_csr_bool(n, X.indptr.astype(np.int64), X.indices.astype(np.int64), y)
    y.flush()
    return y  # condensed form, same layout as scipy.pdist

@njit(parallel=True, fastmath=False) #fastmath=True
def _pdist_euclid_csr_bool(n, indptr, indices, out_vec):
    for i in prange(n - 1):
        ai = indices[indptr[i]:indptr[i+1]]
        ai_len = ai.size
        for j in range(i + 1, n):
            bj = indices[indptr[j]:indptr[j+1]]
            bj_len = bj.size

            inter = _intersect_len(ai, bj)

            # squared L2 distance in 0/1 space:
            # ||a-b||^2 = |a| + |b| - 2*|intersection|
            dist2 = ai_len + bj_len - 2 * inter
            out_vec[_condensed_index(n, i, j)] = np.sqrt(dist2)

def pdist_euclid_sparse_memmap(mat_dict, out_path, dtype=np.float32):
    """
    Euclidean equivalent of scipy.spatial.distance.pdist(cmat, 'euclidean')
    for a bool matrix.
    """
    cmat_bool = zmat_to_cmat(mat_dict)
    X = csr_matrix(cmat_bool.astype(np.bool_, copy=False))
    n = X.shape[0]
    m = n * (n - 1) // 2

    y = np.memmap(out_path, mode='w+', dtype=dtype, shape=(m,))
    _pdist_euclid_csr_bool(
        n,
        X.indptr.astype(np.int64),
        X.indices.astype(np.int64),
        y
    )
    y.flush()
    return y  # condensed form, same layout as scipy.pdist