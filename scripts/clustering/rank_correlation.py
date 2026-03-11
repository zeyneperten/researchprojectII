# Third-party imports
import numpy as np

# Local imports
# Note: core import moved to functions to avoid circular dependency
from ..data_utils import load_nrm

# =============================================================================
# Rank correlation
# =============================================================================

def _block_pairs_upper_full(n, bs=256):
    blocks = []
    for i0 in range(0, n, bs):
        i1 = min(n, i0 + bs)
        # diagonal block
        blocks.append((i0, i1, i0, i1))
        # off-diagonal blocks to the right
        for j0 in range(i1, n, bs):
            j1 = min(n, j0 + bs)
            blocks.append((i0, i1, j0, j1))
    return blocks

def _compute_block_ret(i0, i1, j0, j1, seqs, nrm=None):
    Bi, Bj = i1 - i0, j1 - j0
    cblk = np.zeros((Bi, Bj), dtype=np.float64)
    zblk = np.zeros((Bi, Bj), dtype=np.float64)
    bblk = np.zeros((Bi, Bj), dtype=np.float64)
    bblk2 = np.zeros((Bi, Bj), dtype=np.float64) 

    for ii in range(Bi):
        i = i0 + ii
        s1 = seqs[i]
        for jj in range(Bj):
            j = j0 + jj
            if i >= j:    # strict upper triangle only
                continue
            s2 = seqs[j]
            rc, ln = rankseq_fast(s1, s2)
            mns = choose_nrm_param(ln, nrm=nrm)   # [L, mean, std, thr]
            z = (rc - mns[1]) / mns[2]                   # match original (no guard)
            b = 1.0 * (z > mns[3])
            b2 = 1.0 * (abs(z) >  mns[3]) 
            cblk[ii, jj] = rc
            zblk[ii, jj] = z
            bblk[ii, jj] = b
            bblk2[ii, jj] = b2

    return i0, i1, j0, j1, cblk, zblk, bblk, bblk2

def rankseq(s1, s2):
    """
    Compute the Spearman rank order correlation coefficient between two sequences.

    This function calculates the rank order correlation between two sequences, `s1` and `s2`,
    by first determining the overlap between their elements (interpreted as neurons) and then
    computing the Spearman correlation between the ordinal positions of the overlapping elements.
    If one sequence is shorter than the other, it is used as the reference for ranking.

    Parameters
    ----------
    s1 : array_like
        First input sequence (e.g., a list or 1D numpy array).
    s2 : array_like
        Second input sequence (e.g., a list or 1D numpy array).

    Returns
    -------
    rc : float
        Spearman rank correlation coefficient between the two sequences. Returns NaN if the
        correlation cannot be computed (e.g., due to insufficient overlapping elements).
    ln : int or float
        The number of overlapping elements (neurons) used in the correlation computation.
        Returns NaN if the correlation is not computed.
    """
    # Ensure the sequences are flattened numpy arrays.
    s1 = np.array(s1).flatten()
    s2 = np.array(s2).flatten()
    l1 = len(s1)
    l2 = len(s2)
    
    # Create a difference matrix where each element compares an element from s1 to an element from s2.
    d = np.ones((l1, 1)) * s2 - (np.ones((l2, 1)) * s1).transpose()
    # Convert the difference matrix to a binary identity matrix (True where elements are equal).
    d = (d == 0)
    
    # Choose the shorter sequence as 's0' and the longer as 's'. Adjust the difference matrix accordingly.
    s = s1
    s0 = s2
    ln = l1
    if l1 < l2:
        s = s2
        s0 = s1
        ln = l2
        d = d.transpose()
        
    # Remove elements from s that are not in the overlapping set.
    minseq = s[np.where(np.sum(d, axis=1) > 0)[0]]
    
    # Remove elements from s0 that are not in the overlapping set.
    d0 = np.ones((len(s0), 1)) * minseq - (np.ones((len(minseq), 1)) * s0).transpose()
    d0 = (d0 == 0)
    s0 = s0[np.sum(d0, axis=1) > 0]
    
    # Prepare a matrix to determine ordinal ranking differences.
    dd = np.ones((len(minseq), 1)) * s0 - (np.ones((len(s0), 1)) * minseq).transpose()
    
    # Compute Spearman's rank correlation coefficient if there is more than 4 overlapping element.
    if len(dd) > 4: #1
        ids = np.argmin(np.abs(dd), axis=0)
        rc = np.corrcoef(np.arange(len(ids)), ids)[0, 1]
        ln = len(ids)
    else:
        rc = np.nan
        ln = np.nan
    
    return rc, ln

def rankseq_fast(s1, s2):
    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    # common elements + positions
    _, idx1, idx2 = np.intersect1d(s1, s2, return_indices=True, assume_unique=False)
    k = idx1.size
    if k <= 4:
        return np.nan, np.nan
    s1r = s1[np.sort(idx1)]
    s2r = s2[np.sort(idx2)]
    dd = np.ones((len(s2r), 1)) * s1r - (np.ones((len(s1r), 1)) * s2r).transpose()
    y = np.argmin(np.abs(dd), axis=0)
    x = np.arange(k, dtype=int)
    
    #rc = np.corrcoef(x, y)[0, 1]
    xm = (k - 1) / 2.0
    ym = y.mean()
    num = np.dot(x - xm, y - ym)
    den = np.sqrt(np.dot(x - xm, x - xm) * np.dot(y - ym, y - ym))
    rc = num / den if den > 0 else np.nan
    return rc, k


def choose_nrm_param(ln, nrm=None):
    """
    Pick a row of normalization parameters from `nrm` that matches overlap length `ln`.

    Parameters
    ----------
    ln : int
        Overlap length to match.

    Returns
    -------
    ndarray
        1-D array containing the selected normalization parameters. If no match is found
        (and ln < 50) returns an array of NaNs with length equal to the parameter width.
        If ln >= 50, returns the last row of `nrm`.
    """
    if nrm is None:
        nrm = load_nrm()
    # Extract the first column from nrm to be used for matching the overlap length.
    narr = np.array(nrm)[:, 0]
    if ln >= 50:
        mns = nrm[-1]
    else:
        # Find the matching normalization parameters for the given overlap length.
        whichone = np.array(np.where(ln == narr)).flatten()
        if len(whichone) == 0:
            mns = np.empty(4)
            mns[:] = np.nan
        else:
            mns = nrm[whichone[0]]
    return mns

