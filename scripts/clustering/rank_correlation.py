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
            z = (rc - mns[1]) / mns[2]
            b = 1.0 * (z > mns[3])
            b2 = 1.0 * (abs(z) >  mns[3]) 
            cblk[ii, jj] = rc
            zblk[ii, jj] = z
            bblk[ii, jj] = b
            bblk2[ii, jj] = b2

    return i0, i1, j0, j1, cblk, zblk, bblk, bblk2


def rankseq_fast(s1, s2):
    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    # common elements + positions
    _, idx1, idx2 = np.intersect1d(s1, s2, return_indices=True, assume_unique=False)
    k = idx1.size
    if k < 4: # threshold for reliable correlation
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

