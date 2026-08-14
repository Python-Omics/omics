"""
Helper functions for Mint.

based on: MixOmics/R/internal_mint.block_helpers.R

List of missing items:
- initsvd
- initialisation_by_svd
- soft
- binary_search
- soft_threshold
- soft_thresholding_l1
- rank
- sparsity

***Fix documentation + translate + tests***
"""

import numpy as np
from dataclasses import dataclass

def study_split(data, study, levels=None):
    """
    Splits a matrix into sub-matrices according to a 'study' grouping factor.
    By default (levels=None), levels are sorted alphabetically/numerically,
    same as R's default factor().
    To request a specific order, modify the levels parameter.

    Param:
    data: array-like, shape (n, p)
    study: array-like, shape (n,); study label of each row of 'data'
    levels: list, optional; order of the levels; by default, np.unique(study)

    Return:
    dict: {level: sub-matrix np.ndarray (n_level, p)}
    """
    data = np.asarray(data)
    study = np.asarray(study)
    if data.shape[0] != study.shape[0]:
        raise ValueError("'data' and 'study' must have the same number of rows.")

    if levels is None:
        levels = np.unique(study)

    return {lvl: data[study == lvl] for lvl in levels}


def l2_norm(x):
    """
    Normalizes a vector to L2 norm = 1.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a vector.")
    return x / np.sqrt(x @ x)


def norm2(vec):
    """
    L2 norm, with the same behavior as R: if the norm is 0, returns 0.05.
    """
    vec = np.asarray(vec, dtype=float)
    a = np.sqrt(np.sum(vec ** 2))
    if a == 0:
        a = 0.05
    return a


def cov2(x, y=None, bias=False):
    """
    Covariance (or variance if y=None) with NA handling via
    pairwise-complete observations (cov(x, use="pairwise.complete.obs") in R).

    Param:
    x : array-like, shape (n, p)
    y : array-like, shape (n, q), optional
    bias : bool; if True, applies the (n-1)/n correction
    (***divides by n instead of n-1***)

    Return:
    np.ndarray: (p, p) if y is None; (p, q) if y is given
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]

    if y is None:
        if not np.any(np.isnan(x)):
            C = np.cov(x, rowvar=False)
            if x.shape[1] == 1:
                C = np.array([[C]]) 
        else:
            C = pairwise_cov(x, x)
    else:
        y = np.asarray(y, dtype=float)
        if not np.any(np.isnan(x)) and not np.any(np.isnan(y)):
            xc = x - x.mean(axis=0)
            yc = y - y.mean(axis=0)
            C = (xc.T @ yc) / (n - 1)
        else:
            C = pairwise_cov(x, y)

    if bias:
        C = C * (n - 1) / n

    return C


def pairwise_cov(x, y):
    """
    Pairwise-complete covariance between columns of x and columns of y.
    """
    p, q = x.shape[1], y.shape[1]
    C = np.full((p, q), np.nan)
    for i in range(p):
        xi = x[:, i]
        for j in range(q):
            yj = y[:, j]
            mask = np.isfinite(xi) & np.isfinite(yj)
            m = mask.sum()
            if m < 2:
                continue
            xi_c = xi[mask] - xi[mask].mean()
            yj_c = yj[mask] - yj[mask].mean()
            C[i, j] = (xi_c @ yj_c) / (m - 1)
    return C


def miscrossprod(x, y):
    """
    Cross product ignoring NaN (sum(x*y, na.rm=TRUE) in R).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return np.nansum(x * y)

def tau_estimate(x):
    """
    Strimmer's analytical formula to estimate the optimal shrinkage
    intensity (tau) of the correlation matrix toward the identity.
    """
    x = np.asarray(x, dtype=float)
    n, p = x.shape

    corm = np.corrcoef(x, rowvar=False)
    xs = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    xs2 = xs ** 2

    v = (n / (n - 1) ** 3) * (xs2.T @ xs2 - (1 / n) * (xs.T @ xs) ** 2)
    np.fill_diagonal(v, 0.0)

    I = np.eye(p)
    d = (corm - I) ** 2

    tau = np.sum(v) / np.sum(d)
    tau = max(min(tau, 1.0), 0.0)
    return tau

def deflation(X, y, misdata_q=False, is_na_A_q=None, ind_NA=None):
    """
    Regresses each column of X on the vector y, and returns the
    coefficients of that regression (p) as the residual matrix (R = X - y*p').

    With complete data, each column of R is left exactly orthogonal to y
    (y' @ R ~= 0), because p is chosen as the least-squares projection
    of X onto y.

    Param:
    X: array-like, shape (n, p)
    y: array-like, shape (n,)
    misdata_q: bool; whether block X has missing data
    is_na_A_q: boolean np.ndarray, shape (n, p), required if misdata_q=True;
    NaN mask of X (the NaN of X are assumed to have ALREADY been replaced by 0
    in X itself, following the convention of internal_mint.block.R)
    ind_NA: array-like of column indices with at least one NaN, optional

    Return:
    (p, R): tuple of (np.ndarray shape (p,), np.ndarray shape (n, p))
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    p_cols = X.shape[1]

    if misdata_q:
        if is_na_A_q is None:
            raise ValueError("is_na_A_q is required when misdata_q=True.")

        loadings_temp = X.T @ y
        d_norm = np.full(p_cols, y @ y)

        if ind_NA is not None and len(ind_NA) > 0:
            ind_NA = np.asarray(ind_NA)
            temp = np.outer(y, np.ones(len(ind_NA)))
            temp[is_na_A_q[:, ind_NA]] = 0.0
            d_norm[ind_NA] = np.sum(temp ** 2, axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            p = loadings_temp / d_norm
        p = np.where(np.isnan(p), 0.0, p)
    else:
        p = (X.T @ y) / (y @ y)

    R = X - np.outer(y, p)
    return p, R


def defl_select(yy, rr, nncomp, nn, nbloc, indY=None, mode="canonical",
                 aa=None, misdata=None, is_na_A=None, ind_NA=None):
    """
    Applies the deflation corresponding to each block according to the 'mode'
    of the analysis:
    - canonical (default): each block (including Y) is deflated independently
      on its own variate.
    - classic: block Y is NOT deflated on its own variate; the
      X blocks are deflated using the average of the variates of the other
      blocks.
    - invariant: there is no deflation at all (the residuals are the original
      data unchanged).
    - regression: similar to "classic" but using the average of the
      residuals from deflating against each other block, not a direct average.

    Param:
    yy: np.ndarray, shape (n, nbloc); variates of all blocks in the current
    iteration (column q = block q)
    rr: list of np.ndarray, length nbloc; current residual matrices of
    each block
    nncomp: array-like, length nbloc; number of deflations to apply per
    block (typically ncomp-1)
    nn: int; current component, 1-indexed (so that 'nn <= nncomp[q]'
    matches the R convention)
    nbloc: int
    indY: int or None; 0-based index of block Y (subtract 1 when porting the
    indY from R)
    mode: {"canonical", "classic", "invariant", "regression"}
    aa: list of np.ndarray, required if mode="classic"; current loadings of
    each block
    misdata, is_na_A, ind_NA : per-block lists for NA handling (optional)

    Return:
    (resdefl, pdefl) : tuple of lists of length nbloc
    """
    if misdata is None:
        misdata = [False] * nbloc

    resdefl = [None] * nbloc
    pdefl = [None] * nbloc

    for q in range(nbloc):
        is_na_A_q = is_na_A[q] if misdata[q] else None
        ind_NA_q = ind_NA[q] if ind_NA is not None else None

        if nn <= nncomp[q]:
            if mode == "canonical" or q != indY:
                p, R = deflation(rr[q], yy[:, q], misdata[q], is_na_A_q, ind_NA_q)
                resdefl[q] = R
                pdefl[q] = p

            elif mode == "classic":
                others = [x for x in range(nbloc) if x != q]
                acc = np.zeros_like(rr[q])
                for x in others:
                    acc = acc + (rr[q] - np.outer(yy[:, x], aa[q]))
                resdefl[q] = acc / (nbloc - 1)
                pdefl[q] = np.zeros(rr[q].shape[1])

            elif mode == "invariant":
                resdefl[q] = rr[q]
                pdefl[q] = np.zeros(rr[q].shape[1])

            elif mode == "regression":
                others = [x for x in range(nbloc) if x != q]
                acc = np.zeros_like(rr[q])
                for x in others:
                    _, R_x = deflation(rr[q], yy[:, x], misdata[q], is_na_A_q, ind_NA_q)
                    acc = acc + R_x
                resdefl[q] = acc / (nbloc - 1)
                pdefl[q] = np.zeros(rr[q].shape[1])

            else:
                raise ValueError(f"unknown mode: {mode!r}")
        else:
            resdefl[q] = rr[q]
            pdefl[q] = np.zeros(rr[q].shape[1])

    return resdefl, pdefl



# -----------------------------------------------------
# Classes for Function Results:
# scale_function returns ScaleResult
# mean_centering_per_study returns MeanCenteringResult
# -----------------------------------------------------

@dataclass
class ScaleResult:
    data: np.ndarray # centered data (and scaled if scale=True)
    mean: np.ndarray # per-column mean (shape (p,))
    sigma: np.ndarray # per-column sd (shape (p,)), or None if scale=False

@dataclass
class MeanCenteringResult:
    data: np.ndarray # concatenated matrix, centered/scaled per study, same row order as the input
    means: dict # {study_level: vector of means (p,)}
    sigmas: dict # {study_level: vector of sd (p,)} or {study_level: None} if scale=False
    levels: list # study levels, in the order used


def scale_function(temp, scale=True):
    """
    Centers (and optionally scales to variance 1) a matrix, ignoring NaN
    column by column (colMeans(na.rm=TRUE) and matrixStats::colSds(na.rm=TRUE),
    which is the sample sd, ddof=1).
    """
    temp = np.asarray(temp, dtype=float)
    mean_x = np.nanmean(temp, axis=0)
    centered = temp - mean_x

    if scale:
        sqrt_sdx = np.nanstd(temp, axis=0, ddof=1)
        zero_sd = sqrt_sdx == 0
        with np.errstate(invalid="ignore", divide="ignore"):
            data_scale = centered / sqrt_sdx
        if np.any(zero_sd):
            # If a column has sd=0 (zero variance), the NaN resulting from
            # dividing by 0 is avoided by setting that column to 0 directly.
            data_scale[:, zero_sd] = 0.0
    else:
        sqrt_sdx = None
        data_scale = centered

    return ScaleResult(data=data_scale, mean=mean_x, sigma=sqrt_sdx)


def mean_centering_per_study(data, study, scale):
    """
    Centers (and optionally scales) the data per study: each level of
    'study' is centered/scaled using its own mean/sd.

    Param:
    data: array-like, shape (n, p)
    study: array-like, shape (n,)
    scale: bool
    Return:
    MeanCenteringResult
    """
    data = np.asarray(data, dtype=float)
    study = np.asarray(study)
    n = data.shape[0]

    levels = list(np.unique(study))
    concat = np.empty_like(data)
    means, sigmas = {}, {}

    for lvl in levels:
        mask = study == lvl
        sub = data[mask]
        res = scale_function(sub, scale=scale)
        concat[mask] = res.data
        means[lvl] = res.mean
        sigmas[lvl] = res.sigma

    return MeanCenteringResult(data=concat, means=means, sigmas=sigmas, levels=levels)