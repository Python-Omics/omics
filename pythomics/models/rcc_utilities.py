"""
Utility functions for Regularized Canonical Correlation Analysis (rcc).

Provides helper routines for pairwise-complete covariance calculations,
explained variance estimation, and shrinkage covariance estimations matching 
the internal operations of the mixOmics R package.
"""

import numpy as np

def cov_pairwise(A, B=None):
    """
    Compute pairwise-complete covariance matrix.

    Equivalent to R's `var(X, na.rm=TRUE, use="pairwise")` when `B` is None, or
    `cov(X, Y, use="pairwise")`. For each pair of features (i, j), only sample rows 
    where both features have observed (non-NaN) values are used, centering each column 
    by its subset mean.

    Param:
    A: array-like of shape (n_samples, n_features_a); First input dataset.
    B: array-like of shape (n_samples, n_features_b), optional; Second input dataset. 
    If None, computes the auto-covariance matrix of `A`.

    Return:
    np.ndarray of shape (n_features_a, n_features_b); Pairwise-complete covariance matrix.
    """

    A = np.asarray(A, dtype=float)
    symmetric = B is None
    B = A if symmetric else np.asarray(B, dtype=float)

    n, p = A.shape
    q = B.shape[1]

    # Fast path: if no NaNs, use standard covariance matrix formula
    if not np.any(np.isnan(A)) and not np.any(np.isnan(B)):
        Ac = A - A.mean(axis=0)
        Bc = B - B.mean(axis=0)
        return (Ac.T @ Bc) / (n - 1)

    C = np.empty((p, q))
    mask_A = ~np.isnan(A)
    mask_B = ~np.isnan(B)
    for i in range(p):
        mi = mask_A[:, i]
        ai_full = A[:, i]

        for j in range(q):
            if symmetric and j < i:
                C[i, j] = C[j, i]
                continue
            m = mi & mask_B[:, j]
            cnt = int(m.sum())
            if cnt < 2:
                C[i, j] = np.nan
                continue
            ai = ai_full[m] - ai_full[m].mean()
            bj = B[m, j] - B[m, j].mean()
            C[i, j] = (ai @ bj) / (cnt - 1)
    return C


def explained_variance(data, variates, ncomp):
    """
    Calculate the proportion of explained variance for each canonical component.

    Param:
    data: array-like of shape (n_samples, n_features); Input data matrix 
    (uncentered; NaN values are imputed to 0.0).
    variates: array-like of shape (n_samples, ncomp); Canonical variates /
    latent variables.
    ncomp: int; Number of canonical components evaluated.

    Return
    np.ndarray of shape (ncomp,); Proportion of variance explained by each component.
    """

    data = np.asarray(data, dtype=float)
    data = np.where(np.isnan(data), 0.0, data)
    norm2 = np.sum(data ** 2)  # ||data||_F^2

    expl_var = np.zeros(ncomp)
    for h in range(ncomp):
        t_h = variates[:, h]
        a = t_h @ data  # shape (p,)
        expl_var[h] = np.sum(a ** 2) / np.sum(t_h ** 2) / norm2
    return expl_var

def estimate_lambda_var(xc, n):
    """
    Estimate shrinkage parameter for feature variances (diagonal shrinkage target).

    Param:
    xc: np.ndarray of shape (n_samples, n_features); Column-centered data matrix.
    n: int; Number of samples.

    Return:
    lambda_var: float; Optimal variance shrinkage intensity between 0.0 and 1.0.
    target: float; Shrinkage target value (median empirical variance across columns).
    v: np.ndarray of shape (n_features,); Empirical variance for each feature (ddof=1).
    """

    # bias correction factors, w = rep(1/n, n)
    h1 = n / (n - 1) # 
    h1w2 = 1.0 / (n - 1) 
 
    v = h1 * np.mean(xc ** 2, axis=0) # empirical variance (ddof=1)
    target = np.median(v)
 
    zz = xc ** 2
    q1 = np.mean(zz, axis=0)
    q2 = np.mean(zz ** 2, axis=0) - q1 ** 2
    numerator = np.sum(q2)
    denominator = np.sum((q1 - target / h1) ** 2)
 
    if denominator == 0:
        lambda_var = 1.0
    else:
        lambda_var = max(0.0, min(1.0, numerator / denominator * h1w2))
 
    return lambda_var, target, v
 
 
def estimate_lambda_cor(xs, n, p):
    """
    Estimate shrinkage parameter for feature correlation matrix.

    Param:
    xs: np.ndarray of shape (n_samples, n_features); Standardized data matrix 
    (mean=0, variance=1, ddof=1).
    n: int; Number of samples.
    p: int; Number of features.

    Return:
    float: Optimal correlation shrinkage intensity (`lambda_cor`) between 0.0 and 1.0.
    """

    if p == 1:
        return 1.0
 
    h1w2 = 1.0 / (n - 1)
    sw = 1.0 / np.sqrt(n)  # sqrt(w), where w=1/n
 
    xsw = xs * sw
    r0 = xsw.T @ xsw 
    E2R = r0 ** 2
 
    xs2w = (xs ** 2) * sw
    ER2 = xs2w.T @ xs2w
 
    sE2R = np.sum(E2R) - np.trace(E2R)
    sER2 = np.sum(ER2) - np.trace(ER2)
 
    denominator = sE2R
    numerator = sER2 - sE2R
 
    if denominator == 0:
        lam = 1.0
    else:
        lam = max(0.0, min(1.0, numerator / denominator * h1w2))
 
    return lam
 
 
def cov_shrink(X, n, p):
    """
    Compute shrunk (regularized) covariance matrix using Ledoit-Wolf-style estimation.

    Param:
    X: array-like of shape (n_samples, n_features); Input dataset matrix.
    n: int; Number of samples (rows).
    p: int; Number of features (columns).

    Return:
    Cov_shrunk: np.ndarray of shape (p, p); Regularized covariance matrix.
    lambda_cor: float; Estimated correlation shrinkage intensity.
    sc: np.ndarray of shape (p,); Square root of shrunk variance per feature.
    xs: np.ndarray of shape (n, p); Standardized input dataset used in downstream cross-covariance estimations.
    """

    if n < 3:
        raise ValueError("Sample size too small.")
 
    xc = X - X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    xs = xc / std
 
    lambda_var, target_var, emp_var = estimate_lambda_var(xc, n)
    sv = lambda_var * target_var + (1 - lambda_var) * emp_var  # shrunk variance
    sc = np.sqrt(sv)
 
    lambda_cor = estimate_lambda_cor(xs, n, p)
    corr_emp = (xs.T @ xs) / (n - 1)
    corr_shrunk = (1 - lambda_cor) * corr_emp
    np.fill_diagonal(corr_shrunk, 1.0)
 
    Cov_shrunk = np.outer(sc, sc) * corr_shrunk
 
    return Cov_shrunk, lambda_cor, sc, xs
