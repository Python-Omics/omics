"""
Regularized Canonical Correlation Analysis (rcc) Module.

A Python implementation based on the 'rcc' module from the mixOmics R package.
Provides capabilities to perform RCC using both Ridge and Shrinkage regularization methods.

"""

from dataclasses import dataclass
from rcc_utilities import *

@dataclass
class RCCResult:
    cor: np.ndarray # Calculated canonical correlations ($cor).
    xcoef: np.ndarray # Canonical loadings/coefficients for matrix X ($loadings$X).
    ycoef: np.ndarray # Canonical loadings/coefficients for matrix Y ($loadings$Y).
    variates_x: np.ndarray # Canonical variates/scores for matrix X ($variates$X).
    variates_y: np.ndarray # Canonical variates/scores for matrix Y ($variates$Y).
    lambda1: float # Regularization parameter applied to matrix X.
    lambda2: float # Regularization parameter applied to matrix Y.
    ncomp: int # Number of canonical components computed.
    method: str # Regularization method used ('ridge' or 'shrinkage').
    x_names: list # Names assigned to features in matrix X.
    y_names: list # Names assigned to features in matrix Y.
    prop_expl_var_x : np.ndarray #Proportion of explained variance in X per component.
    prop_expl_var_y : np.ndarray # Proportion of explained variance in Y per component.


def _check_matrix(M, name, allow_nan=True):
    """
    Validate matrix shape, dimensions, and presence of infinite or missing values.

    Param:
    M: array-like; Matrix or 2D array to validate.
    name: str; Identifier name of the matrix for exception messages.
    allow_nan: bool, optional; Whether missing (NaN) values are allowed, by default True.

    Return:
    np.ndarray; Validated 2D float NumPy array.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"'{name}' does not match expected shape (n_samples, n_features).")
    if not allow_nan and not np.all(np.isfinite(M)):
        raise ValueError(f"Infinite or NA values found in '{name}' (not supported).")
    return M


def rcc(X, Y, ncomp=2, method="ridge", lambda1=0.0, lambda2=0.0):
    """
    Perform Regularized Canonical Correlation Analysis (rcc) between two datasets.

    Param:
    X: array-like of shape (n_samples, n_features_x); First input dataset.
    Y: array-like of shape (n_samples, n_features_y); Second input dataset.
    ncomp: int, optional; Number of canonical components to calculate and return, by default 2.
    method: {"ridge", "shrinkage"}, optional; Regularization method to use ('ridge' or 'shrinkage'), by default "ridge".
    lambda1: float, optional; Non-negative regularization parameter for X (0 corresponds to classical CCA), by default 0.0.
    lambda2: float, optional; Non-negative regularization parameter for Y (0 corresponds to classical CCA), by default 0.0.

    Return:
    RCCResult; Dataclass containing canonical correlations, loadings, variates, and metrics.
    """
    if method not in ("ridge", "shrinkage"):
        raise ValueError("Method needs to be 'ridge' or 'shrinkage'.")

    X = _check_matrix(X, "X", allow_nan=(method == "ridge"))
    Y = _check_matrix(Y, "Y", allow_nan=(method == "ridge"))

    n, p = X.shape
    n2, q = Y.shape
    if n != n2:
        raise ValueError("X and Y need to have same number of rows.")

    if ncomp is None or not np.isfinite(ncomp) or ncomp < 1:
        raise ValueError("Invalid value for 'ncomp'.")
    ncomp = int(round(ncomp))
    if ncomp > min(p, q):
        raise ValueError(f"'ncomp' must be between 1 and {min(p, q)}.")

    x_names = [f"X{i + 1}" for i in range(p)]
    y_names = [f"Y{i + 1}" for i in range(q)]

    if method == "ridge":
        if lambda1 < 0 or lambda2 < 0:
                raise ValueError("lambda1 and lambda2 must be non-negative.")

        # pairwise-complete covariances (equivalent to var()/cov(..., use="pairwise") in R)
        Cxx = cov_pairwise(X) + lambda1 * np.eye(p)
        Cyy = cov_pairwise(Y) + lambda2 * np.eye(q)
        Cxy = cov_pairwise(X, Y)

    else: # method == "shinkage"
        Cxx, lambda_x, sc_x, xs = cov_shrink(X, n, p)
        Cyy, lambda_y, sc_y, ys = cov_shrink(Y, n, q)
 
        corr_xy_emp = (xs.T @ ys) / (n - 1)
        Cxy = np.sqrt((1 - lambda_x) * (1 - lambda_y)) * np.outer(sc_x, sc_y) * corr_xy_emp
 
        # input lambdas are replaced by estimated lambdas
        lambda1, lambda2 = lambda_x, lambda_y

    # --- Cholesky factorization ---
    # R uses chol() -> UPPER triangular Rx such that Rx' Rx = Cxx.
    # NumPy uses cholesky() -> LOWER triangular Lx such that Lx Lx' = Cxx.
    # Note that Lx = Rx', therefore:
    #   t(Cxx.fac.inv) in R == Lx_inv (= inv(Lx))
    #   Cxx.fac.inv in R == Lx_inv.T
    Lx = np.linalg.cholesky(Cxx)
    Ly = np.linalg.cholesky(Cyy)
    Lx_inv = np.linalg.inv(Lx)
    Ly_inv = np.linalg.inv(Ly)

    mat = Lx_inv @ Cxy @ Ly_inv.T  # == t(Cxx.fac.inv) %*% Cxy %*% Cyy.fac.inv in R

    if p >= q:
        U, s, Vt = np.linalg.svd(mat, full_matrices=False)
        cor = s
        u = U[:, :ncomp]
        v = Vt.T[:, :ncomp]
        xcoef = Lx_inv.T @ u
        ycoef = Ly_inv.T @ v
    else:
        U2, s2, Vt2 = np.linalg.svd(mat.T, full_matrices=False)
        cor = s2
        v2 = Vt2.T[:, :ncomp]  # == result$v from svd(t(mat)) in R
        u2 = U2[:, :ncomp]  # == result$u from svd(t(mat)) in R
        xcoef = Lx_inv.T @ v2
        ycoef = Ly_inv.T @ u2

    Xc = X - np.nanmean(X, axis=0)
    Yc = Y - np.nanmean(Y, axis=0)
    Xc = np.where(np.isnan(Xc), 0.0, Xc)
    Yc = np.where(np.isnan(Yc), 0.0, Yc)

    variates_x = Xc @ xcoef
    variates_y = Yc @ ycoef

    prop_expl_var_x = explained_variance(X, variates_x, ncomp)
    prop_expl_var_y = explained_variance(Y, variates_y, ncomp)

    return RCCResult(
        cor=cor,
        xcoef=xcoef,
        ycoef=ycoef,
        variates_x=variates_x,
        variates_y=variates_y,
        lambda1=lambda1,
        lambda2=lambda2,
        ncomp=ncomp,
        method=method,
        x_names=x_names,
        y_names=y_names,
        prop_expl_var_x=prop_expl_var_x,
        prop_expl_var_y=prop_expl_var_y,
    )
