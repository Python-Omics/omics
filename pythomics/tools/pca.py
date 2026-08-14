"""
NipalsPCA: a scikit-learn compatible PCA estimator ported from the R
package `mixOmics` (functions `pca()` and `nipals()`).
"""

from __future__ import annotations

import warnings
import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

logger = logging.getLogger(__name__)

try:
    check_array(np.zeros((1, 1)), ensure_all_finite="allow-nan")
    _NAN_KW = {"ensure_all_finite": "allow-nan"}
except TypeError:
    _NAN_KW = {"force_all_finite": "allow-nan"}


def _nipals(X, ncomp=2, max_iter=500, tol=1e-6):
    """Port of `nipals()` (R), including the notes that R emits via message()."""
    X = np.asarray(X, dtype=float)
    nr, nc = X.shape

    if ncomp is None:
        ncomp = 2
    ncomp = min(ncomp, min(nr, nc))

    if np.any(np.all(np.isnan(X), axis=0)) or np.any(np.all(np.isnan(X), axis=1)):
        raise ValueError(
            "Some rows or columns are completely empty (NaN). "
            "Remove them before running the PCA."
        )

    p = np.zeros((nc, ncomp))
    t_mat = np.zeros((nr, ncomp))
    eig = np.zeros(ncomp)

    is_na_X = np.isnan(X)
    na_X = np.any(is_na_X)
    X_iter = X.copy()

    for h in range(ncomp):
        if na_X:
            init_col = np.argmin(np.sum(np.isnan(X_iter), axis=0))
        else:
            init_col = np.argmax(np.nanvar(X_iter, axis=0))

        th = X_iter[:, init_col].copy()
        th[np.isnan(th)] = 0
        ph_old = np.full(nc, 1.0 / np.sqrt(nc))
        iter_count = 1
        diff = 1.0

        if na_X:
            X_aux = X_iter.copy()
            X_aux[is_na_X] = 0

        while diff > tol and iter_count <= max_iter:
            if na_X:
                ph_new = X_aux.T @ th
                Th = np.outer(th, np.ones(nc))
                Th[is_na_X] = 0
                th_cross = Th.T @ Th
                ph_new = ph_new / np.diag(th_cross)
            else:
                ph_new = (X_iter.T @ th) / (th @ th)

            ph_new = ph_new / np.sqrt(ph_new @ ph_new)

            if na_X:
                th = X_aux @ ph_new
                P = np.outer(ph_new, np.ones(nr))
                P[is_na_X.T] = 0
                ph_cross = P.T @ P
                th = th / np.diag(ph_cross)
            else:
                th = (X_iter @ ph_new) / (ph_new @ ph_new)

            diff = np.sum((ph_new - ph_old) ** 2)
            ph_old = ph_new.copy()
            iter_count += 1

        if iter_count > max_iter:
            warnings.warn(
                f"Maximum number of iterations reached for component {h + 1}."
            )
            logger.warning(
                f"Maximum number of iterations reached for component {h + 1}."
            )

        X_iter = X_iter - np.outer(th, ph_new)

        if h > 0:
            corr = np.corrcoef(th, t_mat[:, :h], rowvar=False)[0, 1:]
            max_corr = np.max(np.abs(corr)) if corr.size else 0.0
            if max_corr >= 0.05:
                warnings.warn(
                    f"Component {h + 1} is not orthogonal to the previous ones "
                    f"(max corr = {max_corr:.3f}). Consider filtering variables with "
                    "a high rate of missing data or imputing them."
                )
                logger.warning(
                    f"Component {h + 1} is not orthogonal to the previous ones "
                    f"(max corr = {max_corr:.3f}). Consider filtering variables with "
                    "a high rate of missing data or imputing them."
                )

        p[:, h] = ph_new
        t_mat[:, h] = th
        eig[h] = np.nansum(th * th)

    eig = np.sqrt(eig)
    for h in range(ncomp):
        if eig[h] != 0:
            t_mat[:, h] = t_mat[:, h] / eig[h]

    return {"eig": eig, "p": p, "t": t_mat}


def _scale_matrix(X, center=True, scale=False):
    """Port of R's `scale()` applied with boolean center/scale or vectors."""
    nr, nc = X.shape
    X = np.asarray(X, dtype=float)

    if isinstance(center, bool):
        cen = np.nanmean(X, axis=0) if center else np.zeros(nc)
    else:
        cen = np.array(center, dtype=float, copy=True)

    X_centered = X - cen

    if isinstance(scale, bool):
        if scale:
            sc = np.sqrt(np.nansum(X_centered**2, axis=0) / max(1, nr - 1))
            sc[sc == 0] = 1.0
        else:
            sc = np.ones(nc)
    else:
        sc = np.array(scale, dtype=float, copy=True)
        sc[sc == 0] = 1.0

    return X_centered / sc, cen, sc


def _helmert_basis(n):
    V = np.zeros((n, n - 1))
    for i in range(n - 1):
        j = i + 1
        factor = np.sqrt(1.0 / (j * (j + 1)))
        V[:j, i] = -factor
        V[j, i] = j * factor
    return V


def _clr_backtransfo(X):
    return _helmert_basis(X.shape[1])


def _within_variation(X, design):
    design = np.asarray(design).flatten()
    unique_groups = np.unique(design)
    Xw = X.copy()
    global_mean = np.nanmean(X, axis=0)
    for g in unique_groups:
        idx = design == g
        group_mean = np.nanmean(X[idx, :], axis=0)
        Xw[idx, :] = X[idx, :] - group_mean + global_mean
    return Xw


def _match_logratio(value):
    """Matches `value` (case-insensitive, allows unambiguous prefixes) against
    {'CLR', 'ILR', 'none'}. Unlike the original version, here it is properly
    validated (previously 'none' never matched due to case, and it relied on
    a default fallback)."""
    choices = ["CLR", "ILR", "none"]
    v = str(value).upper()
    matches = [c for c in choices if c.upper().startswith(v)]
    if len(matches) == 1:
        return matches[0]
    upper_choices = [c.upper() for c in choices]
    if v in upper_choices:
        return choices[upper_choices.index(v)]
    raise ValueError(
        f"'logratio' must be one of {choices} (or an unambiguous prefix); "
        f"received {value!r}."
    )


class NipalsPCA(BaseEstimator, TransformerMixin):
    """PCA with support for missing data via NIPALS (equivalent to
    `mixOmics::pca()` in R). When there are no NaN values, standard SVD is
    used; when there are, the iterative NIPALS algorithm is used.

    Note that mixOmics uses population variance, while this implementation
    follows Python standards and uses sample variance.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to extract.
    center : bool or array-like of shape (n_features,), default=True
        If True, centers each column at its mean. An explicit centering
        vector can also be provided.
    scale : bool or array-like of shape (n_features,), default=False
        If True, scales each column to unit standard deviation (with a
        prior check for zero variance). An explicit vector of scales can
        also be provided.
    max_iter : int, default=500
        Maximum number of iterations of the NIPALS algorithm per component.
    tol : float, default=1e-9
        Convergence tolerance of NIPALS.
    logratio : {'none', 'CLR', 'ILR'}, default='none'
        Log-ratio transformation to apply before the PCA (for compositional
        data). Allows unambiguous prefixes, case-insensitive.
    ilr_offset : float, default=0.001
        Offset added to the data before the ILR/CLR transformation.
    V : array-like of shape (n_features, n_features - 1), optional
        Helmert basis matrix for the ILR transformation. If None and
        `logratio='ILR'`, it is computed automatically.
    multilevel : array-like of shape (n_samples,), optional
        Design vector for the multilevel analysis (repeated measurements).

    Attributes
    ----------
    n_features_in_ : int
    n_components_ : int
    mean_ : ndarray of shape (n_features,)
    scale_ : ndarray of shape (n_features,)
    components_ : ndarray of shape (n_components, n_features)
        Loadings, scikit-learn convention (rows = components).
    loadings_ : ndarray of shape (n_features, n_components)
        The same loadings, R/mixOmics convention (columns = components).
    variates_ : ndarray of shape (n_samples, n_components)
        Scores/projection of the training data.
    sdev_ : ndarray of shape (n_components,)
    explained_variance_ : ndarray of shape (n_components,)
    explained_variance_ratio_ : ndarray of shape (n_components,)
    singular_values_ : ndarray of shape (n_components,)
    cum_var_ : ndarray of shape (n_components,)
    total_variance_ : float
    Xw_ : ndarray or None
        Data after removing between-level variation (only if `multilevel`
        was provided).
    """

    def __init__(
        self,
        n_components=2,
        center=True,
        scale=False,
        max_iter=500,
        tol=1e-9,
        logratio="none",
        ilr_offset=0.001,
        V=None,
        multilevel=None,
    ):
        self.n_components = n_components
        self.center = center
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.logratio = logratio
        self.ilr_offset = ilr_offset
        self.V = V
        self.multilevel = multilevel

    def fit(self, X, y=None):
        X = check_array(X, dtype=float, **_NAN_KW)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        n_components = self.n_components
        if n_components is None:
            n_components = min(n_samples, n_features)
        n_components = int(round(n_components))
        if n_components < 1 or n_components > min(n_samples, n_features):
            raise ValueError(
                "'n_components' is invalid: it must be between 1 and "
                "min(n_samples, n_features)."
            )

        if self.max_iter is None or self.max_iter < 1:
            raise ValueError("'max_iter' is invalid.")
        max_iter = int(round(self.max_iter))

        logratio = _match_logratio(self.logratio)
        if logratio != "none" and np.any(X < 0):
            raise ValueError(
                "X contains negative values; a log-ratio transformation "
                "cannot be applied."
            )

        if not isinstance(self.center, bool) and len(self.center) != n_features:
            raise ValueError("'center' does not match the number of columns in X.")
        if not isinstance(self.scale, bool) and len(self.scale) != n_features:
            raise ValueError("'scale' does not match the number of columns in X.")

        logger.info(f"Fitting NipalsPCA | n_components: {n_components} | X: {n_samples}x{n_features}")

        # -- log-ratio transformation -------------------------------------
        V = self.V
        if V is None and logratio == "ILR":
            V = _clr_backtransfo(X)

        if logratio in ("CLR", "ILR"):
            offset = self.ilr_offset if logratio == "ILR" else 0.0
            logger.debug(f"Applying {logratio} log-ratio transformation (offset={offset})")
            X_pos = X + offset
            geom_mean = np.exp(np.nanmean(np.log(X_pos), axis=1, keepdims=True))
            X_lr = np.log(X_pos / geom_mean)
            if logratio == "ILR":
                X = X_lr @ V
                if n_components > min(X.shape):
                    raise ValueError(
                        "'n_components' is too large for the data transformed "
                        "with ILR; use a smaller value."
                    )
            else:
                X = X_lr

        # -- multilevel analysis -------------------------------------------
        self.Xw_ = None
        if self.multilevel is not None:
            multilevel = np.asarray(self.multilevel)
            if multilevel.ndim > 1 and multilevel.shape[1] != 1:
                raise ValueError(
                    "'multilevel' must have a single column for the "
                    "repeated measurements."
                )
            if X.shape[0] != multilevel.shape[0]:
                raise ValueError("X and 'multilevel' must have the same number of rows.")
            logger.debug("Applying within-variation (multilevel) correction.")
            _, design_codes = np.unique(multilevel.ravel(), return_inverse=True)
            X = _within_variation(X, design_codes)
            self.Xw_ = X.copy()

        if isinstance(self.scale, bool) and self.scale:
            col_sd = np.nanstd(X, axis=0, ddof=1)
            if np.any(col_sd == 0):
                raise ValueError(
                    "Cannot scale: there are columns with zero variance."
                )

        X_scaled, cen, sc = _scale_matrix(X, center=self.center, scale=self.scale)

        has_na = np.any(np.isnan(X_scaled))
        variates = None
        if has_na:
            logger.debug("Using NIPALS iterative algorithm (missing data detected).")
            res = _nipals(X_scaled, ncomp=n_components, max_iter=max_iter, tol=self.tol)
            sdev_raw, loadings_raw = res["eig"], res["p"]
        elif logratio in ("CLR", "none"):
            logger.debug("Using SVD.")
            U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
            sdev_raw, loadings_raw = S, Vt.T
        else:  # ILR, complete data
            logger.debug("Using SVD (ILR, complete data).")
            U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
            sdev_raw = S[:n_components] / np.sqrt(max(1, n_samples - 1))
            loadings_raw = V @ Vt.T[:, :n_components]
            variates = U[:, :n_components] @ np.diag(S[:n_components])

        Xf = np.nan_to_num(X_scaled, nan=0.0)
        sdev = sdev_raw[:n_components] / np.sqrt(max(1, n_samples - 1))
        loadings = loadings_raw[:, :n_components]
        if variates is None:
            variates = Xf @ loadings

        # -- scikit-learn style attributes ---------------------------------
        self.n_components_ = n_components
        self.mean_ = cen
        self.scale_ = sc
        self.sdev_ = sdev
        self.loadings_ = loadings                 
        self.components_ = loadings.T             
        self.variates_ = variates
        self.rotation_ = loadings
        self.X_ = X_scaled
        self.logratio_ = logratio
        self.V_ = V

        var_tot = np.nansum(Xf**2) / max(1, n_samples - 1)
        self.total_variance_ = var_tot
        self.explained_variance_ = sdev**2
        self.explained_variance_ratio_ = self.explained_variance_ / var_tot
        self.singular_values_ = sdev * np.sqrt(max(1, n_samples - 1))
        self.cum_var_ = np.cumsum(self.explained_variance_ratio_)

        logger.info("NipalsPCA fit completed.")
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X, dtype=float, **_NAN_KW)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of columns than the one used in fit()."
            )
        logger.debug(f"Transforming {X.shape[0]} samples with {self.n_components_} components.")
        X_scaled = (X - self.mean_) / self.scale_
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        return X_scaled @ self.components_.T

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.variates_

    def inverse_transform(self, X_transformed):
        check_is_fitted(self, "components_")
        X_transformed = check_array(X_transformed, dtype=float)
        return X_transformed @ self.components_ * self.scale_ + self.mean_

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "components_")
        return np.array([f"PC{i + 1}" for i in range(self.n_components_)])
