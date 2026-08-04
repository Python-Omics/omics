"""
NipalsPCA: estimador de PCA compatible con scikit-learn, portado desde el
paquete R `mixOmics` (funciones `pca()` y `nipals()`).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

try:
    check_array(np.zeros((1, 1)), ensure_all_finite="allow-nan")
    _NAN_KW = {"ensure_all_finite": "allow-nan"}
except TypeError:
    _NAN_KW = {"force_all_finite": "allow-nan"}


def _nipals(X, ncomp=2, max_iter=500, tol=1e-6):
    """Port de `nipals()` (R), incluyendo los avisos que R emite via message()."""
    X = np.asarray(X, dtype=float)
    nr, nc = X.shape

    if ncomp is None:
        ncomp = 2
    ncomp = min(ncomp, min(nr, nc))

    if np.any(np.all(np.isnan(X), axis=0)) or np.any(np.all(np.isnan(X), axis=1)):
        raise ValueError(
            "Algunas filas o columnas estan completamente vacias (NaN). "
            "Elimínelas antes de ejecutar el PCA."
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
                f"Se alcanzo el numero maximo de iteraciones para el componente {h + 1}."
            )

        X_iter = X_iter - np.outer(th, ph_new)

        if h > 0:
            corr = np.corrcoef(th, t_mat[:, :h], rowvar=False)[0, 1:]
            max_corr = np.max(np.abs(corr)) if corr.size else 0.0
            if max_corr >= 0.05:
                warnings.warn(
                    f"El componente {h + 1} no es ortogonal a los anteriores "
                    f"(corr max = {max_corr:.3f}). Considere filtrar variables con "
                    "alta tasa de datos faltantes o imputarlos."
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
    """Port de `scale()` de R aplicado con center/scale booleanos o vectores."""
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
    """Empareja `value` (case-insensitive, admite prefijos no ambiguos) contra
    {'CLR', 'ILR', 'none'}. A diferencia de la version original, aqui SI se
    valida correctamente (antes 'none' nunca calzaba por mayusculas/minusculas
    y dependia de un fallback por defecto)."""
    choices = ["CLR", "ILR", "none"]
    v = str(value).upper()
    matches = [c for c in choices if c.upper().startswith(v)]
    if len(matches) == 1:
        return matches[0]
    upper_choices = [c.upper() for c in choices]
    if v in upper_choices:
        return choices[upper_choices.index(v)]
    raise ValueError(
        f"'logratio' debe ser uno de {choices} (o un prefijo no ambiguo); "
        f"se recibio {value!r}."
    )


class NipalsPCA(BaseEstimator, TransformerMixin):
    """PCA con soporte para datos faltantes via NIPALS (equivalente a
    `mixOmics::pca()` en R). Cuando no hay NaN se usa SVD estandar; cuando
    los hay, se usa el algoritmo iterativo NIPALS.

    Tener en consideración que mixOmics usa varianza poblacional y la implementación
    se adapta a los estándares de python usando varianza muestral.

    Parameters
    ----------
    n_components : int, default=2
        Numero de componentes a extraer.
    center : bool or array-like of shape (n_features,), default=True
        Si es True, centra cada columna en su media. Tambien puede
        entregarse un vector de centrado explicito.
    scale : bool or array-like of shape (n_features,), default=False
        Si es True, escala cada columna a desviacion estandar 1 (con un
        chequeo previo de varianza cero). Tambien puede entregarse un
        vector de escalas explicito.
    max_iter : int, default=500
        Maximo de iteraciones del algoritmo NIPALS por componente.
    tol : float, default=1e-9
        Tolerancia de convergencia de NIPALS.
    logratio : {'none', 'CLR', 'ILR'}, default='none'
        Transformacion log-ratio a aplicar antes del PCA (para datos
        composicionales). Admite prefijos no ambiguos, case-insensitive.
    ilr_offset : float, default=0.001
        Offset sumado a los datos antes de la transformacion ILR/CLR.
    V : array-like of shape (n_features, n_features - 1), optional
        Matriz de base de Helmert para la transformacion ILR. Si es None
        y `logratio='ILR'`, se calcula automaticamente.
    multilevel : array-like of shape (n_samples,), optional
        Vector de diseño para el analisis multinivel (medidas repetidas).

    Attributes
    ----------
    n_features_in_ : int
    n_components_ : int
    mean_ : ndarray of shape (n_features,)
    scale_ : ndarray of shape (n_features,)
    components_ : ndarray of shape (n_components, n_features)
        Loadings, convencion scikit-learn (filas = componentes).
    loadings_ : ndarray of shape (n_features, n_components)
        Los mismos loadings, convencion R/mixOmics (columnas = componentes).
    variates_ : ndarray of shape (n_samples, n_components)
        Scores/proyeccion de los datos de entrenamiento.
    sdev_ : ndarray of shape (n_components,)
    explained_variance_ : ndarray of shape (n_components,)
    explained_variance_ratio_ : ndarray of shape (n_components,)
    singular_values_ : ndarray of shape (n_components,)
    cum_var_ : ndarray of shape (n_components,)
    total_variance_ : float
    Xw_ : ndarray or None
        Datos tras remover variacion inter-nivel (solo si `multilevel` se
        entrego).
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
                "'n_components' invalido: debe estar entre 1 y "
                "min(n_samples, n_features)."
            )

        if self.max_iter is None or self.max_iter < 1:
            raise ValueError("'max_iter' invalido.")
        max_iter = int(round(self.max_iter))

        logratio = _match_logratio(self.logratio)
        if logratio != "none" and np.any(X < 0):
            raise ValueError(
                "X contiene valores negativos, no se puede aplicar una "
                "transformacion log-ratio."
            )

        if not isinstance(self.center, bool) and len(self.center) != n_features:
            raise ValueError("'center' no coincide con el numero de columnas de X.")
        if not isinstance(self.scale, bool) and len(self.scale) != n_features:
            raise ValueError("'scale' no coincide con el numero de columnas de X.")

        # -- transformacion log-ratio --------------------------------------
        V = self.V
        if V is None and logratio == "ILR":
            V = _clr_backtransfo(X)

        if logratio in ("CLR", "ILR"):
            offset = self.ilr_offset if logratio == "ILR" else 0.0
            X_pos = X + offset
            geom_mean = np.exp(np.nanmean(np.log(X_pos), axis=1, keepdims=True))
            X_lr = np.log(X_pos / geom_mean)
            if logratio == "ILR":
                X = X_lr @ V
                if n_components > min(X.shape):
                    raise ValueError(
                        "'n_components' es demasiado grande para los datos "
                        "transformados con ILR; use un valor menor."
                    )
            else:
                X = X_lr

        # -- analisis multinivel --------------------------------------------
        self.Xw_ = None
        if self.multilevel is not None:
            multilevel = np.asarray(self.multilevel)
            if multilevel.ndim > 1 and multilevel.shape[1] != 1:
                raise ValueError(
                    "'multilevel' debe tener una unica columna para las "
                    "medidas repetidas."
                )
            if X.shape[0] != multilevel.shape[0]:
                raise ValueError("X y 'multilevel' deben tener el mismo numero de filas.")
            _, design_codes = np.unique(multilevel.ravel(), return_inverse=True)
            X = _within_variation(X, design_codes)
            self.Xw_ = X.copy()

        if isinstance(self.scale, bool) and self.scale:
            col_sd = np.nanstd(X, axis=0, ddof=1)
            if np.any(col_sd == 0):
                raise ValueError(
                    "No se puede escalar: hay columnas con varianza cero."
                )

        X_scaled, cen, sc = _scale_matrix(X, center=self.center, scale=self.scale)

        has_na = np.any(np.isnan(X_scaled))
        variates = None
        if has_na:
            res = _nipals(X_scaled, ncomp=n_components, max_iter=max_iter, tol=self.tol)
            sdev_raw, loadings_raw = res["eig"], res["p"]
        elif logratio in ("CLR", "none"):
            U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
            sdev_raw, loadings_raw = S, Vt.T
        else:  # ILR, datos completos
            U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
            sdev_raw = S[:n_components] / np.sqrt(max(1, n_samples - 1))
            loadings_raw = V @ Vt.T[:, :n_components]
            variates = U[:, :n_components] @ np.diag(S[:n_components])

        Xf = np.nan_to_num(X_scaled, nan=0.0)
        sdev = sdev_raw[:n_components] / np.sqrt(max(1, n_samples - 1))
        loadings = loadings_raw[:, :n_components]
        if variates is None:
            variates = Xf @ loadings

        # -- atributos estilo scikit-learn ----------------------------------
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

        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X, dtype=float, **_NAN_KW)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X tiene un numero de columnas distinto al usado en fit()."
            )
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
