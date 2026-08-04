"""
mint/helpers.py

Port de las funciones genericas de array/estadistica de
internal_mint.block_helpers.R (mixOmics), lineas 90-97 (study_split),
331-337 (l2.norm), 176-183 (norm2), 372-389 (cov2), 396-400 (miscrossprod),
343-362 (tau.estimate):
https://github.com/mixOmicsTeam/mixOmics/blob/master/R/internal_mint.block_helpers.R
"""

import numpy as np


def study_split(data, study, levels=None):
    """
    Port de study_split() (R, linea 90).

    Divide una matriz en sub-matrices segun un factor de agrupacion 'study'
    (tipicamente el estudio/batch de cada muestra).

    R usa split.data.frame(), que agrupa segun los NIVELES del factor -- no
    segun el orden de aparicion. Por defecto (levels=None) replicamos eso
    con niveles ordenados alfabeticamente/numericamente, igual que el factor()
    por defecto de R. Si quieres un orden especifico (p.ej. el orden real de
    los estudios), pasalo explicitamente en 'levels'.

    Parameters
    ----------
    data : array-like, shape (n, p)
    study : array-like, shape (n,)
        etiqueta de estudio/grupo de cada fila de 'data'
    levels : list, opcional
        orden de los niveles; por defecto, np.unique(study) (orden ordenado)

    Returns
    -------
    dict: {nivel: sub-matriz np.ndarray (n_nivel, p)}
    """
    data = np.asarray(data)
    study = np.asarray(study)
    if data.shape[0] != study.shape[0]:
        raise ValueError("'data' y 'study' deben tener el mismo numero de filas.")

    if levels is None:
        levels = np.unique(study)

    return {lvl: data[study == lvl] for lvl in levels}


def l2_norm(x):
    """
    Port de l2.norm() (R, linea 331): normaliza un vector a norma L2 = 1.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x tiene que ser un vector (1D).")
    return x / np.sqrt(x @ x)


def norm2(vec):
    """
    Port de norm2() (R, linea 176): norma L2, con el mismo caso especial
    que R -- si la norma es exactamente 0, devuelve 0.05 en vez de 0 (para
    evitar division por cero mas adelante). Es una constante "magica" del
    codigo original de RGCCA, se replica tal cual para mantener paridad.
    """
    vec = np.asarray(vec, dtype=float)
    a = np.sqrt(np.sum(vec ** 2))
    if a == 0:
        a = 0.05
    return a


def cov2(x, y=None, bias=False):
    """
    Port de cov2() (R, linea 372): covarianza (o varianza si y=None) con
    manejo de NA via observaciones pairwise-completas (equivalente a
    cov(x, use="pairwise.complete.obs") de R).

    Si no hay NaN en los datos, usa la ruta rapida vectorizada (np.cov).
    Si hay NaN, cae a un calculo columna-por-columna/par-por-par mas lento
    pero correcto (igual que R: cada par de columnas usa solo las filas
    donde AMBAS columnas son finitas, dividiendo por (n_pares - 1)).

    Parameters
    ----------
    x : array-like, shape (n, p)
    y : array-like, shape (n, q), opcional
    bias : bool
        si True, aplica la correccion (n-1)/n (version "poblacional",
        dividir por n en vez de n-1)

    Returns
    -------
    np.ndarray: (p, p) si y is None, o (p, q) si y esta dado
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]

    if y is None:
        if not np.any(np.isnan(x)):
            C = np.cov(x, rowvar=False)
            if x.shape[1] == 1:
                C = np.array([[C]])  # np.cov colapsa a escalar con 1 columna
        else:
            C = _pairwise_cov(x, x)
    else:
        y = np.asarray(y, dtype=float)
        if not np.any(np.isnan(x)) and not np.any(np.isnan(y)):
            xc = x - x.mean(axis=0)
            yc = y - y.mean(axis=0)
            C = (xc.T @ yc) / (n - 1)
        else:
            C = _pairwise_cov(x, y)

    if bias:
        C = C * (n - 1) / n

    return C


def _pairwise_cov(x, y):
    """Covarianza pairwise-complete entre columnas de x y columnas de y."""
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
    Port de miscrossprod() (R, linea 396): producto cruzado ignorando NaN
    (equivalente a sum(x*y, na.rm=TRUE) de R).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return np.nansum(x * y)


def tau_estimate(x):
    """
    Port de tau.estimate() (R, linea 343): formula analitica de Strimmer
    para estimar la intensidad optima de encogimiento (tau) de la matriz
    de correlacion hacia la identidad. Se usa en sparse.rgcca_iteration
    cuando tau="optimal".

    OJO: esta es una implementacion DISTINTA (aunque con el mismo espiritu
    de Schafer-Strimmer) de la que ya portamos en rcc.py para
    method="shrinkage" -- son dos funciones independientes en el codigo
    fuente real de mixOmics (esta viene de RGCCA, la de rcc.R viene de
    corpcor), con formulas ligeramente distintas. Se replica cada una tal
    como aparece en su archivo fuente, sin intentar unificarlas.

    Nota: el codigo R original calcula una variable 'm' (linea 356,
    apply(xs2, 2, mean)) que nunca se usa en el resto de la funcion --
    es codigo muerto en el original. Se omite aqui a proposito.
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
