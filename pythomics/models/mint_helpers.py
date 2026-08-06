"""
Funciones Helpers para Mint.

basadas en: MixOmics/R/internal_mint.block_helpers.R

Lista de faltantes:
- initsvd
- initialisation_by_svd
- scale_function
- mean_centering_per_study
- soft
- binary_search
- soft_threshold
- soft_thresholding_l1
- rank
- sparsity

***Fixear documentación + traducir + tests***
"""

import numpy as np


def study_split(data, study, levels=None):
    """
    Divide una matriz en sub-matrices según un factor de agrupacion 'study'
    Por defecto (levels=None), niveles ordenados alfabeticamente/numericamente, 
    igual que el factor() por defecto de R.
    Si se quiere un orden específico, se modifica el param levels.

    Param:
    data: array-like, shape (n, p)
    study: array-like, shape (n,); etiqueta de estudio de cada fila de 'data'
    levels: list, opcional; orden de los niveles; por defecto, np.unique(study)

    Return:
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
    Normaliza un vector a norma L2 = 1.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x tiene que ser un vector.")
    return x / np.sqrt(x @ x)


def norm2(vec):
    """
    Norma L2, con el mismo caso que R: si la norma = 0, devuelve 0.05.
    """
    vec = np.asarray(vec, dtype=float)
    a = np.sqrt(np.sum(vec ** 2))
    if a == 0:
        a = 0.05
    return a


def cov2(x, y=None, bias=False):
    """
    Covarianza (o varianza si y=None) con manejo de NA via observaciones 
    pairwise-completas (cov(x, use="pairwise.complete.obs") en R).

    Param:
    x : array-like, shape (n, p)
    y : array-like, shape (n, q), opcional
    bias : bool; si True, aplica la correccion (n-1)/n 
    (***divide por n en vez de n-1***)

    Return:
    np.ndarray: (p, p) si y:is None; (p, q) si y:esta dado
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
    Covarianza pairwise-complete entre columnas de x y columnas de y.
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
    Producto cruzado ignorando NaN
    (sum(x*y, na.rm=TRUE) de R).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return np.nansum(x * y)

def tau_estimate(x):
    """
    Fórmula analítica de Strimmer para estimar la intensidad optima de 
    encogimiento (tau) de la matriz de correlacion hacia la identidad.
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
    Regresa cada columna de X sobre el vector y, y devuelve los
    coeficientes de esa regresion (p) como la matriz residual (R = X - y*p').

    Con datos completos, cada columna de R queda exactamente ortogonal a y 
    (y' @ R ~= 0), porque p se elige como la proyeccion minimos-cuadrados 
    de X sobre y.

    Param:
    X: array-like, shape (n, p)
    y: array-like, shape (n,)
    misdata_q: bool; si el bloque X tiene datos faltantes
    is_na_A_q: np.ndarray booleano, shape (n, p), requerido si misdata_q=True;
    mascara de NaN de X (se asume que los NaN de X YA fueron reemplazados por 0
    en X mismo, siguiendo la convencion de internal_mint.block.R)
    ind_NA: array-like de indices de columna con al menos un NaN, opcional

    Return:
    (p, R): tupla de (np.ndarray shape (p,), np.ndarray shape (n, p))
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    p_cols = X.shape[1]

    if misdata_q:
        if is_na_A_q is None:
            raise ValueError("is_na_A_q es necesario cuando misdata_q=True.")

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
    Aplica la deflacion correspondiente a cada bloque segun el 'mode' del
    análisis:
    - canonical: cada bloque (incluido Y) se defla independientemente
      sobre su propia variate.
    - classic: el bloque Y NO se defla sobre su propia variate; los
      bloques X se deflan usando el promedio de las variates de los otros
      bloques.
    - invariant: no hay deflacion en absoluto (los residuos son los datos
      originales sin cambios).
    - regression: similar a "classic" pero usando el promedio de los
      RESIDUOS de deflactar contra cada otro bloque, no un promedio directo.

    Param:
    yy: np.ndarray, shape (n, nbloc); variates de todos los bloques en la 
    iteracion actual (columna q = bloque q)
    rr: list de np.ndarray, longitud nbloc; matrices residuales actuales de
    cada bloque
    nncomp: array-like, longitud nbloc; numero de deflaciones a aplicar por
    bloque (tipicamente ncomp-1)
    nn: int; componente actual, 1-indexado (para que 'nn <= nncomp[q]' 
    calce con la convencion de R)
    nbloc: int
    indY: int o None; indice 0-based del bloque Y (resta 1 al portar el indY
    de R)
    mode: {"canonical", "classic", "invariant", "regression"}
    aa: list de np.ndarray, requerido si mode="classic"; loadings actuales de
    cada bloque
    misdata, is_na_A, ind_NA : listas por bloque para manejo de NA (opcional)

    Return:
    (resdefl, pdefl) : tupla de listas de longitud nbloc
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
                raise ValueError(f"mode desconocido: {mode!r}")
        else:
            resdefl[q] = rr[q]
            pdefl[q] = np.zeros(rr[q].shape[1])

    return resdefl, pdefl
