"""
Funciones Helpers para Mint.

basadas en: MixOmics/R/internal_mint.block_helpers.R

Lista de faltantes:
- initsvd
- initialisation_by_svd

***Fixear documentación + traducir + tests***
"""

import numpy as np
import warnings
from dataclasses import dataclass

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
    np.ndarray: (p, p) si y is None; (p, q) si y esta dado
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
    Producto cruzado ignorando NaN (sum(x*y, na.rm=TRUE) de R).
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
    mascara de NaN de X (se asume que los NaN de X ya fueron reemplazados por 0
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
    - canonical (default): cada bloque (incluido Y) se defla independientemente
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

def soft(x, d):
    """
    Operador de soft-thresholding (sign(x) * max(0, |x| - d).
    """
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(0.0, np.abs(x) - d)

def binary_search(argu, sumabs):
    """
    Binary Search que busca 'd' (umbral de soft-thresholding) tal
    que la norma L1 del vector soft-thresholdeado y renormalizado 
    a norma L2=1 sea igual a 'sumabs'.

    Param:
    argu: array-like; vector de loadings a penalizar
    sumabs: float; restriccion de norma L1 objetivo (entre 1 y sqrt(len(argu)))

    Return:
    float: el umbral 'd' encontrado (0 si ya se cumple la restriccion sin penalizar)
    """
    argu = np.asarray(argu, dtype=float)

    if norm2(argu) == 0 or np.sum(np.abs(argu / norm2(argu))) <= sumabs:
        return 0.0

    lam1 = 0.0
    lam2 = np.max(np.abs(argu)) - 1e-5
    iter_ = 1
    while iter_ < 500:
        su = soft(argu, (lam1 + lam2) / 2)
        if np.sum(np.abs(su / norm2(su))) < sumabs:
            lam2 = (lam1 + lam2) / 2
        else:
            lam1 = (lam1 + lam2) / 2
        if (lam2 - lam1) < 1e-10:
            return (lam1 + lam2) / 2
        iter_ += 1

    
    warnings.warn("Didn't quite converge.")
    return (lam1 + lam2) / 2

def soft_threshold(x, sumabs=1.0):
    """
    Aplica soft-thresholding con el umbral que satisface la restriccion de norma L1 'sumabs'.
    """
    return soft(x, binary_search(x, sumabs))

def soft_thresholding_l1(x, nx):
    """
    Pone a cero las 'nx' variables con menor valor absoluto en x, y encoge
    las restantes hacia cero en la magnitud del mayor valor absoluto
    eliminado.E

    Param:
    x: array-like; vector de loadings
    nx: int; numero de variables a poner en cero (= len(x) - keepX)

    Returns
    np.ndarray: loadings penalizados
    """
    x = np.asarray(x, dtype=float).copy()

    if nx > 0:
        abs_a = np.abs(x)
        rank_max = rank_max(abs_a)
        select_feature = rank_max > nx
        if not np.all(select_feature):
            thresh = np.max(abs_a[~select_feature]) if np.any(~select_feature) else 0.0
            x = np.where(select_feature, np.sign(x) * (abs_a - thresh), 0.0)

    return x


def rank_max(a):
    """
    rank(a, ties.method="max") de R: a los valores empatados se les
    asigna el rango mas alto del grupo.
    """
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    n = len(a)
    ranks_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        # rango maximo del grupo empatado [i, j] es (j+1) en base-1
        ranks_sorted[i:j + 1] = j + 1
        i = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks

def sparsity(loadings_a, keep_a=None, penalty=None):
    """
    Sparsity correspondiente segun se especifique keep_a (conteo) o
    penalty (norma L1 continua). Si ambos son None, no hace nada.
    """
    loadings_a = np.asarray(loadings_a, dtype=float)

    if keep_a is not None:
        nx = len(loadings_a) - keep_a
        loadings_a = soft_thresholding_l1(loadings_a, nx=nx)
    elif penalty is not None:
        loadings_a = soft_threshold(loadings_a, penalty)

    return loadings_a



# -----------------------------------------------------
# Clases para Resultados de Funciones:
# scale_function returns ScaleResult
# mean_centering_per_study returns MeanCenteringResult
# -----------------------------------------------------

@dataclass
class ScaleResult:
    data: np.ndarray # datos centrados (y escalados si scale=True)
    mean: np.ndarray # media por columna (shape (p,))
    sigma: np.ndarray # sd por columna (shape (p,)), o None si scale=False

@dataclass
class MeanCenteringResult:
    data: np.ndarray # matriz concatenada, centrada/escalada por estudio, mismo orden de filas que el input
    means: dict # {nivel_estudio: vector de medias (p,)}
    sigmas: dict # {nivel_estudio: vector de sd (p,)} o {nivel_estudio: None} si scale=False
    levels: list # niveles de estudio, en el orden usado


def scale_function(temp, scale=True):
    """
    Centra (y opcionalmente escala a varianza 1) una matriz, ignorando NaN
    columna por columna (colMeans(na.rm=TRUE) y matrixStats::colSds(na.rm=TRUE),
    que es la sd muestral, ddof=1).
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
            # Si alguna columna tiene sd=0 (varianza nula), se evita el NaN 
            # resultante de dividir por 0 poniendo esa columna en 0 directamente.
            data_scale[:, zero_sd] = 0.0
    else:
        sqrt_sdx = None
        data_scale = centered

    return ScaleResult(data=data_scale, mean=mean_x, sigma=sqrt_sdx)


def mean_centering_per_study(data, study, scale):
    """
    Centra (y opcionalmente escala) los datos por estudio: cada nivel de
    'study' se centra/escala usando su propia media/sd.

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