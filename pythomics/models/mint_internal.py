from .mint_helpers import *

"""
basado en : internal_mint.block.R 

"""

# --------------------------
# internal_mint.block()
# --------------------------

@dataclass
class InternalMintBlockResult:
    A: list # bloques centrados/escalados (los datos usados realmente)
    indY: int
    ncomp: list
    mode: str
    variates: list # lista de J arrays (n, ncomp[k])
    loadings: list # lista de J arrays (p_k, ncomp[k])
    variates_partial: list # lista de J dicts {nivel_estudio: array}, o None si tau no es None
    loadings_partial: list # lista de J dicts {nivel_estudio: array}, o None si tau no es None
    study: np.ndarray
    crit: list # criterio de convergencia por componente
    iter: list # iteraciones hasta converger, por componente
    tau: list # tau usado (o estimado) por componente, o None si tau=None
    design: np.ndarray
    scheme: str
    tol: float
    max_iter: int
    scale: bool


def _build_tau_matrix(tau, N, J):
    """
    Replica:
    if (is.vector(tau))
        tau = matrix(rep(tau, N), nrow = N, ncol = length(tau), byrow = TRUE)

    si tau es un vector/valor unico, se replica igual para los N
    componentes. Si ya viene dado componente-por-componente (una lista de
    longitud N), se usa tal cual.
    """
    if isinstance(tau, str):
        if tau != "optimal":
            raise ValueError("tau debe ser 'optimal', un vector de largo J, "
                              "o una lista de longitud N de tales vectores.")
        return ["optimal"] * N

    # scalar tau (e.g. tau=0.5) means the same fixed value for every block and component
    if np.isscalar(tau):
        return [[float(tau)] * J for _ in range(N)]

    tau_list = list(tau)
    # si el primer elemento es a su vez una lista/array de longitud J (o el string "optimal"),
    #  asumimos que ya es "por componente"
    first = tau_list[0]
    already_per_comp = isinstance(first, str) or hasattr(first, "__len__")

    if already_per_comp:
        if len(tau_list) != N:
            raise ValueError(f"tau por componente debe tener longitud N={N}.")
        return tau_list
    else:
        if len(tau_list) != J:
            raise ValueError(f"tau debe tener longitud J={J} (uno por bloque).")
        return [list(tau_list)] * N


def internal_mint_block(A, indY=None, design=None, ncomp=None, scheme="horst", scale=True, init="svd.single", tol=1e-6, 
                        mode="canonical", max_iter=100, study=None, keepA=None, penalty=None, all_outputs=False, tau=None):
    """
    **notas**
    - solo one.model=TRUE en R, para el modo multi modelo aun falta tune.mint 
    - se asume ningun bloque con NaN
    - sin 'all_outputs' extendido (AVE, prop_expl_var, loadings.Astar, mat.c)
    (all_outputs=True calcula solo AVE_inner por ahora).

    Param:
    A: list de np.ndarray, longitud J; bloques de datos (sin centrar, esta funcion los centra por estudio)
    indY: int o NoneM indice 0-based del bloque "respuesta" (usado por mode="classic"/"regression"
    en la deflacion, y por init="svd")
    design: np.ndarray (J, J), opcional; matriz de diseño; por defecto, todos los bloques conectados entre
    si (1 - identidad)
    ncomp: list de int, longitud J, opcional; número de componentes a calcular por bloque; por defecto, 1
    para todos
    scheme: {"horst", "factorial", "centroid"}
    scale: bool; si se escala (además de centrar) cada bloque, por estudio
    init: {"svd", "svd.single"}
    tol, max_iter: criterios de convergencia del loop interno
    mode: {"canonical", "classic", "invariant", "regression"}
    study: array-like, shape (n,), opcional; etiqueta de estudio de cada muestra; por defecto, un único estudio
    ficticio (equivale a un analisis multibloque estandar, sin MINT)
    keepA: list de longitud N (=max ncomp), opcional; keepA[comp] es una lista de longitud J (keepX por bloque
    para ese componente, o None si no hay sparsity en ese bloque)
    penalty: list de longitud J, opcional; penalizacion L1 continua por bloque (fija a traves de todos los
    componentes; alternativa a keepA)
    all_outputs: bool
    tau: None, "optimal", vector de largo J, o lista de largo N de tales vectores/"optimal"
        - None (default): usa sparse_mint_block_iteration (param study)
        - cualquier otro valor: usa sparse_rgcca_iteration (RGCCA regularizado tipo ridge; el estudio solo afecta el centrado
          inicial, no la iteracion en si).

    Returns: InternalMintBlockResult
    """

    J = len(A)
    A = [np.asarray(a, dtype=float) for a in A]
    n = A[0].shape[0]

    if design is None:
        design = 1 - np.eye(J)
    else:
        design = np.asarray(design, dtype=float)

    if ncomp is None:
        ncomp = [1] * J

    if study is None:
        study = np.zeros(n, dtype=int)
    study = np.asarray(study)

    # --- centrado (y escalado opcional) por estudio, por bloque ---
    mean_centered = [mean_centering_per_study(a, study, scale) for a in A]
    A = [mc.data for mc in mean_centered]
    levels = mean_centered[0].levels

    pjs = [a.shape[1] for a in A]
    R = [a.copy() for a in A]
    N = max(ncomp)
    ndefl = [c - 1 for c in ncomp]

    tau_matrix = _build_tau_matrix(tau, N, J) if tau is not None else None

    variates_A = [np.full((n, N), np.nan) for _ in range(J)]
    loadings_A = [np.full((pjs[k], N), np.nan) for k in range(J)]

    if tau is None:
        loadings_partial_A = [
            {lvl: np.full((pjs[k], N), np.nan) for lvl in levels} for k in range(J)
        ]
        variates_partial_A = [
            {lvl: np.full(((study == lvl).sum(), N), np.nan) for lvl in levels}
            for k in range(J)
        ]
    else:
        loadings_partial_A = None
        variates_partial_A = None

    crit_all = []
    iter_all = []
    tau_all = []

    for comp in range(N):  # comp: 0-index
        loadings_init = initialisation_by_svd(R, indY=indY, init=init)
        keepA_comp = keepA[comp] if keepA is not None else [None] * J

        if tau is None:
            result = sparse_mint_block_iteration(
                R, design, study, loadings_init, keepA=keepA_comp, penalty=penalty,
                scheme=scheme, max_iter=max_iter, tol=tol, all_outputs=all_outputs,
            )
        else:
            result = sparse_rgcca_iteration(
                R, design, tau=tau_matrix[comp], scheme=scheme, init=init,
                tol=tol, max_iter=max_iter, penalty=penalty, keepA=keepA_comp,
                all_outputs=all_outputs,
            )

        for k in range(J):
            loadings_A[k][:, comp] = result.loadings_A[k]
            variates_A[k][:, comp] = result.variates_A[:, k]
            if tau is None:
                for lvl in levels:
                    loadings_partial_A[k][lvl][:, comp] = result.loadings_partial_A_comp[k][lvl]
                    variates_partial_A[k][lvl][:, comp] = result.variates_partial_A_comp[k][lvl]

        crit_all.append(result.crit)
        iter_all.append(result.iter)
        if tau is not None:
            tau_all.append(result.tau)

        # deflacion, si no es el ultimo componente
        if N != 1 and comp != N - 1:
            resdefl, pdefl = defl_select(
                result.variates_A, R, nncomp=ndefl, nn=comp + 1, nbloc=J,
                indY=indY, mode=mode, aa=result.loadings_A,
            )
            R = resdefl

    # recortar cada bloque a su propio ncomp[k]
    for k in range(J):
        variates_A[k] = variates_A[k][:, :ncomp[k]]
        loadings_A[k] = loadings_A[k][:, :ncomp[k]]
        if tau is None:
            for lvl in levels:
                loadings_partial_A[k][lvl] = loadings_partial_A[k][lvl][:, :ncomp[k]]
                variates_partial_A[k][lvl] = variates_partial_A[k][lvl][:, :ncomp[k]]

    return InternalMintBlockResult(
        A=A,
        indY=indY,
        ncomp=ncomp,
        mode=mode,
        variates=variates_A,
        loadings=loadings_A,
        variates_partial=variates_partial_A,
        loadings_partial=loadings_partial_A,
        study=study,
        crit=crit_all,
        iter=iter_all,
        tau=tau_all if tau is not None else None,
        design=design,
        scheme=scheme,
        tol=tol,
        max_iter=max_iter,
        scale=scale,
    )



# ----------------------------------------
# sparse.mint.block_iteration en R
# ----------------------------------------


@dataclass
class MintIterationResult:
    variates_A: np.ndarray # shape (n, J)
    loadings_A: list # lista de J vectores np.ndarray
    crit: np.ndarray # criterio de convergencia por iteracion
    AVE_inner: float # None si all_outputs=False
    loadings_partial_A_comp: list # lista de J dicts {nivel_estudio: vector}
    variates_partial_A_comp: list # lista de J dicts {nivel_estudio: vector}
    iter: int


def _g(x, scheme):
    if scheme == "horst":
        return x
    elif scheme == "factorial":
        return x ** 2
    elif scheme == "centroid":
        return np.abs(x)
    else:
        raise ValueError(f"scheme desconocido: {scheme!r}")


def sparse_mint_block_iteration(A, design, study, loadings_A, keepA=None, penalty=None, scheme="horst", max_iter=100, 
                                tol=1e-6, all_outputs=False):
    """
    Algoritmo iterativo tipo NIPALS/SGCCA adaptado para múltiples estudios: 
    en cada iteracion, el "loading" de cada bloque se calcula como la suma de 
    las contribuciones de cada estudio por separado
    (A_split[q][estudio].T @ Z_split[estudio]) 
    en vez de una sola pasada sobre todos los datos [mint evita que el efecto de 
    estudio/batch contamine la direccion encontrada].

    *** x ahora se asume que ningun bloque tiene NaN ***
    (en R el manejo d datos faltantes es en otra fase)

    Param:
    A: list de np.ndarray, longitud J; bloques de datos (residuos actuales, ya centrados por estudio)
    design: np.ndarray, shape (J, J); matriz de diseno (que bloques estan "conectados"); diagonal debe ser 0
    study: array-like, shape (n,); etiqueta de estudio de cada muestra
    loadings_A: list de np.ndarray, longitud J; loadings iniciales (de mint_helpers.py/initialisation_by_svd())
    keepA: list de (int o None), longitud J, opcional; número de variables a mantener por bloque (sparsity por conteo)
    penalty: list de (float o None), longitud J, opcional; penalización L1 continua por bloque (alternativa a keepA)
    scheme: {"horst", "factorial", "centroid"}
    max_iter: int
    tol: float
    all_outputs: bool; si True, calcula AVE_inner

    Return: MintIterationResult
    """

    J = len(A)
    n = A[0].shape[0]
    pjs = [a.shape[1] for a in A]

    if penalty is not None:
        penalty = [pen * np.sqrt(p) if pen is not None else None
                   for pen, p in zip(penalty, pjs)]

    levels = list(np.unique(np.asarray(study)))
    nlevels_study = len(levels)

    A_split = [study_split(a, study, levels=levels) for a in A]

    loadings_A = [np.asarray(l, dtype=float).ravel().copy() for l in loadings_A]
    variates_A = np.zeros((n, J))

    loadings_partial_A_comp = [dict() for _ in range(J)]

    # inicializacion de variates.A con los loadings iniciales, luego
    # se normalizan los loadings (en ese orden, igual que R)
    for q in range(J):
        variates_A[:, q] = A[q] @ loadings_A[q]
        loadings_A[q] = l2_norm(loadings_A[q])

    loadings_A_old = [l.copy() for l in loadings_A]

    crit_list = []
    iter_ = 1

    while True:
        for q in range(J):
            if scheme == "horst":
                cbycovq = design[q, :].copy()
            elif scheme == "factorial":
                cbycovq = design[q, :] * cov2(variates_A, variates_A[:, [q]]).ravel()
            elif scheme == "centroid":
                cbycovq = design[q, :] * np.sign(cov2(variates_A, variates_A[:, [q]]).ravel())
            else:
                raise ValueError(f"scheme desconocido: {scheme!r}")

            # --- componente interna ---
            z_q = variates_A @ cbycovq
            z_split = study_split(z_q.reshape(-1, 1), study, levels=levels)

            # --- peso externo (loading), sumando la contribución de cada estudio por separado ---
            temp = np.zeros(pjs[q])
            for m in levels:
                partial = A_split[q][m].T @ z_split[m].ravel()
                loadings_partial_A_comp[q][m] = partial
                temp = temp + partial
            loadings_A[q] = temp

            if penalty is not None:
                loadings_A[q] = sparsity(loadings_A[q], keep_a=None, penalty=penalty[q])
            else:
                loadings_A[q] = sparsity(loadings_A[q], keep_a=keepA[q] if keepA is not None else None, penalty=None)

            loadings_A[q] = l2_norm(loadings_A[q])

            variates_A[:, q] = A[q] @ loadings_A[q]

        crit_list.append(np.sum(design * _g(cov2(variates_A), scheme)))

        diff_value = max(float((loadings_A[x] - loadings_A_old[x]) @ (loadings_A[x] - loadings_A_old[x]))
            for x in range(J)
        )

        if diff_value < tol or iter_ > max_iter:
            if iter_ > max_iter:
                import warnings
                warnings.warn("sparse_mint_block_iteration: no convergio "f"en {max_iter} iteraciones.")
            break

        loadings_A_old = [l.copy() for l in loadings_A]
        iter_ += 1

    variates_partial_A_comp = []
    for q in range(J):
        vpq = study_split(variates_A[:, [q]], study, levels=levels)
        variates_partial_A_comp.append({m: vpq[m].ravel() for m in levels})

    if all_outputs:
        corr = np.corrcoef(variates_A, rowvar=False)
        AVE_inner = np.sum(design * corr ** 2 / 2) / (np.sum(design) / 2)
    else:
        AVE_inner = None

    crit = np.array(crit_list)
    crit = crit[crit != 0]  # R: crit[which(crit != 0)]

    return MintIterationResult(
        variates_A=variates_A,
        loadings_A=loadings_A,
        crit=crit,
        AVE_inner=AVE_inner,
        loadings_partial_A_comp=loadings_partial_A_comp,
        variates_partial_A_comp=variates_partial_A_comp,
        iter=iter_,
    )


# ----------------------------------------
# sparse.rgcca_iteration() en R
# ----------------------------------------


@dataclass
class RgccaIterationResult:
    variates_A: np.ndarray # shape (n, J)
    loadings_A: list # lista de J vectores np.ndarray
    crit: np.ndarray
    AVE_inner: float # None si all_outputs=False
    tau: np.ndarray # tau usado (o estimado) por bloque
    scheme: str
    iter: int
    keepA: list


def _g(x, scheme):
    if scheme == "horst":
        return x
    elif scheme == "factorial":
        return x ** 2
    elif scheme == "centroid":
        return np.abs(x)
    else:
        raise ValueError(f"scheme desconocido: {scheme!r}")


def sparse_rgcca_iteration(A, design, tau="optimal", scheme="horst", init="svd.single", max_iter=100, 
                           tol=np.finfo(float).eps, keepA=None, penalty=None, all_outputs=False):
    """

    Variante del motor con regularizacion tau.

    **A diferencia de sparse_mint_block_iteration(), esta funcion no
    tiene en cuenta el estudio/batch durante la iteracion**

    - bloques donde n >= p : Formulacion primal (espacio de variables, tamaño p x p)
    - bloques donde n < p  : Formulacion DUAL (espacio de muestras, tamaño n x n) 

    ** también asume q no hay nans ** 

    Param:
    A: list de np.ndarray, longitud J; bloques de datos (residuos actuales, ya centrados por estudio)
    design: np.ndarray, shape (J, J);
    tau: "optimal" o list/array de floats, longitud J; intensidad de regularizacion ridge por bloque 
    (0 = sin regularizar, 1 = maxima regularizacion hacia la identidad). "optimal" estima cada tau[j]
    automáticamente via tau_estimate(A[j]).
    scheme: {"horst", "factorial", "centroid"}
    init: solo "svd.single" soportado (R)
    max_iter, tol: criterios de convergencia
    keepA: list de (int o None), longitud J, opcional
    penalty: list de (float o None), longitud J, opcional; la precedencia keepA/penalty aqui es la 
    opuesta a la de sparse_mint_block_iteration(); aqui, si keepA fue provisto (globalmente, no None),
    tiene prioridad sobre penalty (tal cual R)
    all_outputs: bool

    Return: RgccaIterationResult
    """
    if init != "svd.single":
        raise ValueError("init debe ser 'svd.single'.")

    A = [np.asarray(a, dtype=float) for a in A]
    J = len(A)
    n = A[0].shape[0]
    pjs = [a.shape[1] for a in A]

    if penalty is not None:
        penalty = [pen * np.sqrt(p) if pen is not None else None
                   for pen, p in zip(penalty, pjs)]

    if isinstance(tau, str):
        if tau != "optimal":
            raise ValueError("tau debe ser 'optimal' o un vector numerico de largo J.")
        tau = [tau_estimate(a) for a in A]
    tau = np.asarray(tau, dtype=float)

    which_primal = [j for j in range(J) if n >= pjs[j]]
    which_dual = [j for j in range(J) if n < pjs[j]]

    loadings_A = [None] * J
    alpha = [None] * J
    M = [None] * J
    Minv = [None] * J
    K = [None] * J

    # R hace replace(A[[j]], is.na(A[[j]]), 0) antes de initsvd() para
    # tolerar NaN en la inicialización (x ahora solo eso)
    for j in which_primal:
        loadings_A[j] = initsvd(A[j])
    for j in which_dual:
        alpha[j] = initsvd(A[j])
        K[j] = A[j] @ A[j].T

    variates_A = np.zeros((n, J))
    N_ = n  # tamaño muestral, usado para escalar K[j] en la formulacion dual

    for j in range(J):
        if j in which_primal:
            M[j] = np.linalg.pinv(tau[j] * np.eye(pjs[j]) + (1 - tau[j]) * cov2(A[j]))
            raw = M[j] @ loadings_A[j]
            norm_factor = 1.0 / np.sqrt(loadings_A[j] @ M[j] @ loadings_A[j])
            loadings_A[j] = norm_factor * raw
        else:  # j en which_dual
            M[j] = tau[j] * np.eye(n) + (1 - tau[j]) / N_ * K[j]
            Minv[j] = np.linalg.pinv(M[j])
            norm_factor = 1.0 / np.sqrt(alpha[j] @ M[j] @ K[j] @ alpha[j])
            alpha[j] = norm_factor * alpha[j]
            loadings_A[j] = A[j].T @ alpha[j]
        variates_A[:, j] = A[j] @ loadings_A[j]

    loadings_A_old = [l.copy() for l in loadings_A]
    crit_list = []
    iter_ = 1
    order = which_primal + which_dual

    while True:
        for j in order:
            if scheme == "horst":
                cbycovq = design[j, :].copy()
            elif scheme == "factorial":
                cbycovq = design[j, :] * cov2(variates_A, variates_A[:, [j]]).ravel()
            elif scheme == "centroid":
                cbycovq = design[j, :] * np.sign(cov2(variates_A, variates_A[:, [j]]).ravel())
            else:
                raise ValueError(f"scheme desconocido: {scheme!r}")

            z_j = variates_A @ cbycovq

            if j in which_primal:
                raw = M[j] @ (A[j].T @ z_j)
                norm_factor = 1.0 / np.sqrt(z_j @ A[j] @ M[j] @ A[j].T @ z_j)
                loadings_A[j] = norm_factor * raw
            else:  # dual
                raw_alpha = Minv[j] @ z_j
                norm_factor = 1.0 / np.sqrt(z_j @ K[j] @ Minv[j] @ z_j)
                alpha[j] = norm_factor * raw_alpha
                loadings_A[j] = A[j].T @ alpha[j]

            # sparse usando keepA / penalty -- lo q se decía arriba, aqui keepA tiene
            # precedencia sobre penalty (al revés que en sparse_mint_block_iteration()),
            # y se preserva la norma pre-sparsity
            # (no se renormaliza a 1, a diferencia de sparse_mint_block_iteration())
            if keepA is not None or penalty is not None:
                temp_norm = norm2(loadings_A[j])
                if keepA is not None:
                    loadings_A[j] = sparsity(loadings_A[j], keep_a=keepA[j], penalty=None)
                else:
                    loadings_A[j] = sparsity(loadings_A[j], keep_a=None, penalty=penalty[j])
                loadings_A[j] = (loadings_A[j] / norm2(loadings_A[j])) * temp_norm

            variates_A[:, j] = A[j] @ loadings_A[j]

        crit_list.append(np.sum(design * _g(cov2(variates_A), scheme)))

        diff_value = max(
            float((loadings_A[x] - loadings_A_old[x]) @ (loadings_A[x] - loadings_A_old[x]))
            for x in range(J)
        )

        if diff_value < tol or iter_ > max_iter:
            if iter_ > max_iter:
                import warnings
                warnings.warn("sparse_rgcca_iteration: no convergio en "f"{max_iter} iteraciones.")
            break

        loadings_A_old = [l.copy() for l in loadings_A]
        iter_ += 1

    if all_outputs:
        corr = np.corrcoef(variates_A, rowvar=False)
        AVE_inner = np.sum(design * corr ** 2 / 2) / (np.sum(design) / 2)
    else:
        AVE_inner = None

    crit = np.array(crit_list)
    crit = crit[crit != 0]  # R: crit[which(crit != 0)]

    return RgccaIterationResult(
        variates_A=variates_A,
        loadings_A=loadings_A,
        crit=crit,
        AVE_inner=AVE_inner,
        tau=tau,
        scheme=scheme,
        iter=iter_,
        keepA=keepA,
    )
