# Discrepancias verificadas: MINT (Python, pythomics) vs mixOmics (R)

Referencia R: `~/mixOmics/R/internal_mint.block.R`,
`~/mixOmics/R/internal_mint.block_helpers.R`,
`~/mixOmics/R/check_entry.R` (`Check.entry.wrapper.mint.block`),
`~/mixOmics/R/internal_wrapper.mint.block.R`.

Implementacion Python: `pythomics/models/mint.py`,
`pythomics/models/mint_internal.py`, `pythomics/models/mint_helpers.py`.

Cada discrepancia fue **verificada empiricamente** reimplementando la funcion R en
Python puro y comparando contra `pythomics` (ver scripts en `/tmp/opencode/compare_mint*.py`).

Estado: `[FIX]` corregido en esta sesion, `[PEND]` pendiente.

---

## [FIX] 1. Supervisado: `Y` se ignoraba por completo  (CRITICO)

- R (`Check.entry.wrapper.mint.block`): si `Y` no es `NULL`, se anade como **ultimo
  bloque** de `A`, se fija `indY = length(A)`, se construye la matriz de diseno
  `(J+1)x(J+1)` (cada bloque X conectado a Y) y se ajusta `ncomp` del bloque Y
  (`regression`: `max(ncomp)`; `canonical`: `min(ncomp, ncol(Y)-1)`).
- Python (`mint.py` fit): `Y` se convertia a 2D y luego `pass`; `indY` quedaba en
  `None`. El bloque Y nunca entraba.
- **Evidencia**: `fit(X)` y `fit(X, Y)` daban loadings identicos, `max abs diff = 0.0`.
- **Fix**: codificar `Y` (numerico->columna; categorico->one-hot), anadirlo como
  bloque, fijar `indY=J`, extender design/ncomp/keepA/penalty. `self.Y_` conserva el
  `Y` original para `get_expression_means`.
- **Desviaciones pragmaticas respecto a R** (documentadas a proposito):
  - `ncomp` del bloque Y se fija en `max(ncomp_X)` (R usa `min(ncomp, ncol(Y)-1)`
    para `canonical` y `max(ncomp)` para `regression`). Con Y de 1 columna, la
    formula `canonical` de R da `ncol(Y)-1 = 0` (bloque Y degenerado).
  - Si el modo es `canonical` y hay `Y`, se cambia a `regression` con un aviso
    (`warnings`/`logger`). mixOmics usa `regression` para block/splsda; con
    `canonical` la deflacion agota el bloque Y (rank 1) tras el 1er componente.
    En no supervisado (`indY=None`) el modo no influye en la deflacion.

## [FIX] 2. Ruta dual de RGCCA rota cuando `n < p`  (ALTO)

- R (`sparse.rgcca_iteration` + `initsvd`): para bloques duales (`n < p`) usa la
  formulacion en espacio de muestras; `initsvd` devuelve `$u` (largo `n`) si `n < p`.
- Python: `initsvd` **siempre** devolvia `Vt[0,:]` (largo `p`); `alpha[j]` quedaba
  con largo `p` y el producto `alpha[j] @ M[j] @ K[j]` fallaba.
- **Evidencia**: `MINT(tau="optimal")` con bloque `n=20, p=30` ->
  `ValueError: matmul ... size 20 is different from 30`.
- **Fix**: `initsvd` devuelve `U[:,0]` (largo `n`) cuando `n < p`.

## [FIX] 3. `initsvd` sin rama dual                             (ALTO, ligado a 2)

- R: `if (n >= p) return($v) else return($u)`.
- Python no implementaba la rama `n < p`.
- **Fix**: implementada en `initsvd`.

## [FIX] 4. `initialisation_by_svd` no fiel a R                  (MEDIO)

- `init="svd.single"` con `indY=None`: Python **concatenaba** los bloques y hacia un
  SVD conjunto; R hace `initsvd` **por bloque** (con correccion dual
  `loadings = A_j^T alpha_j`).
  - **Evidencia**: `|cos(py_block0, R_block0)| = 0.223` (distinto).
- `init="svd"`: Python ignoraba `indY`; R usa `crossprod(A[[x]], A[[indY]])` y toma
  `u` (no-Y) / `v` (Y).
- **Fix**: reescrita para replicar ambas ramas de R.

## [FIX] 5. `sparsity(keepA)` valores distintos de R             (MEDIO)

- R (`soft_thresholding_L1`): umbral = **mayor magnitud descartada**
  (`max(abs[descartados])`); usa `rank(ties.method="max")`.
- Python (`binary_search_threshold`): umbral hallado por busqueda binaria.
- **Evidencia**: mismo soporte, pero tras normalizar L2 el coseno bajaba hasta
  `0.9485`. Con empates, R conserva **todos** los empatados en la frontera
  (p.ej. `[1,1,0.5], keepA=1 -> [0.5,0.5,0]`), Python daba ~`[0,0,0]`.
- **Fix**: `sparsity` usa `soft_thresholding_l1` (+ `rank` ties max), fiel a R.

## [FIX] 6. `sparsity(penalty)` semantica distinta               (MEDIO)

- R (`soft.threshold` + `BinarySearch`): interpreta `penalty` como cota de la razon
  **L1/L2** (radio de la bola L1) y busca el umbral que la satisface.
- Python: aplicaba `penalty` como **umbral directo** de soft-thresholding.
- **Evidencia**: `penalty=0.5` -> Python `#nz=5`, R `#nz=1` (L1/L2 Python `1.99`, R `1.0`).
- **Fix**: `sparsity` usa `soft_threshold`/`binary_search` (L1) fiel a R.
  (El escalado `penalty*sqrt(p)` ya existia y coincide con R.)

## [FIX] 7. `crit` no filtraba ceros                             (MENOR)

- R devuelve `crit[which(crit != 0)]` en ambas iteraciones.
- Python devolvia `np.array(crit_list)` sin filtrar.
- **Fix**: filtrar ceros.

## [PEND] 8. Sin soporte de datos faltantes (NA)                 (MENOR)

- R maneja `misdata`, `is.na.A`, `ind.NA`, `ind.NA.col` en centrado, inicializacion,
  iteraciones y deflacion (via `miscrossprod`, mascaras, `deflation` con `misdata.q`).
- Python asume que ningun bloque tiene NaN.
- `deflation`/`defl_select` en Python ya tienen firma para NA, pero el pipeline no
  lo alimenta; `mean_centering_per_study` usa `nanmean`/`nanstd` pero sin enmascarar
  las iteraciones.

## [PEND] 9. `tau` escalar                                        (MENOR)

- R: `tau=0.5` (escalar) se convierte en matriz `N x 1`; en `sparse.rgcca_iteration`
  `tau[j]` seria `NA` para `j>1` (R no lo soporta realmente).
- Python: admite escalar y lo replica por bloque/componente (extension, no fiel).

## [PEND] 10. `all_outputs` limitado                             (MENOR)

- R (`all.outputs=TRUE`) calcula `AVE_inner`, `AVE_outer`, `AVE_X`, `prop_expl_var`
  (`explained_variance`), `loadings.Astar`, `mat.c`.
- Python solo calcula `AVE_inner`.

## [FIX] 11. Convencion de signo en `initsvd`                   (MENOR)

- R (`svds`/`svd`) no fija el signo de los vectores singulares.
- Python fija "primer elemento no nulo positivo" para determinismo. Cosmetico
  (puede invertir el signo de un componente respecto a R), pero reproducible.

## [FIX] 12. Bug Python: `rank` con NaN -> bucle infinito        (CRITICO, solo Python)

- No es una discrepancia frente a R, sino un bug propio que aparecio al portar
  fielmente el flujo supervised.
- `rank` (mint_helpers.py) hacia `j = i` y avanzaba mientras
  `abs(x[j]-x[i]) < 1e-12`. Con un `NaN` esa comparacion es `False`, `j` no
  avanzaba y `i = j` no progresaba -> **bucle infinito**.
  Se disparaba cuando un loading quedaba todo-cero (p.ej. bloque Y agotado por
  deflacion `canonical`), `l2_norm` daba `NaN` y `sparsity`->`rank` se colgaba.
- **Fix**: `j = i + 1` (garantiza progreso; mismo agrupamiento de empates).

---

## Verificados como FIELES (sin accion)

`tau_estimate` (ddof=1, `scale` de R), `cov2` (denominador `n-1`, `bias`),
`norm2` (fallback `0.05`), `l2_norm`, `deflation`, `defl_select` (logica de
`canonical`/`classic`/`invariant`/`regression` y el `if mode=="canonical" or q!=indY`),
`_build_tau_matrix` (salvo el caso escalar, ver 9), estructura general de
`internal_mint_block` (centrado por estudio, deflacion, `nn`/`nncomp`), escalado
`penalty*sqrt(p)`, y los esquemas `horst`/`factorial`/`centroid`.
