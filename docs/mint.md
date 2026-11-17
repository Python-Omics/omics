# `pythomics.models.MINT`

**MINT** (*Multi-group INtegrative Tensor*) es un estimador de integración multi-ómica
para datos provenientes de varios estudios, cohortes o lotes. Implementa en Python puro
el algoritmo `mint.block` de **mixOmics** y expone una API compatible con
scikit-learn (`fit`, `transform`, `fit_transform`, `get_params`, `clone`).

MINT corrige los efectos de estudio/batch de dos formas complementarias:

1. **Centrado y escalado por estudio** de cada bloque antes de la descomposición.
2. **Agregación de loadings por estudio** durante la iteración (ruta MINT), de modo
   que cada estudio contribuye de forma independiente a la dirección integrativa.

> `MINT` está pensado para al menos dos bloques de datos. Para un único bloque, la
> matriz de diseño es degenerada y `fit` lanza `ValueError`.

---

## Contenido

- [Instalación e importación](#instalación-e-importación)
- [Inicio rápido](#inicio-rápido)
- [Referencia de la clase](#referencia-de-la-clase)
  - [Firma del constructor](#firma-del-constructor)
  - [Hiperparámetros](#hiperparámetros)
  - [Atributos](#atributos)
  - [Métodos](#métodos)
- [Detalles del algoritmo](#detalles-del-algoritmo)
- [Compatibilidad con scikit-learn](#compatibilidad-con-scikit-learn)
- [Limitaciones y notas](#limitaciones-y-notas)
- [Errores comunes](#errores-comunes)
- [Referencias](#referencias)

---

## Instalación e importación

MINT forma parte de `pythomics`. Solo requiere `numpy` y `pandas`; `mudata` es
opcional y únicamente se necesita para `MINT.from_mudata`.

```python
from pythomics.models import MINT
```

Dependencias opcionales:

```bash
pip install mudata anndata   # para MINT.from_mudata
```

---

## Inicio rápido

```python
import numpy as np
import pandas as pd

from pythomics.models import MINT

rng = np.random.default_rng(0)
n_samples = 120

X_dict = {
    "mRNA": pd.DataFrame(
        rng.normal(size=(n_samples, 500)),
        columns=[f"Gene_{i}" for i in range(500)],
    ),
    "Proteins": pd.DataFrame(
        rng.normal(size=(n_samples, 200)),
        columns=[f"Prot_{i}" for i in range(200)],
    ),
    "Metabolites": pd.DataFrame(
        rng.normal(size=(n_samples, 150)),
        columns=[f"Met_{i}" for i in range(150)],
    ),
}

# Dos cohortes / estudios
study = np.array([0] * 60 + [1] * 60)

# Etiqueta opcional para análisis supervisado
Y = np.array([0, 1] * 60)

model = MINT(
    n_components=2,
    keepA={
        "mRNA": [30, 30],
        "Proteins": [15, 15],
        "Metabolites": [10, 10],
    },
    tau=None,
    scheme="horst",
    scale=True,
    max_iter=200,
)

model.fit(X_dict=X_dict, Y=Y, study=study)

# Loadings y variates por bloque
print(model.loadings_["mRNA"].shape)   # (500, 2)
print(model.variates_["mRNA"].shape)   # (120, 2)

# Biomarcadores seleccionados en el componente 1
biomarkers = model.get_biomarkers("mRNA", comp=1)
print(biomarkers.head(10))

# Proyección de nuevos datos
new_variates = model.transform(X_dict)
```

---

## Referencia de la clase

### Firma del constructor

```python
MINT(
    n_components=2,
    design=None,
    keepA=None,
    penalty=None,
    tau=None,
    scheme="horst",
    mode="canonical",
    scale=True,
    max_iter=100,
    tol=1e-6,
    init="svd.single",
)
```

Todos los parámetros se almacenan como atributos públicos con el mismo nombre y
pueden consultarse o modificarse con `get_params()` / `set_params()`.

---

### Hiperparámetros

| Parámetro | Tipo | Valor por defecto | Descripción |
|---|---|---|---|
| `n_components` | `int` o `list[int]` | `2` | Número de componentes por bloque. |
| `design` | `array-like`, forma `(J, J)` o `(J+1, J+1)` | `None` | Conectividad entre bloques. |
| `keepA` | `dict`, `list` o `None` | `None` | Número exacto de variables activas por bloque y componente. |
| `penalty` | `dict`, `list` o `None` | `None` | Penalización L1 (razón L1/L2) por bloque. |
| `tau` | `None`, `"optimal"`, `float`, `list` | `None` | Regularización ridge y selección de ruta algorítmica. |
| `scheme` | `{"horst", "factorial", "centroid"}` | `"horst"` | Esquema del criterio de optimización. |
| `mode` | `{"canonical", "classic", "invariant", "regression"}` | `"canonical"` | Estrategia de deflación entre componentes. |
| `scale` | `bool` | `True` | Escalado a varianza unitaria por estudio. |
| `max_iter` | `int` | `100` | Iteraciones máximas por componente. |
| `tol` | `float` | `1e-6` | Tolerancia de convergencia. |
| `init` | `{"svd", "svd.single"}` | `"svd.single"` | Método de inicialización de loadings. |

#### `n_components`

Número de componentes latentes a extraer.

- `int`: se usa el mismo valor para todos los bloques. Por ejemplo, `n_components=2`
  produce dos componentes en cada bloque.
- `list[int]`: un valor por bloque, en el mismo orden que `X_dict`. La longitud debe
  ser igual al número de bloques `J`; de lo contrario `fit` lanza `ValueError`.
  Por ejemplo, `[2, 3, 1]` para tres bloques.

En modo supervisado, el bloque `Y` recibe internamente `max(n_components)` componentes
para evitar bloques de respuesta degenerados. El atributo `n_components_` solo contiene
los componentes de los bloques `X`.

```python
model = MINT(n_components=[2, 1])  # bloque A: 2 componentes, bloque B: 1
```

#### `design`

Matriz que indica qué bloques están conectados entre sí. Un valor distinto de cero
en `design[i, j]` permite que el bloque `i` use información del bloque `j` al
actualizar sus loadings.

- En análisis **no supervisado**, el valor por defecto es `1 - I_J`: todos los bloques
  están conectados entre sí y la diagonal es cero.
- En análisis **supervisado**, el valor por defecto es `1 - I_{J+1}` e incluye el bloque
  `Y`. Si el usuario proporciona una matriz `(J, J)`, se amplía automáticamente a
  `(J+1, J+1)` conectando `Y` con todos los bloques `X`. También se acepta directamente
  una matriz `(J+1, J+1)`.
- La diagonal debe ser cero; en modo supervisado se fuerza a cero automáticamente.

```python
# A conecta con B y C; B y C no están conectados directamente
design = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 0],
])

model = MINT(n_components=2, design=design)
```

#### `keepA`

Controla la **selección exacta de variables**: cuántos coeficientes distintos de cero
se conservan en cada bloque y componente.

Formatos admitidos:

```python
# 1. Diccionario (recomendado): {bloque: [comp1, comp2, ...]}
keepA = {
    "mRNA": [30, 30],
    "Proteins": [15, 15],
    "Metabolites": [10, 10],
}

# 2. Diccionario con un valor fijo para todos los componentes
keepA = {"mRNA": 30, "Proteins": 15, "Metabolites": 10}

# 3. Lista plana de longitud J: mismo valor en todos los componentes
keepA = [30, 15, 10]

# 4. Lista de listas: keepA[componente][bloque]
keepA = [[30, 15, 10], [25, 12, 8]]
```

Notas:

- Si `keepA` es mayor que el número de variables del bloque, se conservan todas.
- En caso de empates en la frontera, se conservan todas las variables empatadas
  (comportamiento fiel a `rank(ties.method="max")` de R).
- Si una lista por bloque es más corta que `n_components`, los componentes faltantes
  no aplican sparsity (`None`).
- `None` desactiva la selección de variables.
- `keepA` y `penalty` son alternativas. Aunque el constructor no impide pasarlos
  simultáneamente, la precedencia depende de la ruta interna; se recomienda usar solo
  uno de los dos.

#### `penalty`

Penalización L1 continua por bloque. Internamente se interpreta como una cota de la
razón **L1/L2** del loading (no como un umbral directo de soft-thresholding) y se
escala por `sqrt(p)` antes de aplicarse.

```python
# Diccionario por bloque
penalty = {"mRNA": 0.5, "Proteins": 0.3, "Metabolites": 0.2}

# Lista de longitud J
penalty = [0.5, 0.3, 0.2]

model = MINT(n_components=2, penalty=penalty)
```

- El valor es fijo para todos los componentes; no existe un formato por componente.
- Es una alternativa a `keepA`.

#### `tau`

Selecciona la ruta algorítmica y, si corresponde, la intensidad de regularización
ridge.

| Valor | Ruta | Significado |
|---|---|---|
| `None` (por defecto) | MINT | Sin ridge; agregación de loadings por estudio. |
| `"optimal"` | RGCCA | Estima `tau` por bloque con la fórmula de Strimmer. |
| `float` | RGCCA | Mismo `tau` para todos los bloques y componentes (extensión). |
| `list[float]` de longitud `J` | RGCCA | Un `tau` fijo por bloque. |
| `list[list[float]]` de longitud `N` | RGCCA | `tau` por componente y bloque. |

- `tau=0`: sin regularización (CCA clásico).
- `tau=1`: máxima regularización (covarianza igual a la identidad).
- En la ruta RGCCA, `study` solo afecta al centrado/escalado inicial; la iteración no
  usa los estudios.
- Una cadena distinta de `"optimal"` o una longitud incorrecta lanzan `ValueError`.

```python
MINT(tau=None)                    # ruta MINT (corrección de batch)
MINT(tau="optimal")               # RGCCA con tau automático
MINT(tau=[0.1, 0.2, 0.15])        # RGCCA con tau fijo por bloque
MINT(tau=[[0.1, 0.2], [0.3, 0.4]])  # RGCCA por componente y bloque
```

#### `scheme`

Esquema utilizado para ponderar la contribución de los demás bloques.

- `"horst"` (por defecto): maximiza la suma de covarianzas entre bloques.
- `"factorial"`: maximiza la suma de covarianzas al cuadrado.
- `"centroid"`: maximiza la suma de valores absolutos de las covarianzas.

Un valor no reconocido lanza `ValueError` durante la iteración.

#### `mode`

Estrategia de deflación aplicada a los bloques después de extraer cada componente.
Solo tiene efecto cuando se calcula más de un componente.

- `"canonical"` (por defecto): cada bloque se deflacta sobre su propio variate.
- `"classic"`: el bloque `Y` no se deflacta sobre su propio variate; los bloques `X`
  se deflactan usando el promedio de los variates de los demás bloques.
- `"invariant"`: no se aplica deflación.
- `"regression"`: variante de `"classic"` que usa el promedio de los residuos de
  deflactar contra cada uno de los otros bloques.

> **Análisis supervisado:** si se proporciona `Y` y `mode="canonical"`, MINT cambia
> internamente a `"regression"` y emite una advertencia. La deflación `canonical`
> agotaría un bloque `Y` de una sola columna tras el primer componente.

#### `scale`

Si es `True` (por defecto), cada bloque se centra y se divide por la desviación
estándar muestral (`ddof=1`) **dentro de cada estudio**. Las columnas de varianza cero
se dejan en cero. Si es `False`, solo se centra.

El centrado por estudio es la razón por la que MINT es invariante a desplazamientos
constantes específicos de cada estudio.

#### `max_iter`

Número máximo de iteraciones del bucle interno para cada componente. Si se alcanza
sin converger, se emite un `UserWarning` y se devuelve el último resultado.

#### `tol`

Tolerancia de convergencia. El bucle se detiene cuando el cambio máximo en los
loadings, medido como norma L2 al cuadrado de la diferencia entre iteraciones
sucesivas, es menor que `tol`.

Valores típicos: `1e-6` (por defecto) para precisión, `1e-4` para ajustes rápidos.

#### `init`

Método de inicialización de los loadings.

- `"svd.single"` (por defecto): SVD por bloque. Es el único modo soportado por la ruta
  RGCCA (`tau` distinto de `None`).
- `"svd"`: inicialización supervisada basada en el producto cruzado de cada bloque `X`
  con el bloque `Y`. Requiere que se proporcione `Y`; de lo contrario `fit` lanza
  `ValueError`.

La convención de signo de los vectores singulares se fija de forma determinista
(primer elemento no nulo positivo), lo que hace que ajustes repetidos sean
reproducibles.

---

### Atributos

#### Atributos disponibles antes de `fit`

| Atributo | Tipo | Descripción |
|---|---|---|
| `n_components` | `int` o `list[int]` | Valor solicitado por el usuario. |
| `design` | `array-like` o `None` | Matriz de diseño solicitada. |
| `keepA` | `dict`, `list` o `None` | Configuración de sparsity. |
| `penalty` | `dict`, `list` o `None` | Configuración de penalización L1. |
| `tau` | varios | Configuración de regularización. |
| `scheme` | `str` | Esquema de optimización. |
| `mode` | `str` | Estrategia de deflación. |
| `scale` | `bool` | Escalado por estudio. |
| `max_iter` | `int` | Iteraciones máximas. |
| `tol` | `float` | Tolerancia. |
| `init` | `str` | Inicialización. |
| `r_model_var_` | `str` | Identificador único usado para interoperabilidad con R. |

#### Atributos después de `fit`

| Atributo | Tipo | Descripción |
|---|---|---|
| `loadings_` | `dict[str, np.ndarray]` | Loadings por bloque, forma `(p_k, n_components_k)`. |
| `variates_` | `dict[str, np.ndarray]` | Variates (scores) por bloque, forma `(n, n_components_k)`. |
| `names_` | `dict[str, list[str]]` | Nombres de variables por bloque. |
| `X_dict_` | `dict[str, array-like]` | Bloques de datos originales usados en el ajuste. |
| `Y_` | `np.ndarray` o `None` | Variable respuesta original, sin codificar. |
| `study_` | `np.ndarray` | Etiquetas de estudio por muestra. |
| `block_names_` | `list[str]` | Nombres de los bloques en el orden usado. |
| `n_components_` | `list[int]` | Componentes efectivos por bloque `X`. |
| `indY_` | `int` o `None` | Índice del bloque `Y` dentro de la lista interna de bloques. |
| `_mint_result` | objeto interno | Resultado completo del motor MINT (loadings parciales, criterios, iteraciones, `tau` usado, etc.). |

Cuando un bloque se pasa como `numpy.ndarray`, los nombres de variables se generan
como `Var_0`, `Var_1`, etc. Cuando se pasa un `pandas.DataFrame`, se usan sus columnas.

---

### Métodos

#### `from_mudata`

```python
@classmethod
MINT.from_mudata(mdata, target_col=None, study_col="study", **kwargs)
```

Crea una instancia de `MINT` a partir de un objeto `mudata.MuData`. No ajusta el
modelo: extrae los bloques y las etiquetas, y deja la instancia lista para llamar a
`fit()` sin argumentos.

**Parámetros**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `mdata` | `mudata.MuData` | Contenedor multi-modal. |
| `target_col` | `str`, opcional | Columna de `mdata.obs` con la variable respuesta `Y`. |
| `study_col` | `str` | Columna de `mdata.obs` con las etiquetas de estudio. Por defecto `"study"`. |
| `**kwargs` | `dict` | Argumentos adicionales para el constructor de `MINT`. |

**Devuelve**

`MINT`: instancia sin ajustar con `X_dict_`, `Y_` y `study_` ya poblados.

**Errores**

- `KeyError`: `target_col` no existe ni en `mdata.obs` ni en el `.obs` de la primera
  modalidad.
- Si `study_col` no existe, se emite una advertencia y se asigna un único estudio
  ficticio con etiqueta `0`.

**Ejemplo**

```python
import mudata as md

mdata = md.read_h5mu("multiomics.h5mu")

model = MINT.from_mudata(
    mdata,
    target_col="Subtype",
    study_col="Batch",
    n_components=2,
    keepA={"mRNA": [20, 20], "Proteins": [15, 15]},
)

model.fit()
```

> Los bloques se extraen con `mod.to_df()` y se asume que todas las modalidades están
> alineadas por muestra en el mismo orden que `mdata.obs`.

---

#### `fit`

```python
fit(X_dict=None, Y=None, study=None)
```

Ajusta el modelo MINT.

**Parámetros**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `X_dict` | `dict[str, array-like]`, opcional | Bloques de datos `{nombre: (n_samples, n_features)}`. Si es `None`, se usa `self.X_dict_` (por ejemplo, tras `from_mudata`). |
| `Y` | `array-like`, opcional | Variable respuesta para análisis supervisado. Si es `None`, se usa `self.Y_`. |
| `study` | `array-like`, opcional | Etiquetas de estudio por muestra. Si es `None`, se usa `self.study_` o un único estudio con etiqueta `0`. |

**Devuelve**

`self`: el modelo ajustado.

**Comportamiento**

1. Valida que existan al menos dos bloques y que todos tengan el mismo número de
   muestras.
2. Convierte cada bloque a `float` y guarda el orden de los bloques en `block_names_`.
3. Prepara `n_components`, `design`, `keepA` y `penalty` en formato interno.
4. Si `Y` no es `None`, lo codifica y lo añade como último bloque:
   - `Y` numérico 1-D: una columna continua.
   - `Y` categórico 1-D: matriz one-hot con clases ordenadas.
   - `Y` 2-D: debe ser numérico y se usa tal cual.
5. Fija `indY_` al índice del bloque `Y` y extiende `design`, `n_components`, `keepA`
   y `penalty` para incluirlo.
6. Si el análisis es supervisado y `mode="canonical"`, cambia internamente a
   `"regression"` y emite una advertencia.
7. Ejecuta el motor `internal_mint_block` y almacena loadings, variates, nombres y
   metadatos.

**Errores**

- `ValueError`: no se proporcionan datos, hay menos de dos bloques, los bloques tienen
  distinto número de muestras, `study` tiene longitud incorrecta, `Y` tiene longitud
  incorrecta, `n_components` es una lista de longitud incorrecta o `init="svd"` se usa
  sin `Y`.

**Ejemplo**

```python
model = MINT(n_components=2, keepA={"mRNA": [20, 20], "Proteins": [10, 10]})
model.fit(X_dict=X_dict, Y=labels, study=batch)
```

---

#### `transform`

```python
transform(X_dict)
```

Proyecta nuevos datos sobre los componentes ajustados.

**Parámetros**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `X_dict` | `dict[str, array-like]` | Nuevos bloques con los mismos nombres y el mismo orden de variables que en el ajuste. |

**Devuelve**

`dict[str, np.ndarray]`: variates por bloque `{nombre: (n_nuevas, n_components)}`.

**Errores**

- `RuntimeError`: el modelo no ha sido ajustado.
- `KeyError`: falta un bloque presente durante el ajuste.

**Notas importantes**

- `transform` centra cada bloque nuevo usando la **media de los propios datos nuevos**
  (`np.nanmean` por columna), no la media de entrenamiento. Los `NaN` se sustituyen
  por cero tras el centrado.
- No aplica el escalado de entrenamiento, aunque `scale=True`.
- Alinea las variables **por posición**, no por nombre. El orden de columnas debe ser
  idéntico al usado en `fit`.
- Por lo anterior, `transform(X_dict)` sobre los datos de entrenamiento solo reproduce
  exactamente el **primer** componente de `variates_` en el caso de un único estudio
  con `scale=False`. Los componentes posteriores se calculan en el ajuste sobre datos
  ya deflactados, mientras que `transform` proyecta los datos originales.

**Ejemplo**

```python
test_variates = model.transform(X_test_dict)
mRNA_scores = test_variates["mRNA"]  # forma (n_test, n_components)
```

---

#### `fit_transform`

```python
fit_transform(X_dict, Y=None, study=None)
```

Ajusta el modelo y proyecta los mismos datos en una sola llamada. Equivale a
`fit(X_dict, Y, study).transform(X_dict)`.

**Devuelve**

`dict[str, np.ndarray]`: variates por bloque.

**Ejemplo**

```python
variates = model.fit_transform(X_dict, Y=labels, study=batch)
```

> Debido al comportamiento de `transform`, los variates devueltos pueden diferir de
> `model.variates_` cuando hay varios estudios, cuando `scale=True` o cuando se
> calculan más de un componente.

---

#### `get_biomarkers`

```python
get_biomarkers(block, comp=1)
```

Extrae las variables seleccionadas (loadings distintos de cero) de un bloque y un
componente.

**Parámetros**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `block` | `str` | Nombre del bloque. |
| `comp` | `int` | Índice de componente, base 1. Por defecto `1`. |

**Devuelve**

`pandas.DataFrame` con índice `Variable`, una columna `value` con los pesos de loading
y ordenado por valor absoluto descendente. Solo incluye variables con loading distinto
de cero.

**Errores**

- `RuntimeError`: el modelo no ha sido ajustado.
- `KeyError`: el bloque no existe.
- `ValueError`: el componente está fuera de rango.

**Ejemplo**

```python
biomarkers = model.get_biomarkers("mRNA", comp=1)
print(biomarkers.head(20))
```

---

#### `get_expression_means`

```python
get_expression_means(active_features, groups)
```

Calcula las medias de expresión crudas por grupo para un conjunto de variables activas.
Está pensado para su uso por las funciones de visualización (por ejemplo, el gráfico
circos).

**Parámetros**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `active_features` | `dict` | Diccionario `{bloque: {"X": array (n, p_activas), ...}}` con los datos de las variables activas. |
| `groups` | `dict` | Mapeo `{etiqueta_de_grupo: color}`. El orden de las claves determina el primer grupo cuando no hay `Y`. |

**Devuelve**

`dict`: `{bloque: {grupo: vector_de_medias}}`.

**Comportamiento**

- Si `Y_` está disponible, cada muestra se asigna a un grupo mediante `Y_ == etiqueta`.
- Si `Y_` es 2-D, se aplana con `ravel()` antes de comparar.
- Si no hay `Y_`, todas las muestras se asignan al primer grupo y los demás grupos
  reciben vectores de ceros.
- Los grupos sin muestras reciben un vector de ceros.

**Ejemplo**

```python
active_features = {}
for block in model.block_names_:
    raw = model.loadings_[block][:, :2]
    idx = np.where(np.any(raw != 0, axis=1))[0]
    names = [model.names_[block][i] for i in idx]
    active_features[block] = {
        "names": names,
        "X": model.X_dict_[block][names].values,
    }

groups = {0: "#3498DB", 1: "#E74C3C"}
means = model.get_expression_means(active_features, groups)
```

---

#### `predict`

```python
predict(X_dict)
```

Marcador de posición para predicción supervisada. **No está implementado** en la
versión actual.

**Comportamiento**

- Requiere un modelo supervisado (`Y` proporcionado en `fit`); de lo contrario lanza
  `RuntimeError`.
- Emite una advertencia y devuelve una matriz de ceros de forma
  `(n_samples, n_components_[0])`.

No debe usarse para obtener etiquetas de clase reales. Para predicción en producción,
se recomienda usar `transform` junto con un clasificador externo o el wrapper de R
`MINTRWrapper`.

---

## Objetos auxiliares del módulo

### `MINTResult`

```python
@dataclass
class MINTResult:
    variates: list
    loadings: list
    names: list
    study: np.ndarray
    design: np.ndarray
    ncomp: list
    mode: str
    scheme: str
    crit: list
    iter: list
    tau: list
```

Contenedor de resultados con una estructura compatible con las funciones de
visualización. `MINT.fit` guarda el resultado completo del motor interno en
`self._mint_result` (una instancia de `InternalMintBlockResult`), que expone
`loadings`, `variates`, `crit`, `iter`, `tau` y otra información de diagnóstico.

### `_get_mint_model_attrs`

```python
_get_mint_model_attrs(model)
```

Función auxiliar privada que extrae en un diccionario los atributos que esperan las
funciones de visualización: `loadings_`, `names_`, `X_dict_`, `Y_`, `block_names_`,
`study_`, `n_components_` y `variates_`.

---

## Detalles del algoritmo

### Preprocesado

Cada bloque se centra (y opcionalmente se escala) **dentro de cada estudio**:

```
X_estudio = (X_estudio - media_estudio) / sd_estudio
```

Esto elimina desplazamientos y cambios de escala específicos de cada lote antes de
buscar direcciones integrativas.

### Ruta MINT (`tau=None`)

En cada iteración, el loading de un bloque `q` se calcula agregando las contribuciones
de cada estudio por separado:

```
loading_q = Σ_estudio (X_estudio_qᵀ · z_estudio)
```

donde `z_estudio` es la combinación ponderada por el diseño de los variates de los
demás bloques dentro de ese estudio. La suma por estudio evita que un lote con más
muestras o mayor varianza domine la dirección integrativa.

Después de cada actualización se aplica sparsity (`keepA` o `penalty`) y se normaliza
el loading a norma L2 unitaria.

### Ruta RGCCA (`tau` distinto de `None`)

Se utiliza regularización ridge sobre la covarianza de cada bloque:

```
M_q = pinv(tau_q · I + (1 - tau_q) · Cov(X_q))
loading_q = M_q · (X_qᵀ · z_q)
```

- Si `n >= p`, se usa la formulación primal en el espacio de variables.
- Si `n < p`, se usa la formulación dual en el espacio de muestras.
- `tau="optimal"` estima cada `tau_q` con la fórmula analítica de Strimmer.
- En esta ruta, la sparsity preserva la norma previa al umbralizado.

### Deflación

Tras extraer cada componente, los bloques se deflactan según `mode` para que el
siguiente componente capture información no explicada. La deflación no se aplica en el
último componente.

### Convergencia

El criterio es el cambio máximo en los loadings entre iteraciones consecutivas. Si se
alcanza `max_iter` sin converger, se emite:

```
UserWarning: sparse_mint_block_iteration: no convergio en 100 iteraciones.
```

Para mejorar la convergencia: aumentar `max_iter` (200–500), relajar `tol` (por ejemplo
`1e-4`), reducir el número de componentes o revisar la colinealidad de los datos.

### Esquemas

El esquema controla cómo se ponderan las covarianzas entre bloques:

- `horst`: pesos del diseño, sin reescalado por magnitud.
- `factorial`: pesos proporcionales a la magnitud de la covarianza (criterio de
  covarianzas al cuadrado).
- `centroid`: pesos según el signo de la covarianza.

---

## Compatibilidad con scikit-learn

`MINT` hereda de `BaseEstimator` y `TransformerMixin` (con un fallback interno si
scikit-learn no está instalado):

```python
from sklearn.base import clone

model = MINT(n_components=2, keepA={"A": [5, 5], "B": [4, 4]})
cloned = clone(model)
```

- `get_params()` y `set_params()` funcionan con todos los hiperparámetros del
  constructor.
- `fit` devuelve `self`.
- `transform` devuelve un diccionario, no una matriz; por ello `MINT` no es un
  transformer estándar de pipelines que esperen un único array.
- `predict` es un marcador de posición y no convierte a `MINT` en un clasificador.

---

## Limitaciones y notas

- **Valores faltantes:** el pipeline asume que los bloques no contienen `NaN` durante
  las iteraciones. El centrado usa `nanmean`/`nanstd`, pero el resto del algoritmo no
  implementa el manejo completo de datos faltantes de mixOmics.
- **`transform`:** usa la media de los nuevos datos y no el escalado de entrenamiento;
  alinea variables por posición. Para proyecciones consistentes, preprocesar los datos
  nuevos con la misma lógica y el mismo orden de columnas.
- **`predict`:** no implementado; devuelve ceros.
- **`keepA` y `penalty`:** no deben combinarse; la precedencia difiere entre la ruta
  MINT y la ruta RGCCA.
- **Supervisado:** `mode="canonical"` se sustituye internamente por `"regression"` para
  evitar que el bloque `Y` se agote.
- **`init="svd"`:** requiere `Y`; la ruta RGCCA solo admite `"svd.single"`.
- **`tau` escalar:** es una extensión de la implementación Python; se replica para
  todos los bloques y componentes.
- **Salidas avanzadas:** el motor interno solo calcula `AVE_inner` cuando se solicita
  `all_outputs=True`; no se calculan `AVE_outer`, `prop_expl_var` ni `loadings.Astar`.
- **Signos:** la convención determinista de signo puede invertir un componente respecto
  a una ejecución de R, sin afectar a la interpretación.

---

## Errores comunes

| Error | Causa habitual | Solución |
|---|---|---|
| `ValueError: MINT is a multi-block method and requires at least 2 data blocks` | Solo se pasó un bloque. | Proporcionar al menos dos bloques. |
| `ValueError: Block 'X' has N samples, expected M` | Bloques desalineados. | Asegurar el mismo número y orden de muestras. |
| `ValueError: study has N labels, expected M` | `study` no coincide con el número de muestras. | Revisar la longitud de `study`. |
| `ValueError: n_components list must have length J` | Lista de componentes incompatible con el número de bloques. | Usar un `int` o una lista de longitud `J`. |
| `ValueError: init='svd' requires indY` | `init="svd"` sin `Y`. | Proporcionar `Y` o usar `init="svd.single"`. |
| `RuntimeError: Model has not been fitted` | `transform` o `get_biomarkers` antes de `fit`. | Llamar primero a `fit`. |
| `KeyError: Block 'X' not found in input data` | Falta un bloque en `transform`. | Incluir todos los bloques usados en el ajuste. |
| `UserWarning: ... no convergio ...` | Se alcanzó `max_iter`. | Aumentar `max_iter` o `tol`. |

---

## Referencias

1. Rohart, F., Eslami, A., Mathelin, C., Lé Cao, K.-A. (2017). *MINT: A multivariate
   integrative method to identify a reproducible molecular signature across independent
   experiments and platforms.* BMC Bioinformatics.
2. mixOmics R package: `mint.block()`, `mint.spls()`, `mint.splsda()`,
   `internal_mint.block()`.
3. Tenenhaus, A., Philippe, C., Guillemot, V., Le Cao, K.-A., Grill, J., Frouin, V.
   (2014). *Variable selection for generalized canonical correlation analysis.*
   Biostatistics.
