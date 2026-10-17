"""
mint_pipeline.py
================

Pipeline personalizable para **ingresar datasets multiómicos y ejecutar MINT**
(Multi-group INtegrative Tensor, implementación pura-Python en
``pythomics.models.MINT``). Pensado para probar/explorar el modelo con datos
propios o sintéticos.

Formas de entrada (elegí una):

  1. Datos sintéticos (para pruebas rápidas):
        python mint_pipeline.py --synthetic --output-dir results/mint

  2. Un objeto MuData (.h5mu):
        python mint_pipeline.py --mudata datos.h5mu \\
            --study-col study --target-col subtype --n-components 2 \\
            --output-dir results/mint

  3. Bloques desde CSV (muestras en filas, variables en columnas):
        python mint_pipeline.py \\
            --block mRNA=mRNA.csv --block miRNA=miRNA.csv \\
            --metadata samples.csv --study-col study --target-col subtype \\
            --n-components 2 --output-dir results/mint

Etiquetas / metadatos:
  --metadata    CSV con índice = muestra y columnas de estudio y/o respuesta.
  --study-col   columna de metadatos con el estudio/lote (batch).
  --target-col  columna de metadatos con la respuesta -> ejecuta MINT supervisado.

Modelo (todos opcionales):
  --n-components N            nº de componentes (default: 2)
  --keepA NAME=N[,N,...]      variables a retener por bloque y componente
                              (repetible; p. ej. --keepA mRNA=10,10)
  --penalty NAME=P            penalización L1 por bloque (repetible)
  --tau {none,optimal,VALOR}  regularización ridge (default: none -> ruta MINT)
  --scheme {horst,factorial,centroid}
  --mode {canonical,classic,invariant,regression}
  --init {svd,svd.single}
  --no-scale                  no escalar los bloques
  --max-iter, --tol

Salida:
  --output-dir DIR   Guarda loadings_, variates_ y un resumen (~ summary.txt).
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pythomics.models import MINT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mint_pipeline")


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def read_block_csv(path: str, transpose: bool = False) -> pd.DataFrame:
    """Lee un bloque desde CSV (índice = muestras, columnas = variables)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el bloque: {path}")
    df = pd.read_csv(path, index_col=0)
    if transpose:
        df = df.T
    df.index = df.index.astype(str)
    df = df[~df.index.duplicated(keep="first")]
    return df


def read_metadata(path: str) -> pd.DataFrame:
    """Lee un CSV de metadatos (índice = muestra)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de metadatos: {path}")
    meta = pd.read_csv(path, index_col=0)
    meta.index = meta.index.astype(str)
    return meta


def parse_name_value(items: Optional[Sequence[str]]) -> Dict[str, str]:
    """Convierte ['mRNA=a.csv', 'miRNA=b.csv'] en {'mRNA': 'a.csv', ...}."""
    out: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Formato inválido {item!r}; se espera NAME=ruta.")
        name, value = item.split("=", 1)
        out[name.strip()] = value.strip()
    return out


def parse_keepA(items: Optional[Sequence[str]]) -> Optional[Dict[str, List[int]]]:
    """--keepA mRNA=10,10  ->  {'mRNA': [10, 10]}"""
    if not items:
        return None
    keep: Dict[str, List[int]] = {}
    for item in items:
        name, value = item.split("=", 1)
        keep[name.strip()] = [int(v) for v in value.split(",") if v.strip() != ""]
    return keep


def parse_penalty(items: Optional[Sequence[str]]) -> Optional[Dict[str, float]]:
    """--penalty mRNA=0.5  ->  {'mRNA': 0.5}"""
    if not items:
        return None
    pen: Dict[str, float] = {}
    for item in items:
        name, value = item.split("=", 1)
        pen[name.strip()] = float(value)
    return pen


def parse_tau(value: str):
    """'none' -> None, 'optimal' -> 'optimal', '0.5' -> 0.5 (escalar)."""
    if value is None or value.lower() == "none":
        return None
    if value.lower() == "optimal":
        return "optimal"
    return float(value)


def align_blocks(
    blocks: Dict[str, pd.DataFrame],
    meta: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """Intersecta las muestras comunes a todos los bloques (y metadatos)."""
    sample_sets = [set(df.index) for df in blocks.values()]
    if meta is not None and len(meta) > 0:
        sample_sets.append(set(meta.index))
    common = set.intersection(*sample_sets)
    if not common:
        raise ValueError(
            "No hay muestras comunes entre los bloques/metadatos. "
            "Revisá los identificadores de muestra."
        )
    common = sorted(common)
    logger.info("Muestras comunes alineadas: %d", len(common))
    aligned = {name: df.loc[common].copy() for name, df in blocks.items()}
    aligned_meta = meta.loc[common].copy() if meta is not None else None
    return aligned, aligned_meta


def load_mudata(path: str):
    """Carga un MuData y devuelve (blocks, meta)."""
    try:
        import mudata as md
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "Para usar --mudata se necesita el paquete 'mudata'."
        ) from exc
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el MuData: {path}")
    mdata = md.read(path)
    blocks = {name: mod.to_df() for name, mod in mdata.mod.items()}
    meta = mdata.obs.copy()
    meta.index = meta.index.astype(str)
    for name in blocks:
        blocks[name].index = blocks[name].index.astype(str)
    return blocks, meta


def make_synthetic(
    n_studies: int = 3,
    n_per_study: int = 20,
    seed: int = 0,
) -> Tuple[Dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    """Genera 2 bloques con una señal latente compartida + efecto de estudio.

    Retorna (blocks, study, Y) donde Y es una clasificación de 3 clases derivada
    del factor latente.
    """
    rng = np.random.default_rng(seed)
    n = n_studies * n_per_study
    study = np.repeat([f"study{i + 1}" for i in range(n_studies)], n_per_study)

    z = rng.normal(size=(n, 1))
    w_a = rng.normal(size=(1, 40))
    w_b = rng.normal(size=(1, 30))

    A = z @ w_a + 0.5 * rng.normal(size=(n, 40))
    B = z @ w_b + 0.5 * rng.normal(size=(n, 30))

    # Efecto de lote (por estudio) que MINT debe corregir
    for si, s in enumerate(np.unique(study)):
        mask = study == s
        A[mask] += si * 1.0
        B[mask] += si * 1.0

    blocks = {
        "mRNA": pd.DataFrame(A, columns=[f"mRNA_{i}" for i in range(A.shape[1])]),
        "miRNA": pd.DataFrame(B, columns=[f"miRNA_{i}" for i in range(B.shape[1])]),
    }

    # Respuesta: 3 clases por cuantiles del factor latente
    q = np.quantile(z[:, 0], [1 / 3, 2 / 3])
    Y = np.digitize(z[:, 0], q)  # 0, 1, 2
    Y = np.array([f"C{v}" for v in Y])
    return blocks, study, Y


# ---------------------------------------------------------------------------
# Ejecución de MINT
# ---------------------------------------------------------------------------
def run_mint(
    blocks: Dict[str, pd.DataFrame],
    study: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    n_components: int = 2,
    keepA: Optional[Dict[str, List[int]]] = None,
    penalty: Optional[Dict[str, float]] = None,
    tau=None,
    scheme: str = "horst",
    mode: str = "canonical",
    scale: bool = True,
    init: str = "svd.single",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> MINT:
    """Construye y ajusta un modelo MINT con la configuración dada."""
    model = MINT(
        n_components=n_components,
        keepA=keepA,
        penalty=penalty,
        tau=tau,
        scheme=scheme,
        mode=mode,
        scale=scale,
        init=init,
        max_iter=max_iter,
        tol=tol,
    )
    logger.info(
        "Ejecutando MINT | bloques=%s | n_components=%d | supervisado=%s | tau=%s",
        list(blocks.keys()),
        n_components,
        Y is not None,
        tau,
    )
    model.fit(blocks, Y=Y, study=study)
    return model


# ---------------------------------------------------------------------------
# Guardado de resultados
# ---------------------------------------------------------------------------
def save_results(model: MINT, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for block in model.block_names_:
        loadings = pd.DataFrame(
            model.loadings_[block],
            index=model.names_[block],
            columns=[f"comp{i + 1}" for i in range(model.loadings_[block].shape[1])],
        )
        loadings.to_csv(os.path.join(out_dir, f"loadings_{block}.csv"))

        variates = pd.DataFrame(
            model.variates_[block],
            index=model.X_dict_[block].index
            if isinstance(model.X_dict_[block], pd.DataFrame)
            else None,
            columns=[f"comp{i + 1}" for i in range(model.variates_[block].shape[1])],
        )
        variates.to_csv(os.path.join(out_dir, f"variates_{block}.csv"))

    if model.Y_ is not None:
        pd.Series(np.asarray(model.Y_).ravel(), name="Y").to_csv(
            os.path.join(out_dir, "Y.csv")
        )
    if model.study_ is not None:
        pd.Series(np.asarray(model.study_).ravel(), name="study").to_csv(
            os.path.join(out_dir, "study.csv")
        )

    logger.info("Resultados guardados en: %s", out_dir)


def print_summary(model: MINT) -> None:
    print("\n=== Resumen MINT ===")
    print(f"Bloques: {model.block_names_}")
    print(f"n_components_: {model.n_components_}")
    print(f"Supervisado: {model.indY_ is not None}")
    if model.study_ is not None:
        vals, counts = np.unique(np.asarray(model.study_), return_counts=True)
        print("Estudios: " + ", ".join(f"{v}={c}" for v, c in zip(vals, counts)))
    for block in model.block_names_:
        shapes = model.variates_[block].shape
        print(f"  [{block}] variates={shapes}, n_vars={len(model.names_[block])}")
    print("\nTop biomarcadores (componente 1):")
    for block in model.block_names_:
        top = model.get_biomarkers(block, comp=1).head(5)
        print(f"  [{block}]")
        for name, row in top.iterrows():
            print(f"    {name}: {row['value']:.4f}")
    print("====================\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline personalizable para ejecutar MINT sobre datasets multiómicos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_argument_group("entrada de datos")
    src.add_argument("--synthetic", action="store_true", help="Generar datos sintéticos.")
    src.add_argument("--mudata", help="Ruta a un objeto MuData (.h5mu).")
    src.add_argument(
        "--block",
        action="append",
        metavar="NAME=PATH.csv",
        help="Bloque CSV (repetible). Ej: --block mRNA=mrna.csv",
    )
    src.add_argument("--metadata", help="CSV de metadatos (índice = muestra).")
    src.add_argument(
        "--transpose-blocks",
        action="store_true",
        help="Transponer los bloques CSV (muestras en columnas).",
    )

    lab = p.add_argument_group("etiquetas")
    lab.add_argument("--study-col", default=None, help="Columna de metadatos con el estudio.")
    lab.add_argument("--target-col", default=None, help="Columna de metadatos con Y (supervisado).")

    mdl = p.add_argument_group("modelo")
    mdl.add_argument("--n-components", type=int, default=2)
    mdl.add_argument("--keepA", action="append", metavar="NAME=N[,N...]")
    mdl.add_argument("--penalty", action="append", metavar="NAME=P")
    mdl.add_argument("--tau", default="none", help="none | optimal | valor escalar")
    mdl.add_argument("--scheme", default="horst", choices=["horst", "factorial", "centroid"])
    mdl.add_argument(
        "--mode",
        default="canonical",
        choices=["canonical", "classic", "invariant", "regression"],
    )
    mdl.add_argument("--init", default="svd.single", choices=["svd", "svd.single"])
    mdl.add_argument("--no-scale", dest="scale", action="store_false")
    mdl.add_argument("--max-iter", type=int, default=100)
    mdl.add_argument("--tol", type=float, default=1e-6)

    out = p.add_argument_group("salida")
    out.add_argument("--output-dir", default=None, help="Directorio donde guardar resultados.")
    return p


def _load_inputs(args) -> Tuple[Dict[str, pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
    """Resuelve la fuente de datos y devuelve (blocks, study, Y)."""
    if args.synthetic:
        blocks, study, Y = make_synthetic()
        return blocks, np.asarray(study), np.asarray(Y)

    if args.mudata:
        blocks, meta = load_mudata(args.mudata)
    elif args.block:
        specs = parse_name_value(args.block)
        blocks = {
            name: read_block_csv(path, transpose=args.transpose_blocks)
            for name, path in specs.items()
        }
        meta = read_metadata(args.metadata) if args.metadata else None
        blocks, meta = align_blocks(blocks, meta)
    else:
        raise SystemExit(
            "Indicá una fuente de datos: --synthetic, --mudata o --block ..."
        )

    if len(blocks) < 2:
        raise SystemExit(f"MINT requiere al menos 2 bloques (recibidos {len(blocks)}).")

    study = None
    Y = None
    if meta is not None:
        if args.study_col:
            if args.study_col not in meta.columns:
                raise SystemExit(f"'{args.study_col}' no está en los metadatos.")
            study = meta[args.study_col].values
        if args.target_col:
            if args.target_col not in meta.columns:
                raise SystemExit(f"'{args.target_col}' no está en los metadatos.")
            Y = meta[args.target_col].values

    return blocks, study, Y


def main(argv: Optional[Sequence[str]] = None) -> MINT:
    args = build_parser().parse_args(argv)

    blocks, study, Y = _load_inputs(args)

    model = run_mint(
        blocks,
        study=study,
        Y=Y,
        n_components=args.n_components,
        keepA=parse_keepA(args.keepA),
        penalty=parse_penalty(args.penalty),
        tau=parse_tau(args.tau),
        scheme=args.scheme,
        mode=args.mode,
        scale=args.scale,
        init=args.init,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    print_summary(model)

    if args.output_dir:
        save_results(model, args.output_dir)

    return model


if __name__ == "__main__":
    main()
