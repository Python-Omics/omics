#!/usr/bin/env python3
"""
mint_r_vs_python.py
===================

Versión ejecutable como script único del notebook ``mint_r_vs_python.ipynb``.

Hace todo lo que hace el notebook:

1. Carga (o exporta desde R) el dataset ``nutrimouse``.
2. Ejecuta MINT en R con mixOmics (``mint.block.splsda``).
3. Ejecuta MINT en Python con ``pythomics.models.MINT``.
4. Compara loadings, variates, iteraciones y variables seleccionadas.
5. Ejecuta las rutas adicionales solo-Python (MINT no supervisado y RGCCA).
6. Guarda CSVs, configuración y gráficos en ``results/mint_r_vs_python/``.

Uso:
    python mint_r_vs_python.py
    python mint_r_vs_python.py --study-source artificial --no-plots
    python mint_r_vs_python.py --backend rscript --output-dir results/comparacion

Requisitos:
    - R con mixOmics (``install.packages("mixOmics")``).
    - Python con ``numpy`` y ``pandas`` (mínimo).
    - Opcionales: ``scikit-learn``, ``matplotlib`` y ``seaborn``.
    - ``rpy2`` es opcional; si no está, se usa ``Rscript``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Localizar el proyecto y, si hace falta, agregar el .venv al path.
# Esto permite ejecutar el script con cualquier Python (por ejemplo, el de
# euporie/pipx) sin instalar las dependencias en ese intérprete.
# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "pythomics").is_dir():
        return here

    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    candidates.extend([
        Path.home() / "omics",
        Path.home() / "projects" / "omics",
        Path.home() / "repos" / "omics",
    ])
    for pattern in ("*/pythomics", "*/*/pythomics"):
        candidates.extend(p.parent for p in Path.home().glob(pattern) if p.is_dir())

    for candidate in candidates:
        if (candidate / "pythomics").is_dir():
            return candidate
    return here


def _bootstrap_project_venv(root: Path) -> Path | None:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [root / ".venv", root.parent / ".venv", Path.home() / ".venv"]
    for venv in candidates:
        site = venv / "lib" / version / "site-packages"
        if (site / "numpy").exists() and str(site) not in sys.path:
            sys.path.append(str(site))
            print(f"Usando dependencias de {venv}")
            return venv
    return None


ROOT = _find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_bootstrap_project_venv(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - depende del entorno
    plt = None

try:
    import seaborn as sns
except Exception:  # pragma: no cover - depende del entorno
    sns = None

from pythomics.models import MINT, MINTRWrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def r_mixomics_version() -> str:
    try:
        proc = subprocess.run(
            ["Rscript", "--vanilla", "-e", 'cat(as.character(packageVersion("mixOmics")))'],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception as exc:
        return f"no disponible ({exc})"


def export_nutrimouse_with_r(csv_path: Path) -> None:
    """Exporta el dataset nutrimouse de mixOmics a un CSV combinado."""
    r_code = f"""
    suppressPackageStartupMessages(library(mixOmics))
    data(nutrimouse)
    gene <- as.data.frame(nutrimouse$gene)
    lipid <- as.data.frame(nutrimouse$lipid)
    colnames(gene) <- paste0("gene.", colnames(gene))
    colnames(lipid) <- paste0("lipid.", colnames(lipid))
    df <- cbind(gene, lipid,
                diet = as.character(nutrimouse$diet),
                genotype = as.character(nutrimouse$genotype))
    write.csv(df, {json.dumps(str(csv_path))}, row.names = FALSE)
    """
    proc = subprocess.run(
        ["Rscript", "--vanilla", "-e", r_code], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"No se pudo exportar nutrimouse desde R:\n{proc.stderr}")


def load_nutrimouse(results_dir: Path) -> tuple[dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    """Carga/exporta nutrimouse y construye X_dict, Y y study."""
    candidates = [ROOT / "nutrimouse.csv", results_dir / "nutrimouse.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)

    if csv_path is None:
        csv_path = results_dir / "nutrimouse.csv"
        print("nutrimouse.csv no encontrado; exportando desde R...")
        export_nutrimouse_with_r(csv_path)

    df = pd.read_csv(csv_path)

    gene_cols = [c for c in df.columns if c.startswith("gene.")]
    lipid_cols = [c for c in df.columns if c.startswith("lipid.")]
    if not gene_cols or not lipid_cols:
        raise ValueError("El CSV debe contener columnas 'gene.*' y 'lipid.*'.")

    X_dict = {
        "gene": df[gene_cols].astype(float).copy(),
        "lipid": df[lipid_cols].astype(float).copy(),
    }
    X_dict["gene"].columns = [c.removeprefix("gene.") for c in gene_cols]
    X_dict["lipid"].columns = [c.removeprefix("lipid.") for c in lipid_cols]

    sample_index = [f"S{i + 1:02d}" for i in range(len(df))]
    for block in X_dict:
        X_dict[block].index = sample_index

    Y = df["diet"].astype(str).to_numpy()
    study = df["genotype"].astype(str).to_numpy()
    return X_dict, Y, study


def make_study(source: str, n: int, genotype: np.ndarray) -> np.ndarray:
    if source == "genotype":
        return genotype
    if source == "artificial":
        return np.array(["batch_A" if i % 2 == 0 else "batch_B" for i in range(n)])
    raise ValueError("study_source debe ser 'genotype' o 'artificial'.")


def cosine_abs(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(np.dot(a, b)) / denom) if denom else float("nan")


def corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(abs(np.corrcoef(a, b)[0, 1]))


def compare_models(r_model, py_model) -> pd.DataFrame:
    rows = []
    for block in py_model.block_names_:
        Lr = r_model.loadings_[block]
        Lp = py_model.loadings_[block]
        Vr = r_model.variates_[block]
        Vp = py_model.variates_[block]

        for comp in range(Lp.shape[1]):
            lr, lp = Lr[:, comp], Lp[:, comp]
            vr, vp = Vr[:, comp], Vp[:, comp]

            set_r = set(np.flatnonzero(lr))
            set_p = set(np.flatnonzero(lp))
            union = set_r | set_p

            rows.append(
                {
                    "block": block,
                    "component": comp + 1,
                    "loading_cos_abs": cosine_abs(lr, lp),
                    "loading_corr_abs": corr_abs(lr, lp),
                    "variate_cos_abs": cosine_abs(vr, vp),
                    "n_selected_R": len(set_r),
                    "n_selected_Py": len(set_p),
                    "n_overlap": len(set_r & set_p),
                    "jaccard": len(set_r & set_p) / len(union) if union else 1.0,
                }
            )
    return pd.DataFrame(rows)


def print_selected_features(r_model, py_model, n_components: int) -> None:
    print("\n=== Variables seleccionadas ===")
    for block in py_model.block_names_:
        for comp in range(1, n_components + 1):
            names_r = set(r_model.get_biomarkers(block, comp).index)
            names_py = set(py_model.get_biomarkers(block, comp).index)
            print(
                f"[{block}] comp {comp}: "
                f"R={len(names_r)} | Py={len(names_py)} | "
                f"intersección={len(names_r & names_py)} | "
                f"solo_R={sorted(names_r - names_py)} | "
                f"solo_Py={sorted(names_py - names_r)}"
            )


def save_plots(comparison: pd.DataFrame, r_model, py_model, out_dir: Path) -> None:
    if plt is None or sns is None:
        print("matplotlib/seaborn no están instalados; se omiten los gráficos.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, block in zip(axes, py_model.block_names_):
        data = comparison[comparison["block"] == block]
        sns.barplot(data=data, x="component", y="loading_cos_abs", ax=ax, color="#4C72B0")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, ls="--", c="gray", lw=1)
        ax.set_title(f"Loadings |coseno| — {block}")
        ax.set_xlabel("Componente")
        ax.set_ylabel("|coseno| R vs Python")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_cosines.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, block in zip(axes, py_model.block_names_):
        Vr = r_model.variates_[block]
        Vp = py_model.variates_[block]
        for comp in range(Vp.shape[1]):
            a = Vr[:, comp]
            b = Vp[:, comp]
            if np.dot(a, b) < 0:
                a = -a
            ax.scatter(a, b, s=28, alpha=0.8, label=f"comp {comp + 1}")
        ax.set_xlabel("Variates R (signo alineado)")
        ax.set_ylabel("Variates Python")
        ax.set_title(f"Variates — {block}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_variates.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, block in zip(axes, py_model.block_names_):
        Lr = r_model.loadings_[block].copy()
        Lp = py_model.loadings_[block]
        for comp in range(Lp.shape[1]):
            if np.dot(Lr[:, comp], Lp[:, comp]) < 0:
                Lr[:, comp] *= -1

        columns = {}
        for comp in range(Lp.shape[1]):
            columns[f"R comp{comp + 1}"] = Lr[:, comp]
            columns[f"Py comp{comp + 1}"] = Lp[:, comp]

        frame = pd.DataFrame(columns, index=py_model.names_[block])
        py_cols = [f"Py comp{comp + 1}" for comp in range(Lp.shape[1])]
        order = (
            frame[py_cols]
            .abs()
            .max(axis=1)
            .sort_values(ascending=False)
            .index[:20]
        )
        sns.heatmap(
            frame.loc[order],
            cmap="RdBu_r",
            center=0,
            ax=ax,
            cbar_kws={"label": "loading"},
        )
        ax.set_title(f"Top-20 loadings — {block}")
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_loadings.png", dpi=150)
    plt.close(fig)
    print("Gráficos guardados en:", out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-source", choices=["genotype", "artificial"], default="genotype")
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--keep-gene", default="15,15")
    parser.add_argument("--keep-lipid", default="8,8")
    parser.add_argument("--scale", action="store_true", default=True)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--backend", choices=["auto", "rpy2", "rscript"], default="auto")
    parser.add_argument("--output-dir", default="results/mint_r_vs_python")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def parse_keep(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v.strip()]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = {
        "gene": parse_keep(args.keep_gene),
        "lipid": parse_keep(args.keep_lipid),
    }

    print("Repo root :", ROOT)
    print("Python    :", sys.executable)
    print("Rscript   :", shutil.which("Rscript"))
    print("mixOmics  :", r_mixomics_version())

    X_dict, Y, genotype = load_nutrimouse(out_dir)
    study = make_study(args.study_source, len(Y), genotype)

    print("\n=== Datos ===")
    print("Muestras  :", len(Y))
    print("Bloques   :", {k: v.shape for k, v in X_dict.items()})
    print("Dietas    :", dict(pd.Series(Y).value_counts()))
    print("Estudios  :", dict(pd.Series(study).value_counts()))
    print(pd.crosstab(pd.Series(study, name="study"), pd.Series(Y, name="diet")))

    # ------------------------------------------------------------------
    # R / mixOmics
    # ------------------------------------------------------------------
    r_model = None
    try:
        r_model = MINTRWrapper(
            n_components=args.n_components,
            keepX=keep,
            scale=args.scale,
            max_iter=args.max_iter,
            tol=args.tol,
            backend=args.backend,
        ).fit(X_dict, Y=Y, study=study)
    except Exception as exc:
        msg = str(exc)
        print("\nNo se pudo ejecutar MINT en R:", repr(exc))
        if "mixOmics" in msg or "there is no package" in msg:
            print("R funciona, pero falta mixOmics: install.packages('mixOmics').")
        elif "Rscript" in msg or "No such file" in msg:
            print("No se encontró Rscript. Verificá que R esté instalado y en el PATH.")
        else:
            print("Se continúa solo con la implementación Python.")
    else:
        print("\n=== R / mixOmics ===")
        print("Backend   :", r_model.backend_)
        print("ncomp     :", r_model.n_components_)
        print("iter      :", r_model.iter_)
        for block in r_model.block_names_:
            nz = int((r_model.loadings_[block][:, 0] != 0).sum())
            print(
                f"  [{block}] loadings={r_model.loadings_[block].shape}, "
                f"variates={r_model.variates_[block].shape}, nz_comp1={nz}"
            )

    # ------------------------------------------------------------------
    # Python / pythomics
    # ------------------------------------------------------------------
    py_model = MINT(
        n_components=args.n_components,
        keepA=keep,
        tau=None,
        scheme="horst",
        mode="canonical",
        scale=args.scale,
        max_iter=args.max_iter,
        tol=args.tol,
        init="svd.single",
    ).fit(X_dict, Y=Y, study=study)

    print("\n=== Python / pythomics ===")
    print("ncomp     :", py_model.n_components_)
    print("iter      :", py_model._mint_result.iter)
    for block in py_model.block_names_:
        nz = int((py_model.loadings_[block][:, 0] != 0).sum())
        print(
            f"  [{block}] loadings={py_model.loadings_[block].shape}, "
            f"variates={py_model.variates_[block].shape}, nz_comp1={nz}"
        )

    # ------------------------------------------------------------------
    # Comparación R vs Python
    # ------------------------------------------------------------------
    comparison = None
    if r_model is not None:
        comparison = compare_models(r_model, py_model)
        print("\n=== Comparación R vs Python ===")
        print(comparison.round(4).to_string(index=False))
        print_selected_features(r_model, py_model, args.n_components)
        comparison.to_csv(out_dir / "comparison.csv", index=False)

        for block in py_model.block_names_:
            pd.DataFrame(
                r_model.loadings_[block],
                index=r_model.names_[block],
                columns=[f"comp{i + 1}" for i in range(r_model.loadings_[block].shape[1])],
            ).to_csv(out_dir / f"r_loadings_{block}.csv")

            pd.DataFrame(
                py_model.loadings_[block],
                index=py_model.names_[block],
                columns=[f"comp{i + 1}" for i in range(py_model.loadings_[block].shape[1])],
            ).to_csv(out_dir / f"py_loadings_{block}.csv")
    else:
        print("\nComparación omitida: R no está disponible.")

    # ------------------------------------------------------------------
    # Rutas adicionales solo en Python
    # ------------------------------------------------------------------
    py_unsup = MINT(
        n_components=args.n_components,
        keepA=keep,
        tau=None,
        scale=args.scale,
        max_iter=args.max_iter,
        tol=args.tol,
        init="svd.single",
    ).fit(X_dict, study=study)

    py_rgcca = MINT(
        n_components=args.n_components,
        keepA=keep,
        tau="optimal",
        scale=args.scale,
        max_iter=args.max_iter,
        tol=args.tol,
        init="svd.single",
    ).fit(X_dict, study=study)

    print("\n=== Rutas adicionales solo en Python ===")
    print(
        "MINT no supervisado -> ncomp:",
        py_unsup.n_components_,
        "| iter:",
        py_unsup._mint_result.iter,
    )
    print(
        "RGCCA (tau=optimal) -> ncomp:",
        py_rgcca.n_components_,
        "| iter:",
        py_rgcca._mint_result.iter,
    )

    # ------------------------------------------------------------------
    # Guardado de configuración y gráficos
    # ------------------------------------------------------------------
    config = {
        "dataset": "nutrimouse",
        "n_components": args.n_components,
        "keepX": keep,
        "scale": args.scale,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "study_source": args.study_source,
        "r_backend": r_model.backend_ if r_model is not None else None,
        "r_ncomp": r_model.n_components_ if r_model is not None else None,
        "r_iter": r_model.iter_ if r_model is not None else None,
        "py_ncomp": py_model.n_components_,
        "py_iter": py_model._mint_result.iter,
        "py_unsup_iter": py_unsup._mint_result.iter,
        "py_rgcca_iter": py_rgcca._mint_result.iter,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    if not args.no_plots and comparison is not None:
        save_plots(comparison, r_model, py_model, out_dir)

    print("\nResultados guardados en:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
