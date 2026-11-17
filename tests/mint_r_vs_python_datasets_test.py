"""
Numerical parity tests: mixOmics MINT (R) vs pythomics MINT (Python).

For every dataset we compare, with a maximum absolute difference < 1e-6:

* global loadings and variates,
* per-study partial loadings and variates,
* predictions returned by R's ``predict()`` and Python's ``MINT.predict()``.

R and Python may converge to the same fixed point after a different number of
iterations when the default tolerance is used.  The tests therefore run both
implementations for a fixed number of iterations (``tol`` effectively zero,
``max_iter=100``), which removes that source of noise.  Signs are aligned per
block/component before taking absolute differences because SVD sign is
arbitrary.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pythomics.models import MINT

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "mint_r_vs_python" / "tests"
TOL = 1e-6
MAX_ITER = 100
R_TOL = 1e-300  # non-negative for mixOmics; effectively "run to max_iter"


def _r_has_mixomics() -> bool:
    if shutil.which("Rscript") is None:
        return False
    proc = subprocess.run(
        [
            "Rscript",
            "--vanilla",
            "-e",
            'suppressPackageStartupMessages(library(mixOmics)); cat("OK")',
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and "OK" in proc.stdout


pytestmark = pytest.mark.skipif(
    not _r_has_mixomics(),
    reason="R + mixOmics are required for the R/Python parity tests",
)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def _load_nutrimouse():
    df = pd.read_csv(DATA_DIR / "nutrimouse.csv")
    gene_cols = [c for c in df.columns if c.startswith("gene.")]
    lipid_cols = [c for c in df.columns if c.startswith("lipid.")]
    X = {
        "gene": df[gene_cols].to_numpy(dtype=float),
        "lipid": df[lipid_cols].to_numpy(dtype=float),
    }
    Y = df["diet"].to_numpy(dtype=str)
    study = df["genotype"].to_numpy(dtype=str)
    keepA = {"gene": [15, 15], "lipid": [8, 8]}
    return X, Y, study, keepA


def _load_yeast():
    data = pd.read_csv(DATA_DIR / "yeast_export" / "data.csv", index_col=0)
    X = {"X": data.T.to_numpy(dtype=float)}
    Y = (
        pd.read_csv(DATA_DIR / "yeast_export" / "condition.csv", index_col=0)
        .iloc[:, 0]
        .to_numpy(dtype=str)
    )
    study = (
        pd.read_csv(DATA_DIR / "yeast_export" / "strain.csv", index_col=0)
        .iloc[:, 0]
        .to_numpy(dtype=str)
    )
    keepA = {"X": [10, 10]}
    return X, Y, study, keepA


def _load_multidrug():
    base = DATA_DIR / "multidrug_export"
    abc = pd.read_csv(base / "ABC.trans.csv", index_col=0)
    compound = pd.read_csv(base / "compound.csv", index_col=0)
    # mixOmics can handle missing values internally; Python MINT cannot.
    # Mean-impute once and feed the same matrix to both implementations.
    abc = abc.fillna(abc.mean())
    compound = compound.fillna(compound.mean())
    X = {
        "ABC": abc.to_numpy(dtype=float),
        "compound": compound.to_numpy(dtype=float),
    }
    Y = (
        pd.read_csv(base / "cell.line_Class.csv", index_col=0)
        .iloc[:, 0]
        .to_numpy(dtype=str)
    )
    # multidrug has no independent batch/study column.  A single study keeps
    # the comparison well-defined; nutrimouse/yeast exercise partial components.
    study = np.array(["all"] * len(Y), dtype=str)
    keepA = {"ABC": [10, 10], "compound": [10, 10]}
    return X, Y, study, keepA


LOADERS = {
    "nutrimouse": _load_nutrimouse,
    "yeast": _load_yeast,
    "multidrug": _load_multidrug,
}


# ---------------------------------------------------------------------------
# R reference implementation
# ---------------------------------------------------------------------------
def _write_inputs(tmp_path: Path, X, Y, study):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()

    index = [f"S{i:03d}" for i in range(len(Y))]
    for i, block in enumerate(X):
        pd.DataFrame(X[block], index=index).to_csv(data_dir / f"X_{i}.csv")
    pd.DataFrame({"Y": Y}, index=index).to_csv(data_dir / "Y.csv")
    pd.DataFrame({"study": study}, index=index).to_csv(data_dir / "study.csv")
    return data_dir, out_dir


def _run_r_reference(tmp_path: Path, X, Y, study, keepA):
    data_dir, out_dir = _write_inputs(tmp_path, X, Y, study)
    block_names = list(X)

    keep_lines = ["keepX <- list()"]
    for block in block_names:
        values = ", ".join(str(int(v)) for v in keepA[block])
        keep_lines.append(f'keepX[["{block}"]] <- c({values})')
    keep_code = "\n".join(keep_lines)

    block_vec = "c(" + ", ".join(f'"{b}"' for b in block_names) + ")"
    script = textwrap.dedent(
        f"""
        options(warn = -1)
        suppressPackageStartupMessages(library(mixOmics))

        data_dir <- "{data_dir}"
        out_dir <- "{out_dir}"
        block_names <- {block_vec}

        X <- list()
        for (i in seq_along(block_names)) {{
            b <- block_names[[i]]
            path <- file.path(data_dir, paste0("X_", i - 1, ".csv"))
            X[[b]] <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE))
        }}

        Y_df <- read.csv(file.path(data_dir, "Y.csv"), row.names = 1,
                         check.names = FALSE)
        study <- read.csv(file.path(data_dir, "study.csv"), row.names = 1,
                          check.names = FALSE)[[1]]
        Y <- as.factor(Y_df[[1]])

        {keep_code}

        if (length(block_names) == 1) {{
            res <- mint.splsda(
                X[[1]], Y, study = study, ncomp = 2,
                keepX = keepX[[block_names[[1]]]],
                scale = TRUE, tol = {R_TOL}, max.iter = {MAX_ITER}
            )
            p <- predict(res, newdata = X[[1]], newdata.scale = res$X,
                         study.test = study)
        }} else {{
            res <- mint.block.splsda(
                X, Y, study = study, ncomp = 2, keepX = keepX,
                scale = TRUE, tol = {R_TOL}, max.iter = {MAX_ITER}
            )
            p <- predict(res, newdata = X, newdata.scale = res$X,
                         study.test = study)
        }}

        for (b in block_names) {{
            write.csv(res$loadings[[b]],
                      file.path(out_dir, paste0("global_loadings_", b, ".csv")))
            write.csv(res$variates[[b]],
                      file.path(out_dir, paste0("global_variates_", b, ".csv")))
            for (s in names(res$loadings.partial[[b]])) {{
                write.csv(
                    res$loadings.partial[[b]][[s]],
                    file.path(out_dir, paste0("partial_loadings_", b, "_", s, ".csv"))
                )
                write.csv(
                    res$variates.partial[[b]][[s]],
                    file.path(out_dir, paste0("partial_variates_", b, "_", s, ".csv"))
                )
            }}
        }}

        pred <- p$predict
        if (is.list(pred)) {{
            for (b in names(pred)) {{
                for (k in seq_len(dim(pred[[b]])[3])) {{
                    write.csv(
                        pred[[b]][,,k],
                        file.path(out_dir, paste0("predict_", b, "_comp", k, ".csv"))
                    )
                }}
            }}
        }} else {{
            for (k in seq_len(dim(pred)[3])) {{
                write.csv(
                    pred[,,k],
                    file.path(out_dir, paste0("predict_X_comp", k, ".csv"))
                )
            }}
        }}

        write.csv(
            data.frame(comp = seq_along(res$iter), iter = as.integer(res$iter)),
            file.path(out_dir, "info_iter.csv"),
            row.names = FALSE
        )
        """
    )

    script_path = tmp_path / "fit_mint.R"
    script_path.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        ["Rscript", "--vanilla", str(script_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "R MINT reference failed.\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )

    def read_csv(name):
        return pd.read_csv(out_dir / name, index_col=0).to_numpy(dtype=float)

    ref = {
        "global_loadings": {},
        "global_variates": {},
        "partial_loadings": {},
        "partial_variates": {},
        "predictions": {},
    }
    for block in block_names:
        ref["global_loadings"][block] = read_csv(f"global_loadings_{block}.csv")
        ref["global_variates"][block] = read_csv(f"global_variates_{block}.csv")

        partial_loadings = {}
        partial_variates = {}
        for path in sorted(out_dir.glob(f"partial_loadings_{block}_*.csv")):
            level = path.stem.removeprefix(f"partial_loadings_{block}_")
            partial_loadings[level] = read_csv(path.name)
        for path in sorted(out_dir.glob(f"partial_variates_{block}_*.csv")):
            level = path.stem.removeprefix(f"partial_variates_{block}_")
            partial_variates[level] = read_csv(path.name)
        ref["partial_loadings"][block] = partial_loadings
        ref["partial_variates"][block] = partial_variates

        pred_paths = sorted(out_dir.glob(f"predict_{block}_comp*.csv"))
        if not pred_paths:
            pred_paths = sorted(out_dir.glob("predict_X_comp*.csv"))
        ref["predictions"][block] = np.stack(
            [read_csv(path.name) for path in pred_paths], axis=2
        )

    return ref


# ---------------------------------------------------------------------------
# Comparison figures
# ---------------------------------------------------------------------------
def _sign_alignment(reference, actual):
    sign = np.sign(np.sum(reference * actual, axis=0))
    sign[sign == 0] = 1.0
    return sign


def _max_abs_diff(reference, actual):
    return float(np.max(np.abs(reference - actual)))


def _collect_metrics(dataset, X, ref, model, py_pred):
    rows = []
    for block in X:
        ref_loadings = ref["global_loadings"][block]
        py_loadings = model.loadings_[block]
        sign = _sign_alignment(ref_loadings, py_loadings)

        for comp in range(py_loadings.shape[1]):
            rows.append(
                {
                    "dataset": dataset,
                    "block": block,
                    "metric": "global_loadings",
                    "study": "all",
                    "component": comp + 1,
                    "max_abs_diff": _max_abs_diff(
                        ref_loadings[:, comp] * sign[comp], py_loadings[:, comp]
                    ),
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "block": block,
                    "metric": "global_variates",
                    "study": "all",
                    "component": comp + 1,
                    "max_abs_diff": _max_abs_diff(
                        ref["global_variates"][block][:, comp] * sign[comp],
                        model.variates_[block][:, comp],
                    ),
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "block": block,
                    "metric": "predictions",
                    "study": "all",
                    "component": comp + 1,
                    "max_abs_diff": _max_abs_diff(
                        ref["predictions"][block][:, :, comp],
                        py_pred[block][:, :, comp],
                    ),
                }
            )

        for study in model.loadings_partial_[block]:
            for comp in range(py_loadings.shape[1]):
                rows.append(
                    {
                        "dataset": dataset,
                        "block": block,
                        "metric": "partial_loadings",
                        "study": study,
                        "component": comp + 1,
                        "max_abs_diff": _max_abs_diff(
                            ref["partial_loadings"][block][study][:, comp]
                            * sign[comp],
                            model.loadings_partial_[block][study][:, comp],
                        ),
                    }
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "block": block,
                        "metric": "partial_variates",
                        "study": study,
                        "component": comp + 1,
                        "max_abs_diff": _max_abs_diff(
                            ref["partial_variates"][block][study][:, comp]
                            * sign[comp],
                            model.variates_partial_[block][study][:, comp],
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _scatter_identity(ax, x, y, title, xlabel, ylabel):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    ax.scatter(x, y, s=10, alpha=0.65, edgecolors="none")
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    if hi == lo:
        hi = lo + 1.0
    pad = 0.05 * (hi - lo)
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color="black",
        lw=1,
        ls="--",
        zorder=0,
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def _save_comparison_plots(dataset, X, ref, model, py_pred):
    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _collect_metrics(dataset, X, ref, model, py_pred)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    summary = (
        metrics.groupby(["metric", "block"])["max_abs_diff"]
        .max()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.42 * len(summary))))
    labels = [f"{metric} / {block}" for metric, block in summary.index]
    values = np.maximum(summary.to_numpy(), 1e-16)
    ax.barh(labels, values, color="#4C72B0")
    ax.set_xscale("log")
    ax.axvline(TOL, color="#C44E52", ls="--", lw=1.2, label=f"umbral {TOL:.0e}")
    ax.set_xlabel("max |R - Python|")
    ax.set_title(f"Paridad MINT R vs Python — {dataset}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_max_abs_diff.png", dpi=150)
    plt.close(fig)

    for block in X:
        ref_loadings = ref["global_loadings"][block]
        py_loadings = model.loadings_[block]
        sign = _sign_alignment(ref_loadings, py_loadings)
        ncomp = py_loadings.shape[1]

        fig, axes = plt.subplots(1, ncomp, figsize=(4.2 * ncomp, 4), squeeze=False)
        for comp, ax in enumerate(axes[0]):
            _scatter_identity(
                ax,
                ref_loadings[:, comp] * sign[comp],
                py_loadings[:, comp],
                f"{block} — comp {comp + 1}",
                "Loadings R",
                "Loadings Python",
            )
        fig.tight_layout()
        fig.savefig(out_dir / f"{block}_global_loadings.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, ncomp, figsize=(4.2 * ncomp, 4), squeeze=False)
        for comp, ax in enumerate(axes[0]):
            _scatter_identity(
                ax,
                ref["global_variates"][block][:, comp] * sign[comp],
                model.variates_[block][:, comp],
                f"{block} — comp {comp + 1}",
                "Variates R",
                "Variates Python",
            )
        fig.tight_layout()
        fig.savefig(out_dir / f"{block}_global_variates.png", dpi=150)
        plt.close(fig)

        studies = list(model.loadings_partial_[block])
        fig, axes = plt.subplots(
            len(studies),
            2 * ncomp,
            figsize=(4.2 * 2 * ncomp, 3.4 * len(studies)),
            squeeze=False,
        )
        for row, study in enumerate(studies):
            for comp in range(ncomp):
                _scatter_identity(
                    axes[row, comp],
                    ref["partial_loadings"][block][study][:, comp] * sign[comp],
                    model.loadings_partial_[block][study][:, comp],
                    f"{block} / {study} — comp {comp + 1}",
                    "Loadings parciales R",
                    "Loadings parciales Python",
                )
                _scatter_identity(
                    axes[row, ncomp + comp],
                    ref["partial_variates"][block][study][:, comp] * sign[comp],
                    model.variates_partial_[block][study][:, comp],
                    f"{block} / {study} — comp {comp + 1}",
                    "Variates parciales R",
                    "Variates parciales Python",
                )
        fig.tight_layout()
        fig.savefig(out_dir / f"{block}_partials.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, ncomp, figsize=(4.2 * ncomp, 4), squeeze=False)
        for comp, ax in enumerate(axes[0]):
            _scatter_identity(
                ax,
                ref["predictions"][block][:, :, comp],
                py_pred[block][:, :, comp],
                f"{block} — comp {comp + 1}",
                "Predicciones R",
                "Predicciones Python",
            )
        fig.tight_layout()
        fig.savefig(out_dir / f"{block}_predictions.png", dpi=150)
        plt.close(fig)

    return metrics


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------
def _assert_close(name, reference, actual):
    diff = np.max(np.abs(reference - actual))
    assert diff < TOL, f"{name}: max abs diff {diff:.3e} >= {TOL:.0e}"


@pytest.mark.parametrize("dataset", list(LOADERS), ids=list(LOADERS))
def test_mint_python_matches_r(dataset, tmp_path):
    X, Y, study, keepA = LOADERS[dataset]()

    ref = _run_r_reference(tmp_path, X, Y, study, keepA)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MINT(
            n_components=2,
            keepA=keepA,
            tau=None,
            scheme="horst",
            mode="regression",
            scale=True,
            max_iter=MAX_ITER,
            tol=0.0,
            init="svd.single",
        ).fit(X, Y=Y, study=study)

    py_pred = model.predict(X, study=study)
    _save_comparison_plots(dataset, X, ref, model, py_pred)

    for block in X:
        ref_loadings = ref["global_loadings"][block]
        py_loadings = model.loadings_[block]
        sign = _sign_alignment(ref_loadings, py_loadings)

        _assert_close(
            f"{dataset}/{block}/global_loadings",
            ref_loadings * sign,
            py_loadings,
        )
        _assert_close(
            f"{dataset}/{block}/global_variates",
            ref["global_variates"][block] * sign,
            model.variates_[block],
        )

        for level in model.loadings_partial_[block]:
            _assert_close(
                f"{dataset}/{block}/{level}/partial_loadings",
                ref["partial_loadings"][block][level] * sign,
                model.loadings_partial_[block][level],
            )
            _assert_close(
                f"{dataset}/{block}/{level}/partial_variates",
                ref["partial_variates"][block][level] * sign,
                model.variates_partial_[block][level],
            )

        _assert_close(
            f"{dataset}/{block}/predictions",
            ref["predictions"][block],
            py_pred[block],
        )

        for comp in range(py_loadings.shape[1]):
            assert set(np.flatnonzero(ref_loadings[:, comp])) == set(
                np.flatnonzero(py_loadings[:, comp])
            ), f"{dataset}/{block}/comp{comp + 1}: selected features differ"
