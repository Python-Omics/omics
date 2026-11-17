"""
Pruebas experimentales para acreditar las dos ciencias de la ingeniería (E3):

1. Álgebra lineal
   Se compara la ruta "formar X'X + resolver el problema propio" (memoria O(p^2), 
   SVD/NIPALS, que no materializan la matriz p x p y usan memoria O(n*p) por componente
   para p = 1.000, 5.000 y 20.000. Además se barre n = 20, 40, 80, 160 con p = 1.000, 5.000.

2. Regularización
   Se comparan las correlaciones canónicas de rcc() propias contra mixOmics::rcc() sobre el 
   dataset Nutrimouse, a lo largo de una grilla de lambda. Además se registra el número de condición de Cxx + lambda*I y
   Cyy + lambda*I para evidenciar la estabilización numérica.

py -3.13 -m pytest -s tests/nutrimouse_linalg_regularization_test.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import time
import tracemalloc
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
import scipy
from scipy.sparse.linalg import eigsh

rpy2 = pytest.importorskip("rpy2", reason="rpy2 no está instalado")

import rpy2.robjects as robjects
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import PackageNotInstalledError, importr

from pythomics.models.rcc import rcc
from pythomics.models.rcc_utilities import cov_pairwise
from pythomics.tools.pca import _nipals

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

try:
    import psutil

    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False

try:
    importr("mixOmics")
    _HAS_MIXOMICS = True
except PackageNotInstalledError:
    _HAS_MIXOMICS = False

pytestmark = pytest.mark.skipif(
    not _HAS_MIXOMICS, reason="El paquete de R 'mixOmics' no está instalado"
)

logging.getLogger("pythomics").setLevel(logging.WARNING)

N_SAMPLES = 40
LATENT_DIM = 5
P_VALUES = (1_000, 5_000, 20_000)
N_SWEEP_VALUES = (20, 40, 80, 160)
P_SWEEP_VALUES = (1_000, 5_000)
SWEEP_FIXED_P = P_SWEEP_VALUES[-1]
NCOMP = 2
RNG_SEED = 2026
LAMBDAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
NIPALS_TOL = 1e-6
REGULARIZATION_TOL = 1e-6

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "engineering_sciences"


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _time_call(fn: Callable[[], Any]) -> tuple[float, Any]:
    gc.collect()
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def _peak_memory_call(fn: Callable[[], Any]) -> tuple[float, Any]:
    gc.collect()
    tracemalloc.start()
    try:
        result = fn()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak_bytes / (1024**2), result


def _loglog_slope(x: Any, y: Any) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def _format_tolerance(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-")


def _print_table(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print(df.to_string(index=False, float_format=lambda value: f"{value:.4g}"))


def _environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "cpu_count": os.cpu_count(),
    }
    if _HAS_MPL:
        metadata["matplotlib"] = matplotlib.__version__
    if _HAS_PSUTIL:
        memory = psutil.virtual_memory()
        metadata["ram_total_gb"] = round(memory.total / 1e9, 2)
        metadata["ram_available_gb_at_start"] = round(memory.available / 1e9, 2)
    return metadata


def _write_environment_metadata() -> None:
    _ensure_results_dir()
    path = RESULTS_DIR / "environment.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_environment_metadata(), handle, indent=2)


def _fast_path_guard() -> None:
    if not _HAS_PSUTIL:
        return
    available_gb = psutil.virtual_memory().available / 1e9
    if available_gb < 6.0:
        pytest.skip(
            "El benchmark de p=20.000 requiere ~6 GB libres para X'X; "
            f"disponibles: {available_gb:.1f} GB"
        )


def _make_benchmark_matrix(p: int, n: int = N_SAMPLES) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED + p)
    latent = rng.normal(size=(n, LATENT_DIM))
    loadings = rng.normal(size=(LATENT_DIM, p))
    X = latent @ loadings + 0.5 * rng.normal(size=(n, p))
    X -= X.mean(axis=0)
    return np.ascontiguousarray(X)


def _fit_power_law_exponents(
    df: pd.DataFrame, value_column: str
) -> dict[str, float]:
    n = df["n"].to_numpy(dtype=float)
    p = df["p"].to_numpy(dtype=float)
    y = df[value_column].to_numpy(dtype=float)
    mask = (
        np.isfinite(n)
        & np.isfinite(p)
        & np.isfinite(y)
        & (n > 0)
        & (p > 0)
        & (y > 0)
    )
    design = np.column_stack(
        [np.ones(int(mask.sum())), np.log(n[mask]), np.log(p[mask])]
    )
    target = np.log(y[mask])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    ss_res = float(np.sum((target - predicted) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "intercept": float(coefficients[0]),
        "n_exponent": float(coefficients[1]),
        "p_exponent": float(coefficients[2]),
        "r_squared": r_squared,
    }


def _save_linear_algebra_figure(
    df: pd.DataFrame, slopes: dict[str, float]
) -> Path | None:
    if not _HAS_MPL:
        return None

    p = df["p"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    series = (
        ("X'X (formación)", "gram_form_time_s", "gram_form_peak_mb", "o", "#d62728"),
        ("X'X + eigsh(top-2)", "gram_top2_time_s", "gram_top2_peak_mb", "s", "#ff7f0e"),
        ("SVD (NumPy)", "svd_time_s", "svd_peak_mb", "^", "#1f77b4"),
        (
            f"NIPALS (2 comps, tol={_format_tolerance(NIPALS_TOL)})",
            "nipals_time_s",
            "nipals_peak_mb",
            "D",
            "#2ca02c",
        ),
    )

    for ax, ylabel, use_time in (
        (axes[0], "tiempo (s)", True),
        (axes[1], "memoria pico (MB)", False),
    ):
        for label, time_col, mem_col, marker, color in series:
            column = time_col if use_time else mem_col
            ax.plot(p, df[column], marker=marker, color=color, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(p)
        ax.set_xticklabels([f"{int(value):,}".replace(",", ".") for value in p])
        ax.set_xlabel("número de variables p")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")

    p0 = p[0]
    axes[0].plot(
        p,
        df["gram_form_time_s"].iloc[0] * (p / p0) ** 2,
        "k--",
        lw=1.2,
        label=r"referencia $\propto p^2$",
    )
    axes[0].plot(
        p,
        df["svd_time_s"].iloc[0] * (p / p0),
        "k:",
        lw=1.2,
        label=r"referencia $\propto p$",
    )
    axes[0].set_title("(a) Tiempo de ejecución")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        p,
        df["gram_top2_peak_mb"].iloc[0] * (p / p0) ** 2,
        "k--",
        lw=1.2,
        label=r"referencia $\propto p^2$",
    )
    axes[1].plot(
        p,
        df["svd_peak_mb"].iloc[0] * (p / p0),
        "k:",
        lw=1.2,
        label=r"referencia $\propto p$",
    )
    axes[1].set_title("(b) Memoria pico (tracemalloc)")
    axes[1].legend(fontsize=8)

    ax = axes[2]
    labels = ["X'X\nforma", "X'X+\ntop-2", "SVD", "NIPALS"]
    time_slopes = [
        slopes["time_gram_form"],
        slopes["time_gram_top2"],
        slopes["time_svd"],
        slopes["time_nipals"],
    ]
    memory_slopes = [
        slopes["mem_gram_form"],
        slopes["mem_gram_top2"],
        slopes["mem_svd"],
        slopes["mem_nipals"],
    ]
    positions = np.arange(len(labels))
    ax.bar(positions - 0.2, time_slopes, width=0.4, label="tiempo", color="#4c72b0")
    ax.bar(positions + 0.2, memory_slopes, width=0.4, label="memoria", color="#dd8452")
    ax.axhline(1.0, color="black", ls=":", lw=1.2, label=r"$O(p)$")
    ax.axhline(2.0, color="black", ls="--", lw=1.2, label=r"$O(p^2)$")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("pendiente log-log observada")
    ax.set_title("(c) Escalamiento medido")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    path = RESULTS_DIR / "linear_algebra_complexity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run_linear_algebra_benchmark() -> tuple[pd.DataFrame, dict[str, Any]]:
    _ensure_results_dir()
    _fast_path_guard()
    _write_environment_metadata()

    rows: list[dict[str, Any]] = []
    for p in P_VALUES:
        Xc = _make_benchmark_matrix(p)

        time_svd, (U, S, Vt) = _time_call(
            lambda: np.linalg.svd(Xc, full_matrices=False)
        )
        singular_values = S[:NCOMP].copy()
        recon_svd = (U[:, :NCOMP] * S[:NCOMP]) @ Vt[:NCOMP, :]
        recon_svd_rel = float(
            np.linalg.norm(Xc - recon_svd) / np.linalg.norm(Xc)
        )
        del U, S, Vt, recon_svd
        gc.collect()

        memory_svd, svd_mem_result = _peak_memory_call(
            lambda: np.linalg.svd(Xc, full_matrices=False)
        )
        del svd_mem_result
        gc.collect()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            time_nipals, nipals_result = _time_call(
                lambda: _nipals(Xc, ncomp=NCOMP, tol=NIPALS_TOL, max_iter=500)
            )
        nipals_warnings = len(caught)
        nipals_eig = np.asarray(nipals_result["eig"], dtype=float)
        nipals_loadings = np.asarray(nipals_result["p"], dtype=float)
        nipals_scores = np.asarray(nipals_result["t"], dtype=float)
        nipals_orthogonality = float(
            np.max(np.abs(nipals_loadings.T @ nipals_loadings - np.eye(NCOMP)))
        )
        recon_nipals = (nipals_scores * nipals_eig) @ nipals_loadings.T
        recon_nipals_rel = float(
            np.linalg.norm(Xc - recon_nipals) / np.linalg.norm(Xc)
        )
        del nipals_result, recon_nipals
        gc.collect()

        memory_nipals, nipals_mem_result = _peak_memory_call(
            lambda: _nipals(Xc, ncomp=NCOMP, tol=NIPALS_TOL, max_iter=500)
        )
        del nipals_mem_result
        gc.collect()

        time_gram_form, gram_time_result = _time_call(lambda: Xc.T @ Xc)
        del gram_time_result
        gc.collect()

        memory_gram_form, gram_mem_result = _peak_memory_call(lambda: Xc.T @ Xc)
        del gram_mem_result
        gc.collect()

        def _gram_top2() -> tuple[np.ndarray, np.ndarray]:
            gram = Xc.T @ Xc
            values = eigsh(
                gram,
                k=NCOMP,
                which="LA",
                tol=1e-6,
                maxiter=2000,
                v0=np.ones(p),
                return_eigenvectors=False,
            )
            return gram, np.sort(values)[::-1]

        time_gram_top2, (gram_top2_time_result, top_values) = _time_call(_gram_top2)
        del gram_top2_time_result
        gc.collect()

        memory_gram_top2, (gram_top2_mem_result, _) = _peak_memory_call(_gram_top2)
        del gram_top2_mem_result
        gc.collect()

        svd_nipals_rel = float(
            np.max(
                np.abs(singular_values - nipals_eig[:NCOMP])
                / np.maximum(singular_values, 1e-300)
            )
        )
        gram_svd_rel = float(
            np.max(
                np.abs(np.sqrt(top_values) - singular_values)
                / np.maximum(singular_values, 1e-300)
            )
        )

        gram_gflop = 2.0 * N_SAMPLES * p**2 / 1e9
        rows.append(
            {
                "p": p,
                "n": N_SAMPLES,
                "ncomp": NCOMP,
                "input_mb": Xc.nbytes / (1024**2),
                "gram_form_time_s": time_gram_form,
                "gram_form_peak_mb": memory_gram_form,
                "gram_top2_time_s": time_gram_top2,
                "gram_top2_peak_mb": memory_gram_top2,
                "svd_time_s": time_svd,
                "svd_peak_mb": memory_svd,
                "nipals_time_s": time_nipals,
                "nipals_peak_mb": memory_nipals,
                "gram_est_gflop": gram_gflop,
                "gram_effective_gflop_s": gram_gflop / time_gram_form,
                "svd_nipals_rel_sv_diff": svd_nipals_rel,
                "gram_svd_rel_sv_diff": gram_svd_rel,
                "svd_recon_rel_error": recon_svd_rel,
                "nipals_recon_rel_error": recon_nipals_rel,
                "nipals_loading_orthogonality_error": nipals_orthogonality,
                "nipals_warnings": nipals_warnings,
                "gram_vs_nipals_time_ratio": time_gram_form / time_nipals,
                "gram_top2_vs_svd_time_ratio": time_gram_top2 / time_svd,
            }
        )

        del Xc
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "linear_algebra_complexity.csv", index=False)

    slopes = {
        "time_gram_form": _loglog_slope(df["p"], df["gram_form_time_s"]),
        "time_gram_top2": _loglog_slope(df["p"], df["gram_top2_time_s"]),
        "time_svd": _loglog_slope(df["p"], df["svd_time_s"]),
        "time_nipals": _loglog_slope(df["p"], df["nipals_time_s"]),
        "mem_gram_form": _loglog_slope(df["p"], df["gram_form_peak_mb"]),
        "mem_gram_top2": _loglog_slope(df["p"], df["gram_top2_peak_mb"]),
        "mem_svd": _loglog_slope(df["p"], df["svd_peak_mb"]),
        "mem_nipals": _loglog_slope(df["p"], df["nipals_peak_mb"]),
    }

    _print_table(df, "ÁLGEBRA LINEAL: coste de X'X frente a SVD/NIPALS")
    print("\nPendientes log-log observadas (aprox. 1 lineal en p, aprox. 2 cuadratica en p):")
    for name, value in slopes.items():
        print(f"  {name:>24}: {value:.3f}")

    figure = _save_linear_algebra_figure(df, slopes)
    details = {
        "design": (
            "Matriz sintética con n=40, 5 factores latentes y ruido gaussiano; "
            "misma semilla por p. La matriz de entrada se crea antes de medir, "
            "por lo que la memoria reportada es la adicional usada por el método."
        ),
        "p_values": list(P_VALUES),
        "n_samples": N_SAMPLES,
        "nipals_tolerance": NIPALS_TOL,
        "slopes": slopes,
        "figure": str(figure) if figure else None,
    }
    with (RESULTS_DIR / "linear_algebra_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(details, handle, indent=2)

    return df, details


def _save_dimension_sweep_figure(
    df: pd.DataFrame, exponents: dict[str, dict[str, float]]
) -> Path | None:
    if not _HAS_MPL:
        return None

    fixed = df[df["p"] == SWEEP_FIXED_P].sort_values("n")
    n = fixed["n"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    ax = axes[0]
    series = (
        ("Matriz X (n×p)", "input_mb", "o", "#7f7f7f"),
        ("SVD", "svd_peak_mb", "^", "#1f77b4"),
        (
            f"NIPALS (tol={_format_tolerance(NIPALS_TOL)})",
            "nipals_peak_mb",
            "D",
            "#2ca02c",
        ),
        ("X'X (covarianza completa)", "gram_form_peak_mb", "s", "#d62728"),
    )
    for label, column, marker, color in series:
        ax.plot(
            n,
            fixed[column].to_numpy(),
            marker=marker,
            color=color,
            linewidth=1.6,
            label=label,
        )
    n0 = n[0]
    ax.plot(
        n,
        fixed["input_mb"].iloc[0] * (n / n0),
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=r"referencia $\propto n$",
    )
    ax.plot(
        n,
        np.full_like(n, fixed["gram_form_peak_mb"].iloc[0]),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"referencia $\propto n^0$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(int(value)) for value in n])
    ax.set_xlabel("número de muestras n")
    ax.set_ylabel("memoria pico adicional (MB)")
    ax.set_title(f"(a) Memoria vs n con p={SWEEP_FIXED_P}".replace(",", "."))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["Matriz X", "SVD", "NIPALS", "X'X"]
    columns = ["input_mb", "svd_peak_mb", "nipals_peak_mb", "gram_form_peak_mb"]
    n_exponents = [exponents[column]["n_exponent"] for column in columns]
    p_exponents = [exponents[column]["p_exponent"] for column in columns]
    positions = np.arange(len(labels))
    ax.bar(
        positions - 0.2,
        n_exponents,
        width=0.4,
        label="exponente de n",
        color="#4c72b0",
    )
    ax.bar(
        positions + 0.2,
        p_exponents,
        width=0.4,
        label="exponente de p",
        color="#dd8452",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="black", linestyle=":", linewidth=1.0)
    ax.axhline(2, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("exponente ajustado")
    ax.set_title(r"(b) Ajuste $\log M = a + b\log n + c\log p$")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = RESULTS_DIR / "dimension_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _save_memory_collapse_figure(df: pd.DataFrame) -> Path | None:
    if not _HAS_MPL:
        return None

    theoretical_mb_per_entry = 8 / (1024**2)
    fixed = df[df["p"] == SWEEP_FIXED_P].sort_values("n")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    ax = axes[0]
    entries = fixed["n"] * fixed["p"]
    for label, column, marker, color in (
        ("Matriz X", "input_mb", "o", "#7f7f7f"),
        ("SVD", "svd_peak_mb", "^", "#1f77b4"),
        (
            f"NIPALS (tol={_format_tolerance(NIPALS_TOL)})",
            "nipals_peak_mb",
            "D",
            "#2ca02c",
        ),
    ):
        ax.plot(
            fixed["n"],
            fixed[column] / entries,
            marker=marker,
            color=color,
            linewidth=1.6,
            label=label,
        )
    ax.axhline(
        theoretical_mb_per_entry,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="8 bytes por entrada",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(fixed["n"])
    ax.set_xticklabels([str(int(value)) for value in fixed["n"]])
    ax.set_xlabel("número de muestras n")
    ax.set_ylabel(r"memoria pico / $(n\cdot p)$ (MB por entrada)")
    ax.set_title(r"(a) Datos y bajo rango: $O(n\cdot p)$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    for n_value, marker in zip(sorted(df["n"].unique()), ("o", "s", "^", "D")):
        subset = df[df["n"] == n_value].sort_values("p")
        ax.plot(
            subset["p"],
            subset["gram_form_peak_mb"] / subset["p"] ** 2,
            marker=marker,
            linewidth=1.5,
            label=f"n={int(n_value)}",
        )
    ax.axhline(
        theoretical_mb_per_entry,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="8 bytes por entrada",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sorted(df["p"].unique()))
    ax.set_xticklabels([str(int(value)) for value in sorted(df["p"].unique())])
    ax.set_xlabel("número de variables p")
    ax.set_ylabel(r"memoria pico de X'X / $p^2$ (MB por entrada)")
    ax.set_title(r"(b) Covarianza completa: $O(p^2)$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = RESULTS_DIR / "memory_collapse.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_dimension_sweep() -> tuple[pd.DataFrame, dict[str, Any]]:
    _ensure_results_dir()
    rows: list[dict[str, Any]] = []

    for n in N_SWEEP_VALUES:
        for p in P_SWEEP_VALUES:
            Xc = _make_benchmark_matrix(p, n)
            input_mb = Xc.nbytes / (1024**2)

            time_svd, (U, S, Vt) = _time_call(
                lambda: np.linalg.svd(Xc, full_matrices=False)
            )
            svd_values = S[:NCOMP].copy()
            del U, S, Vt
            gc.collect()

            memory_svd, svd_memory_result = _peak_memory_call(
                lambda: np.linalg.svd(Xc, full_matrices=False)
            )
            del svd_memory_result
            gc.collect()

            time_nipals, nipals_result = _time_call(
                lambda: _nipals(Xc, ncomp=NCOMP, tol=NIPALS_TOL, max_iter=500)
            )
            nipals_values = np.asarray(nipals_result["eig"], dtype=float)[:NCOMP]
            del nipals_result
            gc.collect()

            memory_nipals, nipals_memory_result = _peak_memory_call(
                lambda: _nipals(Xc, ncomp=NCOMP, tol=NIPALS_TOL, max_iter=500)
            )
            del nipals_memory_result
            gc.collect()

            time_gram, gram_result = _time_call(lambda: Xc.T @ Xc)
            del gram_result
            gc.collect()

            memory_gram, gram_memory_result = _peak_memory_call(lambda: Xc.T @ Xc)
            del gram_memory_result
            gc.collect()

            relative_singular_value_difference = float(
                np.max(
                    np.abs(svd_values - nipals_values)
                    / np.maximum(svd_values, 1e-300)
                )
            )
            rows.append(
                {
                    "n": n,
                    "p": p,
                    "input_mb": input_mb,
                    "svd_time_s": time_svd,
                    "svd_peak_mb": memory_svd,
                    "nipals_time_s": time_nipals,
                    "nipals_peak_mb": memory_nipals,
                    "gram_form_time_s": time_gram,
                    "gram_form_peak_mb": memory_gram,
                    "svd_nipals_rel_sv_diff": relative_singular_value_difference,
                }
            )
            del Xc, svd_values, nipals_values
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "dimension_sweep.csv", index=False)

    memory_columns = (
        "input_mb",
        "svd_peak_mb",
        "nipals_peak_mb",
        "gram_form_peak_mb",
    )
    time_columns = ("svd_time_s", "nipals_time_s", "gram_form_time_s")
    memory_exponents = {
        column: _fit_power_law_exponents(df, column) for column in memory_columns
    }
    time_exponents = {
        column: _fit_power_law_exponents(df, column) for column in time_columns
    }

    expected_law = {
        "input_mb": "O(n·p)",
        "svd_peak_mb": "O(n·p)",
        "nipals_peak_mb": "O(n·p)",
        "gram_form_peak_mb": "O(p²)",
        "svd_time_s": "O(n²·p) para SVD denso",
        "nipals_time_s": "O(n·p) por iteración",
        "gram_form_time_s": "O(n·p²) teórico; domina p² si n << p",
    }
    exponent_rows = []
    for column in memory_columns + time_columns:
        fit = memory_exponents.get(column) or time_exponents[column]
        exponent_rows.append(
            {
                "variable": column,
                "n_exponent": fit["n_exponent"],
                "p_exponent": fit["p_exponent"],
                "r_squared": fit["r_squared"],
                "expected_law": expected_law[column],
            }
        )
    exponent_df = pd.DataFrame(exponent_rows)

    _print_table(
        df,
        "BARRIDO n×p: memoria y tiempo de SVD/NIPALS frente a X'X",
    )
    _print_table(
        exponent_df,
        "EXPONENTES AJUSTADOS: log(valor) = a + b·log(n) + c·log(p)",
    )

    dimension_figure = _save_dimension_sweep_figure(df, memory_exponents)
    collapse_figure = _save_memory_collapse_figure(df)

    details = {
        "design": (
            "Matriz sintética con 5 factores latentes y ruido gaussiano para "
            "cada combinación de n y p; la matriz se crea antes de medir, por "
            "lo que la memoria reportada es la adicional usada por el método."
        ),
        "n_values": list(N_SWEEP_VALUES),
        "p_values": list(P_SWEEP_VALUES),
        "fixed_p_for_cross_section": SWEEP_FIXED_P,
        "nipals_tolerance": NIPALS_TOL,
        "fit_model": "log(value) = a + b*log(n) + c*log(p)",
        "memory_exponents": memory_exponents,
        "time_exponents": time_exponents,
        "figures": {
            "dimension_sweep": str(dimension_figure) if dimension_figure else None,
            "memory_collapse": str(collapse_figure) if collapse_figure else None,
        },
    }
    with (RESULTS_DIR / "dimension_sweep_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(details, handle, indent=2)

    return df, details


def _prepare_r_reference_matrices(X: np.ndarray, Y: np.ndarray) -> None:
    n, p = X.shape
    q = Y.shape[1]
    robjects.r.assign("flat_X", FloatVector(X.ravel().tolist()))
    robjects.r.assign("flat_Y", FloatVector(Y.ravel().tolist()))
    robjects.r.assign("n_ref", int(n))
    robjects.r.assign("p_ref", int(p))
    robjects.r.assign("q_ref", int(q))
    robjects.r("X_ref <- matrix(flat_X, nrow=n_ref, ncol=p_ref, byrow=TRUE)")
    robjects.r("Y_ref <- matrix(flat_Y, nrow=n_ref, ncol=q_ref, byrow=TRUE)")


def _rcc_r_reference(
    lambda1: float, lambda2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robjects.r.assign("lambda1_ref", float(lambda1))
    robjects.r.assign("lambda2_ref", float(lambda2))
    robjects.r.assign("ncomp_ref", int(NCOMP))
    result = robjects.r(
        "r <- rcc(X_ref, Y_ref, ncomp=ncomp_ref, method='ridge', "
        "lambda1=lambda1_ref, lambda2=lambda2_ref); r"
    )
    cor = np.asarray(result.rx2("cor"), dtype=float)
    xcoef = np.asarray(result.rx2("loadings").rx2("X"), dtype=float)
    ycoef = np.asarray(result.rx2("loadings").rx2("Y"), dtype=float)
    return cor, xcoef, ycoef


def _save_regularization_figure(df: pd.DataFrame) -> Path | None:
    if not _HAS_MPL:
        return None

    lambdas = df["lambda"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axes[0, 0]
    for component in range(1, NCOMP + 1):
        ax.semilogx(
            lambdas,
            df[f"cor{component}_py"],
            "-o",
            label=f"Python c{component}",
        )
        ax.semilogx(
            lambdas,
            df[f"cor{component}_r"],
            "--s",
            mfc="none",
            label=f"mixOmics c{component}",
        )
    ax.set_xlabel(r"$\lambda=\lambda_1=\lambda_2$")
    ax.set_ylabel("correlación canónica")
    ax.set_title("(a) rCCA: correlaciones vs λ")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    ax.loglog(
        lambdas,
        df["max_abs_cor_diff"].clip(lower=1e-18),
        "-o",
        color="#c44e52",
    )
    ax.axhline(
        REGULARIZATION_TOL,
        color="black",
        ls="--",
        lw=1.2,
        label=f"tolerancia {_format_tolerance(REGULARIZATION_TOL)}",
    )
    ax.set_xlabel(r"$\lambda=\lambda_1=\lambda_2$")
    ax.set_ylabel(r"$\max_h |cor_h^{Python} - cor_h^{mixOmics}|$")
    ax.set_title("(b) Acuerdo numérico con mixOmics")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    ax.loglog(lambdas, df["cond_Cxx"], "-o", label="Cxx + λI (120×120)")
    ax.loglog(lambdas, df["cond_Cyy"], "-s", label="Cyy + λI (21×21)")
    ax.set_xlabel(r"$\lambda=\lambda_1=\lambda_2$")
    ax.set_ylabel("número de condición")
    ax.set_title("(c) Estabilización numérica")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    ax.semilogx(lambdas, df["py_time_s"], "-o", label="Python")
    ax.semilogx(lambdas, df["r_time_s"], "-s", label="R / mixOmics")
    ax.set_xlabel(r"$\lambda=\lambda_1=\lambda_2$")
    ax.set_ylabel("tiempo (s)")
    ax.set_title("(d) Coste de ejecución")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    path = RESULTS_DIR / "regularization_vs_lambda.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run_regularization_experiment(
    X: np.ndarray, Y: np.ndarray
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _ensure_results_dir()
    _prepare_r_reference_matrices(X, Y)

    n, p = X.shape
    q = Y.shape[1]
    covariance_x = cov_pairwise(X)
    covariance_y = cov_pairwise(Y)
    rank_x = int(np.linalg.matrix_rank(X - X.mean(axis=0)))
    rank_y = int(np.linalg.matrix_rank(Y - Y.mean(axis=0)))

    rows: list[dict[str, Any]] = []
    for lambda_value in LAMBDAS:
        time_python, python_result = _time_call(
            lambda value=lambda_value: rcc(
                X,
                Y,
                ncomp=NCOMP,
                method="ridge",
                lambda1=value,
                lambda2=value,
            )
        )
        time_r, (cor_r, xcoef_r, ycoef_r) = _time_call(
            lambda value=lambda_value: _rcc_r_reference(value, value)
        )

        max_cor_diff = float(np.max(np.abs(python_result.cor - cor_r)))
        max_cor_diff_ncomp = float(
            np.max(np.abs(python_result.cor[:NCOMP] - cor_r[:NCOMP]))
        )
        row: dict[str, Any] = {
            "lambda": lambda_value,
            "cond_Cxx": float(np.linalg.cond(covariance_x + lambda_value * np.eye(p))),
            "cond_Cyy": float(np.linalg.cond(covariance_y + lambda_value * np.eye(q))),
            "rank_X": rank_x,
            "rank_Y": rank_y,
            "py_time_s": time_python,
            "r_time_s": time_r,
            "max_abs_cor_diff": max_cor_diff,
            "max_abs_cor_diff_ncomp": max_cor_diff_ncomp,
        }
        for component in range(NCOMP):
            row[f"cor{component + 1}_py"] = float(python_result.cor[component])
            row[f"cor{component + 1}_r"] = float(cor_r[component])
            row[f"xcoef_corr_c{component + 1}"] = float(
                abs(
                    np.corrcoef(
                        python_result.xcoef[:, component], xcoef_r[:, component]
                    )[0, 1]
                )
            )
            row[f"ycoef_corr_c{component + 1}"] = float(
                abs(
                    np.corrcoef(
                        python_result.ycoef[:, component], ycoef_r[:, component]
                    )[0, 1]
                )
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "regularization_vs_lambda.csv", index=False)
    _print_table(df, "REGULARIZACIÓN rCCA: Nutrimouse frente a mixOmics")

    figure = _save_regularization_figure(df)
    coefficient_columns = [column for column in df.columns if "coef_corr" in column]
    details = {
        "dataset": "mixOmics::nutrimouse",
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "rank_X": rank_x,
        "rank_Y": rank_y,
        "lambdas": list(LAMBDAS),
        "ncomp": NCOMP,
        "agreement_tolerance": REGULARIZATION_TOL,
        "max_abs_cor_diff": float(df["max_abs_cor_diff"].max()),
        "min_coefficient_correlation": float(
            df[coefficient_columns].to_numpy().min()
        ),
        "figure": str(figure) if figure else None,
    }
    with (RESULTS_DIR / "regularization_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(details, handle, indent=2)

    return df, details


@pytest.fixture(scope="session")
def nutrimouse_data() -> tuple[np.ndarray, np.ndarray]:
    robjects.r("data(nutrimouse, package='mixOmics')")
    gene = np.asarray(robjects.r("nutrimouse$gene"), dtype=float).T
    lipid = np.asarray(robjects.r("nutrimouse$lipid"), dtype=float).T
    return gene, lipid


@pytest.fixture(scope="session")
def benchmark_results() -> tuple[pd.DataFrame, dict[str, Any]]:
    return run_linear_algebra_benchmark()


@pytest.fixture(scope="session")
def dimension_sweep_results() -> tuple[pd.DataFrame, dict[str, Any]]:
    return run_dimension_sweep()


@pytest.fixture(scope="session")
def regularization_results(
    nutrimouse_data: tuple[np.ndarray, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return run_regularization_experiment(*nutrimouse_data)


def test_linear_algebra_scales_as_p2_vs_p(benchmark_results) -> None:
    df, details = benchmark_results
    p = df["p"].to_numpy()
    assert list(p) == list(P_VALUES)

    slopes = details["slopes"]
    assert slopes["mem_gram_top2"] > 1.7
    assert slopes["mem_gram_form"] > 1.7
    assert slopes["mem_svd"] < 1.5
    assert slopes["mem_nipals"] < 1.5
    assert slopes["time_gram_form"] > 1.5

    last = df.iloc[-1]
    assert last["gram_top2_peak_mb"] > 100 * last["svd_peak_mb"]
    assert last["gram_top2_peak_mb"] > 100 * last["nipals_peak_mb"]
    assert last["gram_top2_time_s"] > 10 * max(
        last["svd_time_s"], last["nipals_time_s"]
    )

    expected_quadratic_ratio = (P_VALUES[-1] / P_VALUES[0]) ** 2
    gram_memory_ratio = (
        df["gram_top2_peak_mb"].iloc[-1] / df["gram_top2_peak_mb"].iloc[0]
    )
    assert 0.5 * expected_quadratic_ratio < gram_memory_ratio < 2.0 * expected_quadratic_ratio

    expected_linear_ratio = P_VALUES[-1] / P_VALUES[0]
    for column in ("svd_peak_mb", "nipals_peak_mb"):
        linear_ratio = df[column].iloc[-1] / df[column].iloc[0]
        assert 0.5 * expected_linear_ratio < linear_ratio < 2.5 * expected_linear_ratio


def test_linear_algebra_numerical_agreement(benchmark_results) -> None:
    df, _ = benchmark_results
    assert df["svd_nipals_rel_sv_diff"].max() < 1e-4
    assert df["gram_svd_rel_sv_diff"].max() < 1e-8
    assert df["nipals_loading_orthogonality_error"].max() < 1e-8
    assert df["svd_recon_rel_error"].max() < 1.0
    assert df["nipals_recon_rel_error"].max() < 1.0
    assert (
        np.max(
            np.abs(
                df["svd_recon_rel_error"].to_numpy()
                - df["nipals_recon_rel_error"].to_numpy()
            )
        )
        < 1e-4
    )


def test_memory_scales_as_np_for_data_and_low_rank_vs_p2_for_gram(
    dimension_sweep_results,
) -> None:
    df, details = dimension_sweep_results
    assert list(df["n"].unique()) == list(N_SWEEP_VALUES)
    assert list(df["p"].unique()) == list(P_SWEEP_VALUES)
    assert len(df) == len(N_SWEEP_VALUES) * len(P_SWEEP_VALUES)

    expected_exponents = {
        "input_mb": (1.0, 1.0),
        "svd_peak_mb": (1.0, 1.0),
        "nipals_peak_mb": (1.0, 1.0),
        "gram_form_peak_mb": (0.0, 2.0),
    }
    for column, (expected_n, expected_p) in expected_exponents.items():
        fit = details["memory_exponents"][column]
        assert abs(fit["n_exponent"] - expected_n) < 0.25, (column, fit)
        assert abs(fit["p_exponent"] - expected_p) < 0.25, (column, fit)
        assert fit["r_squared"] > 0.98, (column, fit)

    fixed = df[df["p"] == SWEEP_FIXED_P].sort_values("n")
    gram_memory_ratio = (
        fixed["gram_form_peak_mb"].max() / fixed["gram_form_peak_mb"].min()
    )
    assert gram_memory_ratio < 1.05

    expected_n_ratio = N_SWEEP_VALUES[-1] / N_SWEEP_VALUES[0]
    for column in ("svd_peak_mb", "nipals_peak_mb"):
        ratio = fixed[column].iloc[-1] / fixed[column].iloc[0]
        assert 0.5 * expected_n_ratio < ratio < 1.5 * expected_n_ratio

    assert df["svd_nipals_rel_sv_diff"].max() < 1e-3


def test_nutrimouse_svd_and_nipals_agree(nutrimouse_data) -> None:
    _ensure_results_dir()
    X, Y = nutrimouse_data
    rows: list[dict[str, Any]] = []

    for block_name, block in (("gene", X), ("lipid", Y)):
        centered = block - block.mean(axis=0)
        ncomp = min(NCOMP, min(centered.shape))

        time_svd, (U, S, Vt) = _time_call(
            lambda: np.linalg.svd(centered, full_matrices=False)
        )
        singular_values = S[:ncomp].copy()
        recon_svd = (U[:, :ncomp] * S[:ncomp]) @ Vt[:ncomp, :]
        recon_svd_rel = float(
            np.linalg.norm(centered - recon_svd) / np.linalg.norm(centered)
        )
        del U, S, Vt, recon_svd
        gc.collect()

        memory_svd, svd_mem_result = _peak_memory_call(
            lambda: np.linalg.svd(centered, full_matrices=False)
        )
        del svd_mem_result
        gc.collect()

        time_nipals, nipals_result = _time_call(
            lambda: _nipals(centered, ncomp=ncomp, tol=NIPALS_TOL, max_iter=500)
        )
        nipals_eig = np.asarray(nipals_result["eig"], dtype=float)[:ncomp]
        loadings = np.asarray(nipals_result["p"], dtype=float)
        scores = np.asarray(nipals_result["t"], dtype=float)
        orthogonality = float(
            np.max(np.abs(loadings.T @ loadings - np.eye(ncomp)))
        )
        recon_nipals = (scores * nipals_eig) @ loadings.T
        recon_nipals_rel = float(
            np.linalg.norm(centered - recon_nipals) / np.linalg.norm(centered)
        )
        del nipals_result, recon_nipals
        gc.collect()

        memory_nipals, nipals_mem_result = _peak_memory_call(
            lambda: _nipals(centered, ncomp=ncomp, tol=NIPALS_TOL, max_iter=500)
        )
        del nipals_mem_result
        gc.collect()

        rows.append(
            {
                "block": block_name,
                "n": centered.shape[0],
                "p": centered.shape[1],
                "ncomp": ncomp,
                "svd_time_s": time_svd,
                "svd_peak_mb": memory_svd,
                "nipals_time_s": time_nipals,
                "nipals_peak_mb": memory_nipals,
                "svd_nipals_rel_sv_diff": float(
                    np.max(
                        np.abs(singular_values - nipals_eig)
                        / np.maximum(singular_values, 1e-300)
                    )
                ),
                "svd_recon_rel_error": recon_svd_rel,
                "nipals_recon_rel_error": recon_nipals_rel,
                "nipals_loading_orthogonality_error": orthogonality,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "nutrimouse_svd_nipals.csv", index=False)
    _print_table(df, "NUTRIMOUSE: SVD frente a NIPALS por bloque")

    assert df["svd_nipals_rel_sv_diff"].max() < 1e-5
    assert df["nipals_loading_orthogonality_error"].max() < 1e-8
    assert (
        np.max(
            np.abs(
                df["svd_recon_rel_error"].to_numpy()
                - df["nipals_recon_rel_error"].to_numpy()
            )
        )
        < 1e-6
    )


def test_nipals_handles_missing_nutrimouse_values(nutrimouse_data) -> None:
    _ensure_results_dir()
    X, _ = nutrimouse_data
    centered = X - X.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)

    rng = np.random.default_rng(RNG_SEED + 7)
    missing_mask = rng.random(centered.shape) < 0.01
    missing_data = centered.copy()
    missing_data[missing_mask] = np.nan

    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.svd(missing_data, full_matrices=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        time_nipals, nipals_result = _time_call(
            lambda: _nipals(missing_data, ncomp=NCOMP, tol=NIPALS_TOL, max_iter=500)
        )

    assert np.isfinite(nipals_result["t"]).all()
    assert np.isfinite(nipals_result["p"]).all()

    eigen_values = np.asarray(nipals_result["eig"], dtype=float)
    loadings = np.asarray(nipals_result["p"], dtype=float)
    scores = np.asarray(nipals_result["t"], dtype=float)
    relative_singular_value_error = float(
        np.max(
            np.abs(eigen_values[:NCOMP] - singular_values[:NCOMP])
            / np.maximum(singular_values[:NCOMP], 1e-300)
        )
    )
    reconstruction = (scores * eigen_values) @ loadings.T
    observed = ~missing_mask
    observed_reconstruction_error = float(
        np.linalg.norm((centered - reconstruction)[observed])
        / np.linalg.norm(centered[observed])
    )

    df = pd.DataFrame(
        [
            {
                "block": "gene",
                "n": centered.shape[0],
                "p": centered.shape[1],
                "ncomp": NCOMP,
                "missing_fraction": float(missing_mask.mean()),
                "n_missing": int(missing_mask.sum()),
                "nipals_time_s": time_nipals,
                "svd_nipals_rel_sv_diff": relative_singular_value_error,
                "observed_recon_rel_error": observed_reconstruction_error,
                "warnings": len(caught),
            }
        ]
    )
    df.to_csv(RESULTS_DIR / "nutrimouse_missing_nipals.csv", index=False)
    _print_table(df, "NUTRIMOUSE con 1% de valores faltantes: NIPALS")

    assert relative_singular_value_error < 0.05
    assert observed_reconstruction_error < 1.0


def test_classical_cca_is_infeasible_without_regularization(nutrimouse_data) -> None:
    X, Y = nutrimouse_data
    assert X.shape[1] > X.shape[0] - 1

    with pytest.raises(np.linalg.LinAlgError):
        rcc(X, Y, ncomp=NCOMP, method="ridge", lambda1=0.0, lambda2=0.0)

    regularized = rcc(X, Y, ncomp=NCOMP, method="ridge", lambda1=0.1, lambda2=0.1)
    assert np.isfinite(regularized.cor).all()
    assert np.all(regularized.cor >= -1e-8)
    assert np.all(regularized.cor <= 1.0 + 1e-8)


def test_regularization_matches_mixomics_and_conditions_matrices(
    regularization_results,
) -> None:
    df, details = regularization_results
    assert len(df) == len(LAMBDAS)
    assert df["max_abs_cor_diff"].max() < REGULARIZATION_TOL
    assert df["max_abs_cor_diff_ncomp"].max() < REGULARIZATION_TOL

    coefficient_columns = [column for column in df.columns if "coef_corr" in column]
    assert df[coefficient_columns].to_numpy().min() > 0.999999

    correlations = df["cor1_py"].to_numpy()
    assert np.all(np.diff(correlations) <= 1e-12)
    assert correlations[0] > 0.99
    assert correlations[-1] < 0.1

    condition_x = df["cond_Cxx"].to_numpy()
    condition_y = df["cond_Cyy"].to_numpy()
    assert np.all(np.diff(condition_x) <= 0)
    assert np.all(np.diff(condition_y) <= 0)
    assert details["rank_X"] < details["X_shape"][1]
    assert details["rank_Y"] == details["Y_shape"][1]

    assert (df["py_time_s"] > 0).all()
    assert (df["r_time_s"] > 0).all()


def test_reports_are_written(
    benchmark_results, dimension_sweep_results, regularization_results
) -> None:
    assert (RESULTS_DIR / "linear_algebra_complexity.csv").exists()
    assert (RESULTS_DIR / "dimension_sweep.csv").exists()
    assert (RESULTS_DIR / "dimension_sweep_metadata.json").exists()
    assert (RESULTS_DIR / "regularization_vs_lambda.csv").exists()
    assert (RESULTS_DIR / "environment.json").exists()
    if _HAS_MPL:
        assert (RESULTS_DIR / "linear_algebra_complexity.png").exists()
        assert (RESULTS_DIR / "dimension_sweep.png").exists()
        assert (RESULTS_DIR / "memory_collapse.png").exists()
        assert (RESULTS_DIR / "regularization_vs_lambda.png").exists()
