"""
Tests for the R-backed MINT wrapper (``pythomics.models.MINTRWrapper``).

The whole module is skipped when Rscript or the R package ``mixOmics`` is not
available. The parity test against the pure-Python ``MINT`` uses the
``nutrimouse.csv`` file when present and is skipped otherwise.

Execution:
    python -m pytest -s tests/mint_r_wrapper_test.py
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pythomics.models import MINT, MINTRWrapper


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
    reason="R + mixOmics are required to test MINTRWrapper",
)


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    n = 30
    index = [f"S{i}" for i in range(n)]
    X = {
        "A": pd.DataFrame(
            rng.normal(size=(n, 10)), index=index, columns=[f"A{i}" for i in range(10)]
        ),
        "B": pd.DataFrame(
            rng.normal(size=(n, 7)), index=index, columns=[f"B{i}" for i in range(7)]
        ),
    }
    Y = np.array([f"c{i % 3}" for i in range(n)])
    study = np.array(["s0"] * 15 + ["s1"] * 15)
    return X, Y, study


def test_fit_shapes_and_variable_selection(synthetic_data):
    X, Y, study = synthetic_data
    keepX = {"A": [4, 4], "B": [3, 3]}

    model = MINTRWrapper(
        n_components=2, keepX=keepX, scale=True, max_iter=100, backend="rscript"
    ).fit(X, Y=Y, study=study)

    assert model.backend_ == "rscript"
    assert model.block_names_ == ["A", "B"]
    assert model.n_components_ == [2, 2]
    assert len(model.iter_) == 2

    for block in ["A", "B"]:
        assert model.loadings_[block].shape == (X[block].shape[1], 2)
        assert model.variates_[block].shape == (len(X[block]), 2)
        assert model.names_[block] == list(X[block].columns)

        for comp, keep in enumerate(keepX[block], start=1):
            assert len(model.get_biomarkers(block, comp)) == keep


def test_continuous_response(synthetic_data):
    X, _, study = synthetic_data
    rng = np.random.default_rng(1)
    Y = rng.normal(size=len(X["A"]))

    model = MINTRWrapper(
        n_components=2, keepX={"A": [4, 4], "B": [3, 3]}, backend="rscript"
    ).fit(X, Y=Y, study=study)

    for block in ["A", "B"]:
        assert model.loadings_[block].shape == (X[block].shape[1], 2)
        assert np.isfinite(model.loadings_[block]).all()


def test_wrapper_matches_python_on_nutrimouse():
    csv = Path(__file__).resolve().parents[1] / "nutrimouse.csv"
    if not csv.exists():
        pytest.skip("nutrimouse.csv not available")

    df = pd.read_csv(csv)
    gene_cols = [c for c in df.columns if c.startswith("gene.")]
    lipid_cols = [c for c in df.columns if c.startswith("lipid.")]
    index = [f"S{i:02d}" for i in range(len(df))]

    X = {
        "gene": df[gene_cols].astype(float).copy(),
        "lipid": df[lipid_cols].astype(float).copy(),
    }
    X["gene"].columns = [c.removeprefix("gene.") for c in gene_cols]
    X["lipid"].columns = [c.removeprefix("lipid.") for c in lipid_cols]
    for block in X:
        X[block].index = index

    Y = df["diet"].astype(str).to_numpy()
    study = df["genotype"].astype(str).to_numpy()
    keep = {"gene": [15, 15], "lipid": [8, 8]}

    r_model = MINTRWrapper(
        n_components=2, keepX=keep, scale=True, max_iter=100, backend="rscript"
    ).fit(X, Y=Y, study=study)

    py_model = MINT(
        n_components=2,
        keepA=keep,
        tau=None,
        scheme="horst",
        mode="canonical",
        scale=True,
        max_iter=100,
        tol=1e-6,
        init="svd.single",
    ).fit(X, Y=Y, study=study)

    assert r_model.iter_ == py_model._mint_result.iter

    for block in ["gene", "lipid"]:
        for comp in range(2):
            a = r_model.loadings_[block][:, comp]
            b = py_model.loadings_[block][:, comp]
            cos = abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
            assert cos == pytest.approx(1.0, abs=1e-6)

            selected_r = set(np.flatnonzero(a))
            selected_py = set(np.flatnonzero(b))
            assert selected_r == selected_py
