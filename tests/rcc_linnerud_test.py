"""
Unit tests for rcc.py, using the linnerud dataset from sklearn.

This is the exact same dataset used in the official mixOmics rcc.R example:
    data(linnerud)
    X <- linnerud$exercise       # Chins, Situps, Jumps
    Y <- linnerud$physiological  # Weight, Waist, Pulse

sklearn.datasets.load_linnerud() exposes 'data' (exercise) and 'target'
(physiological), in the same order and with the same values as the R dataset
(both originate from Tenenhaus 1998 / Rousseeuw & Leroy).
"""

import numpy as np
import pytest
from sklearn.datasets import load_linnerud

from pythomics.models.rcc import *

@pytest.fixture(scope="module")
def linnerud():
    data = load_linnerud()
    X = data["data"].astype(float) # exercise
    Y = data["target"].astype(float) # physiological
    return X, Y
 
 
def _classical_cca_cor(X, Y):
    """
    INDEPENDENT reference implementation for classical CCA (lambda1=lambda2=0):
    solves the generalized eigenvalue problem
        Cxx^-1 Cxy Cyy^-1 Cyx w = rho^2 w
    using np.linalg.eigvals instead of Cholesky+SVD, to avoid relying on the
    same algebraic path used by rcc().
    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    Cxx = Xc.T @ Xc / (n - 1)
    Cyy = Yc.T @ Yc / (n - 1)
    Cxy = Xc.T @ Yc / (n - 1)
    M = np.linalg.inv(Cxx) @ Cxy @ np.linalg.inv(Cyy) @ Cxy.T
    eigvals = np.linalg.eigvals(M)
    eigvals = np.sort(np.real(eigvals))[::-1]
    return np.sqrt(np.clip(eigvals, 0, None))
 
 
# -------------------------------------
# Classical CCA (lambda1 = lambda2 = 0)
# -------------------------------------
 
class TestClassicalCCA:
    def test_shapes(self, linnerud):
        X, Y = linnerud
        res = rcc(X, Y, ncomp=2, method="ridge")
        n, p = X.shape
        _, q = Y.shape
        assert res.cor.shape == (min(p, q),)
        assert res.xcoef.shape == (p, 2)
        assert res.ycoef.shape == (q, 2)
        assert res.variates_x.shape == (n, 2)
        assert res.variates_y.shape == (n, 2)
        assert res.prop_expl_var_x.shape == (2,)
        assert res.prop_expl_var_y.shape == (2,)
 
    def test_cor_matches_independent_reference(self, linnerud):
        X, Y = linnerud
        res = rcc(X, Y, ncomp=2, method="ridge")
        ref = _classical_cca_cor(X, Y)
        np.testing.assert_allclose(res.cor, ref, atol=1e-8)
 
    def test_cor_is_full_length_not_truncated_to_ncomp(self, linnerud):
        # in R, svd(mat, nu=ncomp, nv=ncomp)$d ALWAYS
        # returns the complete vector of singular values (length
        # min(p,q)); only u/v (and thus xcoef/ycoef) are truncated.
        X, Y = linnerud
        res = rcc(X, Y, ncomp=1, method="ridge")
        assert len(res.cor) == 3  # min(p, q) = 3, not ncomp=1
        assert res.xcoef.shape[1] == 1
        assert res.ycoef.shape[1] == 1
 
    def test_cor_between_zero_and_one(self, linnerud):
        X, Y = linnerud
        res = rcc(X, Y, ncomp=2, method="ridge")
        assert np.all(res.cor >= -1e-8)
        assert np.all(res.cor <= 1 + 1e-8)
 
    def test_variate_correlation_equals_canonical_correlation(self, linnerud):
        X, Y = linnerud
        ncomp = 2
        res = rcc(X, Y, ncomp=ncomp, method="ridge")
        for h in range(ncomp):
            r = np.corrcoef(res.variates_x[:, h], res.variates_y[:, h])[0, 1]
            assert abs(r) == pytest.approx(res.cor[h], abs=1e-6)
 
    def test_variates_x_uncorrelated_across_components(self, linnerud):
        # xcoef.T @ Cxx @ xcoef == I by construction (whitening + SVD with
        # orthonormal vectors) -> X variates must remain mutually uncorrelated.
        X, Y = linnerud
        res = rcc(X, Y, ncomp=2, method="ridge")
        c = np.cov(res.variates_x.T)
        assert c[0, 1] == pytest.approx(0.0, abs=1e-6)
 
    def test_variates_y_uncorrelated_across_components(self, linnerud):
        X, Y = linnerud
        res = rcc(X, Y, ncomp=2, method="ridge")
        c = np.cov(res.variates_y.T)
        assert c[0, 1] == pytest.approx(0.0, abs=1e-6)
 
 
# --------------------------------------------
# Ridge regularization (lambda1, lambda2 > 0)
# --------------------------------------------
 
class TestRidgeRegularization:
    def test_zero_lambda_equals_classical_cca(self, linnerud):
        X, Y = linnerud
        res = rcc(X, Y, ncomp=2, method="ridge", lambda1=0.0, lambda2=0.0)
        ref = _classical_cca_cor(X, Y)
        np.testing.assert_allclose(res.cor, ref, atol=1e-8)
 
    def test_lambdas_stored_in_result(self, linnerud):
        X, Y = linnerud
        res = rcc(X, Y, ncomp=2, method="ridge", lambda1=0.3, lambda2=0.5)
        assert res.lambda1 == 0.3
        assert res.lambda2 == 0.5
 
    def test_large_lambda_shrinks_first_correlation(self, linnerud):
        X, Y = linnerud
        res_weak = rcc(X, Y, ncomp=1, method="ridge", lambda1=0.01, lambda2=0.01)
        res_strong = rcc(X, Y, ncomp=1, method="ridge", lambda1=50.0, lambda2=50.0)
        assert res_strong.cor[0] < res_weak.cor[0]
 
    def test_negative_lambda_raises(self, linnerud):
        X, Y = linnerud
        with pytest.raises(ValueError):
            rcc(X, Y, method="ridge", lambda1=-0.1)
        with pytest.raises(ValueError):
            rcc(X, Y, method="ridge", lambda2=-0.1)


# -----------
# Shrinkage
# -----------
 
class TestShrinkage:
    def test_shapes_and_lambda_bounds(self, linnerud):
        X, Y = linnerud
        res = rcc(X, Y, ncomp=2, method="shrinkage")
        assert res.method == "shrinkage"
        assert 0.0 <= res.lambda1 <= 1.0
        assert 0.0 <= res.lambda2 <= 1.0
        assert res.cor.shape == (3,)
        assert res.xcoef.shape == (3, 2)
 
    def test_cor_between_zero_and_one(self, linnerud):
        X, Y = linnerud
        res = rcc(X, Y, ncomp=2, method="shrinkage")
        assert np.all(res.cor >= -1e-8)
        assert np.all(res.cor <= 1 + 1e-8)
 
    def test_input_lambda1_lambda2_are_ignored_and_estimated(self, linnerud):
        X, Y = linnerud
        res_a = rcc(X, Y, ncomp=2, method="shrinkage", lambda1=0.9, lambda2=0.9)
        res_b = rcc(X, Y, ncomp=2, method="shrinkage", lambda1=0.0, lambda2=0.0)
        assert res_a.lambda1 == pytest.approx(res_b.lambda1)
        assert res_a.lambda2 == pytest.approx(res_b.lambda2)
        np.testing.assert_allclose(res_a.cor, res_b.cor)


# ------------------------------
# Missing value handling (NaN)
# ------------------------------
 
class TestMissingValues:
    def test_nan_allowed_with_ridge(self, linnerud):
        X, Y = linnerud
        Xnan = X.copy()
        Xnan[0, 0] = np.nan
        Xnan[5, 1] = np.nan
        res = rcc(Xnan, Y, ncomp=2, method="ridge")
        assert np.all(np.isfinite(res.cor))
        assert np.all(np.isfinite(res.variates_x))
 
    def test_nan_rejected_with_shrinkage(self, linnerud):
        X, Y = linnerud
        Xnan = X.copy()
        Xnan[0, 0] = np.nan
        with pytest.raises(ValueError):
            rcc(Xnan, Y, ncomp=2, method="shrinkage")
 
    def test_infinite_always_rejected(self, linnerud): #check
        X, Y = linnerud
        Xinf = X.copy()
        Xinf[0, 0] = np.inf
        with pytest.raises(ValueError):
            rcc(Xinf, Y, ncomp=2, method="ridge")
        with pytest.raises(ValueError):
            rcc(Xinf, Y, ncomp=2, method="shrinkage")
 
    def test_variates_finite_where_data_was_missing(self, linnerud):
        # X.aux[is.na(X.aux)] = 0 in R: after centering, a NaN must contribute 0
        # (not NaN) to the corresponding variate.
        X, Y = linnerud
        Xnan = X.copy()
        Xnan[2, 0] = np.nan
        res = rcc(Xnan, Y, ncomp=2, method="ridge")
        assert np.isfinite(res.variates_x[2, :]).all()
 
    def test_pairwise_cov_matches_numpy_when_no_nan(self, linnerud):
        X, _ = linnerud
        C = cov_pairwise(X)
        C_ref = np.cov(X, rowvar=False, ddof=1)
        np.testing.assert_allclose(C, C_ref, atol=1e-10)
 
    def test_pairwise_cov_matches_manual_pairwise_complete(self):
        # Small hand-crafted example with interleaved NaNs: validates against
        # manual pairwise-complete calculation (equivalent to
        # cov(X, use="pairwise") in R).
        A = np.array([
            [1.0, 2.0],
            [2.0, np.nan],
            [3.0, 4.0],
            [4.0, 5.0],
            [np.nan, 6.0],
        ])
        C = cov_pairwise(A)
 
        col0 = A[[0, 1, 2, 3], 0]  # rows where col0 is observed
        assert C[0, 0] == pytest.approx(np.var(col0, ddof=1))
 
        rows = [0, 2, 3]  # rows where BOTH columns are observed
        a = A[rows, 0] - A[rows, 0].mean()
        b = A[rows, 1] - A[rows, 1].mean()
        expected_01 = (a @ b) / (len(rows) - 1)
        assert C[0, 1] == pytest.approx(expected_01)
        assert C[1, 0] == pytest.approx(expected_01) 


# --------------------------------------
# Parameter validation / Error handling
# --------------------------------------
 
class TestValidation:
    def test_invalid_method_raises(self, linnerud):
        X, Y = linnerud
        with pytest.raises(ValueError):
            rcc(X, Y, method="not_a_method")
 
    def test_mismatched_rows_raises(self, linnerud):
        X, Y = linnerud
        with pytest.raises(ValueError):
            rcc(X[:-1], Y)
 
    def test_non_2d_input_raises(self, linnerud):
        X, Y = linnerud
        with pytest.raises(ValueError):
            rcc(X[:, 0], Y)  # 1D
 
    def test_ncomp_below_one_raises(self, linnerud):
        X, Y = linnerud
        with pytest.raises(ValueError):
            rcc(X, Y, ncomp=0)
 
    def test_ncomp_above_min_pq_raises(self, linnerud):
        X, Y = linnerud
        with pytest.raises(ValueError):
            rcc(X, Y, ncomp=10)  # min(p, q) = 3
 
    @pytest.mark.parametrize("ncomp_in,ncomp_out", [(1.4, 1), (1.6, 2), (2.0, 2)])
    def test_ncomp_is_rounded(self, linnerud, ncomp_in, ncomp_out):
        X, Y = linnerud
        res = rcc(X, Y, ncomp=ncomp_in, method="ridge")
        assert res.ncomp == ncomp_out
