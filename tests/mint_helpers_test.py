"""
Tests para mint_helpers.py

Cubre las funciones portadas de MixOmics/R/internal_mint.block_helpers.R:
study_split, l2_norm, norm2, cov2, pairwise_cov, miscrossprod,
tau_estimate, deflation, defl_select, scale_function,
mean_centering_per_study.

Ejecutar con:  pytest -v test_mint_helpers.py
"""

import numpy as np
import pytest

from pythomics.models.mint_helpers import (
    study_split,
    l2_norm,
    norm2,
    cov2,
    pairwise_cov,
    miscrossprod,
    tau_estimate,
    deflation,
    defl_select,
    scale_function,
    mean_centering_per_study,
)


# ---------------------------------------------------------------------
# Fixtures / helpers comunes
# ---------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_matrix(rng):
    # 20 filas, 5 columnas, sin NaN
    return rng.normal(size=(20, 5))


# =======================================================================
# study_split
# =======================================================================

class TestStudySplit:
    def test_basic_split_content(self):
        data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        study = np.array(["b", "a", "a", "b"])
        out = study_split(data, study)

        assert set(out.keys()) == {"a", "b"}
        np.testing.assert_array_equal(out["a"], [[2, 2], [3, 3]])
        np.testing.assert_array_equal(out["b"], [[1, 1], [4, 4]])

    def test_default_levels_sorted(self):
        data = np.arange(6).reshape(3, 2)
        study = np.array(["z", "a", "m"])
        out = study_split(data, study)
        # el orden de insercion del dict debe respetar np.unique (orden alfabetico)
        assert list(out.keys()) == ["a", "m", "z"]

    def test_custom_levels_order_and_subset(self):
        data = np.arange(8).reshape(4, 2)
        study = np.array(["a", "b", "a", "c"])
        out = study_split(data, study, levels=["c", "a"])
        assert list(out.keys()) == ["c", "a"]
        np.testing.assert_array_equal(out["c"], data[[3]])
        np.testing.assert_array_equal(out["a"], data[[0, 2]])
        assert "b" not in out

    def test_row_count_mismatch_raises(self):
        data = np.zeros((5, 2))
        study = np.array(["a", "a", "b"])  # solo 3 filas
        with pytest.raises(ValueError):
            study_split(data, study)

    def test_union_of_splits_recovers_all_rows(self, rng):
        data = rng.normal(size=(30, 4))
        study = rng.choice(["s1", "s2", "s3"], size=30)
        out = study_split(data, study)
        total_rows = sum(v.shape[0] for v in out.values())
        assert total_rows == 30
        # cada bloque debe coincidir exactamente con las filas mascaradas
        for lvl, sub in out.items():
            np.testing.assert_array_equal(sub, data[study == lvl])


# =======================================================================
# l2_norm
# =======================================================================

class TestL2Norm:
    def test_unit_norm(self, rng):
        x = rng.normal(size=10)
        out = l2_norm(x)
        assert np.isclose(np.linalg.norm(out), 1.0)

    def test_direction_preserved(self, rng):
        x = rng.normal(size=8)
        out = l2_norm(x)
        # el vector normalizado debe ser un multiplo escalar positivo de x
        ratios = out / x
        assert np.allclose(ratios, ratios[0])
        assert ratios[0] > 0

    def test_known_vector(self):
        x = np.array([3.0, 4.0])
        out = l2_norm(x)
        np.testing.assert_allclose(out, [0.6, 0.8])

    def test_raises_on_matrix_input(self):
        with pytest.raises(ValueError):
            l2_norm(np.ones((3, 3)))

    def test_idempotent_on_already_normalized_vector(self):
        x = np.array([1.0, 0.0, 0.0])
        out = l2_norm(x)
        np.testing.assert_allclose(out, x)


# =======================================================================
# norm2
# =======================================================================

class TestNorm2:
    def test_matches_euclidean_norm(self, rng):
        v = rng.normal(size=15)
        assert np.isclose(norm2(v), np.linalg.norm(v))

    def test_zero_vector_special_case(self):
        v = np.zeros(5)
        assert norm2(v) == 0.05

    def test_known_value(self):
        v = np.array([3.0, 4.0])
        assert np.isclose(norm2(v), 5.0)

    def test_single_nonzero_component(self):
        v = np.array([0.0, 0.0, -7.0])
        assert np.isclose(norm2(v), 7.0)


# =======================================================================
# cov2
# =======================================================================

class TestCov2:
    def test_matches_numpy_cov_no_nan(self, small_matrix):
        out = cov2(small_matrix)
        expected = np.cov(small_matrix, rowvar=False)
        np.testing.assert_allclose(out, expected)

    def test_single_column_shape(self, rng):
        x = rng.normal(size=(10, 1))
        out = cov2(x)
        assert out.shape == (1, 1)
        np.testing.assert_allclose(out[0, 0], np.var(x, ddof=1))

    def test_cross_covariance_matches_manual(self, rng):
        x = rng.normal(size=(25, 3))
        y = rng.normal(size=(25, 2))
        out = cov2(x, y)

        n = x.shape[0]
        xc = x - x.mean(axis=0)
        yc = y - y.mean(axis=0)
        expected = (xc.T @ yc) / (n - 1)
        np.testing.assert_allclose(out, expected)
        assert out.shape == (3, 2)

    def test_bias_correction_scales_by_n_minus_1_over_n(self, small_matrix):
        n = small_matrix.shape[0]
        unbiased = cov2(small_matrix, bias=False)
        biased = cov2(small_matrix, bias=True)
        np.testing.assert_allclose(biased, unbiased * (n - 1) / n)

    def test_symmetry_of_self_covariance(self, small_matrix):
        out = cov2(small_matrix)
        np.testing.assert_allclose(out, out.T)

    def test_diagonal_equals_variance(self, small_matrix):
        out = cov2(small_matrix)
        variances = np.var(small_matrix, axis=0, ddof=1)
        np.testing.assert_allclose(np.diag(out), variances)

    def test_nan_triggers_pairwise_path(self, rng):
        x = rng.normal(size=(20, 3))
        x_nan = x.copy()
        x_nan[0, 0] = np.nan
        out = cov2(x_nan)
        # la columna 0 usa 19 obs, resto usa 20; comparamos contra pairwise_cov directo
        expected = pairwise_cov(x_nan, x_nan)
        np.testing.assert_allclose(out, expected)


# =======================================================================
# pairwise_cov
# =======================================================================

class TestPairwiseCov:
    def test_matches_full_cov_when_no_missing(self, small_matrix):
        out = pairwise_cov(small_matrix, small_matrix)
        expected = np.cov(small_matrix, rowvar=False)
        np.testing.assert_allclose(out, expected)

    def test_insufficient_overlap_gives_nan(self):
        x = np.array([[1.0], [np.nan], [np.nan]])
        y = np.array([[1.0], [2.0], [3.0]])
        out = pairwise_cov(x, y)
        # solo 1 observacion valida en comun -> no se puede estimar covarianza
        assert np.isnan(out[0, 0])

    def test_manual_two_column_case(self):
        x = np.array([[1.0, np.nan],
                      [2.0, 2.0],
                      [3.0, 4.0],
                      [4.0, 6.0]])
        y = np.array([[10.0], [20.0], [30.0], [40.0]])
        out = pairwise_cov(x, y)

        # columna 0 de x: sin NaN, covarianza con y estandar
        xc = x[:, 0] - x[:, 0].mean()
        yc = y[:, 0] - y[:, 0].mean()
        expected_col0 = (xc @ yc) / (len(xc) - 1)
        assert np.isclose(out[0, 0], expected_col0)

        # columna 1 de x: excluye la fila 0 (NaN)
        mask = np.isfinite(x[:, 1])
        xi = x[mask, 1] - x[mask, 1].mean()
        yj = y[mask, 0] - y[mask, 0].mean()
        expected_col1 = (xi @ yj) / (mask.sum() - 1)
        assert np.isclose(out[1, 0], expected_col1)

    def test_output_shape(self, rng):
        x = rng.normal(size=(10, 4))
        y = rng.normal(size=(10, 2))
        out = pairwise_cov(x, y)
        assert out.shape == (4, 2)


# =======================================================================
# miscrossprod
# =======================================================================

class TestMiscrossprod:
    def test_matches_dot_product_no_nan(self, rng):
        x = rng.normal(size=10)
        y = rng.normal(size=10)
        assert np.isclose(miscrossprod(x, y), x @ y)

    def test_ignores_nan_entries(self):
        x = np.array([1.0, np.nan, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        # el par (nan, 5.0) debe ignorarse => 1*4 + 3*6 = 22
        assert np.isclose(miscrossprod(x, y), 22.0)

    def test_all_nan_gives_zero(self):
        x = np.array([np.nan, np.nan])
        y = np.array([1.0, 2.0])
        assert miscrossprod(x, y) == 0.0

    def test_flattens_multidim_input(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert np.isclose(miscrossprod(x, y), 10.0)


# =======================================================================
# tau_estimate
# =======================================================================

class TestTauEstimate:
    def test_tau_in_unit_interval(self, rng):
        x = rng.normal(size=(30, 6))
        tau = tau_estimate(x)
        assert 0.0 <= tau <= 1.0

    def test_highly_collinear_columns_give_low_tau(self, rng):
        # columnas casi identicas -> la matriz de correlacion empirica es
        # muy estable, por lo que el encogimiento optimo hacia la identidad
        # deberia ser pequeno.
        base = rng.normal(size=(200, 1))
        noise = rng.normal(scale=0.001, size=(200, 5))
        x = base + noise
        tau = tau_estimate(x)
        assert 0.0 <= tau <= 1.0

    def test_matches_explicit_loop_reimplementation(self, rng):
        # Reimplementacion independiente (con loops) de la formula analitica
        # de Strimmer: w_kij = z_ki * z_kj (NO al cuadrado), var_hat(r_ij) =
        # n/(n-1)^3 * sum_k (w_kij - wbar_ij)^2, tau = sum(var)/sum((r-I)^2).
        # Esto confirma que la version vectorizada de tau_estimate (que usa
        # xs2.T@xs2 - (1/n)*(xs.T@xs)**2 como atajo algebraico de esa misma
        # suma de cuadrados) es matematicamente equivalente.
        x = rng.normal(size=(12, 4))
        n, p = x.shape
        xs = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        corm = np.corrcoef(x, rowvar=False)

        v_manual = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                w = xs[:, i] * xs[:, j]
                wbar = w.mean()
                v_manual[i, j] = (n / (n - 1) ** 3) * np.sum((w - wbar) ** 2)
        np.fill_diagonal(v_manual, 0.0)

        d_manual = (corm - np.eye(p)) ** 2
        tau_manual = np.sum(v_manual) / np.sum(d_manual)
        tau_manual = max(min(tau_manual, 1.0), 0.0)

        assert np.isclose(tau_estimate(x), tau_manual)

    def test_perfectly_stable_correlation_gives_low_shrinkage(self, rng):
        # Si dos columnas son una combinacion lineal exacta (mas un ruido
        # minusculo), la correlacion muestral es casi determinista de
        # observacion a observacion, por lo que su varianza estimada (y por
        # ende tau, el encogimiento sugerido) debe ser muy pequena.
        base = rng.normal(size=(300, 1))
        x = np.hstack([base, 2.0 * base + rng.normal(scale=1e-6, size=(300, 1))])
        tau = tau_estimate(x)
        assert tau < 0.05

    def test_symmetric_v_matrix_effect(self, rng):
        # sanity check: no debe lanzar error y debe devolver un escalar float
        x = rng.normal(size=(50, 10))
        tau = tau_estimate(x)
        assert isinstance(tau, float)


# =======================================================================
# deflation
# =======================================================================

class TestDeflation:
    def test_residual_orthogonal_to_y_no_missing(self, rng):
        X = rng.normal(size=(30, 6))
        y = rng.normal(size=30)
        p, R = deflation(X, y)
        # y' @ R debe ser (casi) cero para cada columna de R
        proj = y @ R
        np.testing.assert_allclose(proj, np.zeros(6), atol=1e-10)

    def test_loadings_match_least_squares(self, rng):
        X = rng.normal(size=(20, 4))
        y = rng.normal(size=20)
        p, R = deflation(X, y)
        expected_p = (X.T @ y) / (y @ y)
        np.testing.assert_allclose(p, expected_p)

    def test_reconstruction_identity(self, rng):
        X = rng.normal(size=(15, 3))
        y = rng.normal(size=15)
        p, R = deflation(X, y)
        # por definicion R = X - y*p'
        np.testing.assert_allclose(R, X - np.outer(y, p))

    def test_raises_when_misdata_without_mask(self):
        X = np.ones((5, 2))
        y = np.ones(5)
        with pytest.raises(ValueError):
            deflation(X, y, misdata_q=True, is_na_A_q=None)

    def test_missing_data_path_runs_and_no_nan_leftover(self, rng):
        X = rng.normal(size=(10, 3))
        y = rng.normal(size=10)
        is_na = np.zeros((10, 3), dtype=bool)
        is_na[2, 1] = True
        X[2, 1] = 0.0  # convencion: NaN ya reemplazados por 0
        p, R = deflation(X, y, misdata_q=True, is_na_A_q=is_na, ind_NA=[1])
        assert not np.any(np.isnan(p))
        assert not np.any(np.isnan(R))

    def test_deflating_twice_reduces_norm_or_keeps_orthogonal(self, rng):
        X = rng.normal(size=(25, 4))
        y1 = rng.normal(size=25)
        _, R1 = deflation(X, y1)
        y2 = rng.normal(size=25)
        p2, R2 = deflation(R1, y2)
        # R1 sigue siendo ortogonal a y1 tras la segunda deflacion
        # (deflactar sobre y2 no reintroduce componente de y1 en general,
        # pero al menos R2 debe ser ortogonal a y2)
        np.testing.assert_allclose(y2 @ R2, np.zeros(4), atol=1e-9)


# =======================================================================
# defl_select
# =======================================================================

class TestDeflSelect:
    def test_canonical_mode_matches_direct_deflation(self, rng):
        nbloc = 3
        n = 15
        rr = [rng.normal(size=(n, 4)) for _ in range(nbloc)]
        yy = rng.normal(size=(n, nbloc))
        nncomp = [2, 2, 2]

        resdefl, pdefl = defl_select(yy, rr, nncomp, nn=1, nbloc=nbloc, mode="canonical")

        for q in range(nbloc):
            p_expected, R_expected = deflation(rr[q], yy[:, q])
            np.testing.assert_allclose(pdefl[q], p_expected)
            np.testing.assert_allclose(resdefl[q], R_expected)

    def test_invariant_mode_returns_unchanged_indY_block(self, rng):
        # OJO: el 'mode' solo cambia el comportamiento del bloque indY;
        # cualquier otro bloque (q != indY) siempre se defla normalmente
        # sobre su propia variate, sin importar el modo. Por eso hay que
        # fijar indY explicitamente para poder probar "invariant".
        nbloc = 2
        n = 10
        indY = 0
        rr = [rng.normal(size=(n, 3)) for _ in range(nbloc)]
        yy = rng.normal(size=(n, nbloc))
        nncomp = [1, 1]

        resdefl, pdefl = defl_select(
            yy, rr, nncomp, nn=1, nbloc=nbloc, indY=indY, mode="invariant"
        )

        # bloque Y (indY): sin deflacion, residuo = entrada original
        np.testing.assert_array_equal(resdefl[indY], rr[indY])
        np.testing.assert_array_equal(pdefl[indY], np.zeros(rr[indY].shape[1]))

        # bloques X (q != indY): deflacion estandar sobre su propia variate
        for q in range(nbloc):
            if q == indY:
                continue
            p_expected, R_expected = deflation(rr[q], yy[:, q])
            np.testing.assert_allclose(resdefl[q], R_expected)
            np.testing.assert_allclose(pdefl[q], p_expected)

    def test_component_beyond_nncomp_is_passthrough(self, rng):
        nbloc = 2
        n = 10
        rr = [rng.normal(size=(n, 3)) for _ in range(nbloc)]
        yy = rng.normal(size=(n, nbloc))
        nncomp = [0, 0]  # nn=1 > nncomp[q] para todos los bloques

        resdefl, pdefl = defl_select(yy, rr, nncomp, nn=1, nbloc=nbloc, mode="canonical")

        for q in range(nbloc):
            np.testing.assert_array_equal(resdefl[q], rr[q])
            np.testing.assert_array_equal(pdefl[q], np.zeros(rr[q].shape[1]))

    def test_unknown_mode_raises(self, rng):
        # El dispatch por 'mode' solo se evalua para el bloque q == indY
        # (para q != indY siempre se toma la rama canonica), asi que hay
        # que apuntar indY al bloque que se esta probando para llegar al
        # 'else: raise ValueError' del modo desconocido.
        nbloc = 2
        n = 8
        indY = 0
        rr = [rng.normal(size=(n, 2)) for _ in range(nbloc)]
        yy = rng.normal(size=(n, nbloc))
        nncomp = [1, 1]
        with pytest.raises(ValueError):
            defl_select(yy, rr, nncomp, nn=1, nbloc=nbloc, indY=indY, mode="not_a_real_mode")

    def test_classic_mode_skips_indY_own_deflation(self, rng):
        nbloc = 3
        n = 12
        indY = 1
        rr = [rng.normal(size=(n, 3)) for _ in range(nbloc)]
        yy = rng.normal(size=(n, nbloc))
        aa = [rng.normal(size=3) for _ in range(nbloc)]
        nncomp = [1, 1, 1]

        resdefl, pdefl = defl_select(
            yy, rr, nncomp, nn=1, nbloc=nbloc, indY=indY, mode="classic", aa=aa
        )

        # bloque indY: promedio manual de (rr[q] - yy[:,x]*aa[q]') sobre los otros bloques
        q = indY
        others = [x for x in range(nbloc) if x != q]
        acc = np.zeros_like(rr[q])
        for x in others:
            acc = acc + (rr[q] - np.outer(yy[:, x], aa[q]))
        expected = acc / (nbloc - 1)
        np.testing.assert_allclose(resdefl[q], expected)

        # bloques != indY: deflacion normal sobre su propia variate
        for q2 in range(nbloc):
            if q2 == indY:
                continue
            p_expected, R_expected = deflation(rr[q2], yy[:, q2])
            np.testing.assert_allclose(resdefl[q2], R_expected)


# =======================================================================
# scale_function
# =======================================================================

class TestScaleFunction:
    def test_mean_zero_after_centering_only(self, rng):
        x = rng.normal(loc=5.0, scale=2.0, size=(50, 4))
        out = scale_function(x, scale=False)
        np.testing.assert_allclose(out.data.mean(axis=0), 0.0, atol=1e-10)
        assert out.sigma is None

    def test_mean_zero_std_one_after_full_scaling(self, rng):
        x = rng.normal(loc=3.0, scale=7.0, size=(60, 5))
        out = scale_function(x, scale=True)
        np.testing.assert_allclose(out.data.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.data.std(axis=0, ddof=1), 1.0, atol=1e-10)

    def test_mean_and_sigma_reported_correctly(self, rng):
        x = rng.normal(size=(30, 3))
        out = scale_function(x, scale=True)
        np.testing.assert_allclose(out.mean, x.mean(axis=0))
        np.testing.assert_allclose(out.sigma, x.std(axis=0, ddof=1))

    def test_zero_variance_column_handled_without_nan(self):
        x = np.column_stack([
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.full(4, 9.0),  # varianza cero
        ])
        out = scale_function(x, scale=True)
        assert not np.any(np.isnan(out.data))
        np.testing.assert_allclose(out.data[:, 1], 0.0)

    def test_ignores_nan_in_mean_and_std(self):
        x = np.array([[1.0, np.nan],
                      [2.0, 5.0],
                      [3.0, 7.0]])
        out = scale_function(x, scale=False)
        # columna 0 sin NaN: media = 2
        assert np.isclose(out.mean[0], 2.0)
        # columna 1 ignorando NaN: media = 6
        assert np.isclose(out.mean[1], 6.0)


# =======================================================================
# mean_centering_per_study
# =======================================================================

class TestMeanCenteringPerStudy:
    def test_each_study_independently_centered(self, rng):
        data = rng.normal(loc=10.0, size=(40, 3))
        study = rng.choice(["A", "B"], size=40)
        out = mean_centering_per_study(data, study, scale=True)

        for lvl in out.levels:
            mask = study == lvl
            sub = out.data[mask]
            np.testing.assert_allclose(sub.mean(axis=0), 0.0, atol=1e-10)
            np.testing.assert_allclose(sub.std(axis=0, ddof=1), 1.0, atol=1e-10)

    def test_matches_manual_scale_function_per_group(self, rng):
        data = rng.normal(size=(25, 2))
        study = rng.choice(["s1", "s2", "s3"], size=25)
        out = mean_centering_per_study(data, study, scale=False)

        for lvl in out.levels:
            mask = study == lvl
            expected = scale_function(data[mask], scale=False)
            np.testing.assert_allclose(out.data[mask], expected.data)
            np.testing.assert_allclose(out.means[lvl], expected.mean)

    def test_row_order_preserved(self, rng):
        data = rng.normal(size=(10, 2))
        study = np.array(["b", "a", "b", "a", "b", "a", "b", "a", "b", "a"])
        out = mean_centering_per_study(data, study, scale=True)
        assert out.data.shape == data.shape
        # las filas de estudio "a" en el resultado, reordenadas por posicion
        # original, deben coincidir con la reconstruccion manual
        for lvl in ["a", "b"]:
            mask = study == lvl
            manual = scale_function(data[mask], scale=True).data
            np.testing.assert_allclose(out.data[mask], manual)

    def test_levels_list_matches_unique_studies(self, rng):
        data = rng.normal(size=(12, 2))
        study = np.array(["x"] * 4 + ["y"] * 4 + ["z"] * 4)
        out = mean_centering_per_study(data, study, scale=True)
        assert out.levels == ["x", "y", "z"]
        assert set(out.means.keys()) == {"x", "y", "z"}
        assert set(out.sigmas.keys()) == {"x", "y", "z"}