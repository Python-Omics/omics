# Ejecución: python -m pytest -s tests/diablo_wrapper_test.py

import numpy as np
import pytest
from sklearn.base import clone

rpy2 = pytest.importorskip("rpy2", reason="rpy2 no está instalado")

import rpy2.robjects as robjects
from rpy2.robjects import FloatVector, StrVector
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, PackageNotInstalledError

from pythomics.models.diablo_wrapper import DIABLOWrapper

# ---------------------------------------------------------------------
# Salta todo el módulo si mixOmics no está disponible en el R instalado
# ---------------------------------------------------------------------
try:
    importr("mixOmics")
    _HAS_MIXOMICS = True
except PackageNotInstalledError:
    _HAS_MIXOMICS = False

pytestmark = pytest.mark.skipif(
    not _HAS_MIXOMICS, reason="El paquete de R 'mixOmics' no está instalado"
)


# ---------------------------------------------------------------------
# Fixtures / datos sintéticos reutilizables
# ---------------------------------------------------------------------
@pytest.fixture
def two_block_data():
    np.random.seed(42)
    n_samples = 30
    X1 = np.random.randn(n_samples, 5)
    X2 = np.random.randn(n_samples, 6)
    Y = np.random.choice(["A", "B", "C"], size=n_samples)
    return {"block1": X1, "block2": X2}, Y


@pytest.fixture
def three_block_data():
    np.random.seed(7)
    n_samples = 24
    X1 = np.random.randn(n_samples, 4)
    X2 = np.random.randn(n_samples, 5)
    X3 = np.random.randn(n_samples, 3)
    Y = np.random.choice(["ctrl", "treated"], size=n_samples)
    return {"geno": X1, "transcripto": X2, "proteo": X3}, Y


# ---------------------------------------------------------------------
# 1. Test de referencia: comparación numérica directa contra mixOmics
# ---------------------------------------------------------------------
def test_diablo_matches_r_mixomics(two_block_data):
    X_dict, Y = two_block_data
    n_samples = X_dict["block1"].shape[0]
    n_f1 = X_dict["block1"].shape[1]
    n_f2 = X_dict["block2"].shape[1]

    diablo_python = DIABLOWrapper(n_components=2)
    diablo_python.fit(X_dict, Y)
    scores_python_dict = diablo_python.transform(X_dict)

    # ELIMINAR: pandas2ri.activate() (causante del DeprecationWarning)
    
    robjects.r.assign("n_samples", n_samples)
    robjects.r.assign("n_f1", n_f1)
    robjects.r.assign("n_f2", n_f2)
    robjects.r.assign("flat_X1", FloatVector(X_dict["block1"].flatten().tolist()))
    robjects.r.assign("flat_X2", FloatVector(X_dict["block2"].flatten().tolist()))
    robjects.r.assign("flat_Y", StrVector(np.asarray(Y).tolist()))

    r_script = """
    library(mixOmics)

    X1_sim <- matrix(flat_X1, nrow = n_samples, ncol = n_f1, byrow = TRUE)
    X2_sim <- matrix(flat_X2, nrow = n_samples, ncol = n_f2, byrow = TRUE)
    Y_sim <- as.factor(flat_Y)

    Data <- list(block1 = X1_sim, block2 = X2_sim)

    design <- matrix(1, ncol = length(Data), nrow = length(Data))
    diag(design) <- 0

    diablo_mixomics <- block.splsda(X = Data, Y = Y_sim, ncomp = 2, design = design)

    list(block1 = diablo_mixomics$variates$block1,
         block2 = diablo_mixomics$variates$block2)
    """
    res_r = robjects.r(r_script)

    scores_r_dict = {
        "block1": np.array(res_r.rx2("block1")),
        "block2": np.array(res_r.rx2("block2")),
    }

    for block_name in X_dict.keys():
        scores_py = scores_python_dict[block_name]
        scores_r = scores_r_dict[block_name]

        for i in range(scores_py.shape[1]):
            if np.corrcoef(scores_py[:, i], scores_r[:, i])[0, 1] < 0:
                scores_py[:, i] *= -1

        np.testing.assert_allclose(
            scores_py,
            scores_r,
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Los scores de DIABLO para el bloque '{block_name}' no coinciden con mixOmics",
        )


# ---------------------------------------------------------------------
# 2. Contrato básico de sklearn / API
# ---------------------------------------------------------------------
def test_fit_returns_self(two_block_data):
    X_dict, Y = two_block_data
    model = DIABLOWrapper(n_components=2)
    result = model.fit(X_dict, Y)
    assert result is model


def test_transform_before_fit_raises(two_block_data):
    X_dict, _ = two_block_data
    model = DIABLOWrapper(n_components=2)
    with pytest.raises(RuntimeError):
        model.transform(X_dict)


def test_get_params_returns_constructor_args():
    model = DIABLOWrapper(n_components=3, design=None)
    params = model.get_params()
    assert params == {"n_components": 3, "design": None}


def test_set_params_updates_state(two_block_data):
    model = DIABLOWrapper(n_components=2)
    model.set_params(n_components=1)
    assert model.n_components == 1


def test_clone_does_not_require_fit():
    # clone() debe funcionar sin que __init__ dispare importaciones de R
    # ni dependa de estado interno (bug corregido: importr fuera de __init__)
    model = DIABLOWrapper(n_components=2)
    cloned = clone(model)
    assert cloned.n_components == 2
    assert not hasattr(cloned, "r_fit_")


def test_fit_transform_convenience_method(two_block_data):
    X_dict, Y = two_block_data
    model = DIABLOWrapper(n_components=2)
    scores = model.fit_transform(X_dict, Y)
    assert set(scores.keys()) == set(X_dict.keys())


def test_output_keys_match_input_blocks(two_block_data):
    X_dict, Y = two_block_data
    model = DIABLOWrapper(n_components=2).fit(X_dict, Y)
    scores = model.transform(X_dict)
    assert set(scores.keys()) == set(X_dict.keys())


@pytest.mark.parametrize("ncomp", [1, 2, 3])
def test_output_shape_matches_samples_and_components(two_block_data, ncomp):
    X_dict, Y = two_block_data
    n_samples = X_dict["block1"].shape[0]

    model = DIABLOWrapper(n_components=ncomp).fit(X_dict, Y)
    scores = model.transform(X_dict)

    for block_name, block_scores in scores.items():
        assert block_scores.shape[0] == n_samples
        assert block_scores.shape[1] == ncomp


def test_three_blocks_supported(three_block_data):
    X_dict, Y = three_block_data
    model = DIABLOWrapper(n_components=2).fit(X_dict, Y)
    scores = model.transform(X_dict)

    assert set(scores.keys()) == set(X_dict.keys())
    for block_scores in scores.values():
        assert block_scores.shape == (X_dict["geno"].shape[0], 2)


def test_reproducibility_same_input_same_output(two_block_data):
    X_dict, Y = two_block_data

    model_a = DIABLOWrapper(n_components=2).fit(X_dict, Y)
    scores_a = model_a.transform(X_dict)

    model_b = DIABLOWrapper(n_components=2).fit(X_dict, Y)
    scores_b = model_b.transform(X_dict)

    for block_name in X_dict.keys():
        np.testing.assert_allclose(scores_a[block_name], scores_b[block_name])


def test_default_design_equals_explicit_full_design(two_block_data):
    X_dict, Y = two_block_data
    p = len(X_dict)
    explicit_design = np.ones((p, p)) - np.eye(p)

    model_default = DIABLOWrapper(n_components=2).fit(X_dict, Y)
    scores_default = model_default.transform(X_dict)

    model_explicit = DIABLOWrapper(n_components=2, design=explicit_design).fit(X_dict, Y)
    scores_explicit = model_explicit.transform(X_dict)

    for block_name in X_dict.keys():
        np.testing.assert_allclose(
            scores_default[block_name], scores_explicit[block_name]
        )


def test_custom_partial_design_runs_and_changes_result(two_block_data):
    X_dict, Y = two_block_data
    p = len(X_dict)
    weak_design = np.full((p, p), 0.1)
    np.fill_diagonal(weak_design, 0)

    model_weak = DIABLOWrapper(n_components=2, design=weak_design).fit(X_dict, Y)
    scores_weak = model_weak.transform(X_dict)

    model_full = DIABLOWrapper(n_components=2).fit(X_dict, Y)
    scores_full = model_full.transform(X_dict)

    # Un diseño distinto debería, en general, producir variates distintos
    diffs = [
        not np.allclose(scores_weak[b], scores_full[b])
        for b in X_dict.keys()
    ]
    assert any(diffs)


def test_wrong_shaped_design_raises(two_block_data):
    X_dict, Y = two_block_data
    bad_design = np.ones((3, 3))  # hay solo 2 bloques
    model = DIABLOWrapper(n_components=2, design=bad_design)
    with pytest.raises(ValueError):
        model.fit(X_dict, Y)


def test_y_accepts_list_and_ndarray(two_block_data):
    X_dict, Y = two_block_data

    model_list = DIABLOWrapper(n_components=2).fit(X_dict, list(Y))
    scores_list = model_list.transform(X_dict)

    model_array = DIABLOWrapper(n_components=2).fit(X_dict, np.asarray(Y))
    scores_array = model_array.transform(X_dict)

    for block_name in X_dict.keys():
        np.testing.assert_allclose(scores_list[block_name], scores_array[block_name])


def test_transform_with_unknown_block_name_raises(two_block_data):
    X_dict, Y = two_block_data
    model = DIABLOWrapper(n_components=2).fit(X_dict, Y)
    with pytest.raises(KeyError):
        model.transform({"no_existe": X_dict["block1"]})
