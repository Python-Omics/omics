"""
Tests for the MINT model (pythomics/models/mint.py) and its helpers.

MINT is a pure-Python implementation of the mixOmics MINT algorithm, so these
tests validate behaviour rather than comparing against R:

- sklearn-compatible API (fit / transform / get_params / clone)
- output shapes and metadata
- sparsity (keepA / penalty)
- recovery of a known latent multi-omic signal
- per-study centering (batch-effect correction of the loading directions)
- the two algorithmic paths (MINT tau=None vs RGCCA tau given)
- input validation and error handling

Execution:
    python -m pytest -s tests/mint_test.py
    # or from the repo root:
    python -m pytest tests/mint_test.py -v
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pythomics.models import MINT
from pythomics.models.mint_internal import _build_tau_matrix
from pythomics.models.mint_helpers import (
    study_split,
    l2_norm,
    norm2,
    cov2,
    tau_estimate,
    initsvd,
    soft,
    sparsity,
    binary_search_threshold,
    scale_function,
    mean_centering_per_study,
)

# sparse iterations frequently hit max_iter on random data; the warning is
# expected and unrelated to correctness, so keep the test output clean.
pytestmark = pytest.mark.filterwarnings("ignore:.*no convergio.*:UserWarning")


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
def make_blocks(n=60, specs=None, seed=0):
    """Build a dict of {name: DataFrame} with the given (name, n_features)."""
    if specs is None:
        specs = [("A", 12), ("B", 8)]
    rng = np.random.default_rng(seed)
    return {
        name: pd.DataFrame(
            rng.normal(size=(n, n_feat)),
            columns=[f"{name}{i}" for i in range(n_feat)],
        )
        for name, n_feat in specs
    }


def make_signal_data(
    n=120,
    n_features=60,
    n_signal=8,
    strength=2.0,
    noise=1.0,
    seed=0,
    batch=0.0,
    levels=2,
):
    """
    Two blocks A/B that share two latent factors z1/z2.

    Features ``[0:n_signal]`` load on z1, ``[n_signal:2*n_signal]`` load on z2.
    If ``batch > 0`` a per-feature offset with opposite sign is added to each
    half of the samples, simulating a study/batch effect.

    Returns
    -------
    blocks : dict[str, DataFrame]
    latents : (z1, z2)
    study : np.ndarray or None
    """
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)

    blocks = {}
    for name in ("A", "B"):
        X = rng.normal(size=(n, n_features)) * noise
        X[:, :n_signal] += np.outer(z1, np.linspace(1, 2, n_signal)) * strength
        X[:, n_signal : 2 * n_signal] += (
            np.outer(z2, np.linspace(1, 2, n_signal)) * strength
        )
        df = pd.DataFrame(X, columns=[f"{name}{i}" for i in range(n_features)])
        if batch > 0:
            offset = rng.normal(size=n_features) * batch
            half = n // 2
            df.iloc[:half] = df.iloc[:half].values + offset
            df.iloc[half:] = df.iloc[half:].values - offset
        blocks[name] = df

    study = None
    if batch > 0:
        study = np.array([i // (n // levels) for i in range(n)])

    return blocks, (z1, z2), study


def selected_feature_indices(model, block, comp):
    """Integer indices of the features selected as biomarkers for a block."""
    names = model.get_biomarkers(block, comp).index
    return {int(name[len(block):]) for name in names}


@pytest.fixture
def simple_data():
    return make_blocks(n=60, specs=[("A", 12), ("B", 8)], seed=1)


@pytest.fixture
def signal_data():
    return make_signal_data(seed=0)


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------
class TestAPI:
    def test_fit_transform_getbiomarkers_exist(self):
        for method in ("fit", "transform", "fit_transform", "get_biomarkers"):
            assert callable(getattr(MINT, method))

    def test_is_sklearn_compatible(self, simple_data):
        model = MINT(n_components=2, keepA={"A": [5, 5], "B": [4, 4]})
        # BaseEstimator provides clone-ability
        cloned = clone(model)
        assert cloned.n_components == 2
        assert cloned.keepA == {"A": [5, 5], "B": [4, 4]}

    def test_constructor_stores_parameters(self):
        design = np.array([[0, 1], [1, 0]])
        model = MINT(
            n_components=3,
            design=design,
            keepA={"A": [5, 5, 5]},
            penalty=None,
            tau=0.5,
            scheme="factorial",
            mode="classic",
            scale=False,
            max_iter=42,
            tol=1e-4,
            init="svd",
        )
        assert model.n_components == 3
        assert model.scheme == "factorial"
        assert model.mode == "classic"
        assert model.scale is False
        assert model.max_iter == 42
        assert model.tol == 1e-4
        assert model.init == "svd"
        assert model.tau == 0.5
        np.testing.assert_array_equal(model.design, design)

    def test_unfitted_transform_raises(self, simple_data):
        model = MINT()
        with pytest.raises(RuntimeError):
            model.transform(simple_data)

    def test_unfitted_get_biomarkers_raises(self):
        with pytest.raises(RuntimeError):
            MINT().get_biomarkers("A")

    def test_from_mudata_is_classmethod(self):
        assert isinstance(MINT.__dict__["from_mudata"], classmethod)


# ---------------------------------------------------------------------------
# Fit: shapes and metadata
# ---------------------------------------------------------------------------
class TestFitShapes:
    def test_fit_returns_self(self, simple_data):
        model = MINT(n_components=2, keepA={"A": [5, 5], "B": [4, 4]})
        assert model.fit(simple_data) is model

    def test_block_names_preserved(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [5], "B": [4]})
        model.fit(simple_data)
        assert model.block_names_ == ["A", "B"]

    def test_loadings_and_variates_shapes(self, simple_data):
        model = MINT(n_components=2, keepA={"A": [5, 5], "B": [4, 4]})
        model.fit(simple_data)
        for block in ("A", "B"):
            n, p = simple_data[block].shape
            assert model.loadings_[block].shape == (p, 2)
            assert model.variates_[block].shape == (n, 2)

    def test_names_match_columns(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [5], "B": [4]})
        model.fit(simple_data)
        assert model.names_["A"] == list(simple_data["A"].columns)

    def test_names_generated_for_numpy_blocks(self):
        # MINT is a multi-block method: use at least 2 blocks.
        rng = np.random.default_rng(0)
        X = {"A": rng.normal(size=(30, 6)), "B": rng.normal(size=(30, 4))}
        model = MINT(n_components=1, keepA={"A": [3], "B": [2]})
        model.fit(X)
        assert model.names_["A"] == [f"Var_{i}" for i in range(6)]
        assert model.names_["B"] == [f"Var_{i}" for i in range(4)]

    def test_n_components_int_replicated_across_blocks(self, simple_data):
        model = MINT(n_components=2, keepA={"A": [5, 5], "B": [4, 4]})
        model.fit(simple_data)
        assert model.n_components_ == [2, 2]

    def test_n_components_list_per_block(self, simple_data):
        model = MINT(n_components=[2, 1], keepA={"A": [5, 5], "B": [4]})
        model.fit(simple_data)
        assert model.n_components_ == [2, 1]
        assert model.loadings_["A"].shape[1] == 2
        assert model.loadings_["B"].shape[1] == 1

    def test_n_components_list_wrong_length_raises(self, simple_data):
        model = MINT(n_components=[2, 1, 1])
        with pytest.raises(ValueError):
            model.fit(simple_data)

    def test_internal_result_is_stored(self, simple_data):
        model = MINT(n_components=2, keepA={"A": [5, 5], "B": [4, 4]})
        model.fit(simple_data)
        assert model._mint_result is not None
        assert len(model._mint_result.loadings) == 2


# ---------------------------------------------------------------------------
# Sparsity (keepA / penalty) and biomarkers
# ---------------------------------------------------------------------------
class TestSparsity:
    def test_keepA_dict_selects_exact_count(self, simple_data):
        keep = {"A": 5, "B": 3}
        model = MINT(n_components=1, keepA={k: [v] for k, v in keep.items()})
        model.fit(simple_data)
        for block, k in keep.items():
            assert np.count_nonzero(model.loadings_[block]) == k

    def test_keepA_none_keeps_all_features(self, simple_data):
        model = MINT(n_components=1, keepA=None)
        model.fit(simple_data)
        assert np.count_nonzero(model.loadings_["A"]) == simple_data["A"].shape[1]

    def test_keepA_flat_list_replicated_across_components(self, simple_data):
        model = MINT(n_components=2, keepA=[6, 4])
        model.fit(simple_data)
        for comp in range(2):
            assert np.count_nonzero(model.loadings_["A"][:, comp]) == 6
            assert np.count_nonzero(model.loadings_["B"][:, comp]) == 4

    def test_keepA_list_of_lists(self, simple_data):
        model = MINT(n_components=2, keepA=[[6, 4], [3, 2]])
        model.fit(simple_data)
        assert np.count_nonzero(model.loadings_["A"][:, 0]) == 6
        assert np.count_nonzero(model.loadings_["A"][:, 1]) == 3

    def test_keepA_greater_than_features_keeps_all(self, simple_data):
        p = simple_data["B"].shape[1]
        model = MINT(n_components=1, keepA={"A": [99], "B": [99]})
        model.fit(simple_data)
        assert np.count_nonzero(model.loadings_["B"]) == p

    def test_penalty_produces_sparse_loadings(self, simple_data):
        model = MINT(n_components=1, penalty={"A": 0.5, "B": 0.5})
        model.fit(simple_data)
        assert np.count_nonzero(model.loadings_["A"]) < simple_data["A"].shape[1]
        assert np.isfinite(model.loadings_["A"]).all()

    def test_get_biomarkers_only_returns_nonzero(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [5], "B": [3]})
        model.fit(simple_data)
        bio = model.get_biomarkers("A", comp=1)
        assert len(bio) == 5
        assert (bio["value"] != 0).all()

    def test_get_biomarkers_sorted_by_absolute_value(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [8]})
        model.fit(simple_data)
        values = model.get_biomarkers("A", comp=1)["value"].to_numpy()
        assert np.all(np.abs(values[:-1]) >= np.abs(values[1:]) - 1e-12)

    def test_get_biomarkers_block_name_matches_state(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [5]})
        model.fit(simple_data)
        assert "A" in model.block_names_
        assert model.get_biomarkers("A").index.name == "Variable"

    def test_get_biomarkers_unknown_block_raises(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [5]})
        model.fit(simple_data)
        with pytest.raises(KeyError):
            model.get_biomarkers("Z")

    def test_get_biomarkers_out_of_range_component_raises(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [5]})
        model.fit(simple_data)
        with pytest.raises(ValueError):
            model.get_biomarkers("A", comp=5)


# ---------------------------------------------------------------------------
# Recovery of a known latent signal ("does MINT work?")
# ---------------------------------------------------------------------------
class TestSignalRecovery:
    def _fit(self, blocks, study, n_comp=2, keep=8):
        model = MINT(
            n_components=n_comp,
            keepA={b: [keep] * n_comp for b in blocks},
            tau=None,
            max_iter=500,
            tol=1e-9,
        )
        model.fit(X_dict=blocks, study=study)
        return model

    def test_variates_recover_latent_factors(self, signal_data):
        blocks, (z1, z2), _ = signal_data
        model = self._fit(blocks, study=None)

        for block in ("A", "B"):
            comp_corrs = []
            for comp in range(2):
                cors = [
                    abs(np.corrcoef(model.variates_[block][:, comp], z)[0, 1])
                    for z in (z1, z2)
                ]
                comp_corrs.append(max(cors))
            assert all(c > 0.9 for c in comp_corrs), comp_corrs

    def test_components_map_to_distinct_factors(self, signal_data):
        blocks, (z1, z2), _ = signal_data
        model = self._fit(blocks, study=None)

        def best_factor(comp):
            cors = [
                abs(np.corrcoef(model.variates_["A"][:, comp], z)[0, 1])
                for z in (z1, z2)
            ]
            return int(np.argmax(cors))

        assert best_factor(0) != best_factor(1)

    def test_biomarkers_recover_true_signal_features(self, signal_data):
        blocks, _, _ = signal_data
        model = self._fit(blocks, study=None)
        n_signal = 8
        true_sets = {0: set(range(n_signal)), 1: set(range(n_signal, 2 * n_signal))}

        for comp in range(2):
            factor = int(
                np.argmax(
                    [
                        abs(np.corrcoef(model.variates_["A"][:, comp], z)[0, 1])
                        for z in make_signal_data(seed=0)[1]
                    ]
                )
            )
            for block in ("A", "B"):
                assert selected_feature_indices(model, block, comp + 1) == true_sets[factor]

    def test_converges_on_clean_signal(self, signal_data):
        blocks, _, _ = signal_data
        model = self._fit(blocks, study=None)
        assert all(it < model.max_iter for it in model._mint_result.iter)


# ---------------------------------------------------------------------------
# Batch / study correction
# ---------------------------------------------------------------------------
class TestBatchCorrection:
    def test_loadings_are_invariant_to_per_study_offsets(self):
        """Adding a constant offset per study must not change the loadings:
        MINT centers every block within each study."""
        blocks = make_blocks(n=60, specs=[("A", 12), ("B", 8)], seed=5)
        study = np.array([0] * 30 + [1] * 30)

        shifted = {}
        rng = np.random.default_rng(99)
        for name, df in blocks.items():
            arr = df.values.copy()
            off0 = rng.normal(size=arr.shape[1]) * 10
            off1 = rng.normal(size=arr.shape[1]) * 10
            arr[study == 0] += off0
            arr[study == 1] += off1
            shifted[name] = pd.DataFrame(arr, columns=df.columns)

        kwargs = dict(n_components=1, keepA={"A": [6], "B": [4]}, max_iter=300, scale=False)
        base = MINT(**kwargs).fit(blocks, study=study)
        shif = MINT(**kwargs).fit(shifted, study=study)

        np.testing.assert_allclose(base.loadings_["A"], shif.loadings_["A"], atol=1e-8)
        np.testing.assert_allclose(base.loadings_["B"], shif.loadings_["B"], atol=1e-8)
        np.testing.assert_allclose(base.variates_["A"], shif.variates_["A"], atol=1e-8)

    def test_recovers_signal_in_presence_of_batch_effects(self):
        blocks, (z1, z2), study = make_signal_data(seed=2, batch=5.0)
        model = MINT(
            n_components=2,
            keepA={"A": [8, 8], "B": [8, 8]},
            tau=None,
            max_iter=500,
            tol=1e-9,
        )
        model.fit(X_dict=blocks, study=study)

        for comp in range(2):
            cors = [
                abs(np.corrcoef(model.variates_["A"][:, comp], z)[0, 1])
                for z in (z1, z2)
            ]
            assert max(cors) > 0.9, cors

    def test_single_study_defaults_to_zeros(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [5], "B": [4]})
        model.fit(simple_data)  # no study provided
        assert np.array_equal(model.study_, np.zeros(simple_data["A"].shape[0]))


# ---------------------------------------------------------------------------
# Regularization paths: MINT (tau=None) vs RGCCA (tau given)
# ---------------------------------------------------------------------------
class TestRegularizationPaths:
    def _fit(self, simple_data, tau):
        model = MINT(
            n_components=2, keepA={"A": [5, 5], "B": [4, 4]}, tau=tau, max_iter=300
        )
        model.fit(simple_data, study=np.array([0] * 30 + [1] * 30))
        return model

    def test_tau_none_runs(self, simple_data):
        model = self._fit(simple_data, None)
        assert np.isfinite(model.variates_["A"]).all()

    def test_tau_optimal_runs(self, simple_data):
        model = self._fit(simple_data, "optimal")
        assert np.isfinite(model.variates_["A"]).all()

    def test_tau_fixed_per_block_runs(self, simple_data):
        model = self._fit(simple_data, [0.2, 0.3])
        assert np.isfinite(model.variates_["A"]).all()

    def test_tau_scalar_runs(self, simple_data):
        model = self._fit(simple_data, 0.5)
        assert np.isfinite(model.variates_["A"]).all()

    def test_tau_scalar_equals_replicated_vector(self, simple_data):
        scalar = self._fit(simple_data, 0.5)
        vector = self._fit(simple_data, [0.5, 0.5])
        np.testing.assert_allclose(
            scalar.loadings_["A"], vector.loadings_["A"], atol=1e-10
        )

    def test_tau_invalid_string_raises(self, simple_data):
        with pytest.raises(ValueError):
            self._fit(simple_data, "not-a-tau")

    def test_tau_wrong_length_raises(self, simple_data):
        with pytest.raises(ValueError):
            self._fit(simple_data, [0.1])


# ---------------------------------------------------------------------------
# Scheme / mode options
# ---------------------------------------------------------------------------
class TestSchemeAndMode:
    @pytest.mark.parametrize("scheme", ["horst", "factorial", "centroid"])
    def test_schemes_run(self, simple_data, scheme):
        model = MINT(
            n_components=2,
            keepA={"A": [5, 5], "B": [4, 4]},
            scheme=scheme,
            max_iter=200,
        )
        model.fit(simple_data)
        assert np.isfinite(model.variates_["A"]).all()

    @pytest.mark.parametrize(
        "mode", ["canonical", "classic", "invariant", "regression"]
    )
    def test_modes_run(self, simple_data, mode):
        model = MINT(
            n_components=2,
            keepA={"A": [5, 5], "B": [4, 4]},
            mode=mode,
            max_iter=200,
        )
        model.fit(simple_data)
        assert np.isfinite(model.variates_["A"]).all()

    @pytest.mark.parametrize("init", ["svd.single", "svd"])
    def test_init_options_run(self, simple_data, init):
        # init="svd" is the supervised initialisation (needs a Y block).
        model = MINT(n_components=1, keepA={"A": [5], "B": [4]}, init=init)
        if init == "svd":
            Y = np.array([0, 1] * 30)
            model.fit(simple_data, Y=Y)
        else:
            model.fit(simple_data)
        assert np.isfinite(model.variates_["A"]).all()

    def test_init_svd_without_Y_raises(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [5], "B": [4]}, init="svd")
        with pytest.raises(ValueError):
            model.fit(simple_data)

    def test_unknown_scheme_raises(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [5], "B": [4]}, scheme="nope")
        with pytest.raises(ValueError):
            model.fit(simple_data)


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------
class TestTransform:
    def test_transform_returns_dict_per_block(self, simple_data):
        model = MINT(n_components=2, keepA={"A": [5, 5], "B": [4, 4]})
        model.fit(simple_data)
        out = model.transform(simple_data)
        assert set(out.keys()) == {"A", "B"}
        n = simple_data["A"].shape[0]
        assert out["A"].shape == (n, 2)

    def test_transform_projects_new_samples(self, simple_data):
        model = MINT(n_components=2, keepA={"A": [5, 5], "B": [4, 4]})
        model.fit(simple_data)
        new = {k: v.iloc[:5] for k, v in simple_data.items()}
        out = model.transform(new)
        assert out["A"].shape == (5, 2)

    def test_transform_matches_training_variates_first_component(self):
        """With a single study and scale=False, projecting the training data
        onto component 1 reproduces the fitted variates exactly."""
        blocks = make_blocks(n=50, specs=[("A", 10), ("B", 6)], seed=3)
        model = MINT(
            n_components=1, keepA={"A": [5], "B": [3]}, scale=False, max_iter=300
        )
        model.fit(blocks, study=np.zeros(50, dtype=int))
        out = model.transform(blocks)
        for block in ("A", "B"):
            np.testing.assert_allclose(
                out[block].ravel(), model.variates_[block][:, 0], atol=1e-8
            )

    def test_transform_missing_block_raises(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [5], "B": [4]})
        model.fit(simple_data)
        with pytest.raises(KeyError):
            model.transform({"A": simple_data["A"]})

    def test_fit_transform_returns_variates(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [5], "B": [4]})
        out = model.fit_transform(simple_data)
        assert out["A"].shape == (simple_data["A"].shape[0], 1)


# ---------------------------------------------------------------------------
# get_expression_means
# ---------------------------------------------------------------------------
class TestExpressionMeans:
    def test_group_means_shapes(self):
        blocks = make_blocks(n=40, specs=[("A", 10), ("B", 6)], seed=4)
        y = np.array([0, 1] * 20)
        model = MINT(n_components=1, keepA={"A": [5], "B": [3]})
        model.fit(blocks, Y=y)

        raw = model.loadings_["A"][:, :1]
        active_idx = np.where(np.any(raw != 0, axis=1))[0]
        active_names = [model.names_["A"][i] for i in active_idx]
        active = {
            "A": {"names": active_names, "X": model.X_dict_["A"][active_names].values}
        }
        means = model.get_expression_means(active, {0: "#000", 1: "#fff"})
        assert set(means["A"].keys()) == {0, 1}
        assert means["A"][0].shape == (len(active_names),)
        assert np.isfinite(means["A"][1]).all()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_mismatched_sample_counts_raises(self):
        X = {"A": np.zeros((30, 5)), "B": np.zeros((25, 5))}
        with pytest.raises(ValueError):
            MINT().fit(X)

    def test_study_length_mismatch_raises(self):
        X = {"A": np.zeros((30, 5))}
        with pytest.raises(ValueError):
            MINT().fit(X, study=np.zeros(10))

    def test_fit_without_data_raises(self):
        with pytest.raises(ValueError):
            MINT().fit()

    def test_single_block_raises(self):
        # A single block gives a degenerate (all-zero) design matrix.
        with pytest.raises(ValueError):
            MINT(n_components=1).fit({"A": np.random.default_rng(0).normal(size=(20, 5))})

    def test_predict_requires_supervised(self, simple_data):
        model = MINT(n_components=1, keepA={"A": [5], "B": [4]})
        model.fit(simple_data)
        with pytest.raises(RuntimeError):
            model.predict(simple_data)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_repeated_fit_is_deterministic(self, simple_data):
        kwargs = dict(n_components=2, keepA={"A": [6, 6], "B": [4, 4]}, max_iter=300)
        a = MINT(**kwargs).fit(simple_data)
        b = MINT(**kwargs).fit(simple_data)
        np.testing.assert_allclose(a.loadings_["A"], b.loadings_["A"])
        np.testing.assert_allclose(a.variates_["A"], b.variates_["A"])


# ---------------------------------------------------------------------------
# from_mudata (tested with a lightweight stub; real MuData optional)
# ---------------------------------------------------------------------------
@dataclass
class _FakeModality:
    df: pd.DataFrame

    @property
    def obs(self):
        return pd.DataFrame(index=self.df.index)

    def to_df(self):
        return self.df


class _FakeMuData:
    def __init__(self, blocks, obs):
        self.mod = {name: _FakeModality(df) for name, df in blocks.items()}
        self.obs = obs

    @property
    def shape(self):
        return (len(self.obs),)


class TestFromMuData:
    def _mdata(self):
        blocks = make_blocks(n=40, specs=[("A", 8), ("B", 6)], seed=6)
        obs = pd.DataFrame(
            {
                "study": ["s0"] * 20 + ["s1"] * 20,
                "Subtype": ["x", "y"] * 20,
            },
            index=blocks["A"].index,
        )
        return _FakeMuData(blocks, obs)

    def test_extracts_blocks_and_labels(self):
        mdata = self._mdata()
        model = MINT.from_mudata(mdata, target_col="Subtype", study_col="study")
        assert set(model.X_dict_.keys()) == {"A", "B"}
        assert model.Y_ is not None and len(model.Y_) == 40
        assert model.study_ is not None and len(np.unique(model.study_)) == 2

    def test_missing_target_col_raises(self):
        mdata = self._mdata()
        with pytest.raises(KeyError):
            MINT.from_mudata(mdata, target_col="DoesNotExist")

    def test_missing_study_col_falls_back_to_single_study(self):
        mdata = self._mdata()
        model = MINT.from_mudata(mdata, study_col="NoSuchColumn")
        assert np.array_equal(model.study_, np.zeros(40, dtype=int))

    def test_fit_after_from_mudata(self):
        mdata = self._mdata()
        model = MINT.from_mudata(
            mdata,
            target_col="Subtype",
            study_col="study",
            n_components=1,
            keepA={"A": [4], "B": [3]},
        )
        model.fit()
        assert model.loadings_["A"].shape == (8, 1)

    def test_with_real_mudata_if_available(self):
        md = pytest.importorskip("mudata")
        ad = pytest.importorskip("anndata")
        blocks = make_blocks(n=30, specs=[("A", 5), ("B", 4)], seed=7)
        adatas = {
            name: ad.AnnData(
                X=df.values,
                obs=pd.DataFrame(index=df.index.astype(str)),
                var=pd.DataFrame(index=df.columns.astype(str)),
            )
            for name, df in blocks.items()
        }
        mdata = md.MuData(adatas)
        mdata.obs["study"] = ["s0"] * 15 + ["s1"] * 15
        model = MINT.from_mudata(mdata, study_col="study")
        assert set(model.X_dict_.keys()) == {"A", "B"}


# ---------------------------------------------------------------------------
# Helpers (mint_helpers / mint_internal)
# ---------------------------------------------------------------------------
class TestBuildTauMatrix:
    def test_none_is_not_called(self):
        # _build_tau_matrix is only used when tau is not None; document the
        # vector/scalar/optimal behaviour here.
        pass

    def test_optimal(self):
        assert _build_tau_matrix("optimal", N=2, J=3) == ["optimal", "optimal"]

    def test_scalar_replicated_per_block_and_component(self):
        out = _build_tau_matrix(0.5, N=2, J=3)
        assert out == [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        # rows must be independent objects (no aliasing)
        assert out[0] is not out[1]

    def test_vector_replicated_across_components(self):
        out = _build_tau_matrix([0.1, 0.2], N=2, J=2)
        assert out == [[0.1, 0.2], [0.1, 0.2]]

    def test_per_component_matrix(self):
        out = _build_tau_matrix([[0.1, 0.2], [0.3, 0.4]], N=2, J=2)
        assert out == [[0.1, 0.2], [0.3, 0.4]]

    def test_wrong_vector_length_raises(self):
        with pytest.raises(ValueError):
            _build_tau_matrix([0.1], N=1, J=2)

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            _build_tau_matrix("nope", N=1, J=2)


class TestHelpers:
    def test_study_split(self):
        data = np.arange(12).reshape(6, 2).astype(float)
        study = np.array([0, 0, 0, 1, 1, 1])
        parts = study_split(data, study)
        assert parts[0].shape == (3, 2)
        assert parts[1].shape == (3, 2)
        np.testing.assert_array_equal(parts[0], data[:3])

    def test_l2_norm_unit(self):
        v = l2_norm(np.array([3.0, 4.0]))
        assert np.isclose(np.linalg.norm(v), 1.0)

    def test_norm2_zero_returns_default(self):
        assert norm2(np.zeros(5)) == 0.05
        assert np.isclose(norm2(np.array([3.0, 4.0])), 5.0)

    def test_cov2_matches_numpy(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        np.testing.assert_allclose(cov2(X), np.cov(X, rowvar=False), atol=1e-10)

    def test_tau_estimate_in_unit_interval(self):
        rng = np.random.default_rng(1)
        for p in (3, 10):
            tau = tau_estimate(rng.normal(size=(50, p)))
            assert 0.0 <= tau <= 1.0

    def test_initsvd_returns_vector(self):
        rng = np.random.default_rng(2)
        v = initsvd(rng.normal(size=(30, 7)))
        assert v.shape == (7,)
        assert np.isclose(np.linalg.norm(v), 1.0)

    def test_soft_thresholding(self):
        out = soft(np.array([2.0, -1.0, 0.5]), 1.0)
        np.testing.assert_allclose(out, [1.0, 0.0, 0.0])

    def test_sparsity_keeps_exact_count(self):
        v = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        out = sparsity(v, keep_a=2)
        assert np.count_nonzero(out) == 2

    def test_sparsity_penalty_zeroes_small(self):
        v = np.array([5.0, 0.1, 0.05])
        out = sparsity(v, penalty=1.0)
        assert out[0] > 0
        assert out[1] == 0 and out[2] == 0

    def test_binary_search_threshold(self):
        v = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        thr = binary_search_threshold(v, keep_a=2)
        assert np.sum(np.abs(v) > thr) == 2

    def test_scale_function_centers_and_scales(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        res = scale_function(X, scale=True)
        np.testing.assert_allclose(res.data.mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(res.data.std(axis=0, ddof=1), [1.0, 1.0])

    def test_scale_function_zero_variance_column(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        res = scale_function(X, scale=True)
        assert np.isfinite(res.data).all()
        np.testing.assert_allclose(res.data[:, 1], 0.0)

    def test_mean_centering_per_study_levels(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(6, 3))
        study = np.array([0, 0, 0, 1, 1, 1])
        res = mean_centering_per_study(X, study, scale=False)
        assert res.levels == [0, 1]
        # each study block should be centered on its own mean
        np.testing.assert_allclose(res.data[study == 0].mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(res.data[study == 1].mean(axis=0), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Supervised mode (Y as a block, indY set) -- discrepancy 1
# ---------------------------------------------------------------------------
class TestSupervised:
    def test_Y_changes_the_fit(self):
        blocks = make_blocks(n=60, specs=[("A", 12), ("B", 8)], seed=8)
        y = np.array([0, 1] * 30)
        kwargs = dict(n_components=2, keepA={"A": [6, 6], "B": [4, 4]}, max_iter=300)
        unsup = MINT(**kwargs).fit(blocks)
        sup = MINT(**kwargs).fit(blocks, Y=y)
        assert not np.allclose(unsup.loadings_["A"], sup.loadings_["A"])

    def test_supervised_sets_indY_and_shapes(self):
        blocks = make_blocks(n=60, specs=[("A", 12), ("B", 8)], seed=9)
        y = np.array([0, 1] * 30)
        model = MINT(n_components=2, keepA={"A": [6, 6], "B": [4, 4]}).fit(blocks, Y=y)
        assert model.indY_ == 2  # Y appended after A, B
        for block in ("A", "B"):
            assert model.loadings_[block].shape == (blocks[block].shape[1], 2)
            assert model.variates_[block].shape == (60, 2)
        assert np.array_equal(model.Y_, y)

    def test_categorical_Y_is_one_hot_encoded(self):
        blocks = make_blocks(n=60, specs=[("A", 12), ("B", 8)], seed=10)
        y = np.array(["ctrl", "case"] * 30)
        model = MINT(n_components=1, keepA={"A": [6], "B": [4]}).fit(blocks, Y=y)
        assert model.indY_ == 2
        # original labels preserved for downstream (get_expression_means/predict)
        assert set(model.Y_) == {"ctrl", "case"}

    def test_supervised_recovers_class_signal(self):
        rng = np.random.default_rng(21)
        n, p = 120, 40
        y = np.array([0] * 60 + [1] * 60)
        blocks = {}
        for name in ("A", "B"):
            X = rng.normal(size=(n, p))
            X[:, :5] += np.outer(y - 0.5, np.linspace(1, 2, 5)) * 4.0
            blocks[name] = pd.DataFrame(X, columns=[f"{name}{i}" for i in range(p)])
        model = MINT(
            n_components=1, keepA={"A": [5], "B": [5]}, max_iter=500, tol=1e-9
        ).fit(blocks, Y=y)
        # the first variate should separate the two classes
        corr = abs(np.corrcoef(model.variates_["A"][:, 0], y)[0, 1])
        assert corr > 0.8

    def test_Y_wrong_length_raises(self):
        blocks = make_blocks(n=60, specs=[("A", 12), ("B", 8)], seed=11)
        with pytest.raises(ValueError):
            MINT(n_components=1).fit(blocks, Y=np.array([0, 1, 0]))


# ---------------------------------------------------------------------------
# Dual RGCCA path (n < p block) -- discrepancies 2 & 3
# ---------------------------------------------------------------------------
class TestDualPath:
    def test_rgcca_dual_block_runs(self):
        rng = np.random.default_rng(30)
        X = {"A": rng.normal(size=(20, 30)), "B": rng.normal(size=(20, 8))}  # A is dual
        model = MINT(n_components=1, keepA={"A": [10], "B": [4]}, tau="optimal")
        model.fit(X)
        assert np.isfinite(model.loadings_["A"]).all()
        assert model.loadings_["A"].shape == (30, 1)

    def test_rgcca_all_dual_runs(self):
        rng = np.random.default_rng(31)
        X = {"A": rng.normal(size=(10, 25)), "B": rng.normal(size=(10, 20))}
        model = MINT(n_components=1, keepA={"A": [8], "B": [6]}, tau="optimal").fit(X)
        assert np.isfinite(model.variates_["A"]).all()


# ---------------------------------------------------------------------------
# R-faithful sparsity helpers -- discrepancies 5 & 6
# ---------------------------------------------------------------------------
class TestSparsityFaithful:
    def test_keep_a_matches_R_soft_thresholding_l1(self):
        x = np.array([0.9, -1.2, 0.23, 0.11])
        out = sparsity(x, keep_a=2)
        np.testing.assert_allclose(out, [0.67, -0.97, 0.0, 0.0], atol=1e-12)

    def test_keep_a_second_R_example(self):
        x = np.array([0.9, -1.2, 0.23, 0.89])
        out = sparsity(x, keep_a=2)
        np.testing.assert_allclose(out, [0.01, -0.31, 0.0, 0.0], atol=1e-12)

    def test_keep_a_ties_keep_boundary(self):
        # R's rank(ties.method="max") keeps both tied maxima.
        x = np.array([1.0, 1.0, 0.5])
        out = sparsity(x, keep_a=1)
        np.testing.assert_allclose(out, [0.5, 0.5, 0.0], atol=1e-12)

    def test_penalty_is_L1_constrained_not_direct_threshold(self):
        # penalty semantics: L1/L2 ratio target, NOT a direct soft threshold.
        x = np.array([5.0, 0.1, 0.05])
        out = sparsity(x, penalty=1.0)
        nz = out[out != 0]
        assert len(nz) == 1
        assert np.sum(np.abs(out)) / np.linalg.norm(out) == pytest.approx(1.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
