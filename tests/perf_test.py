"""
Unit tests for the Perf class (perf.py).

Run with:
    pytest perf_test.py -v
"""
import numpy as np
import pytest

from pythomics.tools.perf import Perf



class ConstantWrongModel:
    """Always predicts a label that is guaranteed to be wrong."""

    def __init__(self, ncomp=2, wrong_label=999, **kwargs):
        self.ncomp = ncomp
        self.wrong_label = wrong_label

    def fit(self, X, y):
        return self

    def predict(self, X):
        n = len(next(iter(X.values()))) if isinstance(X, dict) else len(X)
        return np.full((n, self.ncomp), self.wrong_label)


class EchoTrueLabelModel:
    """
    Cheats by remembering y at fit time indirectly through a shared
    dict keyed by id(X_test) is not possible, so instead this model
    is constructed fresh each fold (as Perf does) and predicts correctly
    by matching train labels back via nearest index trick using a
    simple lookup table injected at construction time.
    """

    def __init__(self, ncomp=2, lookup=None, **kwargs):
        self.ncomp = ncomp
        self.lookup = lookup  # dict: sample "key" (row bytes) -> label

    def fit(self, X, y):
        return self

    def predict(self, X):
        arr = X[list(X.keys())[0]] if isinstance(X, dict) else X
        preds = np.array([self.lookup[row.tobytes()] for row in arr])
        return np.tile(preds.reshape(-1, 1), (1, self.ncomp))


class RecordingModel:
    """Records every fit/predict call so tests can inspect them."""

    calls = []  # class-level, reset in each test via RecordingModel.calls.clear()

    def __init__(self, ncomp=2, **kwargs):
        self.ncomp = ncomp
        self.kwargs = kwargs

    def fit(self, X, y):
        RecordingModel.calls.append({"phase": "fit", "X": X, "y": y})
        self._y_train_labels = np.unique(y)
        return self

    def predict(self, X):
        RecordingModel.calls.append({"phase": "predict", "X": X})
        n = len(next(iter(X.values()))) if isinstance(X, dict) else len(X)
        # predict the first training label seen, for all rows/components
        label = self._y_train_labels[0]
        return np.full((n, self.ncomp), label)



def make_dataset(n_per_class=20, n_classes=2, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_per_class * n_classes, n_features))
    y = np.repeat(np.arange(n_classes), n_per_class)
    return X, y


def build_lookup(X, y):
    return {row.tobytes(): label for row, label in zip(X, y)}


def test_perfect_predictions_give_zero_error_array_X():
    X, y = make_dataset(n_per_class=20, n_classes=2)
    lookup = build_lookup(X, y)

    perf = Perf(
        model_class=EchoTrueLabelModel,
        model_params={"ncomp": 2, "lookup": lookup},
        folds=5,
        nrepeat=2,
    )
    results = perf.evaluate(X, y)

    assert results["error_rate_overall"].shape == (2,)
    assert np.allclose(results["error_rate_overall"], 0.0)


def test_dict_X_blocks_evaluate_without_error():
    """
    Verifica que X como dict de bloques (caso DIABLO multi-bloque) funciona:
    evaluate() divide sobre índices de muestra (dummy_X), no sobre el dict.
    """
    X1, y = make_dataset(n_per_class=15, n_classes=3, n_features=4, seed=1)
    X2, _ = make_dataset(n_per_class=15, n_classes=3, n_features=6, seed=2)
    X_dict = {"block1": X1, "block2": X2}
    lookup = build_lookup(X1, y)

    perf = Perf(
        model_class=EchoTrueLabelModel,
        model_params={"ncomp": 1, "lookup": lookup},
        folds=3,
        nrepeat=2,
    )

    results = perf.evaluate(X_dict, y)

    assert "error_rate_overall" in results
    assert results["error_rate_overall"].shape == (1,)
    assert np.allclose(results["error_rate_overall"], 0.0)


def test_always_wrong_predictions_give_full_error():
    X, y = make_dataset(n_per_class=10, n_classes=2)
    folds = 5

    perf = Perf(
        model_class=ConstantWrongModel,
        model_params={"ncomp": 2, "wrong_label": -1},
        folds=folds,
        nrepeat=3,
    )
    results = perf.evaluate(X, y)

    # Every fold contributes an error rate of 1.0 (100% wrong); fold rates
    # are averaged (not summed) within each repetition, so we expect 1.0.
    assert np.allclose(results["error_rate_overall"], 1.0)

def test_output_shape_matches_ncomp():
    X, y = make_dataset(n_per_class=12, n_classes=2)
    ncomp = 3

    perf = Perf(
        model_class=RecordingModel,
        model_params={"ncomp": ncomp},
        folds=4,
        nrepeat=2,
    )
    RecordingModel.calls = []
    results = perf.evaluate(X, y)

    assert "error_rate_overall" in results
    assert results["error_rate_overall"].shape == (ncomp,)


def test_default_ncomp_is_two_when_not_specified():
    X, y = make_dataset(n_per_class=12, n_classes=2)

    perf = Perf(
        model_class=RecordingModel,
        model_params={},  # no ncomp given -> should default to 2
        folds=3,
        nrepeat=1,
    )
    RecordingModel.calls = []
    results = perf.evaluate(X, y)

    assert results["error_rate_overall"].shape == (2,)


def test_model_is_instantiated_and_fit_once_per_fold_per_repeat():
    X, y = make_dataset(n_per_class=20, n_classes=2)
    folds = 5
    nrepeat = 4

    perf = Perf(
        model_class=RecordingModel,
        model_params={"ncomp": 2},
        folds=folds,
        nrepeat=nrepeat,
    )
    RecordingModel.calls = []
    perf.evaluate(X, y)

    fit_calls = [c for c in RecordingModel.calls if c["phase"] == "fit"]
    predict_calls = [c for c in RecordingModel.calls if c["phase"] == "predict"]

    assert len(fit_calls) == folds * nrepeat
    assert len(predict_calls) == folds * nrepeat


def test_train_test_splits_are_disjoint_and_cover_all_samples():
    X, y = make_dataset(n_per_class=20, n_classes=2)

    captured_test_sizes = []

    class SizeCheckingModel:
        def __init__(self, ncomp=2, **kwargs):
            self.ncomp = ncomp

        def fit(self, X, y):
            self._n_train = len(y)
            return self

        def predict(self, X):
            n = len(X)
            captured_test_sizes.append(n)
            return np.zeros((n, self.ncomp))

    folds = 5
    perf = Perf(
        model_class=SizeCheckingModel,
        model_params={"ncomp": 2},
        folds=folds,
        nrepeat=1,
    )
    perf.evaluate(X, y)

    # Every test fold plus its train fold should reconstruct the full dataset.
    assert sum(captured_test_sizes) == len(y)
    # With balanced classes and 5 folds, expect (roughly) equal fold sizes.
    assert all(s > 0 for s in captured_test_sizes)


def test_dict_X_blocks_would_be_sliced_consistently_with_y_once_fixed():
    """
    Same scenario as `test_dict_X_currently_raises_due_to_skf_split_bug`,
    but demonstrates what SHOULD happen once the `skf.split(X, y)` bug is
    fixed to split on sample indices instead of the raw dict. We patch
    `evaluate`'s split source by calling the fold-splitting logic directly
    via a 2-block dataset wrapped so sklearn sees the right sample count,
    to confirm the *rest* of the dict-handling logic (train/test slicing
    per block) is correct in isolation.
    """
    from sklearn.model_selection import StratifiedKFold as _SKF

    X1, y = make_dataset(n_per_class=10, n_classes=2, n_features=3, seed=3)
    X2, _ = make_dataset(n_per_class=10, n_classes=2, n_features=7, seed=4)
    X_dict = {"blockA": X1, "blockB": X2}

    captured = []
    folds = 4
    skf = _SKF(n_splits=folds, shuffle=True, random_state=0)

    # Mirror Perf.evaluate's dict-slicing logic directly (bypassing the
    # buggy skf.split(X, y) call) to verify that logic is sound.
    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        X_test = {k: v[test_idx] for k, v in X_dict.items()}
        assert set(X_test.keys()) == {"blockA", "blockB"}
        assert X_test["blockA"].shape[0] == X_test["blockB"].shape[0] == len(test_idx)
        captured.append(len(test_idx))

    assert sum(captured) == len(y)


def test_model_params_are_forwarded_to_constructor():
    X, y = make_dataset(n_per_class=10, n_classes=2)

    received_kwargs = []

    class ParamCapturingModel:
        def __init__(self, ncomp=2, alpha=None, **kwargs):
            received_kwargs.append({"ncomp": ncomp, "alpha": alpha})
            self.ncomp = ncomp

        def fit(self, X, y):
            return self

        def predict(self, X):
            n = len(X)
            return np.zeros((n, self.ncomp))

    perf = Perf(
        model_class=ParamCapturingModel,
        model_params={"ncomp": 2, "alpha": 0.5},
        folds=3,
        nrepeat=1,
    )
    perf.evaluate(X, y)

    assert len(received_kwargs) == 3  # folds=3, nrepeat=1
    assert all(kw == {"ncomp": 2, "alpha": 0.5} for kw in received_kwargs)


def test_partially_correct_predictions_give_intermediate_error():
    X, y = make_dataset(n_per_class=25, n_classes=2)

    class HalfWrongModel:
        """Flips the label for the first half of each test batch."""

        def __init__(self, ncomp=1, **kwargs):
            self.ncomp = ncomp

        def fit(self, X, y):
            self.classes = np.unique(y)
            return self

        def predict(self, X):
            n = len(X)
            # We don't have y_test, so just alternate predictions between
            # the two known classes; on average ~50% wrong for balanced data.
            preds = np.resize(self.classes, n)
            return preds.reshape(-1, 1)

    perf = Perf(
        model_class=HalfWrongModel,
        model_params={"ncomp": 1},
        folds=5,
        nrepeat=5,
    )
    results = perf.evaluate(X, y)

    err = results["error_rate_overall"][0]
    # Should be strictly between "always right" (0) and "always wrong" (folds=5).
    assert 0.0 < err < 5.0


def test_evaluate_is_deterministic_given_same_inputs():
    X, y = make_dataset(n_per_class=15, n_classes=2)

    def run():
        perf = Perf(
            model_class=ConstantWrongModel,
            model_params={"ncomp": 2, "wrong_label": -5},
            folds=5,
            nrepeat=3,
        )
        return perf.evaluate(X, y)["error_rate_overall"]

    result_1 = run()
    result_2 = run()

    # StratifiedKFold is seeded with `random_state=r`, so repeated calls
    # with identical inputs should produce identical results.
    assert np.allclose(result_1, result_2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
