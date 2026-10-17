# Ejecución: python -m pytest -s tests/pca_test.py
#
# Originalmente este test comparaba NipalsPCA contra mixOmics::pca() vía rpy2,
# lo que obligaba a tener R + rpy2 instalados para siquiera recolectar el test.
# La referencia de R se reemplazó por una implementación pura-NumPy equivalente
# (SVD), de modo que el test corre sin dependencias de R. Además se añade una
# verificación cruzada contra scikit-learn.

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pythomics.tools import NipalsPCA


def _svd_reference(X, ncomp):
    """Referencia pura-NumPy equivalente a mixOmics::pca(center=FALSE, scale=FALSE).

    mixOmics calcula en R:

        scores    <- pca$variates$X
        varianzas <- colSums(scores^2) / (N - 1)

    Con ``center = FALSE`` y ``scale = FALSE``, los variates de un PCA son la
    proyección SVD ``U[:, :ncomp] * S[:ncomp]``. Reproducimos ese cálculo aquí
    sin depender de R.
    """
    X = np.asarray(X, dtype=float)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    scores = U[:, :ncomp] * S[:ncomp]
    varianzas = np.sum(scores**2, axis=0) / (X.shape[0] - 1)
    total_variance = np.sum(np.var(X, ddof=1, axis=0))
    prop_var = varianzas / total_variance
    return varianzas, prop_var


def test_pca_variance_matches_svd_reference():
    """Las varianzas explicadas deben coincidir con la referencia SVD (mixOmics)."""
    np.random.seed(42)
    data = np.random.randn(20, 4)  # 20 muestras, 4 variables

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Implementación nuestra
    pca_test = NipalsPCA(n_components=2)
    pca_test.fit_transform(X_scaled)

    prop_var_python = pca_test.explained_variance_ratio_
    total_variance = np.sum(np.var(X_scaled, ddof=1, axis=0))
    varianzas_python = prop_var_python * total_variance

    varianzas_ref, prop_var_ref = _svd_reference(X_scaled, ncomp=2)

    np.testing.assert_allclose(
        varianzas_python,
        varianzas_ref,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Las varianzas de NipalsPCA no coinciden con la referencia SVD",
    )
    np.testing.assert_allclose(
        prop_var_python,
        prop_var_ref,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Las proporciones de varianza no coinciden con la referencia SVD",
    )


def test_pca_variance_matches_sklearn():
    """Verificación cruzada independiente contra scikit-learn."""
    np.random.seed(0)
    data = np.random.randn(30, 6)

    X_scaled = StandardScaler().fit_transform(data)

    pca_ours = NipalsPCA(n_components=3).fit(X_scaled)
    pca_sklearn = PCA(n_components=3).fit(X_scaled)

    np.testing.assert_allclose(
        pca_ours.explained_variance_,
        pca_sklearn.explained_variance_,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Las varianzas explicadas no coinciden con scikit-learn",
    )
    np.testing.assert_allclose(
        pca_ours.explained_variance_ratio_,
        pca_sklearn.explained_variance_ratio_,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Las proporciones de varianza no coinciden con scikit-learn",
    )
