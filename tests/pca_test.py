# Ejecución: python -m pytest -s tests/pca_test.py

import numpy as np
import pytest
import rpy2.robjects as robjects
from rpy2.robjects import FloatVector, r
from sklearn.preprocessing import StandardScaler

from pythomics.tools import NipalsPCA
#from sklearn.decomposition import PCA

def test_pca_variance_matches_r():
    np.random.seed(42)
    data = np.random.randn(20, 4)  # 20 muestras, 4 variables

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    n_samples, n_features = X_scaled.shape

    # Implementación nuestra
    pca_test = NipalsPCA(n_components=2)
    pca_test.fit_transform(X_scaled)
    
    # Scikit-Learn
    #pca_test = PCA(n_components=2)
    #pca_test.fit(X_scaled)
    #varianzas_python = pca_test.explained_variance_
    
    prop_var_python = pca_test.explained_variance_ratio_
    total_variance = np.sum(np.var(X_scaled, ddof=1, axis=0))
    varianzas_python = prop_var_python * total_variance

    flat_data = FloatVector(X_scaled.flatten().tolist())

    robjects.r.assign("flat_data", flat_data)
    robjects.r.assign("n_samples", n_samples)
    robjects.r.assign("n_features", n_features)
        
    r_script = """
    X_simulated <- matrix(flat_data, nrow = n_samples, ncol = n_features, byrow = TRUE)
    
    library(mixOmics)
    pca_mixomics <- pca(X_simulated, ncomp = 2, center = FALSE, scale = FALSE)
    
    scores <- pca_mixomics$variates$X
    
    N <- nrow(X_simulated)
    varianzas <- colSums(scores^2) / (N - 1)
    
    prop_var <- pca_mixomics$explained_variance
    
    list(varianzas = varianzas, prop_var = prop_var)
    """

    res_r = robjects.r(r_script)
    prop_var_r = np.array(res_r.rx2("prop_var"))
    varianzas_r = np.array(res_r.rx2("varianzas"))

    res_r = robjects.r(r_script)
    varianzas_r = np.array(res_r.rx2("varianzas"))

    np.testing.assert_allclose(
        varianzas_python,
        varianzas_r,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Las varianzas de NipalsPCA no coinciden con mixOmics",
    )
