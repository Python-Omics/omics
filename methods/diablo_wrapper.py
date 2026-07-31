import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Importaciones de rpy2
import rpy2.robjects as robjects
from rpy2.robjects import FloatVector, StrVector, IntVector


class DIABLOWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, design=None):
        self.n_components = n_components
        self.design = design

    def fit(self, X_dict, Y):
        self.block_names_ = list(X_dict.keys())
        p = len(X_dict)

        if self.design is None:
            design_mat = np.ones((p, p)) - np.eye(p)
        else:
            design_mat = np.asarray(self.design, dtype=float)

        if design_mat.shape != (p, p):
            raise ValueError(
                f"design debe tener forma ({p}, {p}), recibido "
                f"{design_mat.shape}"
            )

        # Preparar Y asegurando que sea una lista de strings limpia
        y_values = pd.Series(Y).astype(str).tolist()

        # Asignar variables globales a R
        robjects.r.assign("n_comp", int(self.n_components))
        robjects.r.assign("flat_y", StrVector(y_values))
        
        # Asignar la matriz de diseño aplanada
        design_flat = design_mat.flatten(order='F')
        robjects.r.assign("flat_design", FloatVector(design_flat.tolist()))
        robjects.r.assign("p_size", p)

        # Asignar cada bloque de X convirtiéndolo a formato plano con sus dimensiones
        block_names = list(X_dict.keys())
        robjects.r.assign("block_names", StrVector(block_names))

        for name in block_names:
            block_arr = np.asarray(X_dict[name], dtype=float)
            n_samples, n_features = block_arr.shape
            robjects.r.assign(f"n_samples_{name}", int(n_samples))
            robjects.r.assign(f"n_features_{name}", int(n_features))
            robjects.r.assign(f"flat_x_{name}", FloatVector(block_arr.flatten().tolist()))

        # Script de R para armar los bloques, el diseño y ajustar block.splsda
        r_script = """
        library(mixOmics)

        # Reconstruir la matriz de diseño
        design_matrix <- matrix(flat_design, nrow = p_size, ncol = p_size)

        # Reconstruir la lista de bloques X
        X_list <- list()
        for (name in block_names) {
            n_s <- get(paste0("n_samples_", name))
            n_f <- get(paste0("n_features_", name))
            flat_vals <- get(paste0("flat_x_", name))
            
            X_list[[name]] <- matrix(flat_vals, nrow = n_s, ncol = n_f, byrow = TRUE)
        }

        # Convertir Y a factor
        Y_factor <- as.factor(flat_y)

        # Ajustar block.splsda
        model_fit <- block.splsda(
            X = X_list,
            Y = Y_factor,
            ncomp = n_comp,
            design = design_matrix
        )

        model_fit
        """

        self.r_fit_ = robjects.r(r_script)
        return self

    def transform(self, X_dict):
        if getattr(self, "r_fit_", None) is None:
            raise RuntimeError(
                "El modelo no ha sido ajustado. Ejecuta .fit() primero."
            )

        # Validar si los bloques de entrada coinciden con los guardados o están en el modelo
        # (DIABLO en transform usa los variates calculados del modelo ajustado, 
        # pero validamos que los nombres existan)
        
        r_script = """
        r_variates <- model_fit$variates
        variates_names <- names(r_variates)
        
        # Devolver una lista de los variates almacenados en el objeto ajustado
        variates_list <- list()
        for (name in variates_names) {
            if (!is.null(r_variates[[name]])) {
                variates_list[[name]] <- r_variates[[name]]
            }
        }
        variates_list
        """

        res_r = robjects.r(r_script)
        
        # Extraer los nombres disponibles en R
        available_names = list(res_r.names) if hasattr(res_r, 'names') else list(res_r.keys())

        variates_dict = {}
        for name in X_dict.keys():
            if name not in available_names:
                raise KeyError(
                    f"El bloque '{name}' no está presente en el "
                    f"modelo ajustado. Bloques disponibles: {available_names}"
                )
            r_matrix = res_r.rx2(name)
            variates_dict[name] = np.array(r_matrix)

        return variates_dict

    def fit_transform(self, X_dict, Y):
        return self.fit(X_dict, Y).transform(X_dict)