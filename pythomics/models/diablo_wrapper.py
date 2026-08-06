import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Importaciones de rpy2
import rpy2.robjects as robjects
from rpy2.robjects import FloatVector, StrVector, IntVector


class DIABLOWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, design=None, keepX=None):
        self.n_components = n_components
        self.design = design
        self.keepX = keepX
        
        self.X_dict_ = None
        self.Y_ = None

    @classmethod
    def from_mudata(cls, mdata, target_col, **kwargs):
        """
        Extrae los bloques de datos de un objeto MuData y devuelve una instancia de DIABLOWrapper
        """
        # -- se extraen las matrices/bloques de MuData --
        X_dict = {name: mod.to_df() for name, mod in mdata.mod.items()}
        
        # -- se extrae Y --
        # NOTE: se intenta obtener Y desde mdata.obs[target_col],
        # pero si no está se accede a la primera modalidad y se extra de ahí
        try:
            Y = mdata.obs[target_col]
        except KeyError:
            # Obtenemos la primera modalidad (ej. 'mRNA')
            first_mod_name = list(mdata.mod.keys())[0]
            Y = mdata.mod[first_mod_name].obs[target_col]
            print(f"DEBUG: 'Temporada' no encontrada en mdata.obs. Obtenida desde {first_mod_name}.obs")
        
        # -- creación de instancia --
        instance = cls(**kwargs)
        instance.X_dict_ = X_dict
        instance.Y_ = Y
        
        return instance

    def fit(self, X_dict=None, Y=None):
        # NOTE: si no se pasan X_dict e Y, es porque se construyó desde MuData y ya están guardados en la instancia
        X_dict = X_dict if X_dict is not None else self.X_dict_
        Y = Y if Y is not None else self.Y_
        
        if X_dict is None or Y is None:
            raise ValueError("Se debe proporcionar X_dict e Y, o inicializar desde MuData.")
        

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

            col_names = list(X_dict[name].columns) if isinstance(X_dict[name], pd.DataFrame) else [f"Var_{i}" for i in range(n_features)]

            robjects.r.assign(f"n_samples_{name}", int(n_samples))
            robjects.r.assign(f"n_features_{name}", int(n_features))
            robjects.r.assign(f"flat_x_{name}", FloatVector(block_arr.flatten().tolist()))
            robjects.r.assign(f"col_names_{name}", StrVector(col_names))
        
        # -- traduccion de keepX a lista de R --
        if self.keepX is not None:
            # ej: list('mRNA'=c(20,20), 'Proteinas'=c(20,20))
            keepX_str = "list(" + ", ".join([f"'{k}'=c({','.join(map(str, v))})" for k, v in self.keepX.items()]) + ")"
        else:
            keepX_str = "NULL"

        # Script de R para armar los bloques, el diseño y ajustar block.splsda
        r_script = f"""
        library(mixOmics)

        # Reconstruir la matriz de diseño
        design_matrix <- matrix(flat_design, nrow = p_size, ncol = p_size)

        # Reconstruir la lista de bloques X
        X_list <- list()
        for (name in block_names) {{
            n_s <- get(paste0("n_samples_", name))
            n_f <- get(paste0("n_features_", name))
            flat_vals <- get(paste0("flat_x_", name))
            c_names <- get(paste0("col_names_", name))
            mat <- matrix(flat_vals, nrow = n_s, ncol = n_f, byrow = TRUE)
            colnames(mat) <- c_names
            X_list[[name]] <- mat
        }}

        # Convertir Y a factor
        Y_factor <- as.factor(flat_y)

        # Ajustar block.splsda
        model_fit <- block.splsda(
            X = X_list,
            Y = Y_factor,
            ncomp = n_comp,
            design = design_matrix,
            keepX = {keepX_str},
            near.zero.var = TRUE
        )

        model_fit
        """

        self.r_fit_ = robjects.r(r_script)

        # -- extraer loadings y names/variates --
        r_script_loadings = """
        r_loadings <- model_fit$loadings
        loadings_names <- names(r_loadings)
        
        loadings_list <- list()
        for (name in loadings_names) {
            if (!is.null(r_loadings[[name]])) {
                mat <- r_loadings[[name]]
                loadings_list[[name]] <- list(matrix = mat, names = rownames(mat))
            }
        }
        loadings_list
        """
        res_loadings_r = robjects.r(r_script_loadings)
        available_loadings = list(res_loadings_r.names) if hasattr(res_loadings_r, 'names') else list(res_loadings_r.keys())

        self.loadings_ = {}
        self.names_ = {}
        
        for name in block_names:
            if name in available_loadings:
                r_list = res_loadings_r.rx2(name)
                r_matrix = np.array(r_list[0])
                r_names = list(r_list[1])
                
                self.loadings_[name] = r_matrix
                self.names_[name] = r_names
        
        self.X_dict_ = X_dict
        self.Y_ = Y
        
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

    # se añade temporalmente para satisfacer al plot_circos()
    def get_expression_means(self, active_features, groups):
        """
        Calcula las medias de expresión crudas por grupo
        """
        group_means = {}
        blocks = list(active_features.keys())
        Y_array = np.array(self.Y_)
        
        for block in blocks:
            group_means[block] = {}
            X_active = active_features[block]['X']
            
            for group_name in groups.keys():
                mask = (Y_array == group_name)
                if mask.sum() > 0:
                    group_means[block][group_name] = X_active[mask].mean(axis=0)
                else:
                    group_means[block][group_name] = np.zeros(X_active.shape[1])
                    
        return group_means

    def get_biomarkers(self, block, comp=1):
        """
        Extrae las variables biomarcadoras seleccionadas para un bloque y componente específico
        """
        if getattr(self, "loadings_", None) is None:
            raise RuntimeError("El modelo no ha sido ajustado.")

        # se extraen los loadings del componente especificado
        loadings = self.loadings_[block][:, comp - 1]
        names = np.array(self.names_[block])

        # se filtran las variables que sobrevivieron a keepX
        active_mask = loadings != 0
        active_loadings = loadings[active_mask]
        active_names = names[active_mask]

        # se ordenan por valor absoluto de loading
        sort_idx = np.argsort(np.abs(active_loadings))[::-1]
        df = pd.DataFrame({
            'value': active_loadings[sort_idx]
        }, index=active_names[sort_idx])
        
        df.index.name = 'Variable'
        return df

    def predict(self, X_dict):
        if getattr(self, "r_fit_", None) is None:
            raise RuntimeError("El modelo no ha sido ajustado. Ejecuta .fit() primero.")

        p = len(X_dict)
        block_names = list(X_dict.keys())
        
        for name in block_names:
            if name not in self.block_names_:
                raise KeyError(f"El bloque '{name}' no fue visto durante el entrenamiento.")
                
            block_arr = np.asarray(X_dict[name], dtype=float)
            n_samples, n_features = block_arr.shape
            col_names = list(X_dict[name].columns) if isinstance(X_dict[name], pd.DataFrame) else [f"Var_{i}" for i in range(n_features)]

            robjects.r.assign(f"test_n_samples_{name}", int(n_samples))
            robjects.r.assign(f"test_n_features_{name}", int(n_features))
            robjects.r.assign(f"test_flat_x_{name}", FloatVector(block_arr.flatten().tolist()))
            robjects.r.assign(f"test_col_names_{name}", StrVector(col_names))

        robjects.r.assign("test_block_names", StrVector(block_names))
        robjects.r.assign("test_n_comp", int(self.n_components))

        r_script_predict = """
        library(mixOmics)

        X_test_list <- list()
        for (name in test_block_names) {
            n_s <- get(paste0("test_n_samples_", name))
            n_f <- get(paste0("test_n_features_", name))
            flat_vals <- get(paste0("test_flat_x_", name))
            c_names <- get(paste0("test_col_names_", name))
            mat <- matrix(flat_vals, nrow = n_s, ncol = n_f, byrow = TRUE)
            colnames(mat) <- c_names
            X_test_list[[name]] <- mat
        }

        # Ejecutar predicción protegiendo contra errores de folds pequeños
        pred <- tryCatch({
            predict(model_fit, newdata = X_test_list, ncomp = test_n_comp)
        }, error = function(e) {
            message("PREDICT ERROR: ", conditionMessage(e))
            NULL
        })

        res <- NULL
        if (!is.null(pred)) {
            # Intentar extraer AveragedPredict (multiómico integrado) o max.dist estándar
            if (!is.null(pred$AveragedPredict.class$max.dist)) {
                res <- pred$AveragedPredict.class$max.dist
            } else if (!is.null(pred$predict.class$max.dist)) {
                res <- pred$predict.class$max.dist
            } else {
                # Si devuelve predicciones por bloque individual, tomar el primer bloque disponible
                first_b <- test_block_names[1]
                if (!is.null(pred$class[[first_b]]$max.dist)) {
                    res <- pred$class[[first_b]]$max.dist
                }
            }
        }

        # Asegurar formato matriz (muestras x componentes) si no es nulo
        if (!is.null(res)) {
            if (is.vector(res)) {
                res <- matrix(res, ncol = test_n_comp)
            }
        }
        res
        """

        res_preds = robjects.r(r_script_predict)
        
        # Si R devolvió NULL o un objeto vacío debido a un fold extremo
        if res_preds is robjects.NULL or res_preds is None:
            n_test_samples = len(next(iter(X_dict.values())))
            return np.zeros((n_test_samples, self.n_components), dtype=int)

        arr = np.array(res_preds)   
        
        if arr.ndim == 0:
            n_test_samples = len(next(iter(X_dict.values())))
            return np.zeros((n_test_samples, self.n_components), dtype=int)
            
        if arr.ndim == 1:
            arr = arr.reshape(-1, self.n_components)
            
        return arr