import numpy as np
import pandas as pd
from .._sklearn_compat import BaseEstimator, TransformerMixin
import uuid
import logging

# rpy2 imports (optional: only required when the DIABLO model is actually fitted/used)
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import FloatVector, StrVector, IntVector
    _HAS_RPY2 = True
except ImportError:  # pragma: no cover - depends on the environment
    robjects = None
    FloatVector = StrVector = IntVector = None
    _HAS_RPY2 = False

logger = logging.getLogger(__name__)


def _require_rpy2():
    """Raise a clear error if DIABLO is used without rpy2 installed."""
    if not _HAS_RPY2:
        raise ImportError(
            "DIABLOWrapper requires 'rpy2' and a working R installation with the "
            "'mixOmics' package. Install them with 'pip install rpy2' and "
            "install.packages('mixOmics') inside R."
        )


class DIABLOWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, design=None, keepX=None):
        self.n_components = n_components
        self.design = design
        self.keepX = keepX
        
        self.X_dict_ = None
        self.Y_ = None
        self.r_model_var_ = f"model_fit_{uuid.uuid4().hex[:8]}"

    @classmethod
    def from_mudata(cls, mdata, target_col, **kwargs):
        """
        Extracts the data blocks from a MuData object and returns a DIABLOWrapper instance
        """
        logger.info(f"Creating model instance from MuData with target_col='{target_col}'")
        # -- extract the data matrices/blocks from MuData --
        X_dict = {name: mod.to_df() for name, mod in mdata.mod.items()}
        
        # -- extract Y --
        # NOTE: try to obtain Y from mdata.obs[target_col],
        # but if not present, fall back to the first modality and extract it from there
        try:
            Y = mdata.obs[target_col]
        except KeyError:
            # Obtain the first modality (e.g. 'mRNA')
            first_mod_name = list(mdata.mod.keys())[0]
            Y = mdata.mod[first_mod_name].obs[target_col]
            logger.warning(f"'{target_col}' not found in mdata.obs. Obtained from {first_mod_name}.obs")
        
        # -- create instance --
        instance = cls(**kwargs)
        instance.X_dict_ = X_dict
        instance.Y_ = Y
        
        return instance

    def fit(self, X_dict=None, Y=None):
        _require_rpy2()
        logger.info("Fitting DIABLO model...")
        # NOTE: if X_dict and Y are not passed, the model was built from MuData and they are already stored in the instance
        X_dict = X_dict if X_dict is not None else self.X_dict_
        Y = Y if Y is not None else self.Y_
        
        if X_dict is None or Y is None:
            raise ValueError("X_dict and Y must be provided, or the model must be initialized from MuData.")
        

        self.block_names_ = list(X_dict.keys())
        p = len(X_dict)

        if self.design is None:
            design_mat = np.ones((p, p)) - np.eye(p)
        else:
            design_mat = np.asarray(self.design, dtype=float)

        if design_mat.shape != (p, p):
            raise ValueError(
                f"design must have shape ({p}, {p}), received "
                f"{design_mat.shape}"
            )

        # Prepare Y ensuring it is a clean list of strings
        y_values = pd.Series(Y).astype(str).tolist()

        # Assign global variables to R
        robjects.r.assign("n_comp", int(self.n_components))
        robjects.r.assign("flat_y", StrVector(y_values))
        
        # Assign the flattened design matrix
        design_flat = design_mat.flatten(order='F')
        robjects.r.assign("flat_design", FloatVector(design_flat.tolist()))
        robjects.r.assign("p_size", p)

        # Assign each X block converting it to flat format with its dimensions
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
            logger.debug(f"Block '{name}': {n_samples} samples x {n_features} features")
        
        # -- translate keepX to an R list --
        if self.keepX is not None:
            # e.g. list('mRNA'=c(20,20), 'Proteinas'=c(20,20))
            keepX_str = "list(" + ", ".join([f"'{k}'=c({','.join(map(str, v))})" for k, v in self.keepX.items()]) + ")"
        else:
            keepX_str = "NULL"

        logger.info(f"Blocks: {len(block_names)}; samples: {X_dict[block_names[0]].shape[0]}; features per block: {[X_dict[name].shape[1] for name in block_names]}")
        
        logger.info("Running DIABLO (block.splsda) in R...")
        # R script to build the blocks, the design, and fit block.splsda
        r_script = f"""
        library(mixOmics)

        design_matrix <- matrix(flat_design, nrow = p_size, ncol = p_size)

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

        Y_factor <- as.factor(flat_y)

        {self.r_model_var_} <- block.splsda(
            X = X_list,
            Y = Y_factor,
            ncomp = n_comp,
            design = design_matrix,
            keepX = {keepX_str},
            near.zero.var = TRUE
        )

        {self.r_model_var_}
        """
        self.r_fit_ = robjects.r(r_script)

        

        # -- extract loadings and names/variates --
        r_script_loadings = f"""
        r_loadings <- {self.r_model_var_}$loadings
        loadings_names <- names(r_loadings)

        loadings_list <- list()
        for (name in loadings_names) {{
            if (!is.null(r_loadings[[name]])) {{
                mat <- r_loadings[[name]]
                loadings_list[[name]] <- list(matrix = mat, names = rownames(mat))
            }}
        }}
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
        
        logger.info(f"Loadings extracted for {len(self.loadings_)} blocks.")

        return self

    def transform(self, X_dict):
        _require_rpy2()
        if getattr(self, "r_fit_", None) is None:
            raise RuntimeError(
                "The model has not been fitted. Run .fit() first."
            )
        logger.info(f"Extracting variates from {len(X_dict)} blocks...")


        # Validate whether the input blocks match the stored ones or are in the model
        # (DIABLO in transform uses the variates computed from the fitted model, 
        # but we validate that the names exist)
        
        r_script = f"""
        r_variates <- {self.r_model_var_}$variates
        variates_names <- names(r_variates)

        variates_list <- list()
        for (name in variates_names) {{
            if (!is.null(r_variates[[name]])) {{
                variates_list[[name]] <- r_variates[[name]]
            }}
        }}
        variates_list
        """

        res_r = robjects.r(r_script)
        
        # Extract the names available in R
        available_names = list(res_r.names) if hasattr(res_r, 'names') else list(res_r.keys())

        variates_dict = {}
        for name in X_dict.keys():
            if name not in available_names:
                raise KeyError(
                    f"The block '{name}' is not present in the "
                    f"fitted model. Available blocks: {available_names}"
                )
            r_matrix = res_r.rx2(name)
            variates_dict[name] = np.array(r_matrix)

        return variates_dict

    def fit_transform(self, X_dict, Y):
        return self.fit(X_dict, Y).transform(X_dict)

    # temporarily added to satisfy plot_circos()
    def get_expression_means(self, active_features, groups):
        """
        Computes raw expression means per group
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
        Extracts the biomarker variables selected for a specific block and component
        """
        if getattr(self, "loadings_", None) is None:
            raise RuntimeError("The model has not been fitted.")
        
        logger.info(f"Extracting biomarkers for block '{block}' (comp {comp})")


        # extract the loadings of the specified component
        loadings = self.loadings_[block][:, comp - 1]
        names = np.array(self.names_[block])

        # filter the variables that survived keepX
        active_mask = loadings != 0
        active_loadings = loadings[active_mask]
        active_names = names[active_mask]

        # sort by absolute loading value
        sort_idx = np.argsort(np.abs(active_loadings))[::-1]
        df = pd.DataFrame({
            'value': active_loadings[sort_idx]
        }, index=active_names[sort_idx])
        
        df.index.name = 'Variable'
        return df

    def predict(self, X_dict):
        _require_rpy2()
        if getattr(self, "r_fit_", None) is None:
            raise RuntimeError("The model has not been fitted. Run .fit() first.")

        logger.info(f"Making predictions with {len(X_dict)} blocks (n_comp={self.n_components})...")


        p = len(X_dict)
        block_names = list(X_dict.keys())
        
        for name in block_names:
            if name not in self.block_names_:
                raise KeyError(f"The block '{name}' was not seen during training.")
                
            block_arr = np.asarray(X_dict[name], dtype=float)
            n_samples, n_features = block_arr.shape
            col_names = list(X_dict[name].columns) if isinstance(X_dict[name], pd.DataFrame) else [f"Var_{i}" for i in range(n_features)]

            robjects.r.assign(f"test_n_samples_{name}", int(n_samples))
            robjects.r.assign(f"test_n_features_{name}", int(n_features))
            robjects.r.assign(f"test_flat_x_{name}", FloatVector(block_arr.flatten().tolist()))
            robjects.r.assign(f"test_col_names_{name}", StrVector(col_names))

        robjects.r.assign("test_block_names", StrVector(block_names))
        robjects.r.assign("test_n_comp", int(self.n_components))

        r_script_predict = f"""
        library(mixOmics)

        X_test_list <- list()
        for (name in test_block_names) {{
            n_s <- get(paste0("test_n_samples_", name))
            n_f <- get(paste0("test_n_features_", name))
            flat_vals <- get(paste0("test_flat_x_", name))
            c_names <- get(paste0("test_col_names_", name))
            mat <- matrix(flat_vals, nrow = n_s, ncol = n_f, byrow = TRUE)
            colnames(mat) <- c_names
            X_test_list[[name]] <- mat
        }}

        pred <- tryCatch({{
            predict({self.r_model_var_}, newdata = X_test_list, ncomp = test_n_comp)
        }}, error = function(e) {{
            message("PREDICT ERROR: ", conditionMessage(e))
            NULL
        }})

        res <- NULL
        if (!is.null(pred)) {{
            if (!is.null(pred$AveragedPredict.class$max.dist)) {{
                res <- pred$AveragedPredict.class$max.dist
            }} else if (!is.null(pred$predict.class$max.dist)) {{
                res <- pred$predict.class$max.dist
            }} else {{
                first_b <- test_block_names[1]
                if (!is.null(pred$class[[first_b]]$max.dist)) {{
                    res <- pred$class[[first_b]]$max.dist
                }}
            }}
        }}

        if (!is.null(res)) {{
            if (is.vector(res)) {{
                res <- matrix(res, ncol = test_n_comp)
            }}
        }}
        res
        """

        res_preds = robjects.r(r_script_predict)
        
        # If R returned NULL or an empty object due to an extreme fold
        if res_preds is robjects.NULL or res_preds is None:
            logger.warning("R returned NULL in predict (possible extreme fold); generating zeros fallback.")
            n_test_samples = len(next(iter(X_dict.values())))
            return np.zeros((n_test_samples, self.n_components), dtype=int)

        arr = np.array(res_preds)   
        
        if arr.ndim == 0:
            logger.warning("R returned an empty scalar prediction in predict; generating zeros fallback.")
            n_test_samples = len(next(iter(X_dict.values())))
            return np.zeros((n_test_samples, self.n_components), dtype=int)
            
        if arr.ndim == 1:
            arr = arr.reshape(-1, self.n_components)
            
        return arr
