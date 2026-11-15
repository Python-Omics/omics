"""
MINT: Python-native Multi-group INtegrative Tensor analysis.

Pure Python implementation based on the mixOmics MINT algorithm.
No R/rpy2 dependency required.
"""

import numpy as np
import pandas as pd
from .._sklearn_compat import BaseEstimator, TransformerMixin
from dataclasses import dataclass
import logging
import uuid

from .mint_internal import internal_mint_block
from .mint_helpers import mean_centering_per_study, study_split

logger = logging.getLogger(__name__)


@dataclass
class MINTResult:
    """Container for MINT results compatible with visualization functions."""
    variates: list
    loadings: list
    names: list
    study: np.ndarray
    design: np.ndarray
    ncomp: list
    mode: str
    scheme: str
    crit: list
    iter: list
    tau: list


class MINT(BaseEstimator, TransformerMixin):
    """
    MINT (Multi-group INtegrative Tensor) analysis for multi-omics data with batch/study effects.
    
    This is a pure Python implementation of the MINT algorithm from mixOmics,
    supporting both the MINT path (tau=None, per-study loading aggregation)
    and RGCCA path (tau='optimal' or fixed, ridge regularization).
    
    Parameters
    ----------
    n_components : int or list of int, default=2
        Number of components to compute. If int, same for all blocks.
        If list, length J (n_blocks) specifying components per block.
    design : array-like, shape (J, J), optional
        Design matrix specifying block connections. Default: full design (1 - I).
    keepA : dict or list of list, optional
        Sparsity parameter. If dict: {block_name: [keepX_comp1, keepX_comp2, ...]}.
        If list of lists: keepA[comp][block]. Number of variables to keep per block per component.
    penalty : dict or list, optional
        L1 penalty parameter. If dict: {block_name: penalty_value}.
        If list: length J, penalty per block. Alternative to keepA.
    tau : None, "optimal", list, or list of list, default=None
        Ridge regularization parameter.
        - None (default): MINT path (no ridge, per-study aggregation)
        - "optimal": auto-estimate per block
        - list of float (length J): fixed tau per block
        - list of list (length N): tau per component per block
    scheme : {"horst", "factorial", "centroid"}, default="horst"
        Scheme for the optimization criterion.
    mode : {"canonical", "classic", "invariant", "regression"}, default="canonical"
        Deflation mode.
    scale : bool, default=True
        Whether to scale each block to unit variance (per study).
    max_iter : int, default=100
        Maximum iterations per component.
    tol : float, default=1e-6
        Convergence tolerance.
    init : {"svd", "svd.single"}, default="svd.single"
        Initialization method for loadings.
    
    Attributes
    ----------
    loadings_ : dict
        Loadings per block: {block_name: array (p_k, ncomp)}
    variates_ : dict
        Variates (scores) per block: {block_name: array (n, ncomp)}
    names_ : dict
        Variable names per block: {block_name: list of str}
    X_dict_ : dict
        Original (aligned) data blocks: {block_name: DataFrame}
    Y_ : array-like or None
        Response variable (if supervised)
    study_ : array-like
        Study/batch labels for each sample
    block_names_ : list
        Names of blocks in order
    n_components_ : list
        Number of components per block
    variates_partial_ : dict or None
        Per-study variates: {block_name: {study_level: array (n_study, ncomp)}}.
        Available for the MINT path (``tau=None``).
    loadings_partial_ : dict or None
        Per-study loadings: {block_name: {study_level: array (p_k, ncomp)}}.
        Available for the MINT path (``tau=None``).
    study_levels_ : list
        Study levels in the order used for scaling.
    Y_names_ : list or None
        Class labels for a categorical response.
    means_Y_, sigma_Y_ : ndarray or None
        First-study response centering/scaling used by :meth:`predict`.
    """
    
    def __init__(
        self,
        n_components=2,
        design=None,
        keepA=None,
        penalty=None,
        tau=None,
        scheme="horst",
        mode="canonical",
        scale=True,
        max_iter=100,
        tol=1e-6,
        init="svd.single"
    ):
        self.n_components = n_components
        self.design = design
        self.keepA = keepA
        self.penalty = penalty
        self.tau = tau
        self.scheme = scheme
        self.mode = mode
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        
        self.X_dict_ = None
        self.Y_ = None
        self.study_ = None
        self.loadings_ = None
        self.variates_ = None
        self.names_ = None
        self.block_names_ = None
        self.n_components_ = None
        self.indY_ = None
        self.variates_partial_ = None
        self.loadings_partial_ = None
        self.study_levels_ = None
        self.Y_names_ = None
        self.means_Y_ = None
        self.sigma_Y_ = None
        self._X_scale_info_ = None
        self._Y_scale_info_ = None
        self._mint_result = None
        self.r_model_var_ = f"mint_fit_{uuid.uuid4().hex[:8]}"

    @classmethod
    def from_mudata(cls, mdata, target_col=None, study_col="study", **kwargs):
        """
        Create MINT instance from a MuData object.
        
        Parameters
        ----------
        mdata : mudata.MuData
            Multi-modal data container
        target_col : str, optional
            Column in mdata.obs for supervised analysis (Y)
        study_col : str, default="study"
            Column in mdata.obs containing study/batch labels
        **kwargs : dict
            Additional arguments passed to MINT constructor
            
        Returns
        -------
        MINT
            Initialized but unfitted instance with X_dict_ and Y_ set
        """
        logger.info(f"Creating MINT from MuData with study_col='{study_col}'")
        
        # Extract data blocks
        X_dict = {name: mod.to_df() for name, mod in mdata.mod.items()}
        
        # Extract study labels
        if study_col in mdata.obs.columns:
            study = mdata.obs[study_col].values
        else:
            logger.warning(f"study_col '{study_col}' not found in mdata.obs. Using single study.")
            n_samples = mdata.shape[0]
            study = np.zeros(n_samples, dtype=int)
        
        # Extract Y if target_col provided
        Y = None
        if target_col is not None:
            if target_col in mdata.obs.columns:
                Y = mdata.obs[target_col].values
            else:
                # Try first modality's obs
                first_mod = list(mdata.mod.keys())[0]
                if target_col in mdata.mod[first_mod].obs.columns:
                    Y = mdata.mod[first_mod].obs[target_col].values
                    logger.warning(f"'{target_col}' not in mdata.obs, obtained from {first_mod}.obs")
                else:
                    raise KeyError(f"target_col '{target_col}' not found in mdata.obs or any modality.obs")
        
        instance = cls(**kwargs)
        instance.X_dict_ = X_dict
        instance.Y_ = Y
        instance.study_ = study
        
        return instance

    def _prepare_keepA(self, J, N):
        """
        Convert keepA from user format (dict or list) to internal format (list of lists).
        
        Internal format: keepA[comp][block] for comp in 0..N-1, block in 0..J-1
        """
        if self.keepA is None:
            return None
        
        if isinstance(self.keepA, dict):
            # Dict format: {block_name: [keepX_c1, keepX_c2, ...]}
            keepA_internal = []
            for comp in range(N):
                comp_keepA = []
                for block in self.block_names_:
                    block_keep = self.keepA.get(block, [None] * N)
                    if isinstance(block_keep, list) and len(block_keep) > comp:
                        comp_keepA.append(block_keep[comp])
                    else:
                        comp_keepA.append(block_keep if not isinstance(block_keep, list) else None)
                keepA_internal.append(comp_keepA)
            return keepA_internal
        
        elif isinstance(self.keepA, list):
            # Already in internal format (list of lists)
            # Ensure it's N x J
            if len(self.keepA) == J and not isinstance(self.keepA[0], list):
                # It's per-block only [keepX_block1, keepX_block2, ...] -> replicate across components
                return [self.keepA[:] for _ in range(N)]
            return self.keepA
        
        return None

    def _prepare_penalty(self, J):
        """Convert penalty from user format to internal list format."""
        if self.penalty is None:
            return None
        
        if isinstance(self.penalty, dict):
            return [self.penalty.get(block, None) for block in self.block_names_]
        elif isinstance(self.penalty, list):
            return self.penalty
        return None

    def _prepare_ncomp(self, J):
        """Convert n_components to list format."""
        if isinstance(self.n_components, int):
            return [self.n_components] * J
        elif isinstance(self.n_components, list):
            if len(self.n_components) != J:
                raise ValueError(f"n_components list must have length J={J}")
            return self.n_components
        return [2] * J

    def _prepare_design(self, J):
        """Prepare design matrix."""
        if self.design is None:
            return 1 - np.eye(J)
        return np.asarray(self.design, dtype=float)

    @staticmethod
    def _encode_response(Y):
        """
        Encode the response Y into a numeric block (n, q).

        - numeric 1-D          -> single continuous column
        - numeric 2-D          -> used as is
        - non-numeric object   -> one-hot indicator matrix (classification)

        Mirrors mixOmics' `Check.entry.wrapper.mint.block`, which requires a
        numeric Y matrix (splsda builds the dummy/indicator matrix upstream).
        """
        Y = np.asarray(Y)
        if Y.ndim == 1:
            if Y.dtype.kind in "iufb":  # numeric / bool
                return Y.reshape(-1, 1).astype(float)
            classes = np.unique(Y)
            return (Y[:, None] == classes[None, :]).astype(float)
        if Y.dtype.kind not in "iufb":
            raise ValueError("A 2-D 'Y' must be numeric.")
        return Y.astype(float)

    def _extend_design_supervised(self, design_x, J):
        """
        Build the (J+1)x(J+1) design including the Y block.

        Mirrors R's Check.entry.wrapper.mint.block:
        - default                     -> 1 - I_{J+1} (full design, Y included)
        - user (J x J)                -> rbind(cbind(design, 1), 1); diag = 0
        - user (J+1 x J+1)            -> used as is
        """
        if self.design is None:
            return 1 - np.eye(J + 1)

        if design_x.shape == (J + 1, J + 1):
            design = design_x.copy()
        elif design_x.shape == (J, J):
            design = np.zeros((J + 1, J + 1), dtype=float)
            design[:J, :J] = design_x
            design[J, :J] = 1.0
            design[:J, J] = 1.0
        else:
            raise ValueError(
                f"design must be ({J}x{J}) or ({J+1}x{J+1}); got {design_x.shape}"
            )
        np.fill_diagonal(design, 0.0)
        return design

    def fit(self, X_dict=None, Y=None, study=None):
        """
        Fit the MINT model.
        
        Parameters
        ----------
        X_dict : dict of {str: array-like}, optional
            Data blocks. If None, uses self.X_dict_ (set by from_mudata).
        Y : array-like, optional
            Response variable for supervised mode. If None, uses self.Y_.
        study : array-like, optional
            Study/batch labels. If None, uses self.study_.
            
        Returns
        -------
        self
        """
        logger.info("Fitting MINT model...")
        
        X_dict = X_dict if X_dict is not None else self.X_dict_
        Y = Y if Y is not None else self.Y_
        study = study if study is not None else self.study_
        
        if X_dict is None:
            raise ValueError("X_dict must be provided or model must be initialized from MuData.")
        
        # Store block names in consistent order
        self.block_names_ = list(X_dict.keys())
        J = len(self.block_names_)
        
        if J < 1:
            raise ValueError("MINT requires at least one data block.")
        if J < 2 and Y is None:
            raise ValueError(
                "MINT with a single data block requires a supervised response Y. "
                "An unsupervised single block has a degenerate design matrix."
            )
        
        # Convert to list of arrays in block order
        A = [np.asarray(X_dict[block], dtype=float) for block in self.block_names_]
        n = A[0].shape[0]
        
        # Validate shapes
        for i, a in enumerate(A):
            if a.shape[0] != n:
                raise ValueError(f"Block {self.block_names_[i]} has {a.shape[0]} samples, expected {n}")
        
        # Prepare study labels
        if study is None:
            study = np.zeros(n, dtype=int)
        study = np.asarray(study)
        if study.shape[0] != n:
            raise ValueError(f"study has {study.shape[0]} labels, expected {n}")
        self.study_levels_ = list(np.unique(study))
        
        # Keep the per-study scaling parameters so predict() can apply the
        # training transform to new samples.
        self._X_scale_info_ = [
            mean_centering_per_study(a, study, self.scale) for a in A
        ]
        
        # Prepare ncomp / design for the X blocks (Y added below if supervised)
        ncomp = self._prepare_ncomp(J)
        design = self._prepare_design(J)
        
        # Supervised mode: append Y as the last block and set indY
        indY = None
        Y_original = Y
        if Y is not None:
            Y_block = self._encode_response(Y)
            if Y_block.shape[0] != n:
                raise ValueError(
                    f"Y has {Y_block.shape[0]} samples, expected {n}"
                )
            self._Y_scale_info_ = mean_centering_per_study(
                Y_block, study, self.scale
            )
            if np.asarray(Y_original).ndim == 1 and np.asarray(Y_original).dtype.kind not in "iufb":
                self.Y_names_ = [str(v) for v in np.unique(np.asarray(Y_original))]
            else:
                self.Y_names_ = None
            first_study = self.study_levels_[0]
            self.means_Y_ = self._Y_scale_info_.means[first_study]
            if self.scale:
                sigma = self._Y_scale_info_.sigmas[first_study]
                self.sigma_Y_ = np.where(sigma == 0, 1.0, sigma)
            else:
                self.sigma_Y_ = np.ones(Y_block.shape[1])
            indY = J
            A = A + [Y_block]
            design = self._extend_design_supervised(design, J)
            # R uses min(ncomp, ncol(Y)-1) for 'canonical' and max(ncomp) for
            # 'regression'; we always use max(ncomp) so the Y block is never
            # degenerate (see MINT_DISCREPANCIES.md, item 1).
            ncomp = ncomp + [max(ncomp)]
        
        N = max(ncomp)
        
        # Prepare keepA (X blocks) and append the Y column if supervised
        keepA_internal = self._prepare_keepA(J, N)
        if keepA_internal is not None and indY is not None:
            for comp_row in keepA_internal:
                comp_row.append(None)
        
        # Prepare penalty and append None for the Y block if supervised
        penalty_internal = self._prepare_penalty(J)
        if penalty_internal is not None and indY is not None:
            penalty_internal = list(penalty_internal) + [None]
        
        # Supervised MINT requires 'regression' deflation (mixOmics default for
        # block/splsda); 'canonical' would fully deflate a 1-column Y after the
        # first component (`ncol(Y)-1 == 0`).
        mode = self.mode
        if indY is not None and mode == "canonical":
            logger.warning(
                "Supervised MINT requires mode='regression'; overriding "
                "mode='canonical' -> 'regression'."
            )
            mode = "regression"
        
        # Run internal MINT
        result = internal_mint_block(
            A=A,
            indY=indY,
            design=design,
            ncomp=ncomp,
            scheme=self.scheme,
            scale=self.scale,
            init=self.init,
            tol=self.tol,
            mode=mode,
            max_iter=self.max_iter,
            study=study,
            keepA=keepA_internal,
            penalty=penalty_internal,
            all_outputs=False,
            tau=self.tau
        )
        
        # Store results in sklearn-compatible format (X blocks only)
        self.n_components_ = ncomp[:J]
        self.study_ = study
        self.Y_ = Y_original
        self.X_dict_ = X_dict
        self.indY_ = indY
        
        # Loadings: {block_name: array (p, ncomp)}
        self.loadings_ = {}
        for k, block in enumerate(self.block_names_):
            self.loadings_[block] = result.loadings[k]
        
        # Variates: {block_name: array (n, ncomp)}
        self.variates_ = {}
        for k, block in enumerate(self.block_names_):
            self.variates_[block] = result.variates[k]
        
        # Names: {block_name: list of variable names}
        self.names_ = {}
        for k, block in enumerate(self.block_names_):
            if isinstance(X_dict[block], pd.DataFrame):
                self.names_[block] = list(X_dict[block].columns)
            else:
                self.names_[block] = [f"Var_{i}" for i in range(A[k].shape[1])]
        
        # Public access to per-study partial components (MINT path only).
        if result.variates_partial is not None:
            self.variates_partial_ = {
                block: result.variates_partial[k]
                for k, block in enumerate(self.block_names_)
            }
            self.loadings_partial_ = {
                block: result.loadings_partial[k]
                for k, block in enumerate(self.block_names_)
            }
        else:
            self.variates_partial_ = None
            self.loadings_partial_ = None

        # Save internal result for advanced access
        self._mint_result = result
        
        logger.info(f"MINT fit completed. Blocks: {J}, Components per block: {self.n_components_}")
        return self

    def transform(self, X_dict):
        """
        Project new data onto the fitted MINT components.
        
        Parameters
        ----------
        X_dict : dict of {str: array-like}
            New data blocks with same block names and features as training.
            
        Returns
        -------
        variates_dict : dict of {str: array}
            Projected variates for each block: {block_name: (n_new, ncomp)}
        """
        if self.loadings_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        
        variates_dict = {}
        for block in self.block_names_:
            if block not in X_dict:
                raise KeyError(f"Block '{block}' not found in input data.")
            
            X_new = np.asarray(X_dict[block], dtype=float)
            loadings = self.loadings_[block]
            
            # Center using training means (per study not applicable for new data, use global)
            # For simplicity, use global centering (study-specific would require study labels)
            X_centered = X_new - np.nanmean(X_new, axis=0)
            X_centered = np.where(np.isnan(X_centered), 0.0, X_centered)
            
            variates_dict[block] = X_centered @ loadings
        
        return variates_dict

    def fit_transform(self, X_dict, Y=None, study=None):
        """Fit and transform in one step."""
        return self.fit(X_dict, Y, study).transform(X_dict)

    def get_biomarkers(self, block, comp=1):
        """
        Extract biomarkers (selected variables) for a specific block and component.
        
        Parameters
        ----------
        block : str
            Block name
        comp : int, default=1
            Component index (1-based)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with index=variable names, column='value' (loading weights)
        """
        if self.loadings_ is None:
            raise RuntimeError("Model has not been fitted.")
        
        if block not in self.block_names_:
            raise KeyError(f"Block '{block}' not found. Available: {self.block_names_}")
        
        if comp < 1 or comp > self.loadings_[block].shape[1]:
            raise ValueError(f"Component {comp} out of range (1-{self.loadings_[block].shape[1]})")
        
        loadings = self.loadings_[block][:, comp - 1]
        names = np.array(self.names_[block])
        
        # Filter non-zero loadings (selected variables)
        active_mask = loadings != 0
        active_loadings = loadings[active_mask]
        active_names = names[active_mask]
        
        # Sort by absolute loading value (descending)
        sort_idx = np.argsort(np.abs(active_loadings))[::-1]
        
        df = pd.DataFrame({
            'value': active_loadings[sort_idx]
        }, index=active_names[sort_idx])
        
        df.index.name = 'Variable'
        return df

    def get_expression_means(self, active_features, groups):
        """
        Computes raw expression means per group (for circos plot compatibility).
        
        Parameters
        ----------
        active_features : dict
            Output from internal processing with 'X' arrays per block
        groups : dict
            Group name -> color mapping
            
        Returns
        -------
        dict
            {block: {group: mean_vector}}
        """
        group_means = {}
        blocks = list(active_features.keys())
        
        # Flatten Y if it's 2D (e.g., from one-hot or column vector)
        Y_flat = None
        if self.Y_ is not None:
            Y_array = np.array(self.Y_)
            if Y_array.ndim > 1:
                Y_flat = Y_array.ravel()
            else:
                Y_flat = Y_array
        else:
            Y_flat = np.zeros(active_features[blocks[0]]['X'].shape[0])
        
        for block in blocks:
            group_means[block] = {}
            X_active = active_features[block]['X']
            
            for group_name in groups.keys():
                if self.Y_ is not None:
                    mask = (Y_flat == group_name)
                else:
                    # If no Y, assign all to first group
                    mask = np.ones(X_active.shape[0], dtype=bool) if group_name == list(groups.keys())[0] else np.zeros(X_active.shape[0], dtype=bool)
                
                if mask.sum() > 0:
                    group_means[block][group_name] = X_active[mask].mean(axis=0)
                else:
                    group_means[block][group_name] = np.zeros(X_active.shape[1])
        
        return group_means

    def predict(self, X_dict, study=None):
        """
        Predict the response for new data.

        This mirrors ``mixOmics::predict.mint.block.splsda`` / ``predict.mint.splsda``
        when ``newdata.scale`` is the training matrix: each new block is centered
        (and optionally scaled) with the per-study training parameters, projected
        onto the fitted variates, and the response is reconstructed from the
        fitted loadings.

        Parameters
        ----------
        X_dict : dict of {str: array-like}
            New data blocks, samples x features, with the same blocks/features
            as the training data.
        study : array-like, optional
            Study labels for the new samples. Defaults to the training labels
            when the number of samples matches; otherwise a single study is used.

        Returns
        -------
        dict[str, np.ndarray]
            ``{block_name: array(n_samples, n_response_columns, n_components)}``,
            matching the shape of R's ``predict()$predict``.
        """
        if self._mint_result is None or self.indY_ is None:
            raise RuntimeError(
                "Predict requires a fitted supervised model (Y provided during fit)."
            )

        n = len(next(iter(X_dict.values())))
        if study is None:
            if self.study_ is not None and len(self.study_) == n:
                study = self.study_
            else:
                study = np.zeros(n, dtype=int)
        study = np.asarray(study)
        if study.shape[0] != n:
            raise ValueError(f"study has {study.shape[0]} labels, expected {n}.")

        known = set(self.study_levels_)
        unknown = [lvl for lvl in np.unique(study) if lvl not in known]
        if unknown:
            raise ValueError(
                f"New study levels {unknown!r} were not present during training "
                f"({self.study_levels_!r})."
            )

        Y_scaled = self._mint_result.A[self.indY_]
        preds = {}
        for k, block in enumerate(self.block_names_):
            X_new = np.asarray(X_dict[block], dtype=float)
            if X_new.ndim != 2:
                raise ValueError(f"Block {block!r} must be 2-D.")
            if X_new.shape[0] != n:
                raise ValueError(
                    f"Block {block!r} has {X_new.shape[0]} samples, expected {n}."
                )
            if X_new.shape[1] != self._X_scale_info_[k].data.shape[1]:
                raise ValueError(
                    f"Block {block!r} has {X_new.shape[1]} features, expected "
                    f"{self._X_scale_info_[k].data.shape[1]}."
                )

            X_scaled = np.empty_like(X_new, dtype=float)
            for lvl in self.study_levels_:
                mask = study == lvl
                if not mask.any():
                    continue
                info = self._X_scale_info_[k]
                mean = info.means[lvl]
                if self.scale:
                    sigma = info.sigmas[lvl]
                    sigma = np.where(sigma == 0, 1.0, sigma)
                    X_scaled[mask] = (X_new[mask] - mean) / sigma
                else:
                    X_scaled[mask] = X_new[mask] - mean

            t = self._mint_result.variates[k]
            W = self._mint_result.loadings[k]
            P = X_scaled.T @ t
            C = Y_scaled.T @ t
            ncomp = self.n_components_[k]
            block_pred = np.empty((n, Y_scaled.shape[1], ncomp), dtype=float)
            for comp in range(1, ncomp + 1):
                A = P[:, :comp].T @ W[:, :comp]
                coef = np.linalg.solve(A, np.eye(comp))
                Y_pred = X_scaled @ W[:, :comp] @ coef @ C[:, :comp].T
                block_pred[:, :, comp - 1] = (
                    Y_pred * self.sigma_Y_ + self.means_Y_
                )
            preds[block] = block_pred

        return preds


# Compatibility aliases for existing visualization functions
def _get_mint_model_attrs(model):
    """Extract attributes expected by visualization functions."""
    return {
        'loadings_': model.loadings_,
        'names_': model.names_,
        'X_dict_': model.X_dict_,
        'Y_': model.Y_,
        'block_names_': model.block_names_,
        'study_': model.study_,
        'n_components_': model.n_components_,
        'variates_': model.variates_
    }