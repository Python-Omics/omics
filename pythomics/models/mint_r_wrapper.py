"""
R-backed MINT wrapper (mixOmics), analogous to :class:`DIABLOWrapper`.

The pure-Python :class:`pythomics.models.MINT` is the primary implementation.
This module exists to run the *reference* implementation
(``mixOmics::mint.block.splsda`` / ``mixOmics::mint.block.spls``) from Python,
so results can be compared against the Python port.

Two execution backends are supported:

* ``"rpy2"``    : reuse a live R session through rpy2 (like ``DIABLOWrapper``).
* ``"rscript"`` : call ``Rscript`` in a subprocess (no rpy2 needed).
* ``"auto"``    : rpy2 when available, otherwise ``Rscript``.

Both backends write the inputs to a temporary directory, source the same
generated R script and read the results back as pandas objects. This keeps a
single source of truth for the R call.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from .._sklearn_compat import BaseEstimator, TransformerMixin

try:  # optional dependency
    import rpy2.robjects as robjects

    _HAS_RPY2 = True
except ImportError:  # pragma: no cover - depends on the environment
    robjects = None
    _HAS_RPY2 = False

logger = logging.getLogger(__name__)


def _require_r(r_cmd: str = "Rscript") -> None:
    """Raise a clear error when no R executable is available."""
    if shutil.which(r_cmd) is None:
        raise RuntimeError(
            f"MINT R wrapper requires an R installation with the 'mixOmics' "
            f"package. '{r_cmd}' was not found in PATH. Install R, run "
            "install.packages('mixOmics') inside R, or use the pure-Python "
            "pythomics.models.MINT instead."
        )


class MINTRWrapper(BaseEstimator, TransformerMixin):
    """
    Run ``mixOmics`` MINT from Python and expose a scikit-learn-like API.

    Parameters
    ----------
    n_components : int, default=2
        Number of components (``ncomp`` in mixOmics).
    design : array-like, shape (J, J) or (J+1, J+1), optional
        Design matrix for the J data blocks. If ``(J, J)``, mixOmics links
        every block to the outcome block, exactly like the Python ``MINT``.
    keepX : dict or list, optional
        Number of variables to keep per block and component.
        Dict form: ``{'mRNA': [15, 15], 'lipid': [8, 8]}``.
    scale : bool, default=True
        Whether mixOmics should scale each block (per study).
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    backend : {"auto", "rpy2", "rscript"}, default="auto"
        How to execute R.
    r_cmd : str, default="Rscript"
        R executable used by the ``rscript`` backend.
    da : bool or None, default=None
        Force discriminant analysis (``mint.block.splsda``) or continuous
        response (``mint.block.spls``). ``None`` auto-detects: non-numeric
        ``Y`` -> sPLS-DA, numeric ``Y`` -> sPLS.

    Attributes
    ----------
    loadings_ : dict[str, np.ndarray]
        Loadings per data block, shape ``(p_k, ncomp)``.
    variates_ : dict[str, np.ndarray]
        Variates per data block, shape ``(n, ncomp)``.
    names_ : dict[str, list[str]]
        Feature names per block.
    n_components_ : list[int]
        Components actually used per data block.
    iter_ : list[int]
        Iterations per component reported by mixOmics.
    backend_ : str
        Backend actually used (``"rpy2"`` or ``"rscript"``).
    """

    def __init__(
        self,
        n_components: int = 2,
        design=None,
        keepX=None,
        scale: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
        backend: str = "auto",
        r_cmd: str = "Rscript",
        da: Optional[bool] = None,
    ):
        self.n_components = n_components
        self.design = design
        self.keepX = keepX
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.backend = backend
        self.r_cmd = r_cmd
        self.da = da

        self.X_dict_ = None
        self.Y_ = None
        self.study_ = None
        self.loadings_ = None
        self.variates_ = None
        self.names_ = None
        self.block_names_ = None
        self.n_components_ = None
        self.iter_ = None
        self.indY_ = None
        self.backend_ = None
        self.r_stdout_ = None
        self.r_stderr_ = None
        self.r_model_var_ = f"mint_r_fit_{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_mudata(cls, mdata, target_col, study_col="study", **kwargs):
        """Create a wrapper from a MuData object (same contract as MINT)."""
        logger.info("Creating MINTRWrapper from MuData with study_col=%r", study_col)

        X_dict = {name: mod.to_df() for name, mod in mdata.mod.items()}

        if target_col in mdata.obs.columns:
            Y = mdata.obs[target_col].values
        else:
            first_mod = list(mdata.mod.keys())[0]
            if target_col in mdata.mod[first_mod].obs.columns:
                Y = mdata.mod[first_mod].obs[target_col].values
                logger.warning(
                    "%r not in mdata.obs; obtained from %s.obs", target_col, first_mod
                )
            else:
                raise KeyError(
                    f"target_col {target_col!r} not found in mdata.obs or modality.obs"
                )

        if study_col in mdata.obs.columns:
            study = mdata.obs[study_col].values
        else:
            logger.warning(
                "study_col %r not found in mdata.obs. Using a single study.", study_col
            )
            study = np.zeros(mdata.shape[0], dtype=int)

        instance = cls(**kwargs)
        instance.X_dict_ = X_dict
        instance.Y_ = Y
        instance.study_ = study
        return instance

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_index(X_dict: Dict[str, object]) -> List[str]:
        first = next(iter(X_dict.values()))
        n = len(first)
        if isinstance(first, pd.DataFrame):
            idx = [str(v) for v in first.index]
            if len(set(idx)) == n:
                return idx
        return [f"S{i + 1}" for i in range(n)]

    @staticmethod
    def _block_frame(block, index: Sequence[str], name: str) -> pd.DataFrame:
        if isinstance(block, pd.DataFrame):
            arr = block.to_numpy(dtype=float)
            columns = [str(c) for c in block.columns]
        else:
            arr = np.asarray(block, dtype=float)
            columns = [f"{name}_{i}" for i in range(arr.shape[1])]
        if arr.ndim != 2:
            raise ValueError(f"Block {name!r} must be 2-D (samples x features).")
        if arr.shape[0] != len(index):
            raise ValueError(
                f"Block {name!r} has {arr.shape[0]} samples, expected {len(index)}."
            )
        return pd.DataFrame(arr, index=index, columns=columns)

    def _prepare_keepX(self, block_names: Sequence[str], ncomp: int) -> Dict[str, List[int]]:
        if self.keepX is None:
            return {}

        if isinstance(self.keepX, dict):
            out: Dict[str, List[int]] = {}
            for block in block_names:
                value = self.keepX.get(block)
                if value is None:
                    continue
                if np.isscalar(value):
                    out[block] = [int(value)] * ncomp
                else:
                    values = [int(v) for v in value]
                    if len(values) > ncomp:
                        logger.warning(
                            "keepX[%r] has %d values but n_components=%d; "
                            "truncating to the first %d.",
                            block,
                            len(values),
                            ncomp,
                            ncomp,
                        )
                    out[block] = values[:ncomp]
            return out

        if isinstance(self.keepX, (list, tuple)):
            if len(self.keepX) != len(block_names):
                raise ValueError(
                    "When keepX is a list it must have one entry per block "
                    f"({len(block_names)}), got {len(self.keepX)}."
                )
            out = {}
            for block, value in zip(block_names, self.keepX):
                if np.isscalar(value):
                    out[block] = [int(value)] * ncomp
                else:
                    out[block] = [int(v) for v in value][:ncomp]
            return out

        raise TypeError("keepX must be None, a dict or a list.")

    def _prepare_design(self, block_names: Sequence[str]) -> Optional[pd.DataFrame]:
        if self.design is None:
            return None
        design = np.asarray(self.design, dtype=float)
        J = len(block_names)
        if design.shape == (J, J):
            labels = list(block_names)
        elif design.shape == (J + 1, J + 1):
            labels = list(block_names) + ["Y"]
        else:
            raise ValueError(
                f"design must be ({J}x{J}) or ({J + 1}x{J + 1}); got {design.shape}"
            )
        return pd.DataFrame(design, index=labels, columns=labels)

    def _is_discriminant(self, Y) -> bool:
        if self.da is not None:
            return bool(self.da)
        arr = np.asarray(Y)
        return arr.ndim == 1 and arr.dtype.kind not in "iufb"

    # ------------------------------------------------------------------
    # R script
    # ------------------------------------------------------------------
    @staticmethod
    def _r_literal(value) -> str:
        if isinstance(value, str):
            return json.dumps(value)
        if isinstance(value, (bool, np.bool_)):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return repr(float(value))
        return repr(value)

    def _build_r_script(
        self,
        data_dir: str,
        out_dir: str,
        block_names: Sequence[str],
        keepX: Dict[str, List[int]],
        is_da: bool,
        has_design: bool,
    ) -> str:
        block_vec = "c(" + ", ".join(self._r_literal(b) for b in block_names) + ")"
        keep_lines = ["keepX <- list()"]
        for block, values in keepX.items():
            values_str = ", ".join(str(int(v)) for v in values)
            keep_lines.append(f"keepX[[{self._r_literal(block)}]] <- c({values_str})")
        keep_code = "\n".join(keep_lines)

        common = (
            f"ncomp <- {int(self.n_components)}\n"
            f"scale_flag <- {self._r_literal(bool(self.scale))}\n"
            f"max_iter <- {int(self.max_iter)}\n"
            f"tol <- {self._r_literal(float(self.tol))}\n"
        )
        design_arg = "design = design, " if has_design else ""
        call = (
            f"res <- mint.block.splsda(X, Y, study = study, ncomp = ncomp, "
            f"keepX = keepX, {design_arg}scale = scale_flag, "
            f"tol = tol, max.iter = max_iter)"
            if is_da
            else
            f"res <- mint.block.spls(X, Y, study = study, ncomp = ncomp, "
            f"keepX = keepX, {design_arg}mode = 'regression', "
            f"scale = scale_flag, tol = tol, max.iter = max_iter)"
        )

        return f"""
suppressPackageStartupMessages(library(mixOmics))

data_dir <- {self._r_literal(data_dir)}
out_dir <- {self._r_literal(out_dir)}
block_names <- {block_vec}

X <- list()
for (i in seq_along(block_names)) {{
    b <- block_names[[i]]
    path <- file.path(data_dir, paste0("X_", i - 1, ".csv"))
    X[[b]] <- as.matrix(read.csv(path, row.names = 1, check.names = FALSE))
}}

Y_df <- read.csv(file.path(data_dir, "Y.csv"), row.names = 1, check.names = FALSE)

study <- read.csv(file.path(data_dir, "study.csv"), row.names = 1,
                  check.names = FALSE)[[1]]

design <- NULL
design_path <- file.path(data_dir, "design.csv")
if (file.exists(design_path)) {{
    design <- as.matrix(read.csv(design_path, row.names = 1, check.names = FALSE))
}}

{keep_code}

{common}

if ({self._r_literal(is_da)}) {{
    Y <- as.factor(Y_df[[1]])
}} else {{
    Y <- as.matrix(Y_df)
    rownames(Y) <- rownames(X[[1]])
}}

{call}

for (i in seq_along(block_names)) {{
    b <- block_names[[i]]
    write.csv(res$loadings[[b]],
              file.path(out_dir, paste0("loadings_", i - 1, ".csv")))
    write.csv(res$variates[[b]],
              file.path(out_dir, paste0("variates_", i - 1, ".csv")))
}}

info_blocks <- data.frame(
    block = block_names,
    ncomp = as.integer(res$ncomp[seq_along(block_names)]),
    stringsAsFactors = FALSE
)
write.csv(info_blocks, file.path(out_dir, "info_blocks.csv"), row.names = FALSE)

write.csv(
    data.frame(comp = seq_along(res$iter), iter = as.integer(res$iter)),
    file.path(out_dir, "info_iter.csv"),
    row.names = FALSE
)

cat("MINT_R_OK\\n")
"""

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _resolve_backend(self) -> str:
        if self.backend not in {"auto", "rpy2", "rscript"}:
            raise ValueError("backend must be one of {'auto', 'rpy2', 'rscript'}.")
        if self.backend == "auto":
            return "rpy2" if _HAS_RPY2 else "rscript"
        if self.backend == "rpy2" and not _HAS_RPY2:
            raise ImportError(
                "backend='rpy2' requested but rpy2 is not installed. "
                "Use backend='rscript' or install rpy2."
            )
        return self.backend

    def _execute_r(self, script_path: str, backend: str) -> None:
        if backend == "rpy2":
            logger.info("Running MINT in R via rpy2...")
            try:
                robjects.r.source(script_path)
                return
            except Exception as exc:
                if self.backend != "auto":
                    raise
                logger.warning(
                    "rpy2 execution failed (%s); falling back to Rscript.", exc
                )
                backend = "rscript"
                self.backend_ = "rscript"

        _require_r(self.r_cmd)
        logger.info("Running MINT in R via %s...", self.r_cmd)
        proc = subprocess.run(
            [self.r_cmd, "--vanilla", script_path],
            capture_output=True,
            text=True,
            check=False,
        )
        self.r_stdout_ = proc.stdout
        self.r_stderr_ = proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(
                "R MINT execution failed.\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}"
            )

    def fit(self, X_dict=None, Y=None, study=None):
        """Fit the mixOmics MINT model."""
        X_dict = X_dict if X_dict is not None else self.X_dict_
        Y = Y if Y is not None else self.Y_
        study = study if study is not None else self.study_

        if X_dict is None:
            raise ValueError(
                "X_dict must be provided or the model must be initialized from MuData."
            )
        if Y is None:
            raise ValueError(
                "mixOmics MINT requires an outcome Y (mint.block.splsda/spls). "
                "Use the pure-Python pythomics.models.MINT for unsupervised runs."
            )

        block_names = list(X_dict.keys())
        if len(block_names) < 2:
            raise ValueError(
                "MINT is a multi-block method and requires at least 2 data blocks."
            )

        index = self._sample_index(X_dict)
        n = len(index)
        Y_arr = np.asarray(Y)
        if Y_arr.shape[0] != n:
            raise ValueError(f"Y has {Y_arr.shape[0]} samples, expected {n}.")

        if study is None:
            study = np.zeros(n, dtype=int)
        study_arr = np.asarray(study)
        if study_arr.shape[0] != n:
            raise ValueError(f"study has {study_arr.shape[0]} labels, expected {n}.")

        if not np.isscalar(self.n_components):
            raise ValueError(
                "MINTRWrapper requires n_components to be an int; per-block "
                "ncomp is not supported yet."
            )
        ncomp = int(self.n_components)
        if ncomp < 1:
            raise ValueError("n_components must be >= 1.")
        keepX = self._prepare_keepX(block_names, ncomp)
        design = self._prepare_design(block_names)
        is_da = self._is_discriminant(Y)
        backend = self._resolve_backend()

        self.X_dict_ = X_dict
        self.Y_ = Y
        self.study_ = study_arr
        self.block_names_ = block_names
        self.backend_ = backend

        with tempfile.TemporaryDirectory(prefix="mint_r_") as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)

            for i, block in enumerate(block_names):
                frame = self._block_frame(X_dict[block], index, block)
                frame.to_csv(os.path.join(data_dir, f"X_{i}.csv"))

            if Y_arr.ndim == 1:
                Y_frame = pd.DataFrame({"Y": Y_arr}, index=index)
            elif Y_arr.ndim == 2:
                Y_frame = pd.DataFrame(Y_arr, index=index)
                Y_frame.columns = [f"Y{j + 1}" for j in range(Y_frame.shape[1])]
            else:
                raise ValueError("Y must be 1-D or 2-D.")
            Y_frame.to_csv(os.path.join(data_dir, "Y.csv"))

            pd.DataFrame({"study": study_arr}, index=index).to_csv(
                os.path.join(data_dir, "study.csv")
            )

            if design is not None:
                design.to_csv(os.path.join(data_dir, "design.csv"))

            script = self._build_r_script(
                data_dir=data_dir,
                out_dir=out_dir,
                block_names=block_names,
                keepX=keepX,
                is_da=is_da,
                has_design=design is not None,
            )
            script_path = os.path.join(tmpdir, "fit_mint.R")
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(script)

            self._execute_r(script_path, backend)

            self.loadings_ = {}
            self.variates_ = {}
            self.names_ = {}
            for i, block in enumerate(block_names):
                loadings = pd.read_csv(
                    os.path.join(out_dir, f"loadings_{i}.csv"), index_col=0
                )
                variates = pd.read_csv(
                    os.path.join(out_dir, f"variates_{i}.csv"), index_col=0
                )
                frame = self._block_frame(X_dict[block], index, block)
                loadings = loadings.reindex(frame.columns)
                if loadings.isna().any().any():
                    raise RuntimeError(
                        f"R loadings for block {block!r} could not be aligned "
                        "with the input feature names."
                    )
                self.loadings_[block] = loadings.to_numpy(dtype=float)
                self.names_[block] = list(loadings.index)

                variates = variates.reindex(index)
                self.variates_[block] = variates.to_numpy(dtype=float)

            info_blocks = pd.read_csv(os.path.join(out_dir, "info_blocks.csv"))
            self.n_components_ = [int(v) for v in info_blocks["ncomp"].tolist()]

            info_iter = pd.read_csv(os.path.join(out_dir, "info_iter.csv"))
            self.iter_ = [int(v) for v in info_iter["iter"].tolist()]

            self.indY_ = len(block_names)

        logger.info(
            "MINT R fit completed (backend=%s, blocks=%s, ncomp=%s, iter=%s).",
            backend,
            block_names,
            self.n_components_,
            self.iter_,
        )
        return self

    # ------------------------------------------------------------------
    # sklearn / pythomics compatibility
    # ------------------------------------------------------------------
    def transform(self, X_dict):
        """Project new blocks on the fitted loadings (global centering)."""
        if self.loadings_ is None:
            raise RuntimeError("The model has not been fitted. Run .fit() first.")

        variates_dict = {}
        for block in self.block_names_:
            if block not in X_dict:
                raise KeyError(f"Block {block!r} not found in input data.")
            X_new = np.asarray(X_dict[block], dtype=float)
            X_centered = X_new - np.nanmean(X_new, axis=0)
            X_centered = np.where(np.isnan(X_centered), 0.0, X_centered)
            variates_dict[block] = X_centered @ self.loadings_[block]
        return variates_dict

    def fit_transform(self, X_dict=None, Y=None, study=None):
        return self.fit(X_dict, Y=Y, study=study).transform(
            X_dict if X_dict is not None else self.X_dict_
        )

    def get_biomarkers(self, block, comp=1):
        """Selected variables for a block/component, sorted by |loading|."""
        if self.loadings_ is None:
            raise RuntimeError("The model has not been fitted.")
        if block not in self.block_names_:
            raise KeyError(f"Block {block!r} not found. Available: {self.block_names_}")

        if comp < 1 or comp > self.loadings_[block].shape[1]:
            raise ValueError(
                f"Component {comp} out of range (1-{self.loadings_[block].shape[1]})."
            )

        loadings = self.loadings_[block][:, comp - 1]
        names = np.array(self.names_[block])
        active = loadings != 0

        df = pd.DataFrame({"value": loadings[active]}, index=names[active])
        df = df.sort_values("value", key=lambda s: s.abs(), ascending=False)
        df.index.name = "Variable"
        return df

    def get_expression_means(self, active_features, groups):
        """Raw expression means per group (visualization compatibility)."""
        group_means = {}
        blocks = list(active_features.keys())
        Y_array = np.asarray(self.Y_)
        Y_flat = Y_array.ravel() if Y_array.ndim > 1 else Y_array

        for block in blocks:
            group_means[block] = {}
            X_active = active_features[block]["X"]
            for group_name in groups.keys():
                mask = Y_flat == group_name
                if mask.sum() > 0:
                    group_means[block][group_name] = X_active[mask].mean(axis=0)
                else:
                    group_means[block][group_name] = np.zeros(X_active.shape[1])
        return group_means

    def predict(self, X_dict):
        raise NotImplementedError(
            "MINTRWrapper.predict is not implemented yet. Use the R model "
            "object directly (predict.mint.splsda) or the Python MINT wrapper."
        )
