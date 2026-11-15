"""Minimal scikit-learn compatibility layer.

The core models only use ``BaseEstimator`` and ``TransformerMixin`` for API
compatibility.  Keeping a tiny fallback here lets ``pythomics.models`` work in
environments without scikit-learn (for example, a notebook kernel that only
has numpy/pandas).
"""

from __future__ import annotations

import inspect

try:  # pragma: no cover - depends on the environment
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:  # pragma: no cover - exercised in minimal environments
    class BaseEstimator:  # type: ignore[no-redef]
        """Small subset of ``sklearn.base.BaseEstimator``."""

        @classmethod
        def _get_param_names(cls):
            signature = inspect.signature(cls.__init__)
            return [
                name
                for name, param in signature.parameters.items()
                if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
            ]

        def get_params(self, deep=True):
            return {name: getattr(self, name) for name in self._get_param_names()}

        def set_params(self, **params):
            for name, value in params.items():
                setattr(self, name, value)
            return self

    class TransformerMixin:  # type: ignore[no-redef]
        """Small subset of ``sklearn.base.TransformerMixin``."""

        def fit_transform(self, X, y=None, **fit_params):
            if y is None:
                return self.fit(X, **fit_params).transform(X)
            return self.fit(X, y, **fit_params).transform(X)


__all__ = ["BaseEstimator", "TransformerMixin"]
