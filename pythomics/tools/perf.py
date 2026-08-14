import inspect

import numpy as np
from sklearn.model_selection import StratifiedKFold
import logging

logger = logging.getLogger(__name__)


class Perf:
    def __init__(self, model_class, model_params, folds=5, nrepeat=10, dist='centroids.dist'):
        self.model_class = model_class
        self.model_params = model_params
        self.folds = folds
        self.nrepeat = nrepeat
        self.dist = dist

    def evaluate(self, X, y):
        classes = np.unique(y)
        num_classes = len(classes)

        if 'n_components' in self.model_params:
            ncomp = self.model_params['n_components']
        elif 'ncomp' in self.model_params:
            ncomp = self.model_params['ncomp']
        else:
            params = inspect.signature(self.model_class.__init__).parameters
            ncomp = params['ncomp'].default if 'ncomp' in params else None
            if ncomp is None and 'n_components' in params:
                ncomp = params['n_components'].default
            if ncomp is None:
                raise ValueError("model_params must include 'n_components' (or 'ncomp').")

        logger.info(f"Running perf evaluation | model: {self.model_class.__name__} | folds: {self.folds} | nrepeat: {self.nrepeat} | ncomp: {ncomp}")

        error_rate_overall = np.zeros((self.nrepeat, ncomp))
        error_rate_ber = np.zeros((self.nrepeat, ncomp, num_classes))

        for r in range(self.nrepeat):
            logger.debug(f"Repetition {r + 1}/{self.nrepeat} (random_state={r})...")
            skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=r)

            y_array = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
            dummy_X = np.zeros((len(y_array), 1))

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dummy_X, y_array)):
                logger.debug(f"  Fold {fold_idx + 1}/{self.folds}")

                if isinstance(X, dict):
                    X_train = {k: v[train_idx] for k, v in X.items()}
                    X_test = {k: v[test_idx] for k, v in X.items()}
                else:
                    X_train, X_test = X[train_idx], X[test_idx]

                y_train, y_test = y_array[train_idx], y_array[test_idx]

                model = self.model_class(**self.model_params)
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                # normalize types before comparing: predict() from DIABLOWrapper
                # returns strings (R factor levels), y_test may come as int
                y_test_str = y_test.astype(str)

                for comp in range(ncomp):
                    preds_comp = predictions[:, comp].astype(str)
                    incorrect = preds_comp != y_test_str
                    error_rate_overall[r, comp] += np.sum(incorrect) / len(y_test)

            # average over the folds of this repetition
            error_rate_overall[r, :] /= self.folds

        mean_overall_error = np.mean(error_rate_overall, axis=0)

        logger.info(f"Perf evaluation completed. error_rate_overall: {np.round(mean_overall_error, 4)}")

        return {
            "error_rate_overall": mean_overall_error,
        }