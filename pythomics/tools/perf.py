import numpy as np
from sklearn.model_selection import StratifiedKFold


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
            raise ValueError("model_params debe incluir 'n_components' (o 'ncomp').")

        error_rate_overall = np.zeros((self.nrepeat, ncomp))
        error_rate_ber = np.zeros((self.nrepeat, ncomp, num_classes))

        for r in range(self.nrepeat):
            skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=r)

            y_array = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
            dummy_X = np.zeros((len(y_array), 1))

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dummy_X, y_array)):

                if isinstance(X, dict):
                    X_train = {k: v[train_idx] for k, v in X.items()}
                    X_test = {k: v[test_idx] for k, v in X.items()}
                else:
                    X_train, X_test = X[train_idx], X[test_idx]

                y_train, y_test = y_array[train_idx], y_array[test_idx]

                model = self.model_class(**self.model_params)
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                # normalizamos tipos antes de comparar: predict() de DIABLOWrapper
                # devuelve strings (niveles de factor de R), y_test puede venir como int
                y_test_str = y_test.astype(str)

                for comp in range(ncomp):
                    preds_comp = predictions[:, comp].astype(str)
                    incorrect = preds_comp != y_test_str
                    error_rate_overall[r, comp] += np.sum(incorrect) / len(y_test)

            # promediar sobre los folds de esta repetición
            error_rate_overall[r, :] /= self.folds

        mean_overall_error = np.mean(error_rate_overall, axis=0)

        return {
            "error_rate_overall": mean_overall_error,
        }