import numpy as np
from sklearn.model_selection import StratifiedKFold


class Perf:
    def __init__(self, model_class, model_params, folds=5, nrepeat=10, dist='centroids.dist'):
        """
        model_class: La clase de tu modelo (ej. MyDiabloWrapper o PLSDA)
        model_params: Diccionario con los hiperparámetros para instanciar el modelo
        folds: Número de pliegues para la validación cruzada (por defecto 5)
        nrepeat: Número de repeticiones de la validación cruzada (por defecto 10)
        dist: Métrica de distancia para la predicción ('centroids.dist', 'max.dist', 'mahalanobis.dist')
        """
        self.model_class = model_class
        self.model_params = model_params
        self.folds = folds
        self.nrepeat = nrepeat
        self.dist = dist

    def evaluate(self, X, y):
        """
        Ejecuta la validación cruzada repetida y calcula los errores.
        X: Matriz o diccionario de bloques de datos (según aplique a DIABLO/PLSDA)
        y: Vector de etiquetas de clase
        """
        num_samples = len(y)
        classes = np.unique(y)
        num_classes = len(classes)

        # Estructuras para almacenar errores por repetición y por fold
        # mixOmics suele evaluar por componente (ncomp)
        ncomp = self.model_params.get('ncomp', 2)

        error_rate_overall = np.zeros((self.nrepeat, ncomp))
        error_rate_ber = np.zeros((self.nrepeat, ncomp, num_classes))  # O un promedio ponderado por BER

        for r in range(self.nrepeat):
            # Usamos StratifiedKFold para mantener la proporción de clases en cada fold
            skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=r)

            y_array = y.to_numpy() if hasattr(y, "to_numpy") else y

            # Use a dummy array matching the sample size of y for splitting
            dummy_X = np.zeros((len(y_array), 1))

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dummy_X, y_array)):

                # 1. Separar datos de entrenamiento y prueba
                # (Si X es un diccionario de bloques para DIABLO, se debe iterar sobre las llaves)
                if isinstance(X, dict):
                    X_train = {k: v[train_idx] for k, v in X.items()}
                    X_test = {k: v[test_idx] for k, v in X.items()}
                else:
                    X_train, X_test = X[train_idx], X[test_idx]

                y_array = y.to_numpy() if hasattr(y, "to_numpy") else y
                y_train, y_test = y_array[train_idx], y_array[test_idx]

                # 2. Instanciar y entrenar el modelo con los datos de entrenamiento
                model = self.model_class(**self.model_params)
                model.fit(X_train, y_train)

                # 3. Predecir sobre el conjunto de prueba (out-of-sample)
                predictions = model.predict(X_test)  # Debe retornar un array de forma (len(test_idx), ncomp)

                # 4. Calcular errores por componente
                for comp in range(ncomp):
                    preds_comp = predictions[:, comp]

                    # Error global en este fold
                    incorrect = preds_comp != y_test
                    error_rate_overall[r, comp] += np.sum(incorrect) / len(y_test)
            error_rate_overall[r, :] /= self.folds
                    
        # Promediar los resultados a lo largo de las repeticiones
        mean_overall_error = np.mean(error_rate_overall, axis=0)

        return {
            "error_rate_overall": mean_overall_error,
        }