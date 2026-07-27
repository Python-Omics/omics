"""
Aquí iré colocando mocks de los métodos para poder probar las visualizaciones


Idea es poder reemplazar luego con las implementaciones reales sin tener que cambiar demasiado código
de visualizaciones
"""


import pandas as pd
import numpy as np

class MockDIABLO:
    def __init__(self, path_rna, path_prot):
        df_rna = pd.read_csv(path_rna, index_col=0)
        df_prot = pd.read_csv(path_prot, index_col=0)
        
        self.names_ = {
            'mRNA': [str(nombre).strip() for nombre in df_rna.index.tolist()],
            'Proteinas': [str(nombre).strip() for nombre in df_prot.index.tolist()]
        }
        
        # Simulamos los 'loadings_' (pesos que asigna el algoritmo a cada variable).
        # Generamos números aleatorios entre -1 y 1 simulando las componentes.
        n_comp = 15
        np.random.seed(42) # Semilla para que el gráfico siempre se vea igual en las pruebas
        self.loadings_ = {
            'mRNA': np.random.uniform(-1, 1, size=(len(self.names_['mRNA']), n_comp)),
            'Proteinas': np.random.uniform(-1, 1, size=(len(self.names_['Proteinas']), n_comp))
        }
