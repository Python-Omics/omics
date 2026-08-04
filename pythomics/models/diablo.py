"""
Aquí iré colocando mocks de los métodos para poder probar las visualizaciones


Idea es poder reemplazar luego con las implementaciones reales sin tener que cambiar demasiado código
de visualizaciones
"""


import pandas as pd
import numpy as np

class MockDIABLO:
    def __init__(self, path_rna, path_prot, path_proteomics, path_vsd, path_metadata):
        # -- datos crudos --
        df_prot_raw = pd.read_csv(path_proteomics, sep=';')
        df_vsd_raw = pd.read_csv(path_vsd, sep=',')
        df_meta_raw = pd.read_csv(path_metadata, sep=';')
        
        # -- resultados (features seleccionados por modelo) --
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
        
        # -- procesamiento muestras y etiquetas --
        df_meta = df_meta_raw.copy()
        df_meta['Sample_Base'] = df_meta['SampleID'].apply(lambda x: str(x).split('_')[0])
        
        # Mapeo Muestra -> Temporada (Clase Y) sin duplicados
        meta_grouped = df_meta.drop_duplicates(subset=['Sample_Base']).set_index('Sample_Base')
        
        # Obtenemos las muestras únicas presentes en la matriz proteómica (Muestra01, Muestra12...)
        sample_cols = [c for c in df_prot_raw.columns if c.startswith('Muestra')]
        
        # Filtramos y alineamos el vector Y asegurando el mismo orden
        self.Y = meta_grouped.loc[sample_cols, 'Temporada']
        self.Y.name = 'Class'

        # 4. Generar Matrices X de Expresión (Filas = Muestras, Columnas = Variables filtradas)
        # Usamos distribución normal para simular los valores Z-Score de las variables
        n_samples = len(sample_cols)

        self.X = {
            'mRNA': pd.DataFrame(
                np.random.normal(0, 1, size=(n_samples, len(self.names_['mRNA']))),
                index=sample_cols,
                columns=self.names_['mRNA']
            ),
            'Proteinas': pd.DataFrame(
                np.random.normal(0, 1, size=(n_samples, len(self.names_['Proteinas']))),
                index=sample_cols,
                columns=self.names_['Proteinas']
            )
        }



