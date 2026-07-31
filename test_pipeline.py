import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import anndata as ad
import mudata as md
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import seaborn as sns

from pythomics.models import MockDIABLO
from pythomics.tools import NipalsPCA
from pythomics.visualizations import plot_circos, plot_cim, plot_volcano, plot_pca

md.set_options(pull_on_update=False) # quita advertencias de futuras versiones

def load_and_preprocess_data():
    """
    Se carga, limpia y se construye el objeto MuData
    """
    # -- carga metadatos --
    metadata = pd.read_csv("replica/data/metadata.csv", sep=";", index_col=0)
    
    # NOTE: en metadata.csv cada muestra tiene forma "Muestra_01_R1", "Muestra_01_R2",
    # pero en los otros datos solo se tiene "Muestra_01". Además de que hay duplicados
    metadata.index = metadata.index.str.split('_').str[0]
    metadata = metadata[~metadata.index.duplicated(keep='first')]

    # se convierten las columnas categóricas a tipo 'category' de pandas (equivalente a factor en R)
    cols_to_factor = ["Regimen", "Empresa", "Orden", "Region", "Centro",
                      "Temporada", "Mediciones", "Organismo", "Estado",
                      "Primera_Medicion_Innovex", "Ultima_Medicion_Innovex", 
                      "Primera_Fecha_Seleccionada", "Ultima_Fecha_Seleccionada"]

    for col in cols_to_factor:
        if col in metadata.columns:
            metadata[col] = metadata[col].astype('category')
    
    # -- carga datos transcriptómicos --
    rna_raw = pd.read_csv("replica/data/salmon_counts.txt", sep="\t", skiprows=1, index_col=0)
    
    # limpieza nombres de columnas
    clean_columns = [os.path.basename(col).replace("_Aligned.sortedByCoord.out.bam", "") for col in rna_raw.columns]
    rna_raw.columns = clean_columns

    rna_raw.index.name = "GeneID"
    rna_raw.columns.name = "SampleID"

    # NOTE: IMPORTANTE!!!!!!: en R cada fila representa un gen y cada columna una muestra de cada gen. En Python es al revés,
    # cada fila es una muestra y cada columna un gen. Por eso se traspone.
    rna_counts = rna_raw.T

    # -- carga y limpieza de datos proteómicos --
    prot_raw = pd.read_csv("replica/data/matrix_proteomico.csv", sep=";", decimal=",",index_col=0)
    
    prot_raw.index.name = "ProteinID"
    prot_raw.columns.name = "SampleID"

    # se filtran solo columnas que digan "Muestra"
    prot_samples = [col for col in prot_raw.columns if "Muestra" in col]
    prot_matrix = prot_raw[prot_samples]
    
    # se filtran proteínas con más de 30% de valores faltantes
    na_percentage = prot_matrix.isna().mean(axis=1)
    prot_filtered = prot_matrix[na_percentage < 0.30].copy()

    # se reemplazan 0s por NaN
    prot_filtered = prot_filtered.replace(0, np.nan).astype(float)

    # transformación log2
    prot_log2 = prot_filtered.apply(np.log2)

    # imputación (se reemplazan NAs por valor mínimo de cada proteína)
    prot_imputed = prot_log2.apply(lambda row: row.fillna(row.min()), axis=1)
    
    # se traspone al igual que en transcriptómica := filas = muestras | columnas = proteínas
    prot_final = prot_imputed.T

    # -- uso de estructuras de datos anndata y mudata para empaquetar datos ómicos --
    
    # se busca la intersección real de muestras entre transcriptómica y proteómica
    samples_common = metadata.index.intersection(rna_counts.index).intersection(prot_final.index)

    # se filtra para usar solo muestras comunes
    common_metadata = metadata.loc[samples_common].copy()
    common_rna = rna_counts.loc[samples_common].copy()
    common_prot = prot_final.loc[samples_common].copy()
  
    # se castean como listas
    obs_names = list(samples_common.astype(str))
    var_names_rna = list(common_rna.columns.astype(str))
    var_names_prot = list(common_prot.columns.astype(str))
    
    X_rna = np.array(common_rna.values, dtype=np.float32)
    X_prot = np.array(common_prot.values, dtype=np.float32)
    
    # se limpia df de metadatos para poder guardar correctamente
    obs_df = pd.DataFrame(index=obs_names)
    for col in common_metadata.columns:
        obs_df[col] = common_metadata[col].values
    
    var_df_rna = pd.DataFrame(index=var_names_rna)
    var_df_prot = pd.DataFrame(index=var_names_prot)
    
    # se usa estructura de datos anndata para empaquetar cada ómica de forma individual
    adata_rna = ad.AnnData(X=X_rna, obs=obs_df, var=var_df_rna)
    adata_rna.layers["counts"] = X_rna.copy()
    adata_prot = ad.AnnData(X=X_prot, obs=obs_df, var=var_df_prot)
    adata_prot.layers["counts"] = X_prot.copy()

    for adata in [adata_rna, adata_prot]:
        if None in adata.layers:
            del adata.layers[None]

    # se usa estructura de datos mudata para empaquetar todas las ómicas en un solo objeto
    mdata = md.MuData({
        'mRNA': adata_rna,
        'Proteinas': adata_prot
    })

    # se guarda objeto mudata en disco, en formato h5mu
    os.makedirs("results", exist_ok=True)    
    output_path = "results/mudata_omics.h5mu"
    if os.path.exists(output_path):
        os.remove(output_path)
    mdata.write("results/mudata_omics.h5mu")

    print(mdata)
    return mdata

def run_deseq2_analysis(mdata):
    """
    Se realiza análisis diferencial de expresión (DESeq2) sobre el bloque 'mRNA' de objeto MuData
    """
    # se extraen el bloque de datos transcriptómicos
    adata_rna = mdata.mod['mRNA']
    
    if "counts" in adata_rna.layers:
        counts_df = adata_rna.to_df(layer="counts")
    elif adata_rna.X is not None:
        counts_df = adata_rna.to_df()
    else:
        raise ValueError("Matriz de transcriptómica vacía.")

    counts_df = counts_df.astype(int) # DeSeq2 requiere enteros
    metadata_df = adata_rna.obs.copy()
    metadata_df = metadata_df.dropna(subset=["Regimen", "Temporada"])

    # ~ Regimen + Temporada
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata_df,
        design_factors=["Regimen", "Temporada"]
    )
    
    # se ejecuta DESeq2
    dds.deseq2()

    # -- pca (idea es replicar comportamiento de pca en R) --
    dds_norm = dds.layers['normed_counts']
    vsd = np.log2(dds_norm + 1)

    # calcular la varianza de cada gen a través de las muestras
    gene_vars = np.var(vsd, axis=0)
    
    # seleccionar los índices de los 500 genes más variables (ntop=500 en R)
    top_500_idx = np.argsort(gene_vars)[-500:]
    
    # filtrar la matriz para usar solo esos 500 genes
    vsd_top500 = vsd[:, top_500_idx]

    pca = NipalsPCA(n_components=2)
    pca_results = pca.fit_transform(vsd_top500)
    
    plot_df = pd.DataFrame({
        'PC1': pca_results[:, 0],
        'PC2': pca_results[:, 1],
        'Temporada': metadata_df['Temporada'].values,
        'Regimen': metadata_df['Regimen'].values
    })
    
    # plot pca
    fig, ax = plot_pca(
        model=pca,
        metadata=metadata_df,
        hue='Temporada',
        style='Regimen',
    )

    os.makedirs("results/plots", exist_ok=True)
    fig.savefig("results/plots/PCA_Estandarizado_Final.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # -- iterar sobre combinaciones de temporada --
    seasons = metadata_df['Temporada'].unique()
    seasons_combinations = list(itertools.combinations(seasons, 2))

    for season_a, season_b in seasons_combinations:
        print(f"Comparando {season_a} vs {season_b}")
        
        # se hacen comparaciones entre par de temporadas
        stat_res = DeseqStats(dds, contrast=["Temporada", season_a, season_b])
        stat_res.summary()
        res_df = stat_res.results_df

        # se guardan tablas resultado
        os.makedirs("results/tables", exist_ok=True)
        res_df.to_csv(f"results/tables/Tabla_Temporada_{season_a}_vs_{season_b}.csv")

        # -- volcano plot --
        os.makedirs("results/plots", exist_ok=True)
        res_df['gene_symbol'] = res_df.index
        fig, ax = plot_volcano(
            df=res_df,
            title=f"Temporada {season_a} vs {season_b}",
            gene_name_col='gene_symbol',
            log2fc_col='log2FoldChange',
            pval_col='padj'
        )
        fig.savefig(f"results/plots/Volcano_Temporada_{season_a}_vs_{season_b}.png")
        plt.close()

if __name__ == "__main__":
    # -- carga y preprocesamiento de datos multiómicos --
    mdata = load_and_preprocess_data()
    
    # -- análisis diferencial de expresión (transcriptómica) --
    run_deseq2_analysis(mdata)


    # instanciacion de modelo a usar
    modelo_diablo = MockDIABLO(
        path_rna="replica/results/tables/Biomarcadores_RNA_Elite.csv", 
        path_prot="replica/results/tables/Biomarcadores_Proteinas_Elite.csv",
        path_proteomics="replica/data/matrix_proteomico.csv",
        path_vsd="replica/data/vsd_assay.csv",
        path_metadata="replica/data/metadata.csv"
    )


    # -- visualizaciones --
    block_colors = {
        'mRNA': '#9932CC', 
        'Proteinas': '#90EE90'
    }

    groups_colors = {
        'I': '#1f77b4', 
        'O': '#ff7f0e', 
        'P': '#c5b0d5', 
        'V': '#2ca02c'
    } 
    
    os.makedirs("plots", exist_ok=True) # carpeta para ir guardando visualizaciones de ejemplo
    # -----------------------------------------------
    # fig, ax = plot_circos(
    #     model=modelo_diablo,
    #     figsize=(10, 10),
    #     cutoff=0.7,
    #     block_colors=block_colors,
    #     group_colors=groups_colors,
    #     link_colors=('#CE671C', '#343434'),
    #     title="DIABLO Circos Plot"
    # )
    #
    # # guardado de figura
    # fig.savefig(
    #     "plots/example_circos_plot.png",
    #     dpi=300, # res
    #     bbox_inches="tight", # para que no se recorte nada
    #     facecolor="white" # asegura fondo blanco
    #
    # )
    # # ----------------------------------------------
    # grid_cim = plot_cim(
    #     model=modelo_diablo,
    #     figsize=(12, 12),
    #     block_colors=block_colors,
    #     group_colors=groups_colors,
    #     title="Heatmap Integrado: Transcriptómica y Proteómica",
    # )
    #
    # # guardado de figura
    # grid_cim.savefig(
    #     "plots/example_cim_plot.png",
    #     dpi=300, # res,
    #     bbox_inches="tight",
    #     facecolor="white"
    # )
    # # ----------------------------------------------
    # # NOTE: acá se simulan datos para Volcano plot 
    # np.random.seed(42)
    # n_genes = 10000
    #
    # # 1. Crear Nombres de Genes
    # gene_names = [f"Gene_{i}" for i in range(1, n_genes + 1)]
    #
    # # 2. Simular Log2 Fold Change (mayormente normal, con algunos extremos)
    # # Mezclamos una distribución normal estándar con dos distribuciones anchas
    # log2fc = np.concatenate([
    #     np.random.normal(0, 1.5, int(n_genes * 0.8)),     # Masa central
    #     np.random.normal(5, 2, int(n_genes * 0.1)),       # Sobre-expresados
    #     np.random.normal(-5, 2, int(n_genes * 0.1))       # Sub-expresados
    # ])
    # np.random.shuffle(log2fc)
    #
    # # 3. Simular p-values 
    # # Generamos p-values aleatorios entre 0 y 1
    # pvalues = np.random.uniform(0, 1, n_genes)
    #
    # # Hacemos que los genes con fold change extremo tengan p-values muy pequeños
    # # Esto simula la correlación natural en biología donde cambios grandes suelen ser significativos
    # extreme_mask = abs(log2fc) > 2.5
    # num_extremes = extreme_mask.sum()
    #
    # # Asignamos p-values súper pequeños (ej. entre 1e-10 y 1e-2) a los extremos
    # pvalues[extreme_mask] = np.random.uniform(1e-20, 1e-2, num_extremes)
    #
    # # 4. Construir el DataFrame
    # simulated_df = pd.DataFrame({
    #     'gene_symbol': gene_names,
    #     'log2FoldChange': log2fc,
    #     'pvalue': pvalues
    # })
    #
    # # -- volcano plot --
    # fig, ax = plot_volcano(
    #     df=simulated_df,
    #     title="Simulated Data vs Control",
    #     fc_cutoff=2.0,
    #     pval_cutoff=0.01,
    #     top_n_labels=15
    # )
    # # ------------------------------------------------
    #
    # # plot_indiv()
    #
    #
    # plt.show()
