import os
import matplotlib.pyplot as plt

from pythomics.models import MockDIABLO
from pythomics.visualizations import plot_circos, plot_cim


if __name__ == "__main__":
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
    
    # -----------------------------------------------
    fig, ax = plot_circos(
        model=modelo_diablo,
        figsize=(10, 10),
        cutoff=0.7,
        block_colors=block_colors,
        group_colors=groups_colors,
        link_colors=('#CE671C', '#343434'),
        title="DIABLO Circos Plot"
    )

    os.makedirs("plots", exist_ok=True) # carpeta para ir guardando visualizaciones de ejemplo
    
    # guardado de figura
    fig.savefig(
        "plots/example_circos_plot.png",
        dpi=300, # res
        bbox_inches="tight", # para que no se recorte nada
        facecolor="white" # asegura fondo blanco

    )
    # ----------------------------------------------
    grid_cim = plot_cim(
        model=modelo_diablo,
        figsize=(12, 12),
        block_colors=block_colors,
        group_colors=groups_colors,
        title="Heatmap Integrado: Transcriptómica y Proteómica",
    )
    
    # guardado de figura
    grid_cim.savefig(
        "plots/example_cim_plot.png",
        dpi=300, # res,
        bbox_inches="tight",
        facecolor="white"
    )


    # plot_indiv()
    

    plt.show()
