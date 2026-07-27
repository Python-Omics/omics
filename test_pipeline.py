import matplotlib.pyplot as plt

from pythomics.models import MockDIABLO
from pythomics.visualizations import plot_circos


if __name__ == "__main__":
    # instanciacion de modelo a usar
    modelo_diablo = MockDIABLO(
        path_rna="replica/results/tables/Biomarcadores_RNA_Elite.csv", 
        path_prot="replica/results/tables/Biomarcadores_Proteinas_Elite.csv"
    )

    fig, ax = plot_circos(
        modelo_diablo,
        cutoff=0.7,
        block_colors={'mRNA': '#ff9999', 'Proteinas': '#99ccff'},
        title="DIABLO Circos Plot"
    )

    # plot_indiv(modelo_diablo, group=y, legend=True, title="")
    plt.show()
