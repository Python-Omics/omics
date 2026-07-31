"""
Implementación scatter plot 

...
"""
import pandas as pd
from scipy.stats import alpha
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pca(
    model,
    metadata,
    hue=None,
    style=None,
    title="Análisis de Componentes Principales (PCA) - pythomics",
    figsize=(10, 8),
    palette='Set1',
    ax=None
):
    """
    Genera un gráfico de dispersión (scatter plot) para los resultados de un PCA

    Params:
    model: objeto pythomics.tools.NipalsPCA
        Modelo entrenado de PCA
    metadata: pd.DataFrame
        DataFrame que contiene metadatos de las muestras (indice debe coincidir con las muestras del modelo)
    hue: str
        Nombre de columna en 'metadata' para colorear puntos
    style: str
        Nombre de columna en 'metadata' para asignar diferentes formas a los puntos
    title: str
        Título del gráfico
    figsize: tuple
        Tamaño de la figura (ancho, alto)
    palette: str
        Paleta de colores de seaborn a utilizar (por defecto 'Set1')
    ax: matplotlib.axes.Axes
        Eje sobre el cual dibujar. Si None, se crea nueva figura y eje.

    Returns:
    fig, ax: tuple
        Objetos figura y eje de matplotlib, para guardar o modificar
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # -- extracción de datos de modelo pca --
    if hasattr(model, 'variates_'):
        pca_coords = model.variates_
    elif hasattr(model, 'transform'):
        # si se paca un pca estándar de sklearn ya fiteado pero sin X de entrenamiento, fallaría,
        # se asume que se pasa NipalsPCA
        pass
    else:
        raise AttributeError("El modelo proporcionado no parece ser un objeto NipalsPCA válido.")

    # -- construcción DataFrame para plotear --
    plot_df = pd.DataFrame({
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1]
    })

    # -- agregar metadatos --
    plot_df = pd.concat([plot_df.reset_index(drop=True), metadata.reset_index(drop=True)], axis=1)
    
    # -- scatter plot --
    sns.scatterplot(
        data=plot_df,
        x='PC1',
        y='PC2',
        hue=hue,
        style=style,
        palette=palette,
        s=150,
        alpha=0.8,
        ax=ax
    )

    ax.set_title(title, fontsize=14)
    
    # -- se extrae varianza explicada si modelo la tiene --
    if hasattr(model, 'explained_variance_ratio_'):
        ax.set_xlabel(f"PC1: {model.explained_variance_ratio_[0]*100:.0f}% varianza", fontsize=12)
        ax.set_ylabel(f"PC2: {model.explained_variance_ratio_[1]*100:.0f}% varianza", fontsize=12)
    else:
        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)

    # ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    
    # se ajusta leyenda
    if ax.get_legend() is not None:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    
    return fig, ax 
