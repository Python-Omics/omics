"""
scatter plot implementation 

...
"""
import pandas as pd
from scipy.stats import alpha
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

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
    Generates a scatter plot for the results of a PCA

    Parameters:
    model: pythomics.tools.NipalsPCA object
        Trained PCA model
    metadata: pd.DataFrame
        DataFrame containing the sample metadata (index must match the model samples)
    hue: str
        Column name in 'metadata' to color the points
    style: str
        Column name in 'metadata' to assign different shapes to the points
    title: str
        Plot title
    figsize: tuple
        Figure size (width, height)
    palette: str/dict
        Seaborn color palette to use (default 'Set1')
    ax: matplotlib.axes.Axes
        Axis on which to draw. If None, a new figure and axis are created.

    Returns:
    fig, ax: tuple
        Matplotlib figure and axis objects, for saving or modifying
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    logger.info(f"Plotting PCA scatter | hue: {hue} | style: {style}")
    
    # -- extract data from the pca model --
    if hasattr(model, 'variates_'):
        pca_coords = model.variates_
    elif hasattr(model, 'transform'):
        # if a standard sklearn pca that is already fitted but without training X is passed, it would fail,
        # it is assumed that NipalsPCA is passed
        pass
    else:
        raise AttributeError("The provided model does not appear to be a valid NipalsPCA object.")

    # -- build DataFrame for plotting --
    plot_df = pd.DataFrame({
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1]
    })

    # -- add metadata --
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
    
    # -- extract explained variance if the model has it --
    if hasattr(model, 'explained_variance_ratio_'):
        ax.set_xlabel(f"PC1: {model.explained_variance_ratio_[0]*100:.0f}% varianza", fontsize=12)
        ax.set_ylabel(f"PC2: {model.explained_variance_ratio_[1]*100:.0f}% varianza", fontsize=12)
    else:
        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)

    # ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    
    # adjust the legend
    if ax.get_legend() is not None:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    logger.info("PCA scatter plot created.")

    return fig, ax 
