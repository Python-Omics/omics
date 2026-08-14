"""
Volcano Plot implementation

...
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from adjustText import adjust_text
import logging

logger = logging.getLogger(__name__)

def plot_volcano(
    df,
    log2fc_col='log2FoldChange',
    pval_col='pvalue',
    gene_name_col='gene_symbol',
    fc_cutoff=2.0,
    pval_cutoff=0.05,
    top_n_labels=20,
    title='Volcano Plot',
    figsize=(10, 8)
):
    """
    Generates a Volcano Plot
    
    Parameters:
    df: pandas.DataFrame
        DataFrame with the results of the differential analysis
    log2fc_col: str
        Name of the column with the Log2 Fold Change
    pval_col: str
        Name of the column with the raw or adjusted p-values
    gene_name_col: str
        Name of the column with the gene names used for labeling
    fc_cutoff: float
        Threshold for the Log2 Fold Change (vertical lines)
    pval_cutoff: float
        Threshold for the p-value (horizontal line
    top_n_labels: int
        Number of the most significant genes to label automatically
    title: str
        Main title of the plot
    figsize: tuple
        Figure size (width, height
    
    Returns:
    fig, ax: tuple
        Matplotlib figure and axis objects, for saving or modifying
    """
    data = df.copy()

    logger.info(f"Plotting volcano | total: {len(data)} variables | fc_cutoff: {fc_cutoff} | pval_cutoff: {pval_cutoff} | top_n_labels: {top_n_labels}")

    # avoid log10(0) by replacing absolute zeros with the smallest representable float
    min_pval = np.nextafter(0, 1)
    data[pval_col] = data[pval_col].replace(0, min_pval)

    data['-log10(p)'] = -np.log10(data[pval_col])

    # -- category 1: Not significant (grey) --
    ns_mask = (abs(data[log2fc_col]) < fc_cutoff) & (data[pval_col] >= pval_cutoff)
    
    # -- category 2: Only fc significant (green) --
    fc_mask = (abs(data[log2fc_col]) >= fc_cutoff) & (data[pval_col] >= pval_cutoff)

    # -- category 3: Only p-value significant (blue) --
    p_mask = (abs(data[log2fc_col]) < fc_cutoff) & (data[pval_col] < pval_cutoff)

    # -- category 4: Both significant (red) --
    both_mask = (abs(data[log2fc_col]) >= fc_cutoff) & (data[pval_col] < pval_cutoff)

    data['category'] = 'NS'
    data.loc[fc_mask, 'category'] = 'Log2 FC'
    data.loc[p_mask, 'category'] = 'p-value'
    data.loc[both_mask, 'category'] = 'p-value and log2 FC'

    logger.debug(f"Categories -> NS: {np.sum(ns_mask)} | FC only: {np.sum(fc_mask)} | p-value only: {np.sum(p_mask)} | both: {np.sum(both_mask)}")

    # TODO: change colors? parameterize them?
    
    # colors (imitating EnhancedVolcano)
    color_map = {
        'NS': '#B3B3B3',                  # grey
        'Log2 FC': '#4DAF4A',             # green
        'p-value': '#377EB8',             # blue
        'p-value and log2 FC': '#E41A1C'  # red
    }

    fig, ax = plt.subplots(figsize=figsize)
    
    # points are drawn with seaborn for better color handling
    sns.scatterplot(
        data=data,
        x=log2fc_col,
        y='-log10(p)',
        hue='category',
        palette=color_map,
        s=30, # point size
        alpha=0.7,
        edgecolor=None,
        ax=ax,
        legend=False, # default legend is disabled
    )
    
    # horizontal line for the p-value
    ax.axhline(y=-np.log10(pval_cutoff), color='black', linestyle='--', linewidth=1, alpha=0.8)

    # vertical lines for the Log2 Fold Change
    ax.axvline(x=fc_cutoff, color='black', linestyle='--', linewidth=1, alpha=0.8)
    ax.axvline(x=-fc_cutoff, color='black', linestyle='--', linewidth=1, alpha=0.8)

    texts = []
    
    # filter genes significant in both metrics (red)
    sig_genes = data[data['category'] == 'p-value and log2 FC']
    
    # sort by the most extreme p-value to label the "Top N"
    top_genes = sig_genes.sort_values(by=pval_col, ascending=True).head(top_n_labels)

    logger.debug(f"Labeling top {len(top_genes)} significant genes.")

    for _, row in top_genes.iterrows():
        texts.append(
            ax.text(
                row[log2fc_col],
                row['-log10(p)'],
                row[gene_name_col],
                fontsize=8,
                color='black'
            )
        )

    # adjustText is used so that the names do not overlap or cover the points
    if texts:
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
            ax=ax
        )

    ax.set_title(title, fontsize=16, weight='bold', pad=20, loc='left')
    ax.text(0, 1.02, 'pythomics', transform=ax.transAxes, fontsize=12, style='italic', color='grey')
    
    ax.set_xlabel(r'$\mathregular{Log_2}$ fold change', fontsize=14)
    ax.set_ylabel(r'$\mathregular{-Log_{10}\ P}$', fontsize=14)

    # recreate the legend manually
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=10)
        for k, v in color_map.items()
    ]

    # fix the legend so it appears on top
    ax.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.15),
        ncol=4, 
        frameon=False,
        fontsize=12,
    )
    
    # clean up the top and right borders
    sns.despine()
    
    # show the total number of variables at the bottom right
    plt.figtext(0.9, 0.05, f'total = {len(data)} variables', ha='right', fontsize=10)
    
    plt.tight_layout()

    logger.info("Volcano plot created.")

    return fig, ax 




