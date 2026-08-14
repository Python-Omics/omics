"""
Biomarker Loadings Plot implementation

...
"""
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def plot_biomarkers(
    model,
    comp=1,
    top_n=30,
    block_colors=None,
    figsize=None,
    title='Biomarker Loadings - pythomics'
):
    """
    Generates a plot showing the most important biomarkers (top_n) for each block in a model.
    It shows the absolute weight of each variable in the selected component.
    
    Parameters:
    model: pythomics.models object
        trained DIABLO/PLS model
    components: int
        component from which to extract the biomarkers
    top_n: int
        maximum number of biomarkers to show per block
    block_colors: dict
        dictionary mapping block name to color (hexadecimal or matplotlib color)
    
    figsize: tuple
        figure size (width, height)
    title: str
        plot title
    """
    blocks = list(model.names_.keys())
    n_blocks = len(blocks)

    logger.info(f"Plotting biomarker loadings | blocks: {n_blocks} | comp: {comp} | top_n: {top_n}")
    
    if block_colors is None:
        block_colors = {}
    
    if figsize is None:
        # dynamic scaling with width based on n_blocks and height based on top_n
        figsize = (5 * n_blocks, max(6, top_n * 0.2))

    fig, axes = plt.subplots(1, n_blocks, figsize=figsize, sharex=False)

    if n_blocks == 1:
        axes = [axes] # ensures it is iterable

    for ax, block in zip(axes, blocks):
        # the biomarkers are extracted using the model's method
        df = model.get_biomarkers(block=block, comp=comp)
        logger.debug(f"Block '{block}': {len(df)} biomarkers extracted (top_n={top_n}).")

        # filter to top_n and take the absolute value of the loading
        df = df.head(top_n).copy()
        df['abs_value'] = df['value'].abs()
        
        # sort in ascending order by absolute value
        df = df.sort_values('abs_value', ascending=True)
        
        color = block_colors.get(block, 'steelblue')
        
        # the plot is created
        y_positions = np.arange(len(df))
        
        # draw the lines
        ax.hlines(y=y_positions, xmin=0, xmax=df['abs_value'], color=color, alpha=0.6, linewidth=2)
        
        # draw the points
        ax.plot(df['abs_value'], y_positions, "o", color=color, markersize=7)

        # format the axes
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df.index, fontsize=8)
        
        ax.set_title(block, fontsize=14, fontweight='bold')
        ax.set_xlabel('Absolute Loading Weight', fontsize=10)

        # clean up borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('grey')
        ax.spines['bottom'].set_color('grey')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
    fig.suptitle(f"{title} - Component {comp}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    logger.info(f"Biomarker loadings plot created: {blocks}")

    return fig, axes
