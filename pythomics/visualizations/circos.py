"""
circosPlot implementation


...
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pycirclize
import logging

logger = logging.getLogger(__name__)

def get_cross_corr(X, T):
    """
    Calculates Pearson correlation between the columns of X and the columns of T
    """
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    T_std = (T - T.mean(axis=0)) / (T.std(axis=0) + 1e-8)
    return (X_std.T @ T_std) / X.shape[0]

def plot_circos(
    model,
    cutoff=0.5,
    components=(1, 2),
    block_colors=None,
    group_colors=None,
    link_colors=('#CE671C', '#343434'),
    show_intra_links=False,
    show_var_names=True,
    title='Circos Plot - pythomics',
    figsize=(10, 10),
    ax=None
):
    """
    Generates a circos plot from a trained pythomics model.

    Parameters:
    model: pythomics.models object
        trained model...
    cutoff: float
        correlation threshold to show links
    components: int/tuple
        components to show in the plot
    block_colors: dict
        dictionary mapping block name to color (hexadecimal or matplotlib color name).
        The number of colors must match the number of blocks in the model.
    group_colors: dict
        dictionary mapping group name to color (hexadecimal or matplotlib color name).
    link_colors: tuple
        colors for the links (positive, negative)
    show_intra_links: bool
        if True, plots links between variables belonging to the same block.
        if False, only plots links between different blocks (multi-omics)
    show_var_names: bool
        if True, shows label names on the circle edge
    title: str
        plot title
    figsize: tuple
        figure size (width, height)

    ax: matplotlib.axes.Axes
        axis on which to draw the plot. if None, a new one is created
    
    Returns:
    fig, ax: tuple
        matplotlib figure and axis objects, for saving or modifying
    """
    if isinstance(components, int):
        components = (1, components)

    links = []
    blocks = list(model.names_.keys())
    
    logger.info(f"Plotting circos | blocks: {len(blocks)} | components: {components} | cutoff: {cutoff} | show_intra_links: {show_intra_links}")
    
    if block_colors is None:
        block_colors = {}
    if group_colors is None:
        group_colors = {}

    active_features = {}
    for block in blocks:
        raw_loadings = model.loadings_[block][:, components[0]-1:components[1]]
        
        active_index = np.where(np.any(raw_loadings != 0, axis=1))[0]
        
        # sort by loading on component 1
        sort_order = np.argsort(raw_loadings[active_index][:, 0])
        ordered_index = active_index[sort_order]
        
        # extract the real names of the biomarkers
        active_names = [model.names_[block][i] for i in ordered_index]
        
        X_df = model.X_dict_[block]
        X_active = X_df[active_names].values # matrix of samples x active variables

        logger.debug(f"Block '{block}': {len(active_names)} active variables (of {model.X_dict_[block].shape[1]}).")

        active_features[block] = {
            'indices': ordered_index,
            'loadings': raw_loadings[ordered_index],
            'names': active_names,
            'X': X_active
        }

    variates = model.transform(model.X_dict_)

    # calculate correlations between the active biomarkers
    for i in range(len(blocks)):
        start_j = i if show_intra_links else i + 1
        for j in range(start_j, len(blocks)):
            b1, b2 = blocks[i], blocks[j]
            
            X1 = active_features[b1]['X']
            T1 = variates[b1][:, components[0]-1:components[1]]

            X2 = active_features[b2]['X']
            T2 = variates[b2][:, components[0]-1:components[1]]
            
            # calculate correlations
            corr_X1_T1 = get_cross_corr(X1, T1)
            corr_X2_T2 = get_cross_corr(X2, T2)
            
            corr = (corr_X1_T1 @ corr_X2_T2.T)
            
            # add edges according to cutoff
            rows, columns = np.where(np.abs(corr) >= cutoff)
            
            for vi, vj in zip(rows, columns):
                links.append({
                    'block1': b1, 'idx1': vi, 'var1': active_features[b1]['names'][vi],
                    'block2': b2, 'idx2': vj, 'var2': active_features[b2]['names'][vj],
                    'corr': corr[vi, vj]
                })

    logger.debug(f"Computed {len(links)} links above the {cutoff} cutoff.")

    # create circos with pycirclize
    sectors = {b: len(active_features[b]['names']) for b in blocks} # one sector is created per block, sized by the number of biomarker variables
    
    # handle extreme case where everything was discarded
    if sum(sectors.values()) == 0:
        raise ValueError("No variable exceeded the variance threshold. Review keepX or cutoff.")
    
    circos = pycirclize.Circos(sectors, space=5)
    
    # extract the raw means for the expression curves
    all_group_means = model.get_expression_means(active_features, group_colors)
    
    for sector in circos.sectors:
        
        # -- block ring --
        track_block = sector.add_track((70, 75))
        fc_color = block_colors.get(sector.name, 'grey')
        
        # outer and inner block border, without filling the interior
        track_block.axis(fc="none", ec="black", lw=0.5, alpha=0.5)
        
        # large label with the block name
        sector.text(sector.name, r=105, size=14, weight="bold")

        names = active_features[sector.name]['names']

        # paint alternating sub-blocks for each variable (to see the separation between variables)
        for i in range(len(names)):
            current_alpha = 0.8 if i % 2 == 0 else 0.4 # module is used to alternate tonalities
            
            sector.rect(i, i+1, r_lim=(70, 75), fc=fc_color, alpha=current_alpha, ec="none")

        # -- variable labels --
        if show_var_names:
            for i, name in enumerate(names):
                sector.text(name, r=73, x=i + 0.5, orientation="vertical", size=6, color="black")

        # -- outer expression ring for each variable --
        track_expression = sector.add_track((80, 100))
        x_pos = np.arange(len(names)) + 0.5
        
        group_means = all_group_means[sector.name]
        all_means_arr = np.array(list(group_means.values()))
        block_min, block_max = all_means_arr.min(), all_means_arr.max()

        for group_name, color in group_colors.items():
            y_expression = group_means[group_name]
            track_expression.line(x_pos, y_expression, color=color, lw=1.2, vmin=block_min, vmax=block_max)
    
    # draw links
    for link in links:
        color = link_colors[0] if link['corr'] > 0 else link_colors[1]
        alpha = min(max((abs(link['corr']) - cutoff) / (1 - cutoff) * 0.8 + 0.2, 0.2), 0.8)    
        
        region1_start = link['idx1'] + 0.48
        region1_end = link['idx1'] + 0.52
        
        region2_start = link['idx2'] + 0.48
        region2_end = link['idx2'] + 0.52
        
        circos.link(
            (link['block1'], region1_start, region1_end),
            (link['block2'], region2_start, region2_end),
            r1=68, r2=68,
            color=color,
            alpha=alpha
        )
    
    # final rendering
    if ax is None: # if no axis is passed, one is created with the indicated figsize
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'polar': True})
    else:
        fig = ax.figure # retrieve the figure from the passed axis

    circos.plotfig(ax=ax)
    
    # retrieve the axis from the figure or the one passed as an argument
    ax_res = ax
    
    # -- upper corner texts (upper left - upper right) --
    ax_res.text(0.0, 1.05, f"Comp {components[0]}-{components[1]}", transform=ax_res.transAxes, fontsize=10)
    ax_res.text(1.0, 1.05, f"Correlation cut-off\n    r={cutoff}", transform=ax_res.transAxes, fontsize=10, ha='right')
    
    # -- correlation color legend (bottom left) --
    pos_line = mlines.Line2D([], [], color=link_colors[0], label='Positive Correlation', linewidth=2)
    neg_line = mlines.Line2D([], [], color=link_colors[1], label='Negative Correlation', linewidth=2)

    # bbox_to_anchor pushes the legend outside the center of the plot
    leg_corr = ax_res.legend(handles=[pos_line, neg_line], loc='lower left', title='Correlations', bbox_to_anchor=(-0.1, -0.1), frameon=False)

    ax_res.add_artist(leg_corr)

    # -- expression / conditions legend (bottom right) --
    
    expr_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='-', markersize=4, label=est) 
                    for est, color in group_colors.items()]

    ax_res.legend(handles=expr_handles, loc='lower right', title='Expression', bbox_to_anchor=(1.1, -0.1), frameon=False)
   
    ax_res.set_title(title, pad=30)

    logger.info(f"Circos plot created: {len(links)} links, {sum(len(f['names']) for f in active_features.values())} active variables.")

    return fig, ax_res


