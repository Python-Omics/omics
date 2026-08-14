"""
Relevance Network plot implementation

TODO: group_colors handling is missing.
if group_colors is passed, use those colors for the nodes
if block_colors is passed, use those colors for the nodes

only one of the two should be used, not both. if both are passed, use block_colors.

...
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import logging

logger = logging.getLogger(__name__)

def get_cross_corr(X, T):
    """
    Calculates Pearson correlation between the columns of X and the columns of T
    """
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    T_std = (T - T.mean(axis=0)) / (T.std(axis=0) + 1e-8)
    return (X_std.T @ T_std) / X.shape[0]

def plot_network(
    model,
    cutoff=0.5,
    components=(1, 2),
    block_colors=None,
    group_colors=None,
    link_colors=('#CE671C', '#343434'),
    show_intra_links=False,
    title='Relevance Network Plot - pythomics',
    figsize=(12, 12),
    ax=None
):
    """
    Generates a correlation network graph from the model
    
    Parameters:
    model: pythomics.models object
        trained model
    cutoff: float
        correlation threshold to show links
    components: int/tuple
        components to show in the graph
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

    blocks = list(model.names_.keys())

    logger.info(f"Plotting relevance network | blocks: {len(blocks)} | components: {components} | cutoff: {cutoff} | show_intra_links: {show_intra_links}")

    if block_colors is None:
        block_colors = {}

    # -- biomarker extraction --
    active_features = {}
    for block in blocks:
        raw_loadings = model.loadings_[block][:, components[0]-1:components[1]]
        active_index = np.where(np.any(raw_loadings != 0, axis=1))[0]
        active_names = [model.names_[block][i] for i in active_index]

        X_df = model.X_dict_[block]
        X_active = X_df[active_names].values

        logger.debug(f"Block '{block}': {len(active_names)} active variables (of {model.X_dict_[block].shape[1]}).")

        active_features[block] = {
            'indices': active_index,
            'names': active_names,
            'X': X_active
        }

    variates = model.transform(model.X_dict_)
    
    # -- graph initialization --
    G = nx.Graph()

    # add nodes with their attributes
    for block in blocks:
        for name in active_features[block]['names']:
            node_id = f"{block}::{name}"
            G.add_node(node_id, block=block, display_label=name)

    # -- calculate correlations and add edges --
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
                node1_id = f"{b1}::{active_features[b1]['names'][vi]}"
                node2_id = f"{b2}::{active_features[b2]['names'][vj]}"
                weight = corr[vi, vj]
                
                if node1_id != node2_id:
                    G.add_edge(node1_id, node2_id, weight=weight)
        
    # -- remove orphan nodes --
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    logger.debug(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; {len(isolated_nodes)} orphan nodes removed.")
    
    if len(G.nodes) == 0:
        raise ValueError("No correlation exceeded the threshold. Try reducing the cutoff.")
    
    # -- draw graph --
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # algorithm to naturally distribute nodes
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # extract attributes for rendering
    node_colors = [block_colors.get(G.nodes[n]['block'], 'grey') for n in G.nodes()]
    edges = G.edges(data=True)
    
    edge_colors = [link_colors[0] if attr['weight'] > 0 else link_colors[1] for _, _, attr in edges]

    # scale edge thickness according to correlation
    edge_weights = [abs(attr['weight']) for _, _, attr in edges]
    edge_widths = [((w - cutoff) / (1 - cutoff)) * 3 + 0.5 for w in edge_weights]
    
    # map the labels for visualization (the real name is shown, not the internal ID)
    labels = {n: G.nodes[n]['display_label'] for n in G.nodes()}
    
    # drawing
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=600, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_family="sans-serif",
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))

    # -- legends --
    ax.set_title(title, fontsize=16, fontweight='bold')

    # correlation legend
    pos_line = mlines.Line2D([], [], color=link_colors[0], label='Positive Correlation', linewidth=2)
    neg_line = mlines.Line2D([], [], color=link_colors[1], label='Negative Correlation', linewidth=2)
    leg_corr = ax.legend(handles=[pos_line, neg_line], loc='lower left', title='Correlations', frameon=False)
    ax.add_artist(leg_corr)
 
    # node legend (blocks)
    node_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=block) 
                    for block, color in block_colors.items()]
    ax.legend(handles=node_handles, loc='lower right', title='Omic Blocks', frameon=False)
    
    # hide axes
    ax.axis('off')

    logger.info(f"Relevance network plot created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    return fig, ax 
