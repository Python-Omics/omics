"""
CIM (Clustered Image Map) or Clustered Heatmap implementation

...
"""
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines
import logging

logger = logging.getLogger(__name__)

def plot_cim(
    model,
    components=(1, 2),
    block_colors=None,
    group_colors=None,
    standardize=True,
    show_var_names=True,
    show_sample_names=True,
    title="Heatmap integrado - pythomics",
    figsize=(12, 10),
    cmap="Spectral_r"
):
    """
    Generates a Clustered Image Map (CIM) or Clustered Heatmap from a model

    Parameters:
    model: pythomics.models object
        trained model...
    components: int/tuple
        components to show in the plot (used to filter selected variables)
    block_colors: dict
        dictionary mapping block name to color (hexadecimal or matplotlib color name).
        The number of colors must match the number of blocks in the model.
    group_colors: dict
        dictionary mapping group name to color (hexadecimal or matplotlib color name).
    standardize: bool
        if True, applies z-score per column to normalize scales between blocks
    show_var_names: bool
        if True, shows variable names on the x axis
    show_sample_names: bool
        if True, shows sample IDs on the y axis
    title: str
        plot title
    figsize: tuple
        figure size (width, height)
    cmap: LinearSegmentedColormap
        name of the seaborn color palette to use

    Returns:
    g: seaborn.matrix.ClusterGrid
        seaborn grid object
    """
    if block_colors is None:
        block_colors = {}
    if group_colors is None:
        group_colors = {}
    
    # ensures components is an iterable list
    if isinstance(components, int):
        components = [components]


    # -- concatenate the data blocks (X matrices) from all blocks horizontally --
    blocks = list(model.X_dict_.keys())
    logger.info(f"Plotting CIM | blocks: {len(blocks)} | components: {components} | standardize: {standardize}")
    df_concat_list = []
    col_color_list = []

    for block in blocks:
        df_block = model.X_dict_[block]

        # filter the variables selected by DIABLO (union over the requested components);
        # if the model is not fitted or the block has no biomarkers, the full block is used
        selected_features = set()
        if getattr(model, "loadings_", None) is not None:
            if block in model.loadings_:
                for comp in components:
                    if 1 <= comp <= model.loadings_[block].shape[1]:
                        biomarkers = model.get_biomarkers(block=block, comp=comp)
                        selected_features.update(biomarkers.index)
        if selected_features:
            df_block = df_block[[col for col in df_block.columns if col in selected_features]]

        logger.debug(f"Block '{block}': {df_block.shape[1]} selected features (of {model.X_dict_[block].shape[1]}).")
        df_concat_list.append(df_block)

        # assign a block color to each column of the block
        block_color = block_colors.get(block, 'grey')
        col_color_list.extend([block_color] * df_block.shape[1])

    df_concat = pd.concat(df_concat_list, axis=1).dropna()

    logger.debug(f"Concatenated matrix: {df_concat.shape[0]} samples x {df_concat.shape[1]} features (after dropna).")

    # if there are duplicated variable names between blocks, prefix them with the block name
    if df_concat.columns.duplicated().any():
        df_concat = pd.concat(
            [df.add_prefix(f"{block}::") for df, block in zip(df_concat_list, blocks)],
            axis=1
        ).dropna()

    # -- color mapping of rows (samples) by group --
    sample_groups = model.Y_
    if isinstance(sample_groups, pd.Series):
        try:
            sample_groups = sample_groups.loc[df_concat.index]
        except KeyError:
            sample_groups = pd.Series(sample_groups.to_numpy(), index=df_concat.index)
    else:
        sample_groups = pd.Series(sample_groups, index=df_concat.index)
    
    row_colors = sample_groups.map(group_colors)
    row_colors.name = "Grupo"
    
    # map column colors (omic blocks)
    col_colors = pd.Series(col_color_list, index=df_concat.columns, name="Bloque")
    

    # -- render with seaborn --
    z_param = 1 if standardize else None
    g = sns.clustermap(
        df_concat,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        z_score=z_param,
        xticklabels=show_var_names,
        yticklabels=show_sample_names,
        figsize=figsize,
        dendrogram_ratio=(0.15, 0.15),
        cbar_pos=(0.02, 0.75, 0.03, 0.15)
    )
    
    # -- text and legends --
  
    g.figure.subplots_adjust(top=0.9, right=0.75) # adjust space for legends and title

    # title
    g.figure.suptitle(title, y=0.98, fontsize=14, weight="bold")
    
    # color key above the color palette
    g.ax_cbar.set_title("Color Key", fontsize=10, pad=10)
    
    # legend for row and column colors
    row_handles = [
        mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=7, label=label) 
        for label, color in group_colors.items()
    ]
    
    col_handles = [
        mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=7, label=label) 
        for label, color in block_colors.items()
    ]

    # add row legend
    leg_rows = g.ax_heatmap.legend(
        handles=row_handles, 
        title="Rows", 
        loc='center left', 
        bbox_to_anchor=(1.2, 0.6),
        frameon=False,
        title_fontproperties={'weight':'bold', 'size': 14},
        fontsize=12
    )
    
    g.ax_heatmap.add_artist(leg_rows)

    # add column legend below
    leg_cols = g.ax_heatmap.legend(
        handles=col_handles, 
        title="Columns", 
        loc='center left', 
        bbox_to_anchor=(1.2, 0.4), 
        frameon=False, 
        title_fontproperties={'weight':'bold', 'size': 14},
        fontsize=12
    )

    logger.info("CIM plot created.")

    return g



