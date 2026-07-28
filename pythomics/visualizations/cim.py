"""
Implementación CIM (Clustered Image Map) o Clustered Heatmap

...
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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
    Genera un Clustered Image Map (CIM) o Clustered Heatmap a partir de un modelo

    Params:
    model: objeto pythomics.models
        modelo entrenado...
    components: int/tuple
        componentes a mostrar en el plot
    block_colors: dict
        diccionario que mapea nombre de bloque a color (hexadecimal o nombre de color de matplotlib).
        La cantidad de colores debe ser igual a la cantidad de bloques en el modelo.
    group_colors: dict
        diccionario que mapea nombre de grupo a color (hexadecimal o nombre de color de matplotlib).
    standardize: bool
        si True, apliza z-score por columnas para normalizar escalas entre bloques
    show_var_names: bool
        si True, muestra nombres de etiquetas en borde de círculo
    show_sample_names: bool
        si True, muestra los IDs de las muestras en el eje y
    title: str
        título plot
    figsize: tuple
        tamaño de la figura (ancho, alto)
    cmap: LinearSegmentedColormap
        nombre de la paleta de colores de seaborn a usar 

    Returns:
    g: seaborn.matrix.ClusterGrid
        objeto de la cuadrícula de seaborn
    """
    if block_colors is None:
        block_colors = {}
    if group_colors is None:
        group_colors = {}

    # -- concatenar horizontalmente los bloques de datos (matrices X) de todos los bloques --
    blocks = list(model.X.keys())
    df_concat_list = []
    col_color_list = []

    for block in blocks:
        df_block = model.X[block]
        df_concat_list.append(df_block)
        
        # asignar color de bloque a cada columna del bloque
        block_color = block_colors.get(block, 'grey')
        col_color_list.extend([block_color] * df_block.shape[1])
    
    df_concat = pd.concat(df_concat_list, axis=1)
    
    # -- mapeo de colores de filas (muestras) según grupo --
    sample_groups = model.Y
    row_colors = sample_groups.map(group_colors)
    row_colors.name = "Grupo"
    
    # mapear colores de columnas (bloques ómicos)
    col_colors = pd.Series(col_color_list, index=df_concat.columns, name="Bloque")
    

    # -- render con seaborn --
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
    
    # -- texto y leyendas --
  
    g.figure.subplots_adjust(top=0.9, right=0.75) # se ajusta espacio para leyendas y título

    # titulo
    g.figure.suptitle(title, y=0.98, fontsize=14, weight="bold")
    
    # color key arriba de paleta de colores
    g.ax_cbar.set_title("Color Key", fontsize=10, pad=10)
    
    # leyenda de colores de filas y columnas
    row_handles = [
        mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=7, label=label) 
        for label, color in group_colors.items()
    ]
    
    col_handles = [
        mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=7, label=label) 
        for label, color in block_colors.items()
    ]

    # se añade leyenda de filas
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

    # se añade leyenda de columnas abajo
    leg_cols = g.ax_heatmap.legend(
        handles=col_handles, 
        title="Columns", 
        loc='center left', 
        bbox_to_anchor=(1.2, 0.4), 
        frameon=False, 
        title_fontproperties={'weight':'bold', 'size': 14},
        fontsize=12
    )

    return g



