"""
Implementación circosPlot


...
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pycirclize

def get_cross_corr(X, T):
    """
    Calcula correlación de Pearson entre columnas de X y columnas de T
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
    Genera un circos plot a partir de un modelo entrenado de pythomics.

    Params:
    model: objeto pythomics.models
        modelo entrenado...
    cutoff: float
        umbral de correlación para mostrar links
    components: int/tuple
        componentes a mostrar en el plot
    block_colors: dict
        diccionario que mapea nombre de bloque a color (hexadecimal o nombre de color de matplotlib).
        La cantidad de colores debe ser igual a la cantidad de bloques en el modelo.
    group_colors: dict
        diccionario que mapea nombre de grupo a color (hexadecimal o nombre de color de matplotlib).
    link_colors: tuple
        colores para los links (positivo, negativo)
    show_intra_links: bool
        si True, plotea links entre variables pertenecientes al mismo bloque.
        si False, solo plotea links entre bloques diferentes (multiómicas)
    show_var_names: bool
        si True, muestra nombres de etiquetas en borde de círculo
    title: str
        título plot
    figsize: tuple
        tamaño de la figura (ancho, alto)

    ax: matplotlib.axes.Axes
        eje donde dibujar plot. si None, se crea nuevo
    
    Returns:
    fig, ax: tuple
        Objetos figura y eje de matplotlib, para guardar o modificar
    """
    if isinstance(components, int):
        components = (1, components)

    links = []
    blocks = list(model.names_.keys())
    
    if block_colors is None:
        block_colors = {}
    if group_colors is None:
        group_colors = {}

    active_features = {}
    for block in blocks:
        raw_loadings = model.loadings_[block][:, components[0]-1:components[1]]
        
        active_index = np.where(np.any(raw_loadings != 0, axis=1))[0]
        
        # se ordena por loading en la componente 1
        sort_order = np.argsort(raw_loadings[active_index][:, 0])
        ordered_index = active_index[sort_order]
        
        # se extraen los nombres reales de los biomarcadores
        active_names = [model.names_[block][i] for i in ordered_index]
        
        X_df = model.X_dict_[block]
        X_active = X_df[active_names].values # matriz de muestras x variables activas

        active_features[block] = {
            'indices': ordered_index,
            'loadings': raw_loadings[ordered_index],
            'names': active_names,
            'X': X_active
        }

    variates = model.transform(model.X_dict_)

    # calcular correlaciones entre los biomarcadores activos
    for i in range(len(blocks)):
        start_j = i if show_intra_links else i + 1
        for j in range(start_j, len(blocks)):
            b1, b2 = blocks[i], blocks[j]
            
            X1 = active_features[b1]['X']
            T1 = variates[b1][:, components[0]-1:components[1]]

            X2 = active_features[b2]['X']
            T2 = variates[b2][:, components[0]-1:components[1]]
            
            # calcular correlaciones
            corr_X1_T1 = get_cross_corr(X1, T1)
            corr_X2_T2 = get_cross_corr(X2, T2)
            
            corr = (corr_X1_T1 @ corr_X2_T2.T)
            
            # añadir aristas segun cutoff
            rows, columns = np.where(np.abs(corr) >= cutoff)
            
            for vi, vj in zip(rows, columns):
                links.append({
                    'block1': b1, 'idx1': vi, 'var1': active_features[b1]['names'][vi],
                    'block2': b2, 'idx2': vj, 'var2': active_features[b2]['names'][vj],
                    'corr': corr[vi, vj]
                })

    # crear circos con pycirclize
    sectors = {b: len(active_features[b]['names']) for b in blocks} # se crea un sector por bloque, con tamaño cantidad de variables biomarcadores
    
    # manejar caso extremo donde se descartó todo
    if sum(sectors.values()) == 0:
        raise ValueError("Ninguna variable superó el umbral de varianza. Revisa keepX o cutoff.")
    
    circos = pycirclize.Circos(sectors, space=5)
    
    # se extraen las medias crudas para las curvas de expresión
    all_group_means = model.get_expression_means(active_features, group_colors)
    
    for sector in circos.sectors:
        
        # -- anillo de los bloques --
        track_block = sector.add_track((70, 75))
        fc_color = block_colors.get(sector.name, 'grey')
        
        # borde exterior e interior bloque, sin rellenar interior
        track_block.axis(fc="none", ec="black", lw=0.5, alpha=0.5)
        
        # etiqueta grande con nombre del bloque
        sector.text(sector.name, r=105, size=14, weight="bold")

        names = active_features[sector.name]['names']

        # se pintan sub-bloques intercalados para cada variable (para ver separación entre variables)
        for i in range(len(names)):
            current_alpha = 0.8 if i % 2 == 0 else 0.4 # se usa módulo para alternar tonalidades
            
            sector.rect(i, i+1, r_lim=(70, 75), fc=fc_color, alpha=current_alpha, ec="none")

        # -- etiquetas de variables --
        if show_var_names:
            for i, name in enumerate(names):
                sector.text(name, r=73, x=i + 0.5, orientation="vertical", size=6, color="black")

        # -- anillo exterior de expresión de cada variable --
        track_expression = sector.add_track((80, 100))
        x_pos = np.arange(len(names)) + 0.5
        
        group_means = all_group_means[sector.name]
        all_means_arr = np.array(list(group_means.values()))
        block_min, block_max = all_means_arr.min(), all_means_arr.max()

        for group_name, color in group_colors.items():
            y_expression = group_means[group_name]
            track_expression.line(x_pos, y_expression, color=color, lw=1.2, vmin=block_min, vmax=block_max)
    
    # dibujar links
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
    
    # renderizado final
    if ax is None: # si no se pasa un eje, se crea uno con figsize indicado
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'polar': True})
    else:
        fig = ax.figure # se recupera figura del eje que se pasó

    circos.plotfig(ax=ax)
    
    # se extrae eje de la figura o el que se pasó como arg
    ax_res = ax
    
    # -- textos esquinas superiores (superior izquierda - superior derecha) --
    ax_res.text(0.0, 1.05, f"Comp {components[0]}-{components[1]}", transform=ax_res.transAxes, fontsize=10)
    ax_res.text(1.0, 1.05, f"Correlation cut-off\n    r={cutoff}", transform=ax_res.transAxes, fontsize=10, ha='right')
    
    # -- leyenda de colores correlaciones (inferior izquierda) --
    pos_line = mlines.Line2D([], [], color=link_colors[0], label='Positive Correlation', linewidth=2)
    neg_line = mlines.Line2D([], [], color=link_colors[1], label='Negative Correlation', linewidth=2)

    # bbox_to_anchor empuja la leyenda fuera del centro del gráfico
    leg_corr = ax_res.legend(handles=[pos_line, neg_line], loc='lower left', title='Correlations', bbox_to_anchor=(-0.1, -0.1), frameon=False)

    ax_res.add_artist(leg_corr)

    # -- leyenda expresión / estaciones (inferior derecha) --
    
    expr_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='-', markersize=4, label=est) 
                    for est, color in group_colors.items()]

    ax_res.legend(handles=expr_handles, loc='lower right', title='Expression', bbox_to_anchor=(1.1, -0.1), frameon=False)
   
    ax_res.set_title(title, pad=30)

    return fig, ax_res


