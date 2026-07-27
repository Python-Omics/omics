"""
Implementación circosPlot

TODOs: - asegurarse de que se pueda usar con N bloques, no solo 2
       - revisar númers mágicos y ver qué vale la pena dejar como parámetro


...
"""

import numpy as np
import matplotlib.lines as mlines
import pycirclize

def plot_circos(
    model,
    cutoff=0.5,
    components=(1, 2),
    block_colors=None,
    link_colors=('#FF0000', '#0000FF'),
    show_intra_links=False,
    show_var_names=True,
    title='Circos Plot - pythOmics',
    ax=None
):
    """
    Params:
    model: objeto pythomics.models
        modelo entrenado...
    cutoff: float
        umbral de correlación para mostrar links
    components: int/tuple
        componentes a mostrar en el plot
    block_colors: dict
        diccionario que mapea nombre de bloque a color (hexadecimal o nombre de color de matplotlib)
    link_colors: tuple
        colores para los links (positivo, negativo)
    show_intra_links: bool
        si True, plotea links entre variables pertenecientes al mismo bloque.
        si False, solo plotea links entre bloques diferentes (multiómicas)
    show_var_names: bool
        si True, muestra nombres de etiquetas en borde de círculo
    title: str
        título plot
    ax: matplotlib.axes.Axes
        eje donde dibujar plot. si None, se crea nuevo
    
    Returns:
        fig, ax: tuple
        figura y eje, para guardar
    """
    links = []
    blocks = list(model.names_.keys())
    
    if block_colors is None:
        block_colors = {}

    # Calcular correlaciones entre bloques
    for i in range(len(blocks)):
        start_j = i if show_intra_links else i + 1
        for j in range(start_j, len(blocks)):
            b1, b2 = blocks[i], blocks[j]
            
            loadings_i = model.loadings_[b1][:, components[0]-1:components[1]]
            loadings_j = model.loadings_[b2][:, components[0]-1:components[1]]
            
            corr = np.corrcoef(loadings_i, loadings_j)[:len(loadings_i), len(loadings_i):]
            
            rows, columns = np.where(np.abs(corr) >= cutoff)
            
            for vi, vj in zip(rows, columns):
                links.append({
                    'block1': b1, 'idx1': vi, 'var1': model.names_[b1][vi],
                    'block2': b2, 'idx2': vj, 'var2': model.names_[b2][vj],
                    'corr': corr[vi, vj]
                })

    # crear circos con pycirclize
    sectors = {b: len(model.names_[b]) for b in blocks} # se crea un sector por bloque, con tamaño igual a cantidad variables
    circos = pycirclize.Circos(sectors, space=5)
    
    seasons_colors = {'I': '#1f77b4', 'O': '#ff7f0e', 'P': '#c5b0d5', 'V': '#2ca02c'} # TODO: pasar colores a parámetros?
    for sector in circos.sectors:
        
        # -- anillo de los bloques --
        track_block = sector.add_track((70, 75))
        fc_color = block_colors.get(sector.name, 'grey')
        track_block.axis(fc=fc_color, alpha=0.5)
        
        # etiqueta grande con nombre del bloque
        sector.text(sector.name, r=105, size=14, weight="bold")

        names = model.names_[sector.name]

        # -- anillo exterior de expresión de cada variable --
        track_expression = sector.add_track((80, 100))
        x_pos = np.arange(len(names)) + 0.5
        
        for season, color in seasons_colors.items():
            # TODO: cuando se tenga implementación real, extraer promedio real de expresión 
            # y_expression = model.get_expression_mean(block=sector.name, group=season)
            
            base = {'I': 0.8, 'O': 0.6, 'P': 0.4, 'V': 0.2}[season]
            y_expression = base + np.random.uniform(-0.08, 0.08, size=len(names)) # por ahora aleatorio   
            
            # dibujar curva
            track_expression.line(x_pos, y_expression, color=color, lw=1.2)
        
        # -- etiquetas de variables --
        if show_var_names:
            for i, name in enumerate(names):
                sector.text(name, r=73, x=i + 0.5, orientation="vertical", size=6, color="black")
    
    # dibujar links
    for link in links:
        color = link_colors[0] if link['corr'] > 0 else link_colors[1]
        alpha = min(abs(link['corr']), 1.0) # Asegurar que alpha no pase de 1
        
        # TODO: ajustar para N regiones?

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
    fig = circos.plotfig(ax=ax)
    
    # se extrae eje de la figura o el que se pasó como arg
    ax_res = ax if ax is not None else fig.gca()
    
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
                    for est, color in seasons_colors.items()]

    ax_res.legend(handles=expr_handles, loc='lower right', title='Expression', bbox_to_anchor=(1.1, -0.1), frameon=False)
   
    ax_res.set_title(title, pad=30)

    return fig, ax_res


