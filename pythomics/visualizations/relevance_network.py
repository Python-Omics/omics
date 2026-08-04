"""
Implementación Relevance Network plot

TODO: falta manejar group_colors.
si se pasa group_colors, usar esos colores para los nodos
si se pasa block_colores, usar esos colores para los nodos

solo se deberia usar uno de los dos, no ambos. si se pasan ambos, usar block_colors.

...
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx

def get_cross_corr(X, T):
    """
    Calcula correlación de Pearson entre columnas de X y columnas de T
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
    Genera un grafo de red de correlaciones a partir del modelo
    
    Params:
    model: objeto pythomics.models
        modelo entrenado
    cutoff: float
        umbral de correlación para mostrar links
    components: int/tuple
        componentes a mostrar en el grafo
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

    blocks = list(model.names_.keys())

    if block_colors is None:
        block_colors = {}

    # -- extracción biomarcadores --
    active_features = {}
    for block in blocks:
        raw_loadings = model.loadings_[block][:, components[0]-1:components[1]]
        active_index = np.where(np.any(raw_loadings != 0, axis=1))[0]
        active_names = [model.names_[block][i] for i in active_index]

        X_df = model.X_dict_[block]
        X_active = X_df[active_names].values
        
        active_features[block] = {
            'indices': active_index,
            'names': active_names,
            'X': X_active
        }

    variates = model.transform(model.X_dict_)
    
    # -- inicialización grafo --
    G = nx.Graph()

    # añadir nodos con sus atributos
    for block in blocks:
        for name in active_features[block]['names']:
            node_id = f"{block}::{name}"
            G.add_node(node_id, block=block, display_label=name)

    # -- calcular correlaciones y añadir aristas --
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
            
            # añadir aristas según cutoff
            rows, columns = np.where(np.abs(corr) >= cutoff)
            for vi, vj in zip(rows, columns):
                node1_id = f"{b1}::{active_features[b1]['names'][vi]}"
                node2_id = f"{b2}::{active_features[b2]['names'][vj]}"
                weight = corr[vi, vj]
                
                if node1_id != node2_id:
                    G.add_edge(node1_id, node2_id, weight=weight)
        
    # -- eliminar nodos huérfanos --
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    if len(G.nodes) == 0:
        raise ValueError("Ninguna correlación superó el umbral. Intenta reducir el cutoff.")
    
    # -- dibujar grafo --
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # algoritmo para distribuir nodos naturalmente
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # extraer atributos para renderizado
    node_colors = [block_colors.get(G.nodes[n]['block'], 'grey') for n in G.nodes()]
    edges = G.edges(data=True)
    
    edge_colors = [link_colors[0] if attr['weight'] > 0 else link_colors[1] for _, _, attr in edges]

    # escalar grosor de aristas según correlación
    edge_weights = [abs(attr['weight']) for _, _, attr in edges]
    edge_widths = [((w - cutoff) / (1 - cutoff)) * 3 + 0.5 for w in edge_weights]
    
    # mapear los labels para visualización (se muestra el nombre real, no el ID interno)
    labels = {n: G.nodes[n]['display_label'] for n in G.nodes()}
    
    # dibujo
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=600, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.7)
    
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_family="sans-serif",
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))

    # -- leyendas --
    ax.set_title(title, fontsize=16, fontweight='bold')

    # leyenda correlaciones
    pos_line = mlines.Line2D([], [], color=link_colors[0], label='Positive Correlation', linewidth=2)
    neg_line = mlines.Line2D([], [], color=link_colors[1], label='Negative Correlation', linewidth=2)
    leg_corr = ax.legend(handles=[pos_line, neg_line], loc='lower left', title='Correlations', frameon=False)
    ax.add_artist(leg_corr)
 
    # leyenda de nodos (bloques)
    node_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=block) 
                    for block, color in block_colors.items()]
    ax.legend(handles=node_handles, loc='lower right', title='Omic Blocks', frameon=False)
    
    # ocultar ejes
    ax.axis('off')

    return fig, ax 
