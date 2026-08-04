"""
Implementación Volcano Plot

...
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from adjustText import adjust_text

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
    Genera un Volcano Plot
    
    Params:
    df: pandas.DataFrame
        DataFrame con los resultados del análisis diferencial
    log2fc_col: str
        Nombre de la columna con el Log2 Fold Change
    pval_col: str
        Nombre de la columna con los p-values crudos o ajustados
    gene_name_col: str
        Nombre de la columna con los nombres de los genes para etiquetar
    fc_cutoff: float
        Umbral para el Log2 Fold Change (líneas verticales)
    pval_cutoff: float
        Umbral para el p-value (línea horizontal
    top_n_labels: int
        Cantidad de genes más significativos a etiquetar automáticamente
    title: str
        Título principal del gráfico
    figsize: tuple
        Tamaño de la figura (ancho, alto
    
    Returns:
    fig, ax: tuple
        Objetos figura y eje de matplotlib, para guardar o modificar
    """
    data = df.copy()

    # evitar log10(0) reemplazando ceros absolutos con el valor flotante más pequeño
    min_pval = np.nextafter(0, 1)
    data[pval_col] = data[pval_col].replace(0, min_pval)

    data['-log10(p)'] = -np.log10(data[pval_col])

    # -- categoría 1: No significativos (gris) --
    ns_mask = (abs(data[log2fc_col]) < fc_cutoff) & (data[pval_col] >= pval_cutoff)
    
    # -- categoría 2: Solo fc significativo (verde) --
    fc_mask = (abs(data[log2fc_col]) >= fc_cutoff) & (data[pval_col] >= pval_cutoff)

    # -- categoría 3: Solo p-value significativo (azul) --
    p_mask = (abs(data[log2fc_col]) < fc_cutoff) & (data[pval_col] < pval_cutoff)

    # -- categoría 4: Ambos significativos (rojo) --
    both_mask = (abs(data[log2fc_col]) >= fc_cutoff) & (data[pval_col] < pval_cutoff)

    data['category'] = 'NS'
    data.loc[fc_mask, 'category'] = 'Log2 FC'
    data.loc[p_mask, 'category'] = 'p-value'
    data.loc[both_mask, 'category'] = 'p-value and log2 FC'

    # TODO: cambiar colores? parametrizarlos?
    
    # colores (imitando a EnhancedVolcano)
    color_map = {
        'NS': '#B3B3B3',                  # gris
        'Log2 FC': '#4DAF4A',             # verde
        'p-value': '#377EB8',             # azul
        'p-value and log2 FC': '#E41A1C'  # rojo
    }

    fig, ax = plt.subplots(figsize=figsize)
    
    # puntos se dibujan usando seaborn para mejor manejo de colores
    sns.scatterplot(
        data=data,
        x=log2fc_col,
        y='-log10(p)',
        hue='category',
        palette=color_map,
        s=30, # tamaño puntos
        alpha=0.7,
        edgecolor=None,
        ax=ax,
        legend=False, # se desactiva leyenda por defecto
    )
    
    # línea horizontal para p-value
    ax.axhline(y=-np.log10(pval_cutoff), color='black', linestyle='--', linewidth=1, alpha=0.8)

    # líneas verticales para Log2 Fold Change
    ax.axvline(x=fc_cutoff, color='black', linestyle='--', linewidth=1, alpha=0.8)
    ax.axvline(x=-fc_cutoff, color='black', linestyle='--', linewidth=1, alpha=0.8)

    texts = []
    
    # se filtran genes significativos en ambas métricas (rojos)
    sig_genes = data[data['category'] == 'p-value and log2 FC']
    
    # se ordenan por el p-value más extremo para etiquetar los "Top N"
    top_genes = sig_genes.sort_values(by=pval_col, ascending=True).head(top_n_labels)

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

    # se usa adjustText para que nombres no se solapen ni tapen puntos
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

    # se recrea leyenda manualmente
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=10)
        for k, v in color_map.items()
    ]

    # se arregla leyenda para que aparezca arriba
    ax.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.15),
        ncol=4, 
        frameon=False,
        fontsize=12,
    )
    
    # limpia bordes superior y derecho
    sns.despine()
    
    # se muestra total de variables abajo derecha
    plt.figtext(0.9, 0.05, f'total = {len(data)} variables', ha='right', fontsize=10)
    
    plt.tight_layout()

    return fig, ax 




