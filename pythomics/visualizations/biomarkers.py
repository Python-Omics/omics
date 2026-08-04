"""
Implementación de Biomarker Loadings Plot

...
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_biomarkers(
    model,
    comp=1,
    top_n=30,
    block_colors=None,
    figsize=None,
    title='Biomarker Loadings - pythomics'
):
    """
    Genera un gráfico que muestra los biomarcadores más importantes (top_n) para cada bloque en un modelo.
    Se muestra el peso absoluto de cada variable en el componente seleccionado.
    
    Params:
    model: objeto pythomics.models
        modelo DIABLO/PLS entrenado
    components: int
        componente del cual extraer los biomarcadores
    top_n: int
        número máximo de biomarcadores a mostrar por bloque
    block_colors: dict
        diccionario que mapea nombre de bloque a color (hexadecimal o color de matplotlib)
    
    figsize: tuple
        tamaño de la figura (ancho, alto)
    title: str
        título del gráfico
    """
    blocks = list(model.names_.keys())
    n_blocks = len(blocks)
    
    if block_colors is None:
        block_colors = {}
    
    if figsize is None:
        # escala dinámica con ancho basado en n_blocks y alto basado en top_n
        figsize = (5 * n_blocks, max(6, top_n * 0.2))

    fig, axes = plt.subplots(1, n_blocks, figsize=figsize, sharex=False)

    if n_blocks == 1:
        axes = [axes] # se asegura que sea iterable

    for ax, block in zip(axes, blocks):
        # se extraen los biomarcadores usando el método del modelo
        df = model.get_biomarkers(block=block, comp=comp)

        # se filtra para top_n y se toma valor absoluto del loading
        df = df.head(top_n).copy()
        df['abs_value'] = df['value'].abs()
        
        # se ordena de forma ascendente por valor absoluto
        df = df.sort_values('abs_value', ascending=True)
        
        color = block_colors.get(block, 'steelblue')
        
        # se crea el plot
        y_positions = np.arange(len(df))
        
        # se dibujan las lineas
        ax.hlines(y=y_positions, xmin=0, xmax=df['abs_value'], color=color, alpha=0.6, linewidth=2)
        
        # se dibujan los puntos
        ax.plot(df['abs_value'], y_positions, "o", color=color, markersize=7)

        # se formatean los ejes
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df.index, fontsize=8)
        
        ax.set_title(block, fontsize=14, fontweight='bold')
        ax.set_xlabel('Absolute Loading Weight', fontsize=10)

        # se limpian bordes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('grey')
        ax.spines['bottom'].set_color('grey')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
    fig.suptitle(f"{title} - Component {comp}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig, axes
