from .diablo import MockDIABLO
from .diablo_wrapper import DIABLOWrapper

# TODO: hay que reemplazar después los mocks por las implementaciones reales

# se definen las funciones que se exponen al importar paquete
__all__ = [
    'MockDIABLO', # quitar luego
    'DIABLOWrapper'
]

