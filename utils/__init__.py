"""
Modulo de utilidades para Computer Vision.

Este modulo proporciona funciones reutilizables para:
- Deteccion y manejo de camaras
- Procesamiento de imagenes y frames
"""

from .camera_utils import (
    detectar_camaras_disponibles,
    obtener_info_camara,
    mostrar_info_camaras
)

from .image_utils import (
    ajustar_frame_manteniendo_aspect_ratio,
    rotar_frame,
    bbox_relativo_a_absoluto
)

__all__ = [
    'detectar_camaras_disponibles',
    'obtener_info_camara',
    'mostrar_info_camaras',
    'ajustar_frame_manteniendo_aspect_ratio',
    'rotar_frame',
    'bbox_relativo_a_absoluto'
]
