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
    LetterboxMeta,
    ajustar_frame_manteniendo_aspect_ratio,
    bbox_relativo_a_absoluto,
    letterbox_bgr,
    resize_frame,
    rotar_frame,
)

__all__ = [
    'detectar_camaras_disponibles',
    'obtener_info_camara',
    'mostrar_info_camaras',
    'ajustar_frame_manteniendo_aspect_ratio',
    'bbox_relativo_a_absoluto',
    'letterbox_bgr',
    'LetterboxMeta',
    'resize_frame',
    'rotar_frame',
]
