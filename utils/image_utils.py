"""
Utilidades para procesamiento de imagenes y frames.

Este modulo proporciona funciones para ajustar frames manteniendo
el aspect ratio y otras operaciones comunes de procesamiento de imagenes.
"""
from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
from configs import settings as s


class LetterboxMeta(NamedTuple):
    """
    Metadatos del letterbox para volver de coordenadas en el lienzo fijo
    a coordenadas en la imagen original (des-letterbox).

    - aspect_ratio: factor aplicado al redimensionar (mismo para ancho y alto).
    - offset_x, offset_y: desplazamiento del contenido dentro del lienzo (barras de relleno).
    """

    aspect_ratio: float
    offset_x: int
    offset_y: int


def resize_frame(
    frame: np.ndarray,
    out_wh: tuple[int, int],
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Redimensiona un frame (H, W, C). OpenCV por defecto; RGA si ``settings.USE_RGA``.

    Args:
        frame: Array numpy (alto, ancho, canales).
        out_wh: (ancho, alto) destino, convencion OpenCV.
        interpolation: Flag ``cv2.INTER_*`` (solo aplica en ruta OpenCV legacy).
    """
    if s.USE_RGA:
        # TODO: completar con toolkit de Rockchip (RGA resize).
        return cv2.resize(frame, out_wh, interpolation=interpolation)
    return cv2.resize(frame, out_wh, interpolation=interpolation)


def letterbox_bgr(
    image_bgr: np.ndarray,
    out_wh: tuple[int, int],
    fill_value: int,
) -> tuple[np.ndarray, LetterboxMeta]:
    """
    Letterbox en espacio BGR: escala la imagen manteniendo aspect ratio, centra el
    resultado en un lienzo fijo (out_wh) y rellena el resto con ``fill_value`` por canal.

    Contrato de salida
    -------------------
    - ``canvas`` tiene forma ``(out_h, out_w, 3)``, dtype ``uint8``, BGR.
    - ``meta`` permite mapear cajas/puntos del espacio del lienzo al espacio del
      frame original:
        x_orig = (x_canvas - offset_x) / aspect_ratio
        y_orig = (y_canvas - offset_y) / aspect_ratio

    Uso actual en el repo
    ----------------------
    Preproceso auxiliar del detector **RetinaFace** (ejemplo Rockchip rknn_model_zoo,
    MobileNet 0.25 con entrada 320 y relleno **114**). Ese valor de relleno coincide
    con el demo Python oficial del Zoo.

    No es el mismo criterio que ``ajustar_frame_manteniendo_aspect_ratio``: alli el
    tamano objetivo es una ventana (max_ancho, max_alto) y el fondo es negro; aqui
    el tamano es el del tensor de entrada del modelo y el relleno es configurable.

    Otros modelos (p. ej. YOLOv8 en ``use_model_yolov8``) suelen usar **resize cuadrado**
    sin letterbox; no mezclar pipelines sin revisar metricas.

    Args:
        image_bgr: Imagen BGR (H, W, 3), ``uint8``.
        out_wh: (ancho, alto) del lienzo de salida, p. ej. (320, 320).
        fill_value: Entero 0-255 aplicado a los tres canales del fondo (tipico 114 en RetinaFace).

    Returns:
        (canvas, meta) con ``canvas`` BGR listo para convertir a RGB y alimentar la red.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr debe ser (H, W, 3) BGR")
    target_width, target_height = out_wh[0], out_wh[1]
    image_height, image_width = image_bgr.shape[:2]

    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    resized = resize_frame(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

    canvas = (np.ones((target_height, target_width, 3), dtype=np.uint8) * fill_value).astype(
        np.uint8
    )
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized

    meta = LetterboxMeta(aspect_ratio=aspect_ratio, offset_x=offset_x, offset_y=offset_y)
    return canvas, meta


def ajustar_frame_manteniendo_aspect_ratio(frame, max_ancho, max_alto):
    """
    Ajusta el frame manteniendo el aspect ratio original.
    Agrega barras negras (letterboxing/pillarboxing) si es necesario.
    
    Nota sobre rendimiento:
    - cv2.copyMakeBorder(): Mas rapido (optimizado en C++), menos codigo
    - numpy manual: Mas control, mas legible para entender el proceso
    
    Args:
        frame: Frame de video a ajustar (numpy array)
        max_ancho: Ancho maximo de la ventana
        max_alto: Alto maximo de la ventana
    
    Returns:
        Frame ajustado con barras negras si es necesario
    """
    h, w = frame.shape[:2]
    
    # Calcular el factor de escala para que quepa en la ventana
    escala_ancho = max_ancho / w
    escala_alto = max_alto / h
    escala = min(escala_ancho, escala_alto)  # Usar la escala mas pequena para que quepa
    
    # Nuevas dimensiones manteniendo aspect ratio
    nuevo_ancho = int(w * escala)
    nuevo_alto = int(h * escala)
    
    # Redimensionar manteniendo aspect ratio
    frame_redimensionado = resize_frame(
        frame, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR
    )
    
    # Crear imagen negra del tamano de la ventana usando numpy
    frame_final = np.zeros((max_alto, max_ancho, 3), dtype=np.uint8)
    
    # Calcular posicion para centrar la imagen
    y_offset = (max_alto - nuevo_alto) // 2
    x_offset = (max_ancho - nuevo_ancho) // 2
    
    # Colocar la imagen redimensionada en el centro usando indexacion numpy
    frame_final[y_offset:y_offset + nuevo_alto, x_offset:x_offset + nuevo_ancho] = frame_redimensionado
    
    return frame_final


def rotar_frame(frame, grados):
    """
    Rota el frame en el angulo especificado.
    """
    if grados == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif grados == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif grados == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        return frame


def bbox_relativo_a_absoluto(bbox_relativo, img_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """
    Convierte un bounding box de coordenadas relativas [0.0, 1.0] a coordenadas absolutas (pixeles).
    
    Esta funcion es util para convertir resultados de modelos que usan coordenadas normalizadas,
    como MediaPipe, YOLO, o cualquier modelo que devuelva coordenadas en el rango [0.0, 1.0].
    
    Por que los modelos usan coordenadas relativas?
    - Independencia de resolucion: Funciona igual en imagenes de 100x100 o 1000x1000
    - Facilita escalado: Puedes reescalar la imagen sin romper las posiciones
    - Interoperabilidad: Convencion estandar en vision por computadora
    
    Args:
        bbox_relativo: Bounding box en coordenadas relativas. Puede ser:
            - Objeto MediaPipe RelativeBoundingBox (con atributos xmin, ymin, width, height)
            - Dict con claves: xmin, ymin, width, height (todos en [0.0, 1.0])
            - Tupla (xmin, ymin, width, height) con valores en [0.0, 1.0]
        img_shape: Shape de la imagen como (alto, ancho, canales) o (alto, ancho)
    
    Returns:
        Tupla (x, y, w, h) en coordenadas absolutas (pixeles):
        - x, y: Esquina superior izquierda en pixeles
        - w, h: Ancho y alto en pixeles
    
    Ejemplo:
        # Con MediaPipe
        bbox_mp = detection.location_data.relative_bounding_box
        x, y, w, h = bbox_relativo_a_absoluto(bbox_mp, img.shape)
        
        # Con dict
        bbox_dict = {'xmin': 0.2, 'ymin': 0.3, 'width': 0.4, 'height': 0.5}
        x, y, w, h = bbox_relativo_a_absoluto(bbox_dict, img.shape)
    """
    H, W = img_shape[:2]  # Solo necesitamos alto y ancho
    
    # Manejar diferentes tipos de entrada
    if hasattr(bbox_relativo, 'xmin'):
        # Objeto MediaPipe RelativeBoundingBox
        xmin = bbox_relativo.xmin
        ymin = bbox_relativo.ymin
        width = bbox_relativo.width
        height = bbox_relativo.height
    elif isinstance(bbox_relativo, dict):
        # Dict con claves xmin, ymin, width, height
        xmin = bbox_relativo['xmin']
        ymin = bbox_relativo['ymin']
        width = bbox_relativo['width']
        height = bbox_relativo['height']
    elif isinstance(bbox_relativo, (tuple, list)) and len(bbox_relativo) == 4:
        # Tupla (xmin, ymin, width, height)
        xmin, ymin, width, height = bbox_relativo
    else:
        raise ValueError(f"Formato de bbox_relativo no soportado: {type(bbox_relativo)}")
    
    # Convertir coordenadas relativas [0.0, 1.0] a pixeles absolutos
    x = int(xmin * W)
    y = int(ymin * H)
    w = int(width * W)
    h = int(height * H)
    
    return x, y, w, h
