"""
Utilidades para procesamiento de imagenes y frames.

Este modulo proporciona funciones para ajustar frames manteniendo
el aspect ratio y otras operaciones comunes de procesamiento de imagenes.
"""
import cv2
import numpy as np


def ajustar_frame_manteniendo_aspect_ratio(
    frame,
    max_ancho,
    max_alto,
    default_w: int = 800,
    default_h: int = 600,
):
    """
    Ajusta el frame manteniendo el aspect ratio original.
    Agrega barras negras (letterboxing/pillarboxing) si es necesario.

    En macOS, cv2.getWindowImageRect a veces devuelve 0x0 hasta el primer imshow;
    con max_ancho o max_alto en 0 la escala queda 0 y cv2.resize falla
    (inv_scale_x > 0). Aqui se fuerza un tamano valido.

    Args:
        frame: Frame de video a ajustar (numpy array)
        max_ancho: Ancho maximo de la ventana
        max_alto: Alto maximo de la ventana
        default_w, default_h: Fallback si la ventana aun no tiene tamano (0x0)

    Returns:
        Frame ajustado con barras negras si es necesario
    """
    def _dims_ventana(a, b) -> tuple[int, int]:
        try:
            wa = int(a)
            wh = int(b)
        except (TypeError, ValueError):
            return default_w, default_h
        if wa <= 0 or wh <= 0:
            return default_w, default_h
        return wa, wh

    max_ancho, max_alto = _dims_ventana(max_ancho, max_alto)

    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return np.zeros((max_alto, max_ancho, 3), dtype=np.uint8)

    h, w = (int(frame.shape[0]), int(frame.shape[1]))
    if h <= 0 or w <= 0 or frame.size == 0:
        return np.zeros((max_alto, max_ancho, 3), dtype=np.uint8)

    escala_ancho = max_ancho / float(w)
    escala_alto = max_alto / float(h)
    if not np.isfinite(escala_ancho) or not np.isfinite(escala_alto):
        return np.zeros((max_alto, max_ancho, 3), dtype=np.uint8)

    escala = min(escala_ancho, escala_alto)
    nuevo_ancho = max(1, int(round(w * escala)))
    nuevo_alto = max(1, int(round(h * escala)))

    frame_redimensionado = cv2.resize(
        frame, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR
    )

    frame_final = np.zeros((max_alto, max_ancho, 3), dtype=np.uint8)
    y_offset = (max_alto - nuevo_alto) // 2
    x_offset = (max_ancho - nuevo_ancho) // 2
    frame_final[y_offset : y_offset + nuevo_alto, x_offset : x_offset + nuevo_ancho] = (
        frame_redimensionado
    )
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
