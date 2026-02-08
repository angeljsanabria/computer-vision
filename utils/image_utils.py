"""
Utilidades para procesamiento de imagenes y frames.

Este modulo proporciona funciones para ajustar frames manteniendo
el aspect ratio y otras operaciones comunes de procesamiento de imagenes.
"""
import cv2
import numpy as np


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
    frame_redimensionado = cv2.resize(frame, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)
    
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
