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

