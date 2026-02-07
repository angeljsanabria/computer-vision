"""
Script para deteccion de colores en tiempo real desde camara usando OpenCV.

Descripcion:
-----------
Este script detecta objetos por color en tiempo real desde una camara web.
Usa el espacio de color HSV para crear mascaras de color y detectar objetos
de un color especifico (por defecto rojo). Puede mostrar la mascara, aplicar
bitwise AND, o dibujar bounding boxes alrededor de los objetos detectados.

Ejecucion:
---------
python 4_ocv_color.py

Parametros:
----------
- Modifica cap = cv2.VideoCapture(0) para cambiar el indice de la camara
- COLOR_FN: Si True, detecta el color definido en colorFn (por defecto amarillo)
- SHOW_JUST_MASK: Si True, muestra solo la mascara binaria
- SHOW_BITWISE: Si True, muestra el resultado de aplicar la mascara
- SHOW_PIL: Si True, muestra el video con bounding boxes (por defecto activo)
- Presiona 'q' para cerrar
"""

import numpy as np
import cv2
from PIL import Image


# 0 iphone, 1 webcam
cap = cv2.VideoCapture(0)


def get_limits(color_bgr, h_margin=10, s_margin=60, v_margin=60):
    color_bgr = np.uint8([[color_bgr]])
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    lower = np.array([
        max(h - h_margin, 0),
        max(s - s_margin, 50),   # No permitir saturaciones muy bajas
        max(v - v_margin, 50)    # Ni valores muy oscuros
    ], dtype=np.uint8)

    upper = np.array([
        min(h + h_margin, 179),
        min(s + s_margin, 255),
        min(v + v_margin, 255)
    ], dtype=np.uint8)

    return lower, upper

# defino color rojo
redBajo1 = np.array([0, 140, 100], np.uint8)
redAlto1 = np.array([10, 255, 255], np.uint8)

redBajo2 = np.array([170, 140, 100], np.uint8)
redAlto2 = np.array([180, 255, 255], np.uint8)


COLOR_FN = False
colorFn = [0, 255, 255]     # Amarillo BGR


SHOW_JUST_MASK = False
SHOW_BITWISE = False
SHOW_PIL = False
SHOW_PIL = True

while True:
    ret, frame = cap.read()
    # suavizo el frame con un blurring gausiano
    if ret:
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        frameHSV = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)  # help(cv2.cvtColor): Converts an image from one color space to another.

        if COLOR_FN:
            lowerLimit, upperLimit = get_limits(color_bgr= colorFn)
            maskColor = cv2.inRange(frameHSV, lowerLimit, upperLimit )
        else:
            maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
            maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)
            maskColor = cv2.add(maskRed1, maskRed2)

        if SHOW_JUST_MASK:
            cv2.imshow('cam', maskColor)
        elif SHOW_BITWISE:
            # Aplicar la máscara a la imagen original
            result = cv2.bitwise_and(frame, frame, mask=maskColor)
            cv2.imshow('cam', result)
        elif SHOW_PIL:
            mask_ = Image.fromarray(maskColor)  # convierte en np array de ocv a Pillow
                                                # (es la misma info) en otro formato
            bbox = mask_.getbbox()              # del nuevo formato de la mascara, creo el bonding box
            print(bbox)                         # Si no encuentra nada, retorna None
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

            cv2.imshow('cam', frame)
        else:
            cv2.imshow('cam', maskColor)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

