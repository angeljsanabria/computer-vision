'''
 Detección de Color en Imágenes
La detección de color permite encontrar píxeles o regiones de una imagen que estén dentro de un cierto rango de color.
Es una herramienta fundamental en visión por computadora, especialmente cuando se necesita:
Rastrear objetos por color
Crear máscaras de color para segmentación
Detectar señales, luces, prendas, frutas, etc.

El espacio de color HSV (Hue, Saturation, Value / Matiz, Saturación, Brillo), posee 3 componentes, similar al espacio
de color RGB. Pero se prefiere porque de forma sencilla se pueden determinar rangos de colores para detectar.
H: 0 a 179
S: 0 a 255
V: 0 a 255

para ver el mapa hsv en h y s https://omes-va.com/deteccion-de-colores/
para el rojo  0 a 8, y el segundo de 175 a 179 en H, mientras que para los componente S de 100 a 255, y V de 20 a 255

Matematicamente es aplicar una mascara logica:
mask = (Hmin <= pixel <= Hmax) and (Smin <= pixel <= Smax) and (Vmin <= pixel <= Vmax)
cv2.inRange() facilita esto Donde lo que queda dentro del rango se vuelve blanco, y el resto negro.

Esto es fundamental para tareas de segmentación, seguimiento de objetos, detección de señales, clasificación y más.

Pillow (PIL):
Pillow permite manipular imágenes a nivel de píxeles, pero no tiene funciones integradas como cv2.inRange.
En cambio, podés hacer comparaciones directamente con NumPy o pixel por pixel.

'''

import numpy as np
import cv2
from PIL import Image


# 0 iphone, 1 webcam
cap = cv2.VideoCapture(0)


def get_limits(color) -> (np.uint8, np.uint8):
    pass
    c = np.uint8([[color]]) # valor bgr para convertir en hsr
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)


    lowerLimit = hsvC[0][0][0] - 10, 100, 100
    upperLimit = hsvC[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit

# defino color rojo
redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([8, 255, 255], np.uint8)

redBajo2=np.array([175, 100, 20], np.uint8)
redAlto2=np.array([179, 255, 255], np.uint8)


COLOR_FN = False
colorFn = [0, 255, 255]     # Amarillo BGR


SHOW_JUST_MASK = False
SHOW_BITWISE = False
SHOW_PIL = False
SHOW_PIL = True

while True:
    ret, frame = cap.read()
    if ret:
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # help(cv2.cvtColor): Converts an image from one color space to another.

        if COLOR_FN:
            lowerLimit, upperLimit = get_limits(color= colorFn)
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

