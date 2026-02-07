"""
Script para deteccion y anonimizacion de caras en imagenes usando MediaPipe.

Descripcion:
-----------
Este script detecta caras en una imagen usando MediaPipe Face Detection y
permite anonimizarlas aplicando blur (desenfoque) sobre las regiones detectadas.
Tambien puede dibujar bounding boxes y mostrar informacion detallada de las
detecciones (keypoints, confidence scores).

Ejecucion:
---------
python 5_ocv_anonymize_face.py

Parametros:
----------
- Modifica IMG = 'lily2.jpg' para cambiar la imagen a procesar
- La imagen debe estar en la carpeta 'images/'
- PRINT_DETALLE_DETECCION: Si True, imprime informacion de cada cara detectada
- ADD_RECTANGLE_DETECCION: Si True, dibuja un rectangulo verde alrededor de cada cara
- ADD_BLUR: Si True, aplica blur sobre las caras detectadas
- EXPORT: Si True, guarda la imagen procesada en 'images/out.jpg'
"""
import cv2
import os
import mediapipe as mp
import numpy as np

# Nota: Se removio el import incorrecto 'from cv.ocv_img import ADD_RECTANGLE'
# ya que no existe el paquete cv y ADD_RECTANGLE no se usa en este script

relative_keypoints_dict = { 0 : "Ojo derecho", 1 : "Ojo izquierdo", 2 : "Nariz", 3 : "Boca",
                            4 : "Oreja derecha", 5 : "Oreja izquierda" }

# read image
IMG = 'lily2.jpg'
IMG_PATH = os.path.join('.', 'images', IMG) # '.' current dir
img = cv2.imread(IMG_PATH)

IMG_OUT = 'out.jpg'

# detect face
mp_face_detection = mp.solutions.face_detection


PRINT_DETALLE_DETECCION = True
ADD_RECTANGLE_DETECCION = True
ADD_BLUR = True

# Inicializo el modelo de deteccion de caras de mp
#   model_selection 0 or 1:
#       Si 0, detecta caras cerca de la camara (como 2 metros).
#       Si 1, es mas lejos (como 5 metros)
with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Mediapipe trabaja en RGB, por eso transformo el color space
    out = face_detection.process(img_rgb)

    if PRINT_DETALLE_DETECCION:
        if out.detections is not None:
            print(f"Se detectaron {len(out.detections)} caras.")
            cara = 0
            for detection in out.detections:
                cara = cara + 1
                print(f"\r\n####### CARA {cara} ####### ")
                print("Confidence:", detection.score[0])
                bbox = detection.location_data.relative_bounding_box
                print("Bounding Box:")
                print(" - x:", bbox.xmin)
                print(" - y:", bbox.ymin)
                print(" - width:", bbox.width)
                print(" - height:", bbox.height)

                print("Keypoints:")
                for i, kp in enumerate(detection.location_data.relative_keypoints):
                    print(f"#{i} Point {relative_keypoints_dict[i]}: ({kp.x}, {kp.y})")
                    '''
                    # Índice    Parte del rostro
                        0       Ojo derecho
                        1       Ojo izquierdo
                        2       Nariz
                        3       Boca
                        4       Oreja derecha
                        5       Oreja izquierda 
                    '''
        else:
            print("No se detectaron caras.")

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            # mediapipe no devuelve coordenadas absolutas (en píxeles)
            # sino coordenadas relativas, es decir, normalizadas en el rango [0.0, 1.0]
            # por eso hay que multiplicarlas por el h, w de las dimensiones de la imagen (shape).
            # Por que usa coordenadas relativas?
            #   Independencia de resolución: Funciona igual en imágenes de 100×100 o 1000×1000.
            #   Facilita escalado: Podés reescalar la imagen sin romper las posiciones de los puntos clave.
            #   Interoperabilidad con modelos: Muchos modelos usan esta convención en visión por computadora.

            H, W, _ = img.shape  # alto, ancho, canales
            x = int(x * W)  # convertimos la coordenada relativa en píxeles
            y = int(y * H)
            w = int(w * W)
            h = int(h * H)

            if ADD_RECTANGLE_DETECCION:
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 10)

            if ADD_BLUR:
                # repesentacion en numpy de la img:
                #   img[alto, ancho, canales] == img[filas, columnas, canales] == img[y, x, canales]
                img[y : y+h, x : x+w, :] = cv2.blur(img[y : y+h, x : x+w, :], (20, 20), )  # Mas intenso (50, 50)
# blur faces

# save image

EXPORT = False
if EXPORT:
    IMG_PATH_OUT = os.path.join('.', 'images', IMG_OUT)  # '.' current dir
    cv2.imwrite(IMG_PATH_OUT, img)

# show image
cv2.imshow('face', img)
# Esperar hasta que se presione una tecla por 10 segs
cv2.waitKey(10 * 1000)  # 0, indefinido.

# Cerrar todas las ventanas
cv2.destroyAllWindows()
