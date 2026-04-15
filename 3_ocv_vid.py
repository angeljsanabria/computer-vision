"""
Script para cargar y reproducir un video desde archivo con OpenCV.

Descripcion:
-----------
Este script carga un video desde disco y lo reproduce frame por frame.
Muestra informacion del video (FPS, total de frames, dimensiones) y permite
saltar a frames especificos, agregar texto sobre el video, y controlar
la reproduccion.

Ejecucion:
---------
python 3_ocv_vid.py

Parametros:
----------
- MODO: "archivo" (por defecto) o "camara" para USB / V4L2 (en RK3568 el indice suele ser 10 u 11)
- CAMARA_INDICE: solo si MODO == "camara"
- VID / videos/: solo si MODO == "archivo"
- Presiona 'q' para detener
"""
# ==============================================================================
# Módulo: OpenCV Básico
# Descripción: Carga un video con OpenCV
# ==============================================================================

# ---
# En el caso de los videos:
# - Un video es una secuencia de imágenes (frames) reproducidas a cierta velocidad (FPS).
# - Cada frame es un array de NumPy, como una imagen tradicional.
# - Los videos pueden analizarse frame por frame en tiempo real o en procesamiento por lotes.

# OpenCV (Open Source Computer Vision Library)
# - Permite leer, escribir, manipular y mostrar videos.
# - Se accede al video usando `cv2.VideoCapture()`
# - Se puede obtener información como: cantidad de frames, FPS, ancho, alto, etc.
# - Instalación: `pip install opencv-python`

# ---
# 🧪 Funcionalidades implementadas
# - Carga un video desde disco
# - Obtiene propiedades como total de frames, FPS, tamaño, formato
# - Salta a un frame específico (frame 200)
# - Recorre el video mostrando los frames con temporización real
# - Imprime información de cada frame cada 100 frames
# - Escribo texto sobre el video en un rango
# - Permite detener la reproducción presionando la tecla 'q'
# - Libera recursos al finalizar

# ------------------------------------------------------------------------------

import sys

import cv2
import os

# "archivo" = lily.mp4 en videos/; "camara" = captura en vivo (USB suele ser indice 10+ en RK3568)
MODO = "archivo"
CAMARA_INDICE = 10
CALENTAMIENTO_LECTURAS = 25

VID = "lily.mp4"
VID_PATH = os.path.join(".", "videos", VID)


def abrir_camara(indice: int) -> cv2.VideoCapture | None:
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        cap = cv2.VideoCapture(indice, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(indice)
    return cap if cap.isOpened() else None


def preparar_camara(cap: cv2.VideoCapture) -> bool:
    for _ in range(CALENTAMIENTO_LECTURAS):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True
    return False


if MODO == "camara":
    video = abrir_camara(CAMARA_INDICE)
    if video is None:
        print(f"No se pudo abrir la camara indice {CAMARA_INDICE}")
        sys.exit(1)
    if not preparar_camara(video):
        print("La camara abrio pero no entrego frames; prueba otro CAMARA_INDICE")
        video.release()
        sys.exit(1)
    total_frames = 0
    fps = video.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 1 else 33
    print(f"Modo: camara indice {CAMARA_INDICE}")
    print(f"Tipo objeto: {type(video)}")
    print("Total frames: N/A (captura en vivo)")
    print(f"FPS reportado: {fps}")
    print(f"Ancho: {int(video.get(cv2.CAP_PROP_FRAME_WIDTH))}")
    print(f"Alto: {int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
else:
    video = cv2.VideoCapture(VID_PATH)
    if not video.isOpened():
        print("No se pudo abrir el video")
        sys.exit(1)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 40
    print(f"Tipo objeto: {type(video)}")
    print(f"Total frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"FPS: {video.get(cv2.CAP_PROP_FPS)}")
    print(f"Ancho: {int(video.get(cv2.CAP_PROP_FRAME_WIDTH))}")
    print(f"Alto: {int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"Formato (cuatrocc): {int(video.get(cv2.CAP_PROP_FOURCC))}")

    video.set(cv2.CAP_PROP_POS_FRAMES, 200)

# Print info de frame cada 100
INFO_FRAME = False
# Saltear frames
SKIP_FRAMES = 0
# Dibujar sobre video
# Dibujar número de frame sobre el video
ADD_TEXT = True
ADD_TEXT_RANGE = (300, 400)


# Visualize
ret = True
while ret:
    # Se leen frames del video por polling
    # frame es el frame actual < ret me indica si es valido o no *leido exitosamente
    ret, frame = video.read()
    if not ret:
        break

    if INFO_FRAME:
        if not int(video.get(cv2.CAP_PROP_POS_FRAMES)) % 100:
            print(f"Frame Num: {int(video.get(cv2.CAP_PROP_POS_FRAMES))}")
            print(f"Tipo: {type(frame)}")
            print(f"Shape: {frame.shape}")
            print(f"Canales: {frame.shape[2]}")
            print(f"Dtype: {frame.dtype}")
            print(f"Pixel (0,0): {frame[0, 0]}")

    if SKIP_FRAMES > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, video.get(cv2.CAP_PROP_POS_FRAMES) + SKIP_FRAMES)    # Mostrar cada N frames

    if ADD_TEXT:
        n = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        if n >= ADD_TEXT_RANGE[0] and n <= ADD_TEXT_RANGE[1]:
            # Dibujar número de frame sobre el video si estoy en el rango
            cv2.putText(
                frame,
                f"Frame: {n}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),  # Verde
                2,
                cv2.LINE_AA
            )

    cv2.imshow('Frame', frame)
    #cv2.waitKey(40)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

video.release() #libero la memoria donde esta el video
cv2.destroyAllWindows()

