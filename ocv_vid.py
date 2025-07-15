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

import cv2
import os

# Read video
VID = 'lily.mp4'
VID_PATH = os.path.join('.', 'videos', VID) # '.' current dir

video = cv2.VideoCapture(VID_PATH)
# Verificar apertura
if not video.isOpened():
    print("No se pudo abrir el video")
    exit()

# ver info
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 40  # delay entre frames x seg o default fallback
print(f"Tipo objeto: {type(video)}")
print(f"Total frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"FPS: {video.get(cv2.CAP_PROP_FPS)}")
print(f"Ancho: {int(video.get(cv2.CAP_PROP_FRAME_WIDTH))}")
print(f"Alto: {int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"Formato (cuatrocc): {int(video.get(cv2.CAP_PROP_FOURCC))}")


video.set(cv2.CAP_PROP_POS_FRAMES, 200)  # Ir directo frame 200

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

