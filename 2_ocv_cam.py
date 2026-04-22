"""
Script basico para capturar video desde una camara web con OpenCV.

Descripcion:
-----------
Este script abre una camara web y muestra el video en tiempo real.
Es el ejemplo mas basico de captura de video desde una camara USB o webcam.

En Linux embebido (p. ej. RK3568) el indice de la webcam USB suele ser alto
(p. ej. 10 o 11), no 0; en macOS/PC a menudo 0 o 1. Ajusta CAMERA_INDEX.

La apertura (V4L2 + calentamiento) esta en utils/camera_opencv.py y la usan
tambien finals/cams_viewer.py y export_models/detect_yolov8_rknn_lite_cam_person.py.

Ejecucion:
---------
python 2_ocv_cam.py

Parametros:
----------
- CAMERA_INDEX: indice de dispositivo OpenCV (ver abajo)
- Presiona 'q' para cerrar la ventana y finalizar el script
"""
# ==============================================================================
# Modulo: OpenCV Basico
# Descripcion: Uso de Camaras
# ==============================================================================

import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.camera_opencv import abrir_camara, preparar_camara

# En RK3568 + USB UVC suele ser 10 u 11; en Windows/Mac suele 0 o 1
CAMERA_INDEX = 0

# Etiquetas opcionales para el titulo de la ventana (editar segun tu equipo)
cam_labels = {
    0: "iPhone (Continuity Camera)",
    1: "Webcam interna Mac",
    2: "Cam externa USB",
}


def titulo_ventana(indice: int) -> str:
    return cam_labels.get(indice, f"Camara indice {indice}")


webcam = abrir_camara(CAMERA_INDEX)

if webcam is None:
    print(f"No se pudo abrir la camara indice {CAMERA_INDEX}")
    print("Prueba otro CAMERA_INDEX (en RK3568 suele ser 10 u 11).")
    sys.exit(1)

if not preparar_camara(webcam):
    print("La camara abrio pero no entrego frames; prueba otro CAMERA_INDEX.")
    webcam.release()
    sys.exit(1)

nombre = titulo_ventana(CAMERA_INDEX)

while True:
    ret, frame = webcam.read()
    if not ret or frame is None:
        print("No se pudo leer frame; saliendo.")
        break

    cv2.imshow(nombre, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
