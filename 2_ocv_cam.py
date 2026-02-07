"""
Script basico para capturar video desde una camara web con OpenCV.

Descripcion:
-----------
Este script abre una camara web y muestra el video en tiempo real.
Es el ejemplo mas basico de captura de video desde una camara USB o webcam.

Ejecucion:
---------
python 2_ocv_cam.py

Parametros:
----------
- Modifica la variable webcam = cv2.VideoCapture(1) para cambiar el indice
  de la camara (0, 1, 2, etc.)
- Presiona 'q' para cerrar la ventana y finalizar el script
"""
# ==============================================================================
# Módulo: OpenCV Básico
# Descripción: Uso de Camaras
# ==============================================================================

import cv2

cam_labels = {
    0: "iPhone (Continuity Camera)",
    1: "Webcam interna Mac",
    2: "Cam externa USB",
}

webcam = cv2.VideoCapture(1)

# Verificar si se pudo abrir
if not webcam.isOpened():
    print("No se pudo acceder a la webcam")
    exit()

# Siempre hay frame si abrio la camara
while True:
    ret, frame = webcam.read()

    cv2.imshow('Webcam frames', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

