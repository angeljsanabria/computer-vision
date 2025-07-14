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

