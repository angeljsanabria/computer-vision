"""
Script de deteccion de objetos con YOLOv8 en tiempo real desde camara.

Descripcion:
-----------
Este script carga un modelo YOLOv8 y procesa video en tiempo real desde una
camara web. Dibuja bounding boxes manualmente con OpenCV, filtra detecciones
por confianza minima y cambia el color de los rectangulos segun el nivel de
confianza de la deteccion.

Diferencia con test_yolo_basic.py:
----------------------------------
- test_yolo_basic.py: Procesa imagen/video desde archivo, usa visualizacion
  integrada de YOLO (show=True)
- Este script: Procesa video desde camara en tiempo real, dibuja manualmente
  con OpenCV, filtra por confianza y personaliza colores

Ejecucion:
---------
python test_yolo_cam.py

Parametros:
----------
- M_SMALL, M_MEDIUM, M_LARGE, M_XLARGE: Activa diferentes tamaños de modelo
  (por defecto usa nano si todos estan en False)
- CAMERA: Si True, procesa video desde camara (por defecto True)
- Modifica cap = cv2.VideoCapture(1) para cambiar el indice de la camara
- CONFIANZA_TOLERABLE: Confianza minima para mostrar detecciones (default: 75)
- Presiona 'q' o cierra la ventana para finalizar
"""
import numpy
from ultralytics import YOLO
import cv2
import cvzone
from time import sleep

COLOR_REC_VERDE = (255, 0, 255)
CONFIANZA_TOLERABLE = 75

print("Init!")

M_SMALL = False
M_MEDIUM = True
M_LARGE = False
M_XLARGE = False

if M_SMALL:
    model = YOLO('Yolo-Weights/yolov8s.pt')
    print("Modelo Small OK")
elif M_MEDIUM:
    model = YOLO('Yolo-Weights/yolov8m.pt')
    print("Modelo Medium OK")
elif M_LARGE:
    model = YOLO('Yolo-Weights/yolov8l.pt')
    print("Modelo Large OK")
elif M_XLARGE:
    model = YOLO('Yolo-Weights/yolov8x.pt')
    print("Modelo XLarge OK")
else:
    model = YOLO('Yolo-Weights/yolov8n.pt')
    print("Modelo Nano OK")

#results = model("images/messi3.jpg", show=True)


#sleep(10)ls
#cv2.waitKey(10)
print(model.names)
#print(*model.names)

CAMERA = True

if CAMERA:
    cap = cv2.VideoCapture(1)
    cap.set(3, 720)
    cap.set(4, 480)
    print("Camara OK")

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #print(f'1 //{box.xyxy[0]}')
            # Confianza de la predicción
            conf = float(box.conf[0])
            conf_percent = int(conf * 100)
            print(f'confianza = {conf_percent}')

            if conf_percent < CONFIANZA_TOLERABLE:
                break

            #print(f'Is a {model.names[int(box.cls[0])]}')
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if conf_percent > 80:
                color = (255, 0, 0)
            else:
                color = (255, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        else:
            break

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()