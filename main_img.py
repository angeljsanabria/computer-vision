import numpy
from ultralytics import YOLO
import cv2
import cvzone
from time import sleep

COLOR_REC_VERDE = (255, 0, 255)
CONFIANZA_TOLERABLE = 75

print("Init!")

M_SMALL = False
M_MEDIUM = False
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



print(model.names)

IMAGE = False

if IMAGE:
    results = model(source="images/lily2.jpg", show=True)
else:
    results = model(source="videos/lily.mp4", show=True)


# mostrar con openCV
'''
results = model(source="images/lily2.jpg")
annotated = results[0].plot()
cv2.imshow("Detección YOLOv8", cv2.resize(annotated, (1280, 720)))
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

cv2.waitKey(0)  # espera indefinidamente hasta que presiones una tecla
cv2.destroyAllWindows()

print("Fin")

#sleep(10)ls
#print(*model.names)
