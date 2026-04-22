"""
Deteccion con YOLOv8 ONNX sobre una imagen (sin RKNN).

Uso:
  Desde la raiz del repo: python export_models/detect_yolov8_onnx_img.py

Requiere: ultralytics, opencv-python.

Nota: Ultralytics carga el .onnx y aplica el mismo postprocesado que con el .pt.
"""
from __future__ import annotations

from pathlib import Path
import time

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "Yolo-Weights" / "yolov8n.onnx"
IMAGES_DIR = ROOT / "images"

IMG = "lily2.jpg"

CONF_MIN = 0.25

# Segundos que la ventana permanece visible antes de esperar tecla (WSL/GUI a veces cierra muy rapido)
DELAY_SEC = 5


def main() -> None:
    if not ONNX_PATH.is_file():
        raise SystemExit(f"No se encuentra el modelo: {ONNX_PATH}")

    img_path = IMAGES_DIR / IMG
    if not img_path.is_file():
        raise SystemExit(f"No se encuentra la imagen: {img_path}")

    model = YOLO(str(ONNX_PATH))
    print(f"Modelo ONNX: {ONNX_PATH.name}")
    print(f"Imagen: {img_path}")

    frame = cv2.imread(str(img_path))
    if frame is None:
        raise SystemExit(f"No se pudo leer la imagen: {img_path}")

    results = model(frame, conf=CONF_MIN, verbose=False)
    annotated = results[0].plot()

    win = "YOLOv8 ONNX"
    cv2.imshow(win, annotated)
    cv2.waitKey(1)
    time.sleep(DELAY_SEC)
    print("Cualquier tecla para cerrar")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
